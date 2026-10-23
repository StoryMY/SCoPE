"""File-format and camera adapters producing OpenCV world-to-camera records."""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image


class CameraAdapter(Protocol):
    @property
    def output_size(self) -> tuple[int, int] | None:
        """Rectified image size as (width, height), if known."""
        ...

    def prepare(self, image: Image.Image) -> tuple[Image.Image, np.ndarray]: ...


@dataclass
class PinholeCamera:
    intrinsics: np.ndarray
    size: tuple[int, int] | None = None

    @property
    def output_size(self):
        return self.size

    def prepare(self, image):
        if self.size is not None and image.size != self.size:
            raise ValueError(f"Image size {image.size} does not match camera size {self.size}")
        return image, self.intrinsics.copy()


class ColmapCamera:
    """Delegate camera-model-specific distortion math to COLMAP."""

    def __init__(self, camera):
        import pycolmap

        self.camera = camera
        self.options = pycolmap.UndistortCameraOptions()
        self.pinhole = camera.model.name in {"PINHOLE", "SIMPLE_PINHOLE"}
        if camera.model.name == "EQUIRECTANGULAR":
            raise ValueError("EQUIRECTANGULAR input requires perspective view extraction first")
        if camera.width <= 0 or camera.height <= 0 or not np.isfinite(camera.params).all():
            raise ValueError(f"Invalid COLMAP camera {camera.camera_id}")
        if not camera.verify_params():
            raise ValueError(f"Invalid parameters for COLMAP camera {camera.camera_id}")
        if camera.focal_length_x <= 0 or camera.focal_length_y <= 0:
            raise ValueError(f"Non-positive focal length for COLMAP camera {camera.camera_id}")
        try:
            self.rectified = camera if self.pinhole else pycolmap.undistort_camera(self.options, camera)
        except (ValueError, RuntimeError) as error:
            raise ValueError(f"Cannot rectify COLMAP camera model {camera.model.name}") from error

    @property
    def output_size(self):
        return self.rectified.width, self.rectified.height

    def prepare(self, image):
        import pycolmap

        expected = (self.camera.width, self.camera.height)
        if image.size != expected:
            raise ValueError(f"Image size {image.size} does not match COLMAP camera size {expected}")
        if self.pinhole:
            return image, self.camera.calibration_matrix()
        bitmap = pycolmap.Bitmap.from_array(np.asarray(image, dtype=np.uint8))
        try:
            rectified, camera = pycolmap.undistort_image(self.options, bitmap, self.camera)
        except (ValueError, RuntimeError) as error:
            raise ValueError(f"Cannot rectify COLMAP camera model {self.camera.model.name}") from error
        return Image.fromarray(rectified.to_array()), camera.calibration_matrix()


@dataclass
class Frame:
    name: str
    path: Path
    camera: CameraAdapter
    extrinsics: np.ndarray


@dataclass
class SceneSource:
    name: str
    frames: list[Frame]
    # COLMAP cameras can rectify to different sizes. Choose one scene-wide size,
    # independent of which subset is loaded as conditioning or target views.
    output_size: tuple[int, int] | None = None
    pad_pixel: int = 0


def _is_model(path: Path) -> bool:
    return any(all((path / f"{name}.{suffix}").is_file() for name in ("cameras", "images", "points3D"))
               for suffix in ("bin", "txt"))


def find_colmap_model(scene_dir: Path) -> Path:
    candidates = {p.resolve() for p in (scene_dir, scene_dir / "sparse", scene_dir / "colmap") if _is_model(p)}
    sparse = scene_dir / "sparse"
    if sparse.is_dir():
        candidates.update(p.resolve() for p in sparse.iterdir() if p.is_dir() and _is_model(p))
    if len(candidates) != 1:
        if not candidates:
            raise FileNotFoundError(f"No COLMAP model found in {scene_dir}; use --colmap-model to specify it")
        raise ValueError(f"Multiple COLMAP models found in {scene_dir}; select one with --colmap-model")
    return candidates.pop()


def read_colmap(scene_dir: Path, images_dir: Path, model_dir: Path | None) -> SceneSource:
    import pycolmap

    model_dir = model_dir if model_dir is not None else find_colmap_model(scene_dir)
    if not _is_model(model_dir):
        raise FileNotFoundError(f"Incomplete COLMAP model: {model_dir}")
    reconstruction = pycolmap.Reconstruction(str(model_dir))
    images = sorted((i for i in reconstruction.images.values() if i.has_pose), key=lambda i: i.name)
    if not images:
        raise ValueError(f"COLMAP model has no registered images: {model_dir}")
    frames, cameras, names = [], {}, set()
    image_root = images_dir.resolve()
    for image in images:
        path = (image_root / image.name).resolve()
        if not path.is_relative_to(image_root) or image.name in names:
            raise ValueError(f"Invalid or duplicate COLMAP image name: {image.name}")
        names.add(image.name)
        if image.camera_id not in cameras:
            cameras[image.camera_id] = ColmapCamera(reconstruction.cameras[image.camera_id])
        pose = image.cam_from_world()
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = pose.rotation.matrix()
        extrinsics[:3, 3] = pose.translation
        frames.append(Frame(image.name, path, cameras[image.camera_id], extrinsics))
    return SceneSource(scene_dir.name, frames, frames[0].camera.output_size)



def normalize_poses(poses, unit_baseline=False):
    """Express OpenCV c2w poses relative to the first selected camera."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError("Expected finite [N,4,4] poses")
    reference = poses[0].copy()
    relative = np.linalg.inv(reference)[None] @ poses
    baseline = float(np.linalg.norm(relative[:, :3, 3], axis=-1).max())
    scale = 1.0 / baseline if unit_baseline and baseline > 1e-8 else 1.0
    relative[:, :3, 3] *= scale
    return relative.astype(np.float32), {"reference_c2w": reference.tolist(), "translation_scale": scale}


def prepare_scene(scene, output, num_frames=81, width=None, height=None,
                  model=None, images=None, start=0, end=None, unit_baseline=False, reference_index=None):
    source = read_colmap(Path(scene), Path(images) if images else Path(scene) / "images",
                         Path(model) if model else None)
    end = len(source.frames) - 1 if end is None else end
    if not 0 <= start <= end < len(source.frames):
        raise ValueError("Invalid inclusive start/end frame indices")
    target_count = num_frames if reference_index is None else num_frames - 1
    if reference_index is not None and not 0 <= reference_index < len(source.frames):
        raise ValueError("Invalid reference index")
    if end - start + 1 < target_count:
        raise ValueError(f"Select at least {target_count} registered frames")
    indices = np.rint(np.linspace(start, end, target_count)).astype(int)
    if reference_index is not None:
        indices = np.concatenate(([reference_index], indices))
    frames = [source.frames[i] for i in indices]
    if (width is None) != (height is None):
        raise ValueError("Specify both width and height")
    if width is None:
        ref_w, ref_h = frames[0].camera.output_size
        width, height = (480, 832) if ref_h > ref_w else (832, 480)
    if width <= 0 or height <= 0 or width % 16 or height % 16:
        raise ValueError("Width and height must be positive multiples of 16")
    poses, normalization = normalize_poses(
        [np.linalg.inv(frame.extrinsics) for frame in frames], unit_baseline)
    intrinsics = []
    first_image = None
    for i, frame in enumerate(frames):
        camera = frame.camera
        w, h = camera.output_size
        k = camera.rectified.calibration_matrix().copy()
        if i == 0:
            with Image.open(frame.path) as image:
                first_image, k = camera.prepare(image.convert("RGB"))
            w, h = first_image.size
        # Cover resize and center crop: preserve aspect ratio and update calibration.
        scale = max(width / w, height / h)
        resized_w, resized_h = int(np.ceil(w * scale)), int(np.ceil(h * scale))
        left, top = (resized_w - width) // 2, (resized_h - height) // 2
        k[0] *= resized_w / w
        k[1] *= resized_h / h
        k[0, 2] -= left
        k[1, 2] -= top
        if i == 0:
            first_image = first_image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
            first_image = first_image.crop((left, top, left + width, top + height))
        intrinsics.append(k)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    first_image.save(output / "first_frame.png")
    np.save(output / "camera_poses.npy", poses[:, :3])
    np.save(output / "intrinsics.npy", np.asarray(intrinsics, dtype=np.float32))
    (output / "scene.json").write_text(json.dumps({
        "frames": [frame.name for frame in frames], "indices": indices.tolist(),
        "camera_normalization": normalization, "width": width, "height": height,
    }, indent=2), encoding="utf-8")
    return poses[:, :3], np.asarray(intrinsics, dtype=np.float32)
