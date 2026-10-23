"""CPU integration checks for the standalone COLMAP adapter."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import pycolmap

spec = importlib.util.spec_from_file_location("colmap_adapter_test", Path(__file__).resolve().parents[1] / "scope/colmap_scene.py")
scene = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scene
spec.loader.exec_module(scene)


class ColmapTests(unittest.TestCase):
    def test_models_and_normalization(self):
        for model in ("PINHOLE", "OPENCV"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sparse = root / "sparse/0"
                sparse.mkdir(parents=True)
                (root / "images").mkdir()
                camera = pycolmap.Camera.create_from_model_name(1, model, 25., 32, 24)
                if model == "OPENCV":
                    params = camera.params.copy()
                    params[4] = .05
                    camera.params = params
                (sparse / "cameras.txt").write_text("1 " + model + " 32 24 " + " ".join(map(str, camera.params)) + "\n")
                # File order differs from filename order; camera starts away from origin.
                (sparse / "images.txt").write_text("9 1 0 0 0 5 2 3 1 b.png\n\n3 1 0 0 0 1 2 3 1 a.png\n\n")
                (sparse / "points3D.txt").write_text("")
                Image.new("RGB", (32, 24)).save(root / "images/a.png")
                poses, k = scene.prepare_scene(root, root / "out", num_frames=2)
                np.testing.assert_allclose(poses[0], np.eye(4)[:3], atol=1e-6)
                np.testing.assert_allclose(poses[1, :, 3], [-4, 0, 0])
                self.assertEqual(k.shape, (2, 3, 3))
                with Image.open(root / "out/first_frame.png") as exported:
                    self.assertEqual(exported.size, (832, 480))
                scaled, _ = scene.prepare_scene(root, root / "scaled", num_frames=2, unit_baseline=True)
                np.testing.assert_allclose(scaled[1, :, 3], [-1, 0, 0])
                Image.new("RGB", (32, 24)).save(root / "images/b.png")
                chosen, _ = scene.prepare_scene(root, root / "reference", num_frames=2,
                                                 reference_index=1, start=0, end=0)
                np.testing.assert_allclose(chosen[0], np.eye(4)[:3], atol=1e-6)
                np.testing.assert_allclose(chosen[1, :, 3], [4, 0, 0])
                with self.assertRaises(ValueError):
                    scene.prepare_scene(root, root / "invalid", num_frames=2, reference_index=2)
                portrait, portrait_k = scene.prepare_scene(
                    root, root / "portrait", num_frames=2, width=480, height=832)
                with Image.open(root / "portrait/first_frame.png") as exported:
                    self.assertEqual(exported.size, (480, 832))
                np.testing.assert_allclose(portrait, poses)
                if model == "PINHOLE":
                    # 32x24 -> 1110x832 -> center crop to 480x832.
                    np.testing.assert_allclose(portrait_k[0, 0, 0], 25 * 1110 / 32)
                    np.testing.assert_allclose(portrait_k[0, 0, 2], 240)
                with self.assertRaises(ValueError):
                    scene.prepare_scene(root, root / "badsize", num_frames=2, width=480)
                binary = root / "binary"
                binary.mkdir()
                pycolmap.Reconstruction(str(sparse)).write_binary(str(binary))
                poses2, k2 = scene.prepare_scene(root, root / "binout", num_frames=2, model=binary)
                np.testing.assert_allclose(poses2, poses)
                np.testing.assert_allclose(k2, k)
                with self.assertRaises(ValueError):
                    scene.prepare_scene(root, root / "bad", num_frames=81)

    def test_rotated_reference(self):
        first = np.eye(4)
        first[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        first[:3, 3] = [5, 6, 7]
        second = first.copy()
        second[:3, 3] += [0, 2, 0]
        relative, _ = scene.normalize_poses([first, second])
        np.testing.assert_allclose(relative[0], np.eye(4), atol=1e-6)
        np.testing.assert_allclose(relative[1, :3, 3], [2, 0, 0])
        pure, info = scene.normalize_poses([first, first], True)
        self.assertEqual(info["translation_scale"], 1.)
        self.assertTrue(np.isfinite(pure).all())


if __name__ == "__main__":
    unittest.main()
