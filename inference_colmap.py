"""Generate from a COLMAP scene; preparation also works without torch/model weights."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--colmap-model", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--reference-index", type=int, help="Reference image index in filename-sorted registered images")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, help="Inclusive index in filename-sorted registered images")
    parser.add_argument("--unit-baseline", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepared-dir", type=Path, default=Path("outputs/colmap_input"))
    parser.add_argument("--prompt")
    parser.add_argument("--model_path", default="TencentARC/SCoPE")
    parser.add_argument("--output_path", type=Path, default=Path("outputs/colmap.mp4"))
    parser.add_argument("--cache_dir", type=Path)
    parser.add_argument("--vram_limit_gb", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative_prompt", type=Path, default=Path("configs/negative_prompt.txt"))
    args = parser.parse_args()
    if not args.prepare_only and not args.prompt:
        parser.error("--prompt is required for generation")
    # Load the CPU adapter directly: scope.__init__ imports torch-dependent modules.
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("scope_colmap_adapter", Path(__file__).parent / "scope/colmap_scene.py")
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    try:
        pose, intrinsics = adapter.prepare_scene(
            args.scene, args.prepared_dir, model=args.colmap_model,
            images=args.images_dir, start=args.start, end=args.end,
            unit_baseline=args.unit_baseline, reference_index=args.reference_index)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.error(str(error))
    print(f"Prepared {len(pose)} frames in {args.prepared_dir}")
    if args.prepare_only:
        return
    from scope.config import InferenceConfig
    from scope.inference import _prepare_device, generate
    from scope.weights import load_pipeline, resolve_model_dir
    import numpy as np
    config = InferenceConfig(seed=args.seed)
    example = {
        "first_frame": args.prepared_dir / "first_frame.png",
        "caption": args.prompt,
        "intrinsics": intrinsics,
        "x_fov": float(2 * np.arctan(config.width / (2 * intrinsics[0, 0, 0]))),
    }
    pipe = load_pipeline(resolve_model_dir(args.model_path, args.cache_dir), config)
    device = _prepare_device(pipe, args.vram_limit_gb)
    generate(pipe, example, args.output_path, config,
             args.negative_prompt.read_text(encoding="utf-8").strip(), device, pose=pose)


if __name__ == "__main__":
    main()
