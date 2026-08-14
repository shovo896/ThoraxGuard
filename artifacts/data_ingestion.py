"""Create a dataset manifest without copying or moving the source images."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Read and validate the data-ingestion section of the project config."""
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict) or "data_ingestion" not in config:
        raise ValueError(f"Missing data_ingestion configuration in {config_path}")
    return config["data_ingestion"]


def build_manifest(source_dir: Path) -> dict:
    """Return deterministic class and file metadata for the external dataset."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {source_dir}")

    class_dirs = [path for path in sorted(source_dir.iterdir()) if path.is_dir()]
    if len(class_dirs) < 2:
        raise ValueError(f"Expected at least two class folders in {source_dir}")

    class_counts: dict[str, int] = {}
    files: list[dict[str, int | str]] = []
    for class_dir in class_dirs:
        class_files = [
            image_path
            for image_path in sorted(class_dir.rglob("*"))
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        class_counts[class_dir.name] = len(class_files)
        files.extend(
            {
                "path": str(image_path.relative_to(source_dir)),
                "size_bytes": image_path.stat().st_size,
            }
            for image_path in class_files
        )

    if not files:
        raise ValueError(f"No supported image files found in {source_dir}")

    return {
        "source_dir": str(source_dir),
        "class_counts": class_counts,
        "total_images": len(files),
        "total_image_bytes": sum(file["size_bytes"] for file in files),
        "files": files,
    }


def run() -> Path:
    """Scan the external dataset and write its manifest inside artifacts/."""
    data_config = load_config()
    source_dir = Path(data_config["source_dir"]).expanduser().resolve()
    artifacts_root = (PROJECT_ROOT / data_config["root_dir"]).resolve()
    manifest_path = (PROJECT_ROOT / data_config["manifest_file"]).resolve()
    if not manifest_path.is_relative_to(artifacts_root):
        raise ValueError("data_ingestion.manifest_file must be inside data_ingestion.root_dir")
    artifacts_root.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(source_dir)
    temporary_path = manifest_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_path.replace(manifest_path)

    print(f"Dataset kept at: {source_dir}")
    print(f"Manifest written to: {manifest_path}")
    print(f"Class counts: {manifest['class_counts']}; total images: {manifest['total_images']}")
    return manifest_path


if __name__ == "__main__":
    run()
