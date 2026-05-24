from __future__ import annotations

import hashlib
import json
import random
import tarfile
from pathlib import Path
from typing import TypedDict
from urllib.request import urlretrieve


class ManifestRecord(TypedDict):
    sample_id: str
    path: str
    synset: str
    label: int


class PreparedManifest(TypedDict):
    split: str
    path: str
    metadata_path: str
    sample_count: int
    sha256: str


class PreparedImagenette(TypedDict):
    dataset_root: str
    val_root: str
    manifests: list[PreparedManifest]


class DatasetPreparationError(RuntimeError):
    pass


IMAGENETTE_160_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENETTE_160_MD5 = "e793b78cc4c9e9a4ccc0c1155377a412"
IMAGENETTE_160_DIRNAME = "imagenette2-160"
IMAGENETTE_MANIFEST_SEED = 1337
IMAGENETTE_EVAL_SPLITS = {"eval20": 20, "eval500": 500}

IMAGENETTE_LABELS: dict[str, int] = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp"}


def resolve_imagenette160_roots(root: Path) -> tuple[Path, Path]:
    """Return the dataset root and validation root for a root or val directory."""
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    if (root / "val").is_dir():
        return root, root / "val"
    if root.name == "val" and root.is_dir():
        return root.parent, root
    if _contains_label_dirs(root):
        return root, root
    raise FileNotFoundError(f"expected Imagenette val folders under: {root}")


def validate_imagenette_labels(root: Path, *, require_all_labels: bool = False) -> None:
    _, val_root = resolve_imagenette160_roots(root)
    label_dirs = sorted(path.name for path in val_root.iterdir() if path.is_dir())
    unmapped = sorted(set(label_dirs) - set(IMAGENETTE_LABELS))
    if unmapped:
        raise DatasetPreparationError(f"unmapped Imagenette label folders: {', '.join(unmapped)}")

    if require_all_labels:
        missing = sorted(set(IMAGENETTE_LABELS) - set(label_dirs))
        if missing:
            raise DatasetPreparationError(f"missing Imagenette label folders: {', '.join(missing)}")


def build_imagenette_manifest(
    root: Path,
    *,
    limit: int | None,
    seed: int,
    path_root: Path | None = None,
) -> list[ManifestRecord]:
    validate_imagenette_labels(root)
    _, val_root = resolve_imagenette160_roots(root)

    records: list[ManifestRecord] = []
    for synset, label in IMAGENETTE_LABELS.items():
        for image_path in sorted((val_root / synset).glob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            path = image_path
            if path_root is not None:
                path = image_path.relative_to(path_root)
            records.append(
                {
                    "sample_id": image_path.stem,
                    "path": path.as_posix() if path_root is not None else str(path),
                    "synset": synset,
                    "label": label,
                }
            )

    rng = random.Random(seed)
    rng.shuffle(records)
    if limit is not None:
        records = records[:limit]
    return records


def load_imagenette_manifest(manifest_path: Path, *, dataset_root: Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            synset = raw.get("synset")
            label = raw.get("label")
            if synset not in IMAGENETTE_LABELS:
                raise DatasetPreparationError(
                    f"manifest line {line_number} has unmapped Imagenette label: {synset}"
                )
            if label != IMAGENETTE_LABELS[synset]:
                raise DatasetPreparationError(
                    f"manifest line {line_number} has label {label} for {synset}, "
                    f"expected {IMAGENETTE_LABELS[synset]}"
                )

            image_path = Path(raw["path"])
            if not image_path.is_absolute():
                image_path = dataset_root / image_path
            records.append(
                {
                    "sample_id": str(raw["sample_id"]),
                    "path": str(image_path),
                    "synset": synset,
                    "label": label,
                }
            )
    return records


def prepare_imagenette160(
    root: Path,
    *,
    manifests_dir: Path,
    download: bool,
) -> PreparedImagenette:
    dataset_root, val_root = _ensure_imagenette160(root, download=download)
    validate_imagenette_labels(val_root, require_all_labels=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    prepared: list[PreparedManifest] = []
    for split, limit in IMAGENETTE_EVAL_SPLITS.items():
        records = build_imagenette_manifest(
            val_root,
            limit=limit,
            seed=IMAGENETTE_MANIFEST_SEED,
            path_root=dataset_root,
        )
        if len(records) < limit:
            raise DatasetPreparationError(
                f"{split} requires {limit} images, found {len(records)} in {val_root}"
            )

        manifest_path = manifests_dir / f"{IMAGENETTE_160_DIRNAME}-{split}.jsonl"
        _write_jsonl_manifest(manifest_path, records)
        digest = manifest_sha256(manifest_path)
        metadata_path = manifest_path.with_suffix(".metadata.json")
        metadata = {
            "dataset": "imagenette",
            "variant": "160px",
            "split": split,
            "sample_count": len(records),
            "seed": IMAGENETTE_MANIFEST_SEED,
            "manifest_sha256": digest,
            "manifest_path": str(manifest_path),
            "dataset_root": str(dataset_root),
            "val_root": str(val_root),
            "source_url": IMAGENETTE_160_URL,
            "source_md5": IMAGENETTE_160_MD5,
            "label_mapping": IMAGENETTE_LABELS,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        prepared.append(
            {
                "split": split,
                "path": str(manifest_path),
                "metadata_path": str(metadata_path),
                "sample_count": len(records),
                "sha256": digest,
            }
        )

    return {
        "dataset_root": str(dataset_root),
        "val_root": str(val_root),
        "manifests": prepared,
    }


def manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_imagenette160(root: Path, *, download: bool) -> tuple[Path, Path]:
    try:
        return resolve_imagenette160_roots(root)
    except FileNotFoundError:
        if not download:
            raise
        return _download_imagenette160(root)


def _download_imagenette160(root: Path) -> tuple[Path, Path]:
    if root.name == "val":
        raise DatasetPreparationError("download root must be the imagenette2-160 dataset directory")
    if root.name != IMAGENETTE_160_DIRNAME:
        raise DatasetPreparationError(
            f"download root must end with {IMAGENETTE_160_DIRNAME}: {root}"
        )

    root.parent.mkdir(parents=True, exist_ok=True)
    archive_path = root.parent / f"{IMAGENETTE_160_DIRNAME}.tgz"
    if not archive_path.exists():
        urlretrieve(IMAGENETTE_160_URL, archive_path)

    actual_md5 = _file_md5(archive_path)
    if actual_md5 != IMAGENETTE_160_MD5:
        raise DatasetPreparationError(
            f"Imagenette archive md5 mismatch: expected {IMAGENETTE_160_MD5}, got {actual_md5}"
        )

    _safe_extract_tar(archive_path, root.parent)
    return resolve_imagenette160_roots(root)


def _write_jsonl_manifest(path: Path, records: list[ManifestRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")


def _file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract_tar(archive_path: Path, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = (target_dir / member.name).resolve()
            if not member_path.is_relative_to(target_root):
                raise DatasetPreparationError(f"unsafe archive member path: {member.name}")
        archive.extractall(target_dir)


def _contains_label_dirs(root: Path) -> bool:
    if not root.is_dir():
        return False
    return any((root / synset).is_dir() for synset in IMAGENETTE_LABELS)
