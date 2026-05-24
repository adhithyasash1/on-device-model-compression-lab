import json
from pathlib import Path

import pytest

from mlclab.data import (
    IMAGENETTE_LABELS,
    DatasetPreparationError,
    build_imagenette_manifest,
    load_imagenette_manifest,
    manifest_sha256,
    prepare_imagenette160,
)


def test_build_imagenette_manifest(tmp_path: Path) -> None:
    synset = "n01440764"
    folder = tmp_path / synset
    folder.mkdir()
    (folder / "sample.JPEG").write_bytes(b"not a real image")
    (folder / "ignore.txt").write_text("skip", encoding="utf-8")

    records = build_imagenette_manifest(tmp_path, limit=10, seed=1)

    assert records == [
        {
            "sample_id": "sample",
            "path": str(folder / "sample.JPEG"),
            "synset": synset,
            "label": IMAGENETTE_LABELS[synset],
        }
    ]


def test_build_imagenette_manifest_refuses_unmapped_label(tmp_path: Path) -> None:
    mapped = tmp_path / "n01440764"
    mapped.mkdir()
    (mapped / "mapped.JPEG").write_bytes(b"image")
    unmapped = tmp_path / "n00000000"
    unmapped.mkdir()
    (unmapped / "unknown.JPEG").write_bytes(b"image")

    with pytest.raises(DatasetPreparationError, match="unmapped Imagenette label folders"):
        build_imagenette_manifest(tmp_path, limit=1, seed=1)


def test_prepare_imagenette160_writes_eval_manifests_and_hashes(tmp_path: Path) -> None:
    dataset_root = tmp_path / "imagenette2-160"
    _write_fake_imagenette_val(dataset_root, images_per_label=50)
    manifests_dir = tmp_path / "manifests"

    first = prepare_imagenette160(dataset_root, manifests_dir=manifests_dir, download=False)
    second = prepare_imagenette160(dataset_root, manifests_dir=manifests_dir, download=False)

    eval20_path = manifests_dir / "imagenette2-160-eval20.jsonl"
    eval500_path = manifests_dir / "imagenette2-160-eval500.jsonl"
    eval20_metadata_path = manifests_dir / "imagenette2-160-eval20.metadata.json"
    eval500_metadata_path = manifests_dir / "imagenette2-160-eval500.metadata.json"

    eval20_records = _read_jsonl(eval20_path)
    eval500_records = _read_jsonl(eval500_path)
    eval20_metadata = json.loads(eval20_metadata_path.read_text(encoding="utf-8"))
    eval500_metadata = json.loads(eval500_metadata_path.read_text(encoding="utf-8"))

    assert len(eval20_records) == 20
    assert len(eval500_records) == 500
    assert all(record["path"].startswith("val/") for record in eval20_records)
    assert eval20_metadata["manifest_sha256"] == manifest_sha256(eval20_path)
    assert eval500_metadata["manifest_sha256"] == manifest_sha256(eval500_path)
    assert [item["sha256"] for item in first["manifests"]] == [
        item["sha256"] for item in second["manifests"]
    ]

    loaded = load_imagenette_manifest(eval20_path, dataset_root=dataset_root)

    assert Path(loaded[0]["path"]).is_file()


def _write_fake_imagenette_val(dataset_root: Path, *, images_per_label: int) -> None:
    for synset in IMAGENETTE_LABELS:
        folder = dataset_root / "val" / synset
        folder.mkdir(parents=True)
        for index in range(images_per_label):
            (folder / f"{synset}_{index:04d}.JPEG").write_bytes(b"image")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
