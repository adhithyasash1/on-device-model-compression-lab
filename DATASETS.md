# Datasets

## Imagenette

Imagenette is the default laptop-scale dataset for v1. It is a 10-class subset drawn from ImageNet classes, useful for fast local experiments before a full ImageNet validation run. This project maps Imagenette synset folder names back to the original ImageNet-1k class indices so MobileViT can keep its 1000-class head.

Expected local layout for the real smoke config:

```text
data/imagenette2-160/val/
  n01440764/
  n02102040/
  ...
```

Prepare or validate that folder with:

```bash
uv run mlclab prepare-imagenette160
```

The command downloads Imagenette 160px when `data/imagenette2-160` is missing,
refuses validation folders with unmapped labels, and writes fixed `eval20` and
`eval500` JSONL manifests under `data/manifests/`. Each manifest has a
`.metadata.json` sidecar with the SHA-256 hash of the JSONL file.

Dataset files are not committed.

Public release notes:

- This repo does not redistribute Imagenette images, local manifests, or downloaded archives.
- The benchmark tables in `reports/` use `eval20` and `eval500` subsets generated from the local Imagenette 160px validation folder with seed `1337`.
- Check upstream dataset terms before redistributing images or derived artifacts. The AWS Open Data registry for fast.ai image classification datasets states that licenses vary by dataset and points users to upstream documentation.

## ImageNet-1k

ImageNet-1k validation is planned as an optional canonical mode. Users must provide a local dataset path and comply with ImageNet access terms.

## Sources

- [fast.ai Imagenette tutorial](https://docs.fast.ai/tutorial.imagenette.html)
- [fast.ai image classification datasets on AWS Open Data](https://registry.opendata.aws/fast-ai-imageclas)
- [Local Imagenette preparation code](src/mlclab/data/imagenette.py)
