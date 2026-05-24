from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without the optional ML extra
    torch = None


if torch is not None:

    class LogitsOnlyWrapper(torch.nn.Module):
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.model = model

        def forward(self, pixel_values: Any) -> Any:
            return self.model(pixel_values=pixel_values).logits

else:

    class LogitsOnlyWrapper:
        def __init__(self, model: Any) -> None:
            self.model = model

        def eval(self) -> LogitsOnlyWrapper:
            self.model.eval()
            return self

        def __call__(self, pixel_values: Any) -> Any:
            return self.model(pixel_values=pixel_values).logits


@dataclass(frozen=True)
class MobileViTBundle:
    model: Any
    processor: Any
    wrapper: LogitsOnlyWrapper
    input_name: str = "pixel_values"


def load_mobilevit(model_id: str, revision: str | None = None) -> MobileViTBundle:
    try:
        from transformers import AutoImageProcessor, MobileViTForImageClassification
    except ImportError as exc:
        raise RuntimeError(
            "MobileViT loading requires transformers. Install the ml extra."
        ) from exc

    kwargs = {"revision": revision} if revision else {}
    processor = AutoImageProcessor.from_pretrained(model_id, **kwargs)
    model = MobileViTForImageClassification.from_pretrained(model_id, **kwargs)
    model.eval()
    return MobileViTBundle(model=model, processor=processor, wrapper=LogitsOnlyWrapper(model))
