"""Lightweight observability helpers for MT-LNN scripts and benchmarks.

The core model stays framework-agnostic. This module provides small utilities
for structured logs/metrics so demos and benchmarks can expose cache growth,
latency, and compression diagnostics without depending on a production stack.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union


def setup_logging(level: Union[int, str] = "INFO") -> logging.Logger:
    """Configure a simple stderr logger and return the project logger."""
    logger = logging.getLogger("mt_lnn")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level if isinstance(level, int) else level.upper())
    return logger


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


class JsonlMetricWriter:
    """Append one JSON object per metric event.

    Parameters
    ----------
    path:
        Destination JSONL file. Parent directories are created automatically.
    static_fields:
        Fields included in every event, for example benchmark name or device.
    """

    def __init__(
        self,
        path: Union[str, Path],
        static_fields: Optional[Mapping[str, Any]] = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.static_fields = dict(static_fields or {})
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, event: str, fields: Optional[Mapping[str, Any]] = None) -> None:
        row: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **self.static_fields,
        }
        if fields:
            row.update(dict(fields))
        self._fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlMetricWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def cache_summary(cache: Any) -> Dict[str, Any]:
    """Return a small, serializable cache summary.

    Works with ``ModelCacheStruct`` without importing it at module load time.
    """
    if cache is None:
        return {
            "token_count": 0,
            "cache_bytes": 0,
            "layers": 0,
            "has_attention_kv": False,
            "has_recurrent_state": False,
            "has_gwtb_kv": False,
            "has_coherence_kv": False,
        }

    layers = getattr(cache, "layers", []) or []
    has_attention_kv = any(layer is not None and layer[0] is not None for layer in layers)
    has_recurrent_state = any(
        layer is not None and len(layer) > 1 and layer[1] is not None for layer in layers
    )
    has_block_gwtb_kv = any(
        layer is not None and len(layer) > 2 and layer[2] is not None for layer in layers
    )
    cache_bytes = cache.tensor_bytes() if hasattr(cache, "tensor_bytes") else None

    return {
        "token_count": getattr(cache, "token_count", 0),
        "cache_bytes": cache_bytes,
        "layers": len(layers),
        "has_attention_kv": has_attention_kv,
        "has_recurrent_state": has_recurrent_state,
        "has_gwtb_kv": has_block_gwtb_kv or getattr(cache, "gwtb_kv", None) is not None,
        "has_coherence_kv": getattr(cache, "coherence_kv", None) is not None,
    }
