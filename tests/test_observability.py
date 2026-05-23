import json

import torch

from mt_lnn.model import ModelCacheStruct
from mt_lnn.observability import JsonlMetricWriter, cache_summary


def test_cache_summary_distinguishes_recurrent_only_cache():
    cache = ModelCacheStruct(token_count=7)
    h_prev = torch.zeros(1, 13, 5, 8)
    cache.layers.append((None, h_prev, None))

    summary = cache_summary(cache)

    assert summary["token_count"] == 7
    assert summary["cache_bytes"] == h_prev.numel() * h_prev.element_size()
    assert summary["layers"] == 1
    assert summary["has_attention_kv"] is False
    assert summary["has_recurrent_state"] is True
    assert summary["has_gwtb_kv"] is False
    assert summary["has_coherence_kv"] is False


def test_jsonl_metric_writer_appends_structured_events(tmp_path):
    path = tmp_path / "metrics.jsonl"

    with JsonlMetricWriter(path, static_fields={"benchmark": "unit"}) as writer:
        writer.write("cache", {"cache_bytes": 123, "ok": True})

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["event"] == "cache"
    assert rows[0]["benchmark"] == "unit"
    assert rows[0]["cache_bytes"] == 123
    assert rows[0]["ok"] is True
    assert "ts" in rows[0]
