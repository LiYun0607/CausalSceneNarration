"""
Token statistics logger for text-enrichment ablations.

Records per-frame instruction/notice token counts so that the mean number of
tokens per channel can be reported for each text style. Activated by setting the
LMDRIVE_TOKEN_LOG_PATH environment variable; otherwise no-op.

Output format: JSONL, one record per generate() call.
    {"style": "csn", "instruction_tokens": 52, "notice_tokens": 33,
     "total_tokens": 85}
"""

import json
import os
import logging

logger = logging.getLogger(__name__)


def _estimate_tokens(text):
    """Match scene_text_enrichment._estimate_tokens (LLaMA ~4 chars/token)."""
    if not text:
        return 0
    return max(1, (len(text) + 2) // 4)


class TokenLogger:
    """Lightweight JSONL appender; safe to instantiate even when disabled."""

    _instances = {}

    @classmethod
    def get(cls, style):
        path = os.environ.get("LMDRIVE_TOKEN_LOG_PATH", "").strip()
        key = (style, path)
        if key not in cls._instances:
            cls._instances[key] = cls(style, path)
        return cls._instances[key]

    def __init__(self, style, path):
        self.style = style
        self.path = path
        self.enabled = bool(path)
        self._fh = None
        if self.enabled:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self._fh = open(path, "a", buffering=1)
                logger.info("TokenLogger active: style=%s, path=%s", style, path)
            except Exception as e:
                logger.warning("TokenLogger disabled (open failed): %s", e)
                self.enabled = False

    def log(self, instruction, notice):
        if not self.enabled or self._fh is None:
            return
        i_tok = _estimate_tokens(instruction)
        n_tok = _estimate_tokens(notice)
        record = {
            "style": self.style,
            "instruction_tokens": i_tok,
            "notice_tokens": n_tok,
            "total_tokens": i_tok + n_tok,
        }
        try:
            self._fh.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def close(self):
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None


def summarise(path):
    """Compute mean/std/count of tokens per channel from a JSONL file."""
    if not os.path.exists(path):
        return None
    by_style = {}
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            s = rec.get("style", "unknown")
            entry = by_style.setdefault(s, {"i": [], "n": [], "t": []})
            entry["i"].append(rec.get("instruction_tokens", 0))
            entry["n"].append(rec.get("notice_tokens", 0))
            entry["t"].append(rec.get("total_tokens", 0))
    summary = {}
    for s, d in by_style.items():
        n = len(d["t"])
        summary[s] = {
            "frames": n,
            "instruction_mean": sum(d["i"]) / n if n else 0,
            "notice_mean": sum(d["n"]) / n if n else 0,
            "total_mean": sum(d["t"]) / n if n else 0,
        }
    return summary
