# src/utils/llm_trace_logger.py
# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Any, Dict, List

class LLMTraceLogger:

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: List[Dict[str, Any]] = []

    def log(self, record: Dict[str, Any]):
        try:
            self._buffer.append(record)
        except Exception:
            pass

    def close(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self._buffer, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        self._buffer = []