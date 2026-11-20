"""Audit Sink and Dataset Exporter for Continuous Calibration & Active Learning."""

import json
import time
import asyncio
from pathlib import Path
from typing import Any, Optional


class AuditSink:
    """Asynchronous audit logger collecting routing anomalies, rejected chunks, and blocked answers.
    
    Generates training and fine-tuning datasets for domain-specific Jev calibration.
    """

    def __init__(self, log_path: str = "data/audit_events.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def log_event(self, event_type: str, payload: dict[str, Any]):
        """Append an audit record to JSONL asynchronously."""
        record = {
            "timestamp": time.time(),
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_type": event_type,
            "payload": payload,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        async with self._lock:
            # Synchronous append within lock to avoid external aiofiles dependency if not installed
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line)

    def export_jev_finetuning_dataset(self, output_path: str = "data/jev_training_pairs.jsonl") -> int:
        """Parse audit logs and generate Jev-compatible calibration pairs.
        
        Returns:
            Number of extracted training pairs.
        """
        if not self.log_path.exists():
            return 0

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with open(self.log_path, "r", encoding="utf-8") as fin, open(out_file, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                data = json.loads(line)
                event_type = data.get("event_type")
                payload = data.get("payload", {})

                # Format for Jev training
                pair = None
                if event_type == "STAGE1_ARBITRATION":
                    pair = {
                        "primitive": "noul",
                        "context": f"User Query: {payload.get('query')}",
                        "question": "needs_external_knowledge",
                        "target_value": payload.get("needs_external_knowledge"),
                        "calibrated_prob": payload.get("confidence"),
                    }
                elif event_type == "STAGE3_HALLUCINATION_BLOCKED":
                    pair = {
                        "primitive": "noul",
                        "context": (
                            f"Evidence: {payload.get('evidence')}\n"
                            f"Query: {payload.get('query')}\n"
                            f"Answer: {payload.get('answer')}"
                        ),
                        "question": "contains_unsupported_facts",
                        "target_value": True,
                        "calibrated_prob": payload.get("hallucination_prob", 0.9),
                    }

                if pair:
                    fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    count += 1

        return count


# Global sink instance
audit_sink = AuditSink()
