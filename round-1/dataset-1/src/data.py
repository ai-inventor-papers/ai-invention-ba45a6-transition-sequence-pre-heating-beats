# /// script
# requires-python = ">=3.12"
# dependencies = ["loguru"]
# ///
"""Standardize the 5 real room-occupancy datasets (CASAS Aruba/Cairo/Milan/Tulum,
SPHERE Challenge) into exp_sel_data_out.json schema: one example per
(dataset_id, day_id, room_id) row, grouped by dataset."""

import json
import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
Path("logs").mkdir(exist_ok=True)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

WORKDIR = Path(__file__).parent
DATA_DIR = WORKDIR / "temp/datasets"

MAX_TRANSITIONS_PER_EXAMPLE = 40  # caps pathological high-traffic sensors (e.g. hallway doorways)


def _minute_of_day(ts: str) -> str:
    # SPHERE timestamps look like "seq12_t345.67"; CASAS are ISO datetimes.
    if "T" in ts:
        return ts.split("T", 1)[1][:8]
    return ts



# Final 4 (of 5 candidates): SPHERE dropped — only 10 short scripted sequences with
# no real calendar dates (weekday_weekend="unknown"), so it cannot support the
# weekday/weekend K=5 nearest-day matching PreHeat needs; each CASAS house gives
# 57-235 real days across >=27 rooms with a genuine weekday/weekend split.
DATASET_FILES = {
    "casas_aruba": "full_casas_aruba.json",
    "casas_cairo": "full_casas_cairo.json",
    "casas_milan": "full_casas_milan.json",
    "casas_tulum": "full_casas_tulum.json",
}


def build_examples(dataset_id: str, payload: dict) -> list[dict]:
    meta = payload["metadata_fold"]
    transitions_by_day = meta.get("transitions_by_day", {})
    transition_fields = meta.get("transition_fields", ["from_room", "to_room", "timestamp", "dwell_minutes"])

    examples = []
    for row in payload["rows"]:
        day_id = row["day_id"]
        room_id = row["room_id"]
        day_transitions = transitions_by_day.get(day_id, [])
        # compact [from_room, to_room, minute_of_day, dwell_minutes] tuples (timestamps -> minute-of-day int)
        room_transitions = [
            [t[0], t[1], _minute_of_day(t[2]), t[3]]
            for t in day_transitions
            if t[0] == room_id or t[1] == room_id
        ][:MAX_TRANSITIONS_PER_EXAMPLE]

        example = {
            "input": json.dumps(row["input"]),
            "output": json.dumps(row["output"]["next_bin_occupied"]),
            "metadata_dataset_source": meta["source"],
            "metadata_house_id": meta["house_id"],
            "metadata_day_id": day_id,
            "metadata_room_id": room_id,
            "metadata_weekday_weekend": row["weekday_weekend"],
            "metadata_real_or_synthetic": meta["real_or_synthetic"],
            "metadata_adjacency_provenance": meta.get("adjacency_provenance", "unknown"),
            "metadata_room_transitions": room_transitions,
            "metadata_task_type": "binary_sequence_forecasting",
            "metadata_n_classes": 2,
            "metadata_bin_minutes": 15,
        }
        examples.append(example)
    return examples


def main():
    datasets_out = []
    per_dataset_meta = {}
    for dataset_id, filename in DATASET_FILES.items():
        path = DATA_DIR / filename
        logger.info(f"Loading {path}")
        payload = json.loads(path.read_text())
        examples = build_examples(dataset_id, payload)
        logger.info(f"{dataset_id}: {len(examples)} examples")
        datasets_out.append({"dataset": dataset_id, "examples": examples})
        m = payload["metadata_fold"]
        per_dataset_meta[dataset_id] = {
            "rooms": m.get("rooms", []),
            "adjacency_edges": m.get("adjacency_edges", []),
            "adjacency_provenance": m.get("adjacency_provenance", "unknown"),
            "house_id": m.get("house_id"),
            "source": m.get("source"),
        }

    out = {
        "metadata": {
            "description": "Real, room-level, 15-minute-binned occupancy traces for pre-heating "
            "(PreHeat-style per-room K-NN and room-transition models). Each example is one "
            "(house, day, room) row: input = 96-bin binary occupancy vector for the day, "
            "output = same-length vector shifted one bin ahead (next-bin-occupied target). "
            "Room adjacency (per dataset, in per_dataset_meta below) is inferred from observed "
            "transition frequency (not floorplan-verified) except where noted.",
            "sources": [
                "CASAS Zenodo record 17180309 (aruba/cairo/milan/tulum, WSU CASAS project)",
            ],
            "per_dataset_meta": per_dataset_meta,
        },
        "datasets": datasets_out,
    }

    total = sum(len(d["examples"]) for d in datasets_out)
    out_path = WORKDIR / "full_data_out.json"
    out_path.write_text(json.dumps(out))
    logger.info(f"Wrote {total} total examples across {len(datasets_out)} datasets to {out_path} "
                f"({out_path.stat().st_size / 1e6:.1f} MB)")

    # mini (3 examples/dataset) and preview (10 examples/dataset) — every dataset
    # group kept, only the examples list is truncated, so all 4 datasets survive.
    def truncated(n: int) -> dict:
        return {
            "metadata": out["metadata"],
            "datasets": [{"dataset": d["dataset"], "examples": d["examples"][:n]} for d in datasets_out],
        }

    mini_path = WORKDIR / "mini_data_out.json"
    mini_path.write_text(json.dumps(truncated(3), indent=2))
    logger.info(f"Wrote mini ({mini_path.stat().st_size / 1e3:.1f} KB) to {mini_path}")

    preview_path = WORKDIR / "preview_data_out.json"
    preview_path.write_text(json.dumps(truncated(10), indent=2))
    logger.info(f"Wrote preview ({preview_path.stat().st_size / 1e3:.1f} KB) to {preview_path}")


if __name__ == "__main__":
    main()
