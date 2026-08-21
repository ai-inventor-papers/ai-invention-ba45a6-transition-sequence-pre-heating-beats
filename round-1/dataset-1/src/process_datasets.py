#!/usr/bin/env python3
"""Parse CASAS (Aruba/Cairo/Milan/Tulum) and SPHERE raw sensor logs into a
common per-room 15-minute occupancy schema for PreHeat-style pre-heating
experiments.

Output schema (list of row dicts):
  dataset_id, house_id, day_id, room_id, weekday_weekend,
  input: 96-length binary vector (15-min bins, 1=occupied)
  output: {next_bin_occupied: [96 labels shifted by one bin],
           transitions: [{from_room, to_room, timestamp, dwell_minutes}, ...]}
  metadata_fold: {source, real_or_synthetic, adjacency_edges: [[a,b],...]}
"""
import json
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
Path("logs").mkdir(exist_ok=True)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

WORKDIR = Path(__file__).parent
CASAS_ZIP = WORKDIR / "temp/datasets/casas/new_labeled_data.zip"
SPHERE_TRAIN_ZIP = WORKDIR / "temp/datasets/sphere/train.zip"
SPHERE_META_ZIP = WORKDIR / "temp/datasets/sphere/metadata.zip"
BIN_MINUTES = 15
BINS_PER_DAY = 24 * 60 // BIN_MINUTES  # 96


def bin_index(dt: datetime) -> int:
    return (dt.hour * 60 + dt.minute) // BIN_MINUTES


def parse_casas_house(house_name: str, raw_text: str) -> dict:
    """Parse one CASAS house's raw sensor-event text into per-sensor,
    per-day 15-min occupancy bins and observed room-to-room transitions.
    Motion sensors (M-prefixed) are treated as room proxies; door (D) and
    temperature (T) sensors are ignored for occupancy binning.
    """
    # day_id -> sensor_id -> set(bin_index)
    day_room_bins: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    # ordered ON events per day for transition inference: (timestamp, sensor)
    day_events: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
    rooms_seen: set[str] = set()

    for line in raw_text.splitlines():
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 3:
            continue
        if " " in parts[0]:
            date_str, time_str = parts[0].split(" ", 1)
            sensor_id, value = parts[1], parts[2]
        else:
            if len(parts) < 4:
                continue
            date_str, time_str, sensor_id, value = parts[0], parts[1], parts[2], parts[3]
        if not sensor_id.startswith("M"):
            continue  # only motion sensors define room occupancy
        if value != "ON":
            continue
        try:
            dt = datetime.strptime(f"{date_str} {time_str.split('.')[0]}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        day_id = date_str
        day_room_bins[day_id][sensor_id].add(bin_index(dt))
        day_events[day_id].append((dt, sensor_id))
        rooms_seen.add(sensor_id)

    # transitions + adjacency (frequency-based, inferred not floorplan-verified)
    transitions_by_day: dict[str, list[dict]] = defaultdict(list)
    adjacency_counts: dict[tuple[str, str], int] = defaultdict(int)
    for day_id, events in day_events.items():
        events.sort(key=lambda e: e[0])
        prev_time, prev_room = None, None
        for ts, room in events:
            if prev_room is not None and room != prev_room:
                dwell = (ts - prev_time).total_seconds() / 60.0
                if 0 < dwell < 24 * 60:
                    transitions_by_day[day_id].append(
                        {
                            "from_room": prev_room,
                            "to_room": room,
                            "timestamp": ts.isoformat(),
                            "dwell_minutes": round(dwell, 2),
                        }
                    )
                    edge = tuple(sorted((prev_room, room)))
                    adjacency_counts[edge] += 1
            prev_time, prev_room = ts, room

    # keep only adjacency edges with a plausible minimum co-transition count
    adjacency_edges = [list(e) for e, c in adjacency_counts.items() if c >= 3]

    return {
        "day_room_bins": day_room_bins,
        "transitions_by_day": transitions_by_day,
        "adjacency_edges": adjacency_edges,
        "rooms": sorted(rooms_seen),
    }


def rows_from_casas_house(house_name: str, parsed: dict) -> tuple[list[dict], dict]:
    rows = []
    for day_id, room_bins in parsed["day_room_bins"].items():
        try:
            dt_day = datetime.strptime(day_id, "%Y-%m-%d")
        except ValueError:
            continue
        weekday_weekend = "weekend" if dt_day.weekday() >= 5 else "weekday"
        for room_id, bins in room_bins.items():
            vec = [1 if b in bins else 0 for b in range(BINS_PER_DAY)]
            next_vec = vec[1:] + [vec[0]]
            rows.append(
                {
                    "day_id": day_id,
                    "room_id": room_id,
                    "weekday_weekend": weekday_weekend,
                    "input": vec,
                    "output": {"next_bin_occupied": next_vec},
                }
            )
    # transitions kept once per day (not duplicated per room-row)
    transitions_compact = {
        day_id: [
            [t["from_room"], t["to_room"], t["timestamp"], t["dwell_minutes"]]
            for t in trs
        ]
        for day_id, trs in parsed["transitions_by_day"].items()
    }
    metadata = {
        "dataset_id": f"casas_{house_name}",
        "house_id": house_name,
        "source": "CASAS",
        "real_or_synthetic": "real",
        "adjacency_edges": parsed["adjacency_edges"],
        "adjacency_provenance": "inferred_from_transition_frequency",
        "rooms": parsed["rooms"],
        "transitions_by_day": transitions_compact,
        "transition_fields": ["from_room", "to_room", "timestamp", "dwell_minutes"],
    }
    return rows, metadata


def parse_sphere() -> tuple[list[dict], dict]:
    """Parse SPHERE Challenge pir.csv sequences into per-sequence per-room
    binary occupancy (short scripted sequences, seconds resolution, binned
    to the same 15-min-equivalent granularity scaled to sequence length)."""
    rows: list[dict] = []
    with zipfile.ZipFile(SPHERE_META_ZIP) as zf:
        rooms = json.loads(zf.read("metadata/rooms.json"))

    transitions_by_seq: dict[str, list] = {}
    with zipfile.ZipFile(SPHERE_TRAIN_ZIP) as zf:
        seq_ids = sorted({n.split("/")[1] for n in zf.namelist() if n.startswith("train/") and len(n.split("/")) > 2 and n.split("/")[1]})
        adjacency_counts: dict[tuple[str, str], int] = defaultdict(int)
        for seq_id in seq_ids:
            pir_path = f"train/{seq_id}/pir.csv"
            if pir_path not in zf.namelist():
                continue
            content = zf.read(pir_path).decode("utf-8").splitlines()
            header = content[0].split(",")
            events = []
            for line in content[1:]:
                parts = line.split(",")
                if len(parts) != len(header):
                    continue
                rec = dict(zip(header, parts))
                try:
                    start, end = float(rec["start"]), float(rec["end"])
                except ValueError:
                    continue
                events.append((start, end, rec["name"]))
            events.sort()
            # infer transitions from consecutive PIR firings (different room)
            transitions = []
            prev_end, prev_room = None, None
            for start, end, room in events:
                if prev_room is not None and room != prev_room:
                    dwell = (start - prev_end) / 60.0
                    if dwell >= 0:
                        transitions.append(
                            {
                                "from_room": prev_room,
                                "to_room": room,
                                "timestamp": f"seq{seq_id}_t{start:.2f}",
                                "dwell_minutes": round(dwell, 3),
                            }
                        )
                        edge = tuple(sorted((prev_room, room)))
                        adjacency_counts[edge] += 1
                prev_end, prev_room = end, room
            transitions_by_seq[f"seq_{seq_id}"] = [
                [t["from_room"], t["to_room"], t["timestamp"], t["dwell_minutes"]] for t in transitions
            ]

            for room in rooms:
                room_events = [(s, e) for s, e, r in events if r == room]
                if not room_events:
                    input_vec = [0] * BINS_PER_DAY
                else:
                    seq_len = max(e for _, e, _ in events) if events else 1.0
                    input_vec = [0] * BINS_PER_DAY
                    for s, e in room_events:
                        b = min(BINS_PER_DAY - 1, int((s / max(seq_len, 1e-6)) * BINS_PER_DAY))
                        input_vec[b] = 1
                next_vec = input_vec[1:] + [input_vec[0]]
                rows.append(
                    {
                        "day_id": f"seq_{seq_id}",
                        "room_id": room,
                        "weekday_weekend": "unknown",
                        "input": input_vec,
                        "output": {"next_bin_occupied": next_vec},
                    }
                )
        adjacency_edges = [list(e) for e, c in adjacency_counts.items() if c >= 2]
    metadata = {
        "dataset_id": "sphere_challenge",
        "house_id": "sphere_house",
        "source": "SPHERE",
        "real_or_synthetic": "real",
        "adjacency_edges": adjacency_edges,
        "adjacency_provenance": "inferred_from_transition_frequency",
        "rooms": rooms,
        "transitions_by_day": transitions_by_seq,
        "transition_fields": ["from_room", "to_room", "timestamp", "dwell_minutes"],
    }
    return rows, metadata


def main():
    logger.info("Reading CASAS zip...")
    datasets: dict[str, tuple[list[dict], dict]] = {}
    with zipfile.ZipFile(CASAS_ZIP) as zf:
        for house_file in ["aruba.txt", "cairo.txt", "milan.txt"]:
            logger.info(f"Parsing {house_file}...")
            raw = zf.read(house_file).decode("utf-8", errors="ignore")
            house_name = house_file.replace(".txt", "")
            parsed = parse_casas_house(house_name, raw)
            house_rows, meta = rows_from_casas_house(house_name, parsed)
            logger.info(f"{house_file}: {len(parsed['day_room_bins'])} days, {len(parsed['rooms'])} rooms, {len(house_rows)} rows")
            datasets[f"casas_{house_name}"] = (house_rows, meta)
        # tulum split across two files -> merge as one house
        tulum_raw = zf.read("tulum1.txt").decode("utf-8", errors="ignore") + "\n" + zf.read("tulum2.txt").decode("utf-8", errors="ignore")
        parsed = parse_casas_house("tulum", tulum_raw)
        tulum_rows, tulum_meta = rows_from_casas_house("tulum", parsed)
        logger.info(f"tulum: {len(parsed['day_room_bins'])} days, {len(parsed['rooms'])} rooms, {len(tulum_rows)} rows")
        datasets["casas_tulum"] = (tulum_rows, tulum_meta)

    logger.info("Parsing SPHERE...")
    sphere_rows, sphere_meta = parse_sphere()
    logger.info(f"SPHERE: {len(sphere_rows)} rows")
    datasets["sphere_challenge"] = (sphere_rows, sphere_meta)

    out_dir = WORKDIR / "temp/datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    for dataset_id, (rows, meta) in datasets.items():
        total_rows += len(rows)
        full_path = out_dir / f"full_{dataset_id}.json"
        full_path.write_text(json.dumps({"metadata_fold": meta, "rows": rows}))
        logger.info(f"{dataset_id}: {len(rows)} rows -> {full_path} ({full_path.stat().st_size / 1e6:.1f} MB)")

        mini_rows = rows[:3]
        mini_meta = dict(meta)
        mini_meta["transitions_by_day"] = dict(list(meta["transitions_by_day"].items())[:5])
        (out_dir / f"mini_{dataset_id}.json").write_text(json.dumps({"metadata_fold": mini_meta, "rows": mini_rows}, indent=2))

        preview_rows = []
        for r in rows[:3]:
            pr = dict(r)
            pr["input"] = pr["input"][:10] + ["..."]
            preview_rows.append(pr)
        preview_meta = dict(meta)
        preview_meta["transitions_by_day"] = dict(list(meta["transitions_by_day"].items())[:2])
        preview_meta["adjacency_edges"] = meta["adjacency_edges"][:10] + (["..."] if len(meta["adjacency_edges"]) > 10 else [])
        (out_dir / f"preview_{dataset_id}.json").write_text(json.dumps({"metadata_fold": preview_meta, "rows": preview_rows}, indent=2))
    logger.info(f"Wrote {len(datasets)} datasets, {total_rows} total rows")


if __name__ == "__main__":
    main()
