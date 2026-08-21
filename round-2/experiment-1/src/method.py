#!/usr/bin/env python3
"""PreHeat K-NN vs HMM vs Room-Transition-Markov occupancy forecasting + RC
thermal-simulator energy-savings comparison, on real CASAS houses and an
expanded synthetic-topology sweep.
"""

from __future__ import annotations

import gc
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.metrics import roc_auc_score, roc_curve

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
Path("logs").mkdir(exist_ok=True)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKDIR = Path(__file__).parent
DATA_PATH = WORKDIR / "full_data_out.json"
OUT_PATH = WORKDIR / "full_method_out.json"

RAM_BUDGET = 6 * 1024**3  # 6GB, well under 29GB container limit
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3 * 3600, 3 * 3600))

RNG = np.random.default_rng(0)
BIN_MINUTES = 15
N_BINS = 96
LOOKAHEADS_MIN = [15, 30, 45, 60]
LOOKAHEAD_BINS = {m: m // BIN_MINUTES for m in LOOKAHEADS_MIN}
T_OFFSETS = list(range(16, 88, 8))  # 9 sampled within-day evaluation points
KNN_K = 5
N_BOOT_AUC = 300  # reduced from spec's 1000 for compute-budget reasons (logged)
N_BOOT_TOPOLOGY = 5000
CONTROL_LOOKAHEAD_MIN = 30  # lookahead used to drive the thermal controller
FP_RATES = [0.05, 0.10, 0.15, 0.20]
MAX_THERMAL_TEST_DAYS = 10  # per house/config, for compute-budget reasons (logged)
MAX_THERMAL_ROOMS = 6

DEVIATIONS: list[str] = [
    f"AUC bootstrap resamples reduced to {N_BOOT_AUC} (spec: 1000) to fit the compute budget.",
    f"Thermal simulation subsamples to the first {MAX_THERMAL_TEST_DAYS} TEST days and the "
    f"{MAX_THERMAL_ROOMS} most-active rooms per house/config (full grid too slow for the budget).",
    "Room-transition model is a bin-level (15min) first-order Markov chain derived from the "
    "empirical next-room distribution plus an exponential-holding-time approximation "
    "(p_stay = exp(-15/mean_dwell)); this is a deliberate discretization of the continuous "
    "dwell-time model described in the plan, not the raw continuous-time chain.",
    "RC-network thermal simulator + HeatRate rule is a from-scratch minimal reimplementation "
    "(no prior-iteration simulator code was found in the dependency workspace) using a "
    "synthetic diurnal ambient-temperature trace; documented explicitly per the fallback plan.",
]

HMM_FALLBACK_LOG: dict[str, str] = {}

try:
    from hmmlearn import hmm as hmmlearn_hmm

    HAVE_HMMLEARN = True
except ImportError:
    HAVE_HMMLEARN = False
    logger.warning("hmmlearn not importable; ALL HMM baselines will use the Markov fallback")


# ---------------------------------------------------------------------------
# 1. Load + structure data
# ---------------------------------------------------------------------------
def load_data() -> dict:
    assert DATA_PATH.exists(), f"Dependency data file missing: {DATA_PATH}"
    size = DATA_PATH.stat().st_size
    assert size > 1000, f"Dependency data file suspiciously small ({size} bytes) — empty-dependency bug?"
    data = json.loads(DATA_PATH.read_text())
    assert "datasets" in data and len(data["datasets"]) > 0, "No datasets in dependency file"
    total_examples = sum(len(ds["examples"]) for ds in data["datasets"])
    houses_present = {ds["dataset"].replace("casas_", "") for ds in data["datasets"]}
    assert total_examples > 10000, f"Only {total_examples} examples loaded — expected >10000"
    assert {"aruba", "cairo", "milan", "tulum"} <= houses_present, f"Missing houses: {houses_present}"
    logger.info(f"Loaded {total_examples} examples across houses {sorted(houses_present)}")
    return data


def build_house(ds: dict, adjacency_meta: dict) -> dict:
    house_id = ds["dataset"].replace("casas_", "")
    examples = ds["examples"]
    rooms = sorted({e["metadata_room_id"] for e in examples})
    room_idx = {r: i for i, r in enumerate(rooms)}
    n_rooms = len(rooms)

    day_rooms: dict[str, dict] = {}
    for e in examples:
        day = e["metadata_day_id"]
        rec = day_rooms.setdefault(day, {"rooms": {}, "weekday": e["metadata_weekday_weekend"], "transitions": None})
        raw_input = e["input"]
        vec = json.loads(raw_input) if isinstance(raw_input, str) else raw_input
        rec["rooms"][e["metadata_room_id"]] = np.asarray(vec, dtype=np.int8)
        if rec["transitions"] is None:
            rec["transitions"] = e["metadata_room_transitions"]

    day_ids = sorted(day_rooms.keys())
    n_train = max(1, int(len(day_ids) * 0.7))
    train_days, test_days = day_ids[:n_train], day_ids[n_train:]
    if not test_days:
        test_days = train_days[-1:]
        train_days = train_days[:-1] or train_days

    def day_matrix(day_id: str) -> np.ndarray:
        mat = np.zeros((n_rooms, N_BINS), dtype=np.int8)
        for r, vec in day_rooms[day_id]["rooms"].items():
            mat[room_idx[r], : len(vec)] = vec[:N_BINS]
        return mat

    mats = {d: day_matrix(d) for d in day_ids}
    weekday_flag = {d: day_rooms[d]["weekday"] for d in day_ids}
    transitions = {d: (day_rooms[d]["transitions"] or []) for d in day_ids}
    edges = adjacency_meta.get(ds["dataset"], {}).get("adjacency_edges", [])

    logger.info(
        f"[{house_id}] rooms={n_rooms} days={len(day_ids)} train={len(train_days)} "
        f"test={len(test_days)} edges={len(edges)}"
    )
    return dict(
        house_id=house_id,
        rooms=rooms,
        room_idx=room_idx,
        n_rooms=n_rooms,
        mats=mats,
        weekday_flag=weekday_flag,
        transitions=transitions,
        train_days=train_days,
        test_days=test_days,
        edges=edges,
    )


# ---------------------------------------------------------------------------
# 2. Predictor: PreHeat K-NN (per room, weekday/weekend bucketed Hamming K-NN)
# ---------------------------------------------------------------------------
def preheat_scores(house: dict, lookahead_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, labels) pooled across rooms/days/t-offsets for this house."""
    scores_all, labels_all = [], []
    for bucket in ("weekday", "weekend"):
        train_ids = [d for d in house["train_days"] if house["weekday_flag"][d] == bucket]
        test_ids = [d for d in house["test_days"] if house["weekday_flag"][d] == bucket]
        if len(train_ids) < KNN_K or not test_ids:
            continue
        train_stack = np.stack([house["mats"][d] for d in train_ids])  # (n_train, n_rooms, 96)
        test_stack = np.stack([house["mats"][d] for d in test_ids])  # (n_test, n_rooms, 96)
        for room_i in range(house["n_rooms"]):
            train_room = train_stack[:, room_i, :]  # (n_train,96)
            test_room = test_stack[:, room_i, :]  # (n_test,96)
            for t in T_OFFSETS:
                target = t + lookahead_bins
                if target >= N_BINS:
                    continue
                dist = np.abs(test_room[:, None, :t] - train_room[None, :, :t]).sum(axis=-1)  # (n_test,n_train)
                topk = np.argsort(dist, axis=1)[:, :KNN_K]
                pred = train_room[topk, target].mean(axis=1)  # (n_test,)
                y_true = test_room[:, target]
                scores_all.append(pred)
                labels_all.append(y_true)
    if not scores_all:
        return np.array([]), np.array([])
    return np.concatenate(scores_all), np.concatenate(labels_all)


# ---------------------------------------------------------------------------
# 3. Predictor: HMM (2-state, per room) with Markov-chain fallback
# ---------------------------------------------------------------------------
def fit_hmm_room(house_id: str, room: str, train_seq: np.ndarray) -> dict:
    key = f"{house_id}/{room}"
    if HAVE_HMMLEARN and train_seq.std() > 0:
        try:
            model = hmmlearn_hmm.CategoricalHMM(
                n_components=2, n_iter=100, tol=1e-4, random_state=0, init_params="ste"
            )
            model.fit(train_seq.reshape(-1, 1))
            trans, emit = model.transmat_, model.emissionprob_
            if not (np.isfinite(trans).all() and np.isfinite(emit).all()):
                raise ValueError("non-finite HMM params")
            if np.allclose(trans, np.eye(2), atol=1e-6):
                raise ValueError("degenerate (identity) transition matrix")
            return {"trans": trans, "emit": emit}
        except Exception as ex:  # noqa: BLE001 - documented fallback path
            HMM_FALLBACK_LOG[key] = f"hmmlearn failed/degenerate: {ex}"
    else:
        HMM_FALLBACK_LOG[key] = "hmmlearn unavailable or constant sequence"
    counts = np.ones((2, 2))  # Laplace-smoothed bigram counts
    a, b = train_seq[:-1], train_seq[1:]
    for i in range(2):
        for j in range(2):
            counts[i, j] += np.sum((a == i) & (b == j))
    trans = counts / counts.sum(axis=1, keepdims=True)
    return {"trans": trans, "emit": np.eye(2)}


def hmm_scores(house: dict, lookahead_bins: int, models: dict[str, dict]) -> tuple[np.ndarray, np.ndarray]:
    scores_all, labels_all = [], []
    trans_pow_cache: dict[tuple[str, int], np.ndarray] = {}
    for room_i, room in enumerate(house["rooms"]):
        model = models[room]
        cache_key = (room, lookahead_bins)
        if cache_key not in trans_pow_cache:
            trans_pow_cache[cache_key] = np.linalg.matrix_power(model["trans"], lookahead_bins)
        Tk = trans_pow_cache[cache_key]
        emit1 = model["emit"][:, 1]
        for d in house["test_days"]:
            mat = house["mats"][d]
            for t in T_OFFSETS:
                target = t + lookahead_bins
                if target >= N_BINS:
                    continue
                obs = mat[room_i, t - 1]
                like = model["emit"][:, obs]
                like_sum = like.sum()
                post = like / like_sum if like_sum > 0 else np.array([0.5, 0.5])
                state_dist = post @ Tk
                p_occ = float(state_dist @ emit1)
                scores_all.append(p_occ)
                labels_all.append(int(mat[room_i, target]))
    return np.array(scores_all), np.array(labels_all)


# ---------------------------------------------------------------------------
# 4. Predictor: room-transition Markov model
# ---------------------------------------------------------------------------
def fit_transition_model(house: dict, day_ids: list[str], shuffle: bool = False) -> dict:
    rooms = house["rooms"]
    room_idx = house["room_idx"]
    n = len(rooms)
    next_counts = np.ones((n, n))  # Laplace smoothing
    dwell_sums = np.zeros(n)
    dwell_counts = np.zeros(n)
    for d in day_ids:
        trans = list(house["transitions"].get(d, []))
        if shuffle:
            order = RNG.permutation(len(trans))
            trans = [trans[i] for i in order]
        for row in trans:
            try:
                fr, to, _t, dwell = row[0], row[1], row[2], float(row[3])
            except (IndexError, ValueError, TypeError):
                continue
            if fr not in room_idx or to not in room_idx:
                continue
            fi, ti = room_idx[fr], room_idx[to]
            next_counts[fi, ti] += 1
            dwell_sums[fi] += dwell
            dwell_counts[fi] += 1
    next_probs = next_counts / next_counts.sum(axis=1, keepdims=True)
    mean_dwell = np.where(dwell_counts > 0, dwell_sums / np.maximum(dwell_counts, 1), 15.0)
    mean_dwell = np.clip(mean_dwell, 1.0, 24 * 60.0)
    p_stay = np.exp(-BIN_MINUTES / mean_dwell)

    t_bin = np.zeros((n, n))
    for i in range(n):
        t_bin[i, i] = p_stay[i]
        off = next_probs[i].copy()
        off[i] = 0.0
        off_sum = off.sum()
        if off_sum > 0:
            t_bin[i] = t_bin[i] + (1 - p_stay[i]) * off / off_sum
        else:
            t_bin[i] += (1 - p_stay[i]) / max(n - 1, 1)
            t_bin[i, i] = p_stay[i]
    row_sums = t_bin.sum(axis=1, keepdims=True)
    t_bin = t_bin / np.where(row_sums > 0, row_sums, 1.0)
    return {"t_bin": t_bin}


def transition_scores(house: dict, lookahead_bins: int, model: dict) -> tuple[np.ndarray, np.ndarray]:
    n = house["n_rooms"]
    Tk = np.linalg.matrix_power(model["t_bin"], lookahead_bins)
    activity = np.array([house["mats"][d].sum() for d in house["train_days"]]).sum()
    room_activity = np.zeros(n)
    for d in house["train_days"]:
        room_activity += house["mats"][d].sum(axis=1)
    fallback_dist = room_activity / max(room_activity.sum(), 1)

    scores_all, labels_all = [], []
    for d in house["test_days"]:
        mat = house["mats"][d]
        for t in T_OFFSETS:
            target = t + lookahead_bins
            if target >= N_BINS:
                continue
            col = mat[:, t - 1]
            if col.sum() > 0:
                cur = col.astype(float) / col.sum()
            else:
                cur = fallback_dist
            dist = cur @ Tk
            scores_all.append(dist)
            labels_all.append(mat[:, target])
    if not scores_all:
        return np.array([]), np.array([])
    return np.stack(scores_all).ravel(), np.stack(labels_all).ravel()


# ---------------------------------------------------------------------------
# 5. AUC + bootstrap CI (block-bootstrap over evaluation "rows")
# ---------------------------------------------------------------------------
def safe_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def bootstrap_auc_ci(scores: np.ndarray, labels: np.ndarray, n_boot: int = N_BOOT_AUC) -> tuple[float | None, float | None]:
    point = safe_auc(scores, labels)
    if point is None:
        return None, None
    n = len(scores)
    boot_vals = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        v = safe_auc(scores[idx], labels[idx])
        if v is not None:
            boot_vals.append(v)
    if len(boot_vals) < 10:
        return point, point
    lo, hi = np.percentile(boot_vals, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# 6. RC-network thermal simulator + HeatRate rule
# ---------------------------------------------------------------------------
def ambient_temp(t_bin: int) -> float:
    frac = t_bin / N_BINS
    return 5.0 + 5.0 * math.sin(2 * math.pi * (frac - 0.3))


def simulate_thermal(
    room_names: list[str],
    edges: list[tuple[str, str]],
    day_mats: list[np.ndarray],  # each (n_rooms_subset,96) ground-truth occupancy
    forecast_probs: list[np.ndarray],  # each (n_rooms_subset,96) predictor forecast @ CONTROL_LOOKAHEAD_MIN
    threshold: float,
    setpoint: float = 20.0,
    C: float = 2.0e6,
    U_amb: float = 150.0,
    U_adj: float = 80.0,
    P_heat: float = 1500.0,
    dt: float = 900.0,
) -> tuple[float, float]:
    """Returns (total_energy_kwh, total_misstime_minutes) across all provided days."""
    idx = {r: i for i, r in enumerate(room_names)}
    n = len(room_names)
    adj_pairs = [(idx[a], idx[b]) for a, b in edges if a in idx and b in idx]

    total_energy_j = 0.0
    total_miss_min = 0.0
    for occ_mat, fc_mat in zip(day_mats, forecast_probs):
        T = np.full(n, 18.0)
        for b in range(N_BINS):
            T_amb = ambient_temp(b)
            heater_on = np.zeros(n, dtype=bool)
            heater_on |= fc_mat[:, b] >= threshold  # PreHeat/anticipatory rule
            heater_on |= occ_mat[:, b] == 1  # reactive fallback: never let occupied room go unheated
            Q = np.where(heater_on, P_heat, 0.0)
            dT = np.zeros(n)
            for i in range(n):
                loss = U_amb * (T[i] - T_amb)
                for a, c in adj_pairs:
                    if a == i:
                        loss += U_adj * (T[i] - T[c])
                    elif c == i:
                        loss += U_adj * (T[i] - T[a])
                dT[i] = dt / C * (Q[i] - loss)
            T = T + dT
            total_energy_j += float(Q.sum()) * dt
            miss = (occ_mat[:, b] == 1) & (T < setpoint - 1.0)
            total_miss_min += float(miss.sum()) * BIN_MINUTES
    return total_energy_j / 3.6e6, total_miss_min


def forecast_matrix(mat: np.ndarray, predict_fn, lookahead_bins: int) -> np.ndarray:
    """predict_fn(room_i, t) -> prob; builds an (n_rooms,96) forecast matrix (0 before t>=1)."""
    n, _ = mat.shape
    fc = np.zeros((n, N_BINS))
    for t in range(1, N_BINS - lookahead_bins):
        for room_i in range(n):
            fc[room_i, t] = predict_fn(room_i, t)
    return fc


# ---------------------------------------------------------------------------
# 7. Synthetic topology generator
# ---------------------------------------------------------------------------
def build_synth_adjacency(room_count: int, adjacency_type: str) -> list[tuple[str, str]]:
    rooms = [f"S{i}" for i in range(room_count)]
    edges = []
    if adjacency_type == "linear":
        edges = [(rooms[i], rooms[i + 1]) for i in range(room_count - 1)]
    elif adjacency_type == "loop":
        edges = [(rooms[i], rooms[(i + 1) % room_count]) for i in range(room_count)]
    elif adjacency_type == "star":
        edges = [(rooms[0], rooms[i]) for i in range(1, room_count)]
    elif adjacency_type == "hallway":
        hub = rooms[0]
        edges = [(hub, rooms[i]) for i in range(1, room_count)]
        edges += [(rooms[i], rooms[i + 1]) for i in range(1, room_count - 1)]
    else:
        raise ValueError(adjacency_type)
    return edges


def generate_synth_house(room_count: int, adjacency_type: str, regularity: float, seed: int, n_days: int = 40) -> dict:
    rng = np.random.default_rng(seed)
    rooms = [f"S{i}" for i in range(room_count)]
    room_idx = {r: i for i, r in enumerate(rooms)}
    edges = build_synth_adjacency(room_count, adjacency_type)
    adj = {i: set() for i in range(room_count)}
    for a, b in edges:
        adj[room_idx[a]].add(room_idx[b])
        adj[room_idx[b]].add(room_idx[a])

    conc = 1.0 + regularity * 15.0
    base_trans = np.zeros((room_count, room_count))
    for i in range(room_count):
        neighbors = sorted(adj[i]) or [j for j in range(room_count) if j != i]
        alpha = np.full(len(neighbors), conc)
        probs = rng.dirichlet(alpha)
        for k, j in enumerate(neighbors):
            base_trans[i, j] = probs[k]
        stay = 0.3 + 0.5 * regularity
        base_trans[i] = base_trans[i] * (1 - stay)
        base_trans[i, i] += stay
        base_trans[i] /= base_trans[i].sum()

    mats, transitions, weekday_flag = {}, {}, {}
    day_ids = [f"d{k:03d}" for k in range(n_days)]
    for k, day in enumerate(day_ids):
        weekday_flag[day] = "weekend" if k % 7 >= 5 else "weekday"
        mat = np.zeros((room_count, N_BINS), dtype=np.int8)
        cur = int(rng.integers(0, room_count))
        trans_list = []
        clock_min = 0.0
        for b in range(N_BINS):
            mat[cur, b] = 1
            nxt = rng.choice(room_count, p=base_trans[cur])
            if nxt != cur:
                trans_list.append([rooms[cur], rooms[nxt], f"{b:02d}:00:00", float(BIN_MINUTES)])
                if len(trans_list) >= 40:
                    trans_list = trans_list[-40:]
            cur = nxt
        mats[day] = mat
        transitions[day] = trans_list

    n_train = int(n_days * 0.7)
    return dict(
        house_id=f"synth_{adjacency_type}_{room_count}r_{regularity}",
        rooms=rooms,
        room_idx=room_idx,
        n_rooms=room_count,
        mats=mats,
        weekday_flag=weekday_flag,
        transitions=transitions,
        train_days=day_ids[:n_train],
        test_days=day_ids[n_train:],
        edges=edges,
    )


# ---------------------------------------------------------------------------
# 8. Per-house full evaluation (predictors + AUC tables)
# ---------------------------------------------------------------------------
def evaluate_house(house: dict) -> dict:
    hid = house["house_id"]
    auc_rows = []
    shuffle_rows = []

    # --- fit HMM per room on TRAIN ---
    hmm_models = {}
    for room_i, room in enumerate(house["rooms"]):
        seq = np.concatenate([house["mats"][d][room_i] for d in house["train_days"]]) if house["train_days"] else np.zeros(1, dtype=np.int8)
        hmm_models[room] = fit_hmm_room(hid, room, seq)

    trans_model = fit_transition_model(house, house["train_days"], shuffle=False)
    trans_model_shuffled = fit_transition_model(house, house["train_days"], shuffle=True)

    for lookahead_min in LOOKAHEADS_MIN:
        lb = LOOKAHEAD_BINS[lookahead_min]
        s, l = preheat_scores(house, lb)
        lo, hi = bootstrap_auc_ci(s, l)
        auc_rows.append(dict(house=hid, lookahead_min=lookahead_min, predictor="preheat_knn",
                              auc=safe_auc(s, l), ci_low=lo, ci_high=hi, n_pairs=len(s)))

        s, l = hmm_scores(house, lb, hmm_models)
        lo, hi = bootstrap_auc_ci(s, l)
        auc_rows.append(dict(house=hid, lookahead_min=lookahead_min, predictor="hmm",
                              auc=safe_auc(s, l), ci_low=lo, ci_high=hi, n_pairs=len(s)))

        s, l = transition_scores(house, lb, trans_model)
        lo, hi = bootstrap_auc_ci(s, l)
        auc_rows.append(dict(house=hid, lookahead_min=lookahead_min, predictor="transition_markov",
                              auc=safe_auc(s, l), ci_low=lo, ci_high=hi, n_pairs=len(s)))

        s_sh, l_sh = transition_scores(house, lb, trans_model_shuffled)
        lo_sh, hi_sh = bootstrap_auc_ci(s_sh, l_sh)
        shuffle_rows.append(dict(house=hid, lookahead_min=lookahead_min, predictor="transition_markov_shuffled",
                                  auc=safe_auc(s_sh, l_sh), ci_low=lo_sh, ci_high=hi_sh, n_pairs=len(s_sh)))

    return dict(auc_rows=auc_rows, shuffle_rows=shuffle_rows, hmm_models=hmm_models, trans_model=trans_model)


# ---------------------------------------------------------------------------
# 9. Thermal comparison at matched-FPR operating points for one house
# ---------------------------------------------------------------------------
def thermal_comparison(house: dict, hmm_models: dict, trans_model: dict, max_days: int, max_rooms: int) -> list[dict]:
    hid = house["house_id"]
    lb = LOOKAHEAD_BINS[CONTROL_LOOKAHEAD_MIN]
    activity = {r: sum(house["mats"][d][i].sum() for d in house["test_days"]) for i, r in enumerate(house["rooms"])}
    top_rooms = sorted(activity, key=activity.get, reverse=True)[:max_rooms]
    if not top_rooms:
        return []
    room_local_idx = {r: house["room_idx"][r] for r in top_rooms}
    sub_edges = [(a, b) for a, b in house["edges"] if a in room_local_idx and b in room_local_idx]
    test_days = house["test_days"][:max_days]
    if not test_days:
        return []

    def preheat_pred_fn(room_full_i, day, t):
        return None  # not used; PreHeat forecast built separately below

    # Build forecast matrices (subset rooms) per predictor for the chosen days.
    Tk_trans = np.linalg.matrix_power(trans_model["t_bin"], lb)
    room_activity_vec = np.array([activity[r] for r in house["rooms"]])
    fallback_dist = room_activity_vec / max(room_activity_vec.sum(), 1)

    occ_mats, fc_preheat, fc_hmm, fc_trans = [], [], [], []
    # precompute per-house per-room train stacks bucketed by weekday for PreHeat
    bucket_train = {}
    for bucket in ("weekday", "weekend"):
        ids = [d for d in house["train_days"] if house["weekday_flag"][d] == bucket]
        bucket_train[bucket] = (ids, np.stack([house["mats"][d] for d in ids]) if ids else None)

    for d in test_days:
        full_mat = house["mats"][d]
        occ_sub = full_mat[[house["room_idx"][r] for r in top_rooms], :]
        occ_mats.append(occ_sub)

        bucket = house["weekday_flag"][d]
        ids, stack = bucket_train[bucket]
        fc_ph = np.zeros((len(top_rooms), N_BINS))
        fc_h = np.zeros((len(top_rooms), N_BINS))
        fc_tr = np.zeros((len(top_rooms), N_BINS))
        for ri, r in enumerate(top_rooms):
            full_i = house["room_idx"][r]
            model = hmm_models[r]
            Tk_room = np.linalg.matrix_power(model["trans"], lb)
            emit1 = model["emit"][:, 1]
            for t in range(1, N_BINS - lb):
                target = t + lb
                if stack is not None and len(ids) >= KNN_K:
                    train_room = stack[:, full_i, :t]
                    test_room = full_mat[full_i, :t]
                    dist = np.abs(test_room[None, :] - train_room).sum(axis=-1)
                    topk = np.argsort(dist)[:KNN_K]
                    fc_ph[ri, t] = stack[topk, full_i, target].mean()
                obs = full_mat[full_i, t - 1]
                like = model["emit"][:, obs]
                like_sum = like.sum()
                post = like / like_sum if like_sum > 0 else np.array([0.5, 0.5])
                fc_h[ri, t] = float((post @ Tk_room) @ emit1)
                col = full_mat[:, t - 1]
                cur = col.astype(float) / col.sum() if col.sum() > 0 else fallback_dist
                dist_r = cur @ Tk_trans
                fc_tr[ri, t] = dist_r[full_i]
        fc_preheat.append(fc_ph)
        fc_hmm.append(fc_h)
        fc_trans.append(fc_tr)

    results = []
    for predictor_name, fc_list in (("preheat_knn", fc_preheat), ("hmm", fc_hmm), ("transition_markov", fc_trans)):
        all_scores = np.concatenate([m.ravel() for m in fc_list])
        for fp_rate in FP_RATES:
            thresh = float(np.quantile(all_scores, 1 - fp_rate)) if len(all_scores) else 0.5
            energy, miss = simulate_thermal(top_rooms, sub_edges, occ_mats, fc_list, thresh)
            results.append(dict(house=hid, predictor=predictor_name, fp_rate=fp_rate,
                                 threshold=thresh, energy_kwh=energy, misstime_min=miss,
                                 n_days=len(test_days), rooms_used=top_rooms))
    return results


def savings_table(thermal_rows: list[dict], group_key: str) -> list[dict]:
    by_group: dict = {}
    for row in thermal_rows:
        by_group.setdefault(row[group_key], {}).setdefault(row["fp_rate"], {})[row["predictor"]] = row
    out = []
    for group, by_fp in by_group.items():
        for fp_rate, by_pred in by_fp.items():
            if "preheat_knn" not in by_pred:
                continue
            base = by_pred["preheat_knn"]
            for other_name in ("hmm", "transition_markov"):
                if other_name not in by_pred:
                    continue
                other = by_pred[other_name]
                savings_pct = (
                    100.0 * (base["energy_kwh"] - other["energy_kwh"]) / base["energy_kwh"]
                    if base["energy_kwh"] > 0
                    else 0.0
                )
                out.append(dict(
                    group=group, fp_rate=fp_rate, method_vs_baseline=f"{other_name}_vs_preheat_knn",
                    savings_pct=savings_pct, misstime_delta_min=other["misstime_min"] - base["misstime_min"],
                    baseline_energy_kwh=base["energy_kwh"], method_energy_kwh=other["energy_kwh"],
                ))
            if "hmm" in by_pred and "transition_markov" in by_pred:
                h, t = by_pred["hmm"], by_pred["transition_markov"]
                savings_pct = 100.0 * (h["energy_kwh"] - t["energy_kwh"]) / h["energy_kwh"] if h["energy_kwh"] > 0 else 0.0
                out.append(dict(
                    group=group, fp_rate=fp_rate, method_vs_baseline="transition_markov_vs_hmm",
                    savings_pct=savings_pct, misstime_delta_min=t["misstime_min"] - h["misstime_min"],
                    baseline_energy_kwh=h["energy_kwh"], method_energy_kwh=t["energy_kwh"],
                ))
    return out


def bootstrap_mean_ci(values: list[float], n_boot: int = N_BOOT_TOPOLOGY) -> tuple[float, float, float]:
    arr = np.array(values, dtype=float)
    point = float(arr.mean())
    if len(arr) < 2:
        return point, point, point
    boots = [float(RNG.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_examples(rows: list[dict], dataset_name: str) -> dict:
    examples = []
    for row in rows:
        meta = {f"metadata_{k}": v for k, v in row.items()}
        predict_field = "predict_auc" if "auc" in row else ("predict_savings_pct" if "savings_pct" in row else "predict_energy_kwh")
        predict_val = row.get("auc", row.get("savings_pct", row.get("energy_kwh")))
        examples.append({
            "input": f"{dataset_name} row: {json.dumps({k: v for k, v in row.items() if not isinstance(v, (list, dict))})}",
            "output": json.dumps(row, default=str),
            predict_field: json.dumps(predict_val),
            **meta,
        })
    return {"dataset": dataset_name, "examples": examples}


def main() -> None:
    t0 = time.time()
    data = load_data()
    adjacency_meta = data.get("metadata", {}).get("per_dataset_meta", {})

    houses = {}
    for ds in data["datasets"]:
        h = build_house(ds, adjacency_meta)
        houses[h["house_id"]] = h
    del data
    gc.collect()

    all_auc_rows, all_shuffle_rows, all_thermal_rows = [], [], []
    house_savings_summaries = []
    for hid, house in houses.items():
        logger.info(f"=== Evaluating real house: {hid} ===")
        t1 = time.time()
        eval_res = evaluate_house(house)
        all_auc_rows.extend(eval_res["auc_rows"])
        all_shuffle_rows.extend(eval_res["shuffle_rows"])
        logger.info(f"[{hid}] predictor AUC eval done in {time.time()-t1:.1f}s")

        t2 = time.time()
        thermal_rows = thermal_comparison(
            house, eval_res["hmm_models"], eval_res["trans_model"], MAX_THERMAL_TEST_DAYS, MAX_THERMAL_ROOMS
        )
        for r in thermal_rows:
            r["house"] = hid
        all_thermal_rows.extend(thermal_rows)
        logger.info(f"[{hid}] thermal sim done in {time.time()-t2:.1f}s ({len(thermal_rows)} rows)")

    real_savings = savings_table(all_thermal_rows, "house")
    for row in real_savings:
        row["scope"] = "real_casas_house"
    house_savings_summaries = list(real_savings)

    # --- Expanded synthetic topology sweep ---
    synth_configs = []
    for rc in (3, 4, 5, 6):
        for adj in ("linear", "star", "hallway", "loop"):
            synth_configs.append((rc, adj, 0.55))
    for reg in (0.25, 0.85):
        synth_configs.append((4, "hallway", reg))
    logger.info(f"Synthetic sweep: {len(synth_configs)} configs -> {synth_configs}")

    synth_auc_rows, synth_thermal_rows = [], []
    cross_topology_savings = {"transition_markov_vs_preheat_knn": [], "hmm_vs_preheat_knn": [], "transition_markov_vs_hmm": []}
    for seed, (rc, adj, reg) in enumerate(synth_configs):
        cfg_id = f"{adj}_r{rc}_reg{reg}"
        house = generate_synth_house(rc, adj, reg, seed=1000 + seed)
        eval_res = evaluate_house(house)
        for row in eval_res["auc_rows"]:
            row["config"] = cfg_id
        synth_auc_rows.extend(eval_res["auc_rows"])

        thermal_rows = thermal_comparison(
            house, eval_res["hmm_models"], eval_res["trans_model"], max_days=len(house["test_days"]), max_rooms=rc
        )
        for r in thermal_rows:
            r["house"] = cfg_id
        synth_thermal_rows.extend(thermal_rows)

        cfg_savings = savings_table(thermal_rows, "house")
        for row in cfg_savings:
            row["scope"] = "synthetic_topology"
            row["config"] = cfg_id
            row["room_count"], row["adjacency_type"], row["regularity"] = rc, adj, reg
        house_savings_summaries.extend(cfg_savings)

        for row in cfg_savings:
            key = row["method_vs_baseline"]
            if key in cross_topology_savings and row["fp_rate"] == FP_RATES[len(FP_RATES) // 2]:
                cross_topology_savings[key].append(row["savings_pct"])
        logger.info(f"[synth {cfg_id}] done ({time.time()-t0:.1f}s elapsed total)")

    cross_topology_ci = {}
    for key, vals in cross_topology_savings.items():
        if vals:
            point, lo, hi = bootstrap_mean_ci(vals)
            cross_topology_ci[key] = dict(mean_savings_pct=point, ci_low=lo, ci_high=hi, n_configs=len(vals),
                                           ci_excludes_zero=bool(lo > 0 or hi < 0))

    real_house_savings_vals = [
        r["savings_pct"] for r in real_savings
        if r["method_vs_baseline"] == "transition_markov_vs_preheat_knn" and r.get("scope") == "real_casas_house"
    ]
    real_savings_summary = dict(
        n_houses=len(houses), note="n=4 houses, too small for a bootstrap CI to be meaningful; NOT pooled with synthetic CI",
        transition_vs_preheat_mean_pct=float(np.mean(real_house_savings_vals)) if real_house_savings_vals else None,
        raw_values=real_house_savings_vals,
    )

    output = {
        "metadata": {
            "method_name": "PreHeat-KNN vs HMM vs Room-Transition-Markov occupancy forecasting + RC-thermal-sim energy savings",
            "houses_evaluated": list(houses.keys()),
            "hmm_backend": "hmmlearn.CategoricalHMM" if HAVE_HMMLEARN else "markov_chain_fallback_only",
            "hmm_fallback_rooms": HMM_FALLBACK_LOG,
            "lookaheads_min": LOOKAHEADS_MIN,
            "n_bootstrap_auc": N_BOOT_AUC,
            "n_bootstrap_topology": N_BOOT_TOPOLOGY,
            "fp_rates_evaluated": FP_RATES,
            "synthetic_configs_run": [dict(room_count=rc, adjacency_type=adj, regularity=reg) for rc, adj, reg in synth_configs],
            "cross_topology_bootstrap_ci": cross_topology_ci,
            "real_casas_savings_summary": real_savings_summary,
            "deviations_from_plan": DEVIATIONS,
            "runtime_seconds": time.time() - t0,
        },
        "datasets": [
            build_examples(all_auc_rows, "real_casas_auc_evaluation"),
            build_examples(all_shuffle_rows, "real_casas_shuffle_control"),
            build_examples(house_savings_summaries, "energy_savings_real_and_synthetic"),
            build_examples(synth_auc_rows, "synthetic_topology_auc_evaluation"),
        ],
    }

    OUT_PATH.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.2f} MB) in {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
