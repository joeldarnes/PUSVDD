"""TRAIN-only RF10-M4-SHAPE preprocessing, with reusable native PSDs.

Every one-sided periodogram ordinate contributes mass ``PSD * fs/N``.  For
interpolation we spread that mass uniformly over its frequency-bin cell,
whose upper edge is ``min(f + fs/(2*N), fs/2)``.  The first cell starts at
zero; the last ends exactly at Nyquist, also for odd segment lengths.
Interpolating this cumulative mass therefore conserves *all* native power,
including DC and the Nyquist bin.  This explicit convention avoids silently
losing DC when interpolating cumulative masses at the bin centres.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
from scipy.signal import periodogram
from sklearn.preprocessing import StandardScaler

FS, RF, M, CYCLES = 4000, 10, 4, 277
ACC_SCALE = 6.103515625e-5
GYRO_SCALE = 4.76837158203125e-4
VERSION = "RF10-M4-SHAPE-v1-bin-cell-cumulative"
CHANNELS = ("accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ")
PHASES = ("rising", "falling")
TABLE_KEYS = ("recording_id", "cycle_id", "t", "true_label")


def rf_cuts(n: int, rf: int = RF) -> np.ndarray:
    """Integer floor cuts; every sample belongs to exactly one segment."""
    if n < 2 * rf:
        raise ValueError(f"Need at least {2 * rf} samples per phase, got {n}")
    return np.arange(rf + 1, dtype=np.int64) * n // rf


def spectral_mass(segments: np.ndarray, fs: float = FS):
    """Return cell-boundary frequencies and native cumulative PSD masses."""
    segments = np.asarray(segments, dtype=np.float64)
    n = segments.shape[-1]
    frequencies, density = periodogram(
        segments, fs=fs, window="hann", detrend="constant",
        scaling="density", nfft=None, return_onesided=True, axis=-1,
    )
    mass = density * fs / n
    knots = np.r_[0.0, np.minimum(frequencies + fs / (2 * n), fs / 2)]
    knots[-1] = fs / 2
    cumulative = np.concatenate(
        (np.zeros(mass.shape[:-1] + (1,)), np.cumsum(mass, axis=-1)), axis=-1,
    )
    return knots, cumulative


def make_band_edges(width: float, fmax: float = FS / 2) -> np.ndarray:
    if not np.isfinite(width) or width <= 0 or fmax <= 0:
        raise ValueError("Band width and maximum frequency must be positive")
    return np.r_[np.arange(0.0, fmax, width), float(fmax)]


def integrate_bands(knots: np.ndarray, cumulative: np.ndarray,
                    edges: np.ndarray) -> np.ndarray:
    """Interpolate cumulative mass at edges and difference adjacent values."""
    knots, edges = np.asarray(knots), np.asarray(edges)
    if (np.any(np.diff(knots) <= 0) or np.any(np.diff(edges) <= 0)
            or edges[0] != 0 or edges[-1] != knots[-1]):
        raise ValueError("Edges must increase from zero to exact Nyquist")
    upper = np.clip(np.searchsorted(knots, edges, side="right"), 1, len(knots) - 1)
    lower = upper - 1
    fraction = (edges - knots[lower]) / (knots[upper] - knots[lower])
    values = cumulative[..., lower] + fraction * (
        cumulative[..., upper] - cumulative[..., lower]
    )
    return np.maximum(np.diff(values, axis=-1), 0.0)


def shape_features(knots: np.ndarray, cumulative: np.ndarray,
                   edges: np.ndarray) -> np.ndarray:
    power = integrate_bands(knots, cumulative, edges)
    total = cumulative[..., -1, None]
    ratio = np.divide(power, total, out=np.zeros_like(power), where=total > 0)
    return 10 * np.log10(np.maximum(ratio, 1e-30))


def _json(value):
    return json.dumps(value, default=lambda x: x.tolist() if isinstance(x, np.ndarray)
                      else int(x) if isinstance(x, np.integer) else float(x),
                      sort_keys=True, separators=(",", ":"))


def _source_fingerprint(path: Path) -> dict:
    """Hash content as well as path: equal timestamps cannot hide edits."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "size": path.stat().st_size,
            "sha256": digest.hexdigest()}


def _recording_psd(path: Path, recording: dict):
    """Validate SQLite/calibration/clocks and select the last complete pairs."""
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        metadata = {key: json.loads(value) for key, value in
                    connection.execute("SELECT key,value FROM metadata")}
        if metadata["capture"]["first_timestamp_ns"] != recording["recording_id"]:
            raise ValueError(f"Recording identity disagrees with metadata: {path}")
        imu = metadata["imu"]
        if (imu["acc_scale_g_per_count"] != ACC_SCALE
                or imu["gyro_scale_dps_per_count"] != GYRO_SCALE):
            raise ValueError(f"Unexpected physical calibration: {path}")
        windows = connection.execute(
            "SELECT window_id,type,start_timestamp_ns,end_timestamp_ns "
            "FROM windows ORDER BY start_timestamp_ns,window_id"
        ).fetchall()
        dtype = [(name, "<i8") for name in
                 ("window_id", "sample_time_ns", "acc_x", "acc_y", "acc_z",
                  "gyr_x", "gyr_y", "gyr_z")]
        samples = np.fromiter(connection.execute(
            "SELECT window_id,sample_time_ns,acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z "
            "FROM samples ORDER BY window_id,sample_time_ns"
        ), dtype=dtype)
    matrix = samples.view(np.int64).reshape(-1, 8)
    ids, first, counts = np.unique(matrix[:, 0], return_index=True, return_counts=True)
    slices = {int(wid): slice(int(start), int(start + count))
              for wid, start, count in zip(ids, first, counts)}
    complete = {}
    for wid, phase, start, end in windows:
        section = slices.get(wid, slice(0, 0))
        times = matrix[section, 1]
        complete[wid] = bool(
            len(times) >= RF * 2 and end > start
            and 0 <= times[0] - start < 1e9 / FS
            and 0 < end - times[-1] <= 1e9 / FS
            and np.all(np.diff(times) == 1_000_000_000 // FS)
        )
    pairs = [(rising, falling) for rising, falling in zip(windows, windows[1:])
             if rising[1] == "rising" and falling[1] == "falling"
             and rising[3] == falling[2]
             and complete[rising[0]] and complete[falling[0]]]
    if len(pairs) < CYCLES:
        raise ValueError(f"{path.name}: only {len(pairs)} complete cycles, need {CYCLES}")
    selected = pairs[-CYCLES:]
    durations = np.array([[(phase[3] - phase[2]) / 1e9 for phase in pair]
                          for pair in selected], dtype=np.float64)
    result = {"cycle_id": np.array([pair[0][0] for pair in selected], dtype=np.int64),
              "t": np.full(CYCLES, recording["t"], dtype=np.int64),
              "true_label": np.full(CYCLES, recording["true_label"], dtype=np.int64),
              "durations": durations}
    groups = {}
    scale = np.array([ACC_SCALE] * 3 + [GYRO_SCALE] * 3)
    phase_counts = []
    for cycle, pair in enumerate(selected):
        for phase, window in enumerate(pair):
            values = matrix[slices[window[0]], 2:].astype(np.float64) * scale
            phase_counts.append(len(values))
            cuts = rf_cuts(len(values))
            for segment, (start, stop) in enumerate(zip(cuts, cuts[1:])):
                rows, signals = groups.setdefault(stop - start, ([], []))
                rows.append((cycle, phase, segment))
                signals.append(values[start:stop].T)
    result["lengths"] = np.array(sorted(groups), dtype=np.int64)
    for n, (rows, signals) in groups.items():
        knots, cumulative = spectral_mass(np.stack(signals))
        result[f"rows_{n}"] = np.array(rows, dtype=np.int64)
        result[f"knots_{n}"] = knots
        result[f"cumulative_{n}"] = cumulative
    audit = {"recording_id": recording["recording_id"], "t": recording["t"],
             "true_label": recording["true_label"], "complete_cycles": len(pairs),
             "selected_cycles": CYCLES, "first_cycle_id": int(result["cycle_id"][0]),
             "last_cycle_id": int(result["cycle_id"][-1]),
             "incomplete_windows": sum(not value for value in complete.values()),
             "phase_duration_min": durations.min(axis=0).tolist(),
             "phase_duration_median": np.median(durations, axis=0).tolist(),
             "phase_duration_max": durations.max(axis=0).tolist(),
             "phase_samples_min": min(phase_counts), "phase_samples_max": max(phase_counts),
             "segment_lengths": result["lengths"].tolist(),
             "sample_interval_ns": 1_000_000_000 // FS,
             "acc_scale": ACC_SCALE, "gyro_scale": GYRO_SCALE,
             "acquisition_validation_ok": metadata.get("validation", {}).get("ok")}
    return result, audit


def load_psd_cache(data_dir: Path, cache_dir: Path,
                   recordings: list[dict] | None = None) -> dict:
    """Create/load one native PSD file per recording, independent of any split."""
    if recordings is None:
        from .protocol import RECORDINGS
        recordings = RECORDINGS
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {int(path.stem.split("__")[1]): path
             for path in data_dir.glob("*.sqlite")}
    if not recordings or len({r["recording_id"] for r in recordings}) != len(recordings):
        raise ValueError("Recording definitions must be nonempty and unique")
    missing = {r["recording_id"] for r in recordings} - paths.keys()
    if missing:
        raise ValueError(f"Missing SQLite recordings: {sorted(missing)}")
    result = {"recordings": {}, "audit": []}
    arrays = {key: [] for key in (*TABLE_KEYS, "durations")}
    sources = []
    for recording in recordings:
        rid = recording["recording_id"]
        source = _source_fingerprint(paths[rid])
        sources.append(source)
        provenance = {"version": VERSION, "source": source, "recording": recording}
        target = cache_dir / f"{rid}.npz"
        sidecar = target.with_suffix(".json")
        cached = json.loads(sidecar.read_text("utf-8")) if sidecar.exists() else {}
        if not target.exists() or cached.get("provenance") != provenance:
            native, audit = _recording_psd(paths[rid], recording)
            np.savez(target, **native)
            sidecar.write_text(_json({"provenance": provenance, "audit": audit}), "utf-8")
        else:
            audit = cached["audit"]
        with np.load(target, allow_pickle=False) as native:
            arrays["recording_id"].append(np.full(CYCLES, rid, dtype=np.int64))
            for key in (*TABLE_KEYS[1:], "durations"):
                arrays[key].append(native[key])
        result["recordings"][rid] = target
        result["audit"].append(audit)
        print(f"PSD {len(result['audit'])}/{len(recordings)}: {rid}, "
              f"{audit['complete_cycles']} complete -> {CYCLES}", flush=True)
    result.update({key: np.concatenate(parts) for key, parts in arrays.items()})
    result["fingerprint"] = hashlib.sha256(
        _json({"version": VERSION, "sources": sources, "recordings": recordings}).encode()
    ).hexdigest()
    (cache_dir / "audit.json").write_text(_json({
        "preprocessing_version": VERSION, "fingerprint": result["fingerprint"],
        "recordings": result["audit"], "n_cycles": len(result["cycle_id"]),
        "fs": FS, "RF": RF, "acc_scale": ACC_SCALE, "gyro_scale": GYRO_SCALE,
        "duration_source": "(windows.end_timestamp_ns-start_timestamp_ns)/1e9",
        "spectral_mass_interpolation": __doc__,
    }), "utf-8")
    return result


def feature_order(band_edges_rising, band_edges_falling) -> list[str]:
    return [f"{phase[0]}{segment + 1}_{channel}_band{band + 1}"
            for phase, edges in zip(PHASES, (band_edges_rising, band_edges_falling))
            for segment in range(RF) for channel in CHANNELS
            for band in range(len(edges) - 1)]


def transform_features(psd: dict, ids: list[int], band_edges_rising,
                       band_edges_falling) -> dict:
    """Transform native masses without fitting anything; preserve ID alignment."""
    ids = [int(rid) for rid in ids]
    edges = [np.asarray(band_edges_rising), np.asarray(band_edges_falling)]
    dimensions = [RF * len(CHANNELS) * (len(boundaries) - 1) for boundaries in edges]
    offsets = [0, dimensions[0]]
    tables = []
    for rid in ids:
        with np.load(psd["recordings"][rid], allow_pickle=False) as native:
            n_cycles = len(native["cycle_id"])
            x = np.empty((n_cycles, sum(dimensions)), dtype=np.float64)
            for n in native["lengths"]:
                rows = native[f"rows_{n}"]
                cumulative = native[f"cumulative_{n}"]
                knots = native[f"knots_{n}"]
                for phase in range(2):
                    selected = rows[:, 1] == phase
                    view = x[:, offsets[phase]:offsets[phase] + dimensions[phase]].reshape(
                        n_cycles, RF, len(CHANNELS), len(edges[phase]) - 1)
                    view[rows[selected, 0], rows[selected, 2]] = shape_features(
                        knots, cumulative[selected], edges[phase])
            tables.append({"x": x, "recording_id": np.full(n_cycles, rid, dtype=np.int64),
                           **{key: native[key] for key in TABLE_KEYS[1:]}})
    return {key: np.concatenate([table[key] for table in tables])
            for key in ("x", *TABLE_KEYS)}


def transform_with_metadata(psd: dict, ids: list[int], metadata: dict) -> dict:
    """Replay checkpoint preprocessing using its frozen bands and scaler."""
    if metadata["preprocessing_version"] != VERSION:
        raise ValueError("Unsupported checkpoint preprocessing version")
    table = transform_features(psd, ids, metadata["band_edges_rising"],
                               metadata["band_edges_falling"])
    if (table["x"].shape[1] != metadata["n_features"]
            or feature_order(metadata["band_edges_rising"], metadata["band_edges_falling"])
            != metadata["feature_order"]):
        raise ValueError("Checkpoint feature schema disagrees with transform")
    table["x"] = ((table["x"] - np.asarray(metadata["scaler_mean"]))
                  / np.asarray(metadata["scaler_scale"])).astype(np.float32)
    if not np.isfinite(table["x"]).all():
        raise ValueError("Nonfinite checkpoint preprocessing result")
    return table


def prepare_fold(psd: dict, train_ids: list[int], eval_ids: list[int],
                 cache_path: Path):
    """Fit duration/bands/scaler on TRAIN only; return frozen transformed tables."""
    train_ids, eval_ids = list(map(int, train_ids)), list(map(int, eval_ids))
    if (not train_ids or not eval_ids or len(set(train_ids)) != len(train_ids)
            or len(set(eval_ids)) != len(eval_ids) or set(train_ids) & set(eval_ids)
            or not (set(train_ids) | set(eval_ids)) <= psd["recordings"].keys()):
        raise ValueError("TRAIN/evaluation recording IDs must be distinct and disjoint")
    provenance = {"preprocessing_version": VERSION, "fingerprint": psd["fingerprint"],
                  "fit_recording_ids": train_ids, "eval_recording_ids": eval_ids}
    cache_path = Path(cache_path)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadata"]))
            if all(metadata.get(key) == value for key, value in provenance.items()):
                tables = [{key: cache[f"{prefix}_{key}"] for key in ("x", *TABLE_KEYS)}
                          for prefix in ("train", "eval")]
                return *tables, metadata
    fit_mask = np.isin(psd["recording_id"], train_ids)
    tref = np.median(psd["durations"][fit_mask] / RF, axis=0)
    rising, falling = [make_band_edges(M / duration) for duration in tref]
    train = transform_features(psd, train_ids, rising, falling)
    evaluation = transform_features(psd, eval_ids, rising, falling)
    for table, ids in ((train, train_ids), (evaluation, eval_ids)):
        for rid in ids:
            selected = table["recording_id"] == rid
            if (selected.sum() != np.count_nonzero(psd["recording_id"] == rid)
                    or len(np.unique(table["cycle_id"][selected])) != selected.sum()):
                raise ValueError("Cycle counts or identity disagree with native PSD cache")
    scaler = StandardScaler().fit(train["x"])
    train["x"] = scaler.transform(train["x"]).astype(np.float32)
    evaluation["x"] = scaler.transform(evaluation["x"]).astype(np.float32)
    columns = feature_order(rising, falling)
    n_features = len(columns)
    if (train["x"].shape[1] != n_features or evaluation["x"].shape[1] != n_features
            or not np.isfinite(train["x"]).all() or not np.isfinite(evaluation["x"]).all()):
        raise ValueError("Nonfinite features or feature order/width mismatch")
    metadata = {**provenance, "fs": FS, "RF": RF, "M": M,
                "representation": "SHAPE", "Tref_rising": float(tref[0]),
                "Tref_falling": float(tref[1]), "band_edges_rising": rising.tolist(),
                "band_edges_falling": falling.tolist(),
                "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                "n_features": n_features, "feature_order": columns,
                "acc_scale": ACC_SCALE, "gyro_scale": GYRO_SCALE,
                "n_train_cycles": len(train["x"]), "n_eval_cycles": len(evaluation["x"]),
                "expected_3300_matches": n_features == 3300,
                "fit_scope": "all TRAIN cycles (A union U); evaluation excluded",
                "spectral_mass_interpolation": "linear cumulative mass at native bin-cell boundaries"}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, metadata=_json(metadata), **{
        f"{prefix}_{key}": value for prefix, table in (("train", train), ("eval", evaluation))
        for key, value in table.items()})
    cache_path.with_suffix(".json").write_text(_json(metadata), "utf-8")
    return train, evaluation, metadata
