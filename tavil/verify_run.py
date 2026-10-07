"""Independently verify a completed TAVIL run against its saved evidence.

Recompute TRAIN-only preprocessing, two-stage selection, checkpoint predictions,
recording medians and AP/AUROC. A smaller Emax requires --allow-diagnostic and
is explicitly reported as a diagnostic rather than the complete protocol.
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from tavil import preprocessing as prep
from tavil.experiment import code_hash, predict_ensemble, write_json
from tavil.protocol import (A_U_ORIENTATIONS, RECORDINGS,
                            REFIT_A_U_ASSIGNMENTS, make_protocol)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_native(cache):
    audit = read_json(cache / "native" / "audit.json")
    require(audit["n_cycles"] == 13850, "Expected exactly 50 × 277 native cycles")
    require(len(audit["recordings"]) == 50, "Expected 50 audited recordings")
    require(all(r["selected_cycles"] == 277 for r in audit["recordings"]),
            "Native recordings must contain the last 277 complete cycles")
    require(Counter(r["complete_cycles"] for r in audit["recordings"]) == {278: 48, 277: 2},
            "Actual-data cycle audit differs from the documented 48/2 recording counts")
    audited = {row["recording_id"]: row for row in audit["recordings"]}
    psd = {"recordings": {}, "fingerprint": audit["fingerprint"]}
    arrays = {key: [] for key in (*prep.TABLE_KEYS, "durations")}
    for recording in RECORDINGS:
        rid = recording["recording_id"]
        path = cache / "native" / f"{rid}.npz"
        psd["recordings"][rid] = path
        sidecar = read_json(path.with_suffix(".json"))
        require(sidecar["provenance"]["recording"] == recording,
                f"Native-cache recording definition differs: {rid}")
        source = sidecar["provenance"]["source"]
        digest = hashlib.sha256()
        source_path = Path(source["path"])
        with source_path.open("rb") as database:
            for chunk in iter(lambda: database.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        require(source_path.stat().st_size == source["size"] and digest.hexdigest() == source["sha256"],
                f"SQLite data changed after native PSD computation: {rid}")
        with np.load(path, allow_pickle=False) as native:
            require(len(native["cycle_id"]) == len(np.unique(native["cycle_id"])) == 277,
                    f"Native cycle identities are not unique: {rid}")
            require(np.all(native["true_label"] == recording["true_label"]),
                    f"Native ground truth changed: {rid}")
            require(np.all(np.diff(native["cycle_id"]) > 0) and
                    int(native["cycle_id"][0]) == audited[rid]["first_cycle_id"] and
                    int(native["cycle_id"][-1]) == audited[rid]["last_cycle_id"],
                    f"Native last-cycle audit/identity differs: {rid}")
            arrays["recording_id"].append(np.full(277, rid, dtype=np.int64))
            for key in (*prep.TABLE_KEYS[1:], "durations"):
                arrays[key].append(native[key])
    psd.update({key: np.concatenate(parts) for key, parts in arrays.items()})
    return psd


def check_fold_cache(psd, path, train_ids, eval_ids):
    """Independent scaler fit on raw TRAIN features, never on evaluation rows."""
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        train, evaluation = [
            {key: saved[f"{prefix}_{key}"] for key in ("x", *prep.TABLE_KEYS)}
            for prefix in ("train", "eval")]
    require(metadata["fit_recording_ids"] == train_ids, f"Fit IDs differ: {path}")
    require(metadata["eval_recording_ids"] == eval_ids, f"Evaluation IDs differ: {path}")
    require(not set(train_ids) & set(eval_ids), f"Recording leakage: {path}")
    require(metadata["fingerprint"] == psd["fingerprint"], f"Stale cache: {path}")
    require((metadata["fs"], metadata["RF"], metadata["M"]) == (4000, 10, 4),
            f"Preprocessing constants differ: {path}")
    require((metadata["acc_scale"], metadata["gyro_scale"]) ==
            (prep.ACC_SCALE, prep.GYRO_SCALE), f"Physical calibration differs: {path}")
    tref = np.median(psd["durations"][np.isin(psd["recording_id"], train_ids)] / 10, axis=0)
    np.testing.assert_array_equal([metadata["Tref_rising"], metadata["Tref_falling"]], tref)
    edges = [np.r_[np.arange(0., 2000., 4 / duration), 2000.] for duration in tref]
    for phase, boundaries in zip(prep.PHASES, edges):
        np.testing.assert_array_equal(metadata[f"band_edges_{phase}"], boundaries)
    require(metadata["feature_order"] == prep.feature_order(*edges), f"Feature order differs: {path}")
    raw = prep.transform_features(psd, train_ids, *edges)
    scaler = StandardScaler().fit(raw["x"])
    np.testing.assert_allclose(metadata["scaler_mean"], scaler.mean_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(metadata["scaler_scale"], scaler.scale_, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(train["x"], scaler.transform(raw["x"]).astype(np.float32))
    for table, ids in ((train, train_ids), (evaluation, eval_ids)):
        expected_ids = np.repeat(ids, 277)
        np.testing.assert_array_equal(table["recording_id"], expected_ids)
        require(table["x"].shape == (277 * len(ids), metadata["n_features"]),
                f"Feature shape differs: {path}")
        require(np.isfinite(table["x"]).all(), f"Nonfinite features: {path}")
        for rid in ids:
            source = psd["recording_id"] == rid
            selected = table["recording_id"] == rid
            for key in prep.TABLE_KEYS[1:]:
                np.testing.assert_array_equal(table[key][selected], psd[key][source])
    require(len(metadata["feature_order"]) == metadata["n_features"], f"Feature count differs: {path}")
    return train, evaluation, metadata


def check_curve(curve, epochs, validation):
    require(len(curve["train"]) == epochs, "Training epoch count differs")
    require(len(curve["val"]) == (epochs if validation else 0), "Validation epoch count differs")
    require(np.isfinite(curve["train"]).all() and np.isfinite(curve["val"]).all(),
            "Nonfinite saved loss curve")


def computed_alpha(table, labeled_ids):
    unlabeled = ~np.isin(table["recording_id"], labeled_ids)
    return float(np.count_nonzero((table["true_label"] == 1) & unlabeled) / unlabeled.sum())


def check_selection(out, configs, emax, fold, cached_tables):
    computed = []
    for c, config in enumerate(configs):
        ae_curves, svdd_curves = [], []
        for k, split in enumerate(fold["inner"]):
            train, val = cached_tables[k]
            for i, j in A_U_ORIENTATIONS:
                ae = read_json(out / "curves" / f"config{c}_inner{k}_{i}{j}_ae.json")
                require(ae["config"] == config and ae["load_best"] is False,
                        "AE config or local-best policy differs")
                alpha = computed_alpha(train, [split["train_anomaly_ids"][i]])
                require(alpha == computed_alpha(val, [split["val_anomaly_ids"][j]]) == 1 / 19,
                        "Inner alpha differs from true/PU labels")
                require(ae["alpha"] == alpha, "AE alpha metadata differs")
                check_curve(ae["ae"], emax, True)
                ae_curves.append(ae["ae"]["val"])
        ae_mean = np.mean(ae_curves, axis=0)
        e_ae = int(np.argmin(ae_mean)) + 1
        for k, _ in enumerate(fold["inner"]):
            for i, j in A_U_ORIENTATIONS:
                ae = read_json(out / "curves" / f"config{c}_inner{k}_{i}{j}_ae.json")
                svdd = read_json(out / "curves" / f"config{c}_inner{k}_{i}{j}_svdd.json")
                require(svdd["config"] == config and svdd["load_best"] is False,
                        "SVDD config or local-best policy differs")
                require(svdd["pretrain_epoch"] == e_ae and svdd["alpha"] == 1 / 19,
                        "SVDD did not use the aggregate AE epoch/alpha")
                check_curve(svdd["ae"], e_ae, True)
                check_curve(svdd["svdd"], emax, True)
                for key in ("train", "val"):
                    np.testing.assert_allclose(svdd["ae"][key], ae["ae"][key][:e_ae],
                                               rtol=1e-6, atol=1e-6)
                require(np.isfinite(svdd["center"]).all(), "Nonfinite inner center")
                svdd_curves.append(svdd["svdd"]["val"])
        svdd_mean = np.mean(svdd_curves, axis=0)
        e_svdd = int(np.argmin(svdd_mean)) + 1
        computed.append(dict(config=config, config_index=c, pretrain_epoch=e_ae,
                             epoch=e_svdd, val_loss=float(svdd_mean[e_svdd - 1]),
                             ae_mean_curve=ae_mean.tolist(), svdd_mean_curve=svdd_mean.tolist()))
    saved = read_json(out / "curves" / "selection.json")
    require(saved == computed, "Saved configuration selection differs from aggregate curves")
    expected = min(computed, key=lambda result: result["val_loss"])
    selected = read_json(out / "selected.json")
    require(selected == expected, "Selected config/epoch does not respect the first minimum")
    return selected


def recompute_metrics(scores, table):
    rows = []
    for rid in np.unique(table["recording_id"]):
        mask = table["recording_id"] == rid
        values = scores[mask]
        labels = np.unique(table["true_label"][mask])
        require(len(values) == 277 and len(labels) == 1, "Invalid TEST recording cycle/label count")
        rows.append(dict(recording_id=int(rid), t=int(table["t"][mask][0]),
                         true_label=int(labels[0]), n_cycles=277, min=float(values.min()),
                         q25=float(np.quantile(values, .25)), median=float(np.median(values)),
                         q75=float(np.quantile(values, .75)), max=float(values.max()),
                         mean=float(values.mean())))
    y = [row["true_label"] for row in rows]
    require(len(rows) == 10 and sum(y) == 1, "TEST must be 1 anomaly + 9 normal recordings")
    medians = [row["median"] for row in rows]
    return dict(recording_AUROC=float(roc_auc_score(y, medians)),
                recording_AP=float(average_precision_score(y, medians)),
                cycle_AUROC=float(roc_auc_score(table["true_label"], scores)),
                cycle_AP=float(average_precision_score(table["true_label"], scores)),
                recording_statistics=rows)


def verify_run(output, cache, allow_diagnostic=False):
    output, cache = Path(output), Path(cache)
    start = time.perf_counter()
    manifest = read_json(output / "manifest.json")
    require(manifest["seed"] == 42, "Seed must be 42")
    require(manifest["emax"] == 100 or allow_diagnostic,
            "Emax differs from 100; this is a diagnostic, not the complete protocol")
    require(manifest["code_sha256"] == code_hash(), "Implementation differs from the run manifest")
    require(manifest["protocol"] == make_protocol(), "Manifest protocol differs from frozen recording splits")
    require(read_json(output / "progress.json")["status"] == "complete", "Run is not complete")
    psd = load_native(cache)
    require(manifest["data_fingerprint"] == psd["fingerprint"], "Data fingerprint differs")
    torch.set_num_threads(manifest["threads"])
    torch.use_deterministic_algorithms(True)
    device = torch.device(manifest["device"])
    final = read_json(output / "results.json")
    require(len(final["outer_results"]) == 5, "Expected five individual outer results")
    summaries, feature_counts, statistics = [], [], []
    for fold in manifest["protocol"]:
        o = fold["outer_fold"]
        out, fold_cache = output / f"outer{o}", cache / "folds" / f"outer{o}"
        tables = []
        for k, split in enumerate(fold["inner"]):
            train, val, metadata = check_fold_cache(psd, fold_cache / f"inner{k}.npz",
                                                   split["train_ids"], split["val_ids"])
            tables.append((train, val))
            feature_counts.append(metadata["n_features"])
        selected = check_selection(out, manifest["configs"], manifest["emax"], fold, tables)
        del tables
        refit, test, metadata = check_fold_cache(psd, fold_cache / "refit.npz",
                                                fold["refit_ids"], fold["test_ids"])
        feature_counts.append(metadata["n_features"])
        checkpoint = torch.load(out / "ensemble.pt", map_location="cpu", weights_only=True)
        shared = checkpoint["shared"]
        require(len(checkpoint["ensemble_members"]) == 6, "Expected six REFIT ensemble members")
        required = {"model_config", "selected_pretrain_epoch", "selected_epoch", "alpha",
                    "fs", "RF", "M", "representation", "Tref_rising", "Tref_falling",
                    "band_edges_rising", "band_edges_falling", "scaler_mean", "scaler_scale",
                    "n_features", "feature_order", "acc_scale", "gyro_scale", "preprocessing_version"}
        require(required <= shared.keys(), "Checkpoint omits required shared state")
        require(shared["selected_pretrain_epoch"] == selected["pretrain_epoch"] and
                shared["selected_epoch"] == selected["epoch"], "REFIT epochs differ from selection")
        require(shared["model_config"]["n_in"] == metadata["n_features"], "Checkpoint input count differs")
        for key in metadata:
            require(shared[key] == metadata[key], f"Checkpoint preprocessing differs: {key}")
        for m, ((labeled_idx, hidden_idx), member) in enumerate(zip(
                REFIT_A_U_ASSIGNMENTS, checkpoint["ensemble_members"])):
            labeled = [fold["refit_anomaly_ids"][i] for i in labeled_idx]
            hidden = [fold["refit_anomaly_ids"][i] for i in hidden_idx]
            require(member["labeled_refit_ids"] == labeled and member["hidden_refit_ids"] == hidden,
                    "REFIT A/U assignment differs")
            require(member["alpha"] == shared["alpha"] == computed_alpha(refit, labeled) == 1 / 19,
                    "REFIT alpha differs from true/PU labels")
            require(torch.isfinite(member["center"]).all().item(), "Nonfinite REFIT center")
            curves = read_json(out / "members" / f"member{m}_training.json")
            check_curve(curves["ae"], selected["pretrain_epoch"], False)
            check_curve(curves["svdd"], selected["epoch"], False)
        replay_table = prep.transform_with_metadata(psd, fold["test_ids"], shared)
        for key in test:
            np.testing.assert_array_equal(test[key], replay_table[key])
        with np.load(out / "test_scores.npz", allow_pickle=False) as recorded:
            scores, members = recorded["score"], recorded["member_scores"]
            require(members.shape == (6, 2770), "Expected six scores per 2770 TEST cycles")
            np.testing.assert_array_equal(scores, members.mean(axis=0))
            for key in prep.TABLE_KEYS:
                np.testing.assert_array_equal(recorded[key], test[key])
        replay = predict_ensemble(checkpoint, replay_table, device, selected["config"]["batch_size"])
        np.testing.assert_array_equal(scores, replay)
        metrics = recompute_metrics(scores, test)
        result = read_json(out / "results.json")
        require(result == final["outer_results"][o], "Outer result differs from final aggregation")
        for key, expected in metrics.items():
            require(result[key] == expected, f"Saved evaluation differs: outer{o}/{key}")
        summaries.append(dict(outer_fold=o, selected_pretrain_epoch=selected["pretrain_epoch"],
                              selected_epoch=selected["epoch"],
                              **{key: value for key, value in metrics.items() if key != "recording_statistics"}))
        statistics.extend(dict(outer_fold=o, **row) for row in metrics["recording_statistics"])
        print(f"Verified outer {o}: selection, preprocessing, six members, replay and metrics", flush=True)
    require(final["recording_statistics"] == statistics, "Final recording statistics differ")
    for level in ("recording", "cycle"):
        for metric in ("AUROC", "AP"):
            key = f"{level}_{metric}"
            require(final[f"mean_{key}"] == float(np.mean([row[key] for row in summaries])),
                    f"Across-fold mean differs: {key}")
            require(key not in final, "Unexpected pooled raw-score metric")
    config_count = len(manifest["configs"])
    report = dict(status="passed", complete_protocol=manifest["emax"] == 100,
                  elapsed_seconds=time.perf_counter() - start, emax=manifest["emax"],
                  configs=config_count, outer_folds=5, preprocessing_folds=35,
                  ae_selection_runs=120 * config_count, ae_rerun_svdd_runs=120 * config_count,
                  refit_models=30, n_features_per_preprocessing_fold=feature_counts,
                  checkpoint_raw_psd_to_score_replay="exact",
                  checks=["fresh SQLite source SHA256", "recording split/label/cycle identity", "TRAIN-only Tref/bands/scaler",
                          "24-run AE aggregate and exact rerun", "24-run SVDD aggregate and first ties",
                          "six REFIT assignments and exact selected epochs", "complete ensemble checkpoints",
                          "frozen preprocessing and prediction replay", "member score mean",
                          "recording medians/six statistics/AP/AUROC", "five-fold means without pooling"],
                  outer_results=summaries)
    write_json(output / "verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/TAVIL/full"))
    parser.add_argument("--cache", type=Path, default=Path("datasets/TAVIL/.cache"))
    parser.add_argument("--allow-diagnostic", action="store_true")
    args = parser.parse_args()
    try:
        report = verify_run(args.output, args.cache, args.allow_diagnostic)
    except Exception as error:
        write_json(args.output / "verification.json", dict(status="failed", error=str(error)))
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
