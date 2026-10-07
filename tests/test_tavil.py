"""Numerical and protocol regression tests for the TAVIL adaptation.

These use small synthetic inputs; the recorded data audit and complete nested
experiment provide the complementary empirical verification.
"""

import json
import sqlite3
import tempfile
import unittest
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.signal import periodogram
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from models import detectors, losses, trainers, utils
from tavil.protocol import (A_U_ORIENTATIONS, NORMAL_GROUPS, RECORDINGS,
                            REFIT_A_U_ASSIGNMENTS, make_protocol)
from tavil import preprocessing as prep
from tavil import experiment as experiment


def feature_table(ids, cycles=3, n_features=4):
    """Unique feature rows make accidental reordering visible in tests."""
    metadata = {r["recording_id"]: r for r in RECORDINGS}
    recording_id = np.repeat(np.asarray(ids, dtype=np.int64), cycles)
    return {
        "x": np.arange(len(recording_id) * n_features, dtype=np.float32).reshape(-1, n_features),
        "recording_id": recording_id,
        "cycle_id": np.tile(np.arange(cycles), len(ids)),
        "t": np.asarray([metadata[int(i)]["t"] for i in recording_id]),
        "true_label": np.asarray([metadata[int(i)]["true_label"] for i in recording_id]),
    }


class CoreTests(unittest.TestCase):
    def test_dense_architecture_and_symmetric_decoder(self):
        model = detectors.DenseSVDD(n_in=11, n_latent=3, n_h=7)
        encoder = [m for m in model.encoder if isinstance(m, nn.Linear)]
        decoder = [m for m in model.decoder if isinstance(m, nn.Linear)]
        self.assertEqual([(m.in_features, m.out_features) for m in encoder],
                         [(11, 7), (7, 7), (7, 7), (7, 3)])
        self.assertEqual([(m.in_features, m.out_features) for m in decoder],
                         [(3, 7), (7, 7), (7, 7), (7, 11)])
        self.assertTrue(all(m.bias is None for m in encoder + decoder))
        self.assertEqual(sum(isinstance(m, nn.ReLU) for m in model.modules()), 6)
        self.assertFalse(any(isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.Dropout))
                             for m in model.modules()))
        self.assertEqual(tuple(model(torch.randn(5, 11)).shape), (5,))

    def test_original_ae_loss_uses_only_unlabeled(self):
        output = torch.tensor([10., 2., 4.], requires_grad=True)
        loss = losses.AELoss()(output, torch.tensor([1., 0., 0.]))
        self.assertEqual(float(loss.detach()), 3.)
        loss.backward()
        torch.testing.assert_close(output.grad, torch.tensor([0., .5, .5]))

    def test_original_pusvdd_absolute_pu_risk(self):
        distance = torch.tensor([.3, .5, .7, 1.2], dtype=torch.float64)
        target = torch.tensor([1., 0., 1., 0.], dtype=torch.float64)
        alpha = .2
        positive_risk = alpha * (1 / (distance[target == 1] ** 2 + 1e-6)).mean()
        negative_risk = ((distance[target == 0] ** 2).mean()
                         - alpha * (distance[target == 1] ** 2).mean())
        torch.testing.assert_close(losses.PUSVDDLoss(alpha)(distance, target),
                                   positive_risk + negative_risk.abs())
        # Include the branch where the corrected normal risk is negative.
        distance = torch.tensor([10., .1], dtype=torch.float64)
        target = torch.tensor([1., 0.], dtype=torch.float64)
        expected = alpha / (100 + 1e-6) + abs(.01 - alpha * 100)
        self.assertAlmostEqual(float(losses.PUSVDDLoss(alpha)(distance, target)), expected)

    def test_center_uses_all_samples_and_original_epsilon_rule(self):
        class IdentitySVDD(detectors.DeepSVDD):
            n_latent = 3

            def encode(self, x):
                return x

        model = IdentitySVDD()
        x = torch.tensor([[-.04, .01, 0.], [0., .03, 0.]])
        loader = DataLoader(TensorDataset(x, torch.tensor([1., 0.])), batch_size=1)
        utils.set_center(model, loader, torch.device("cpu"), eps=.1)
        torch.testing.assert_close(model.c, torch.tensor([-.1, .1, 0.]))
        torch.testing.assert_close(model.estimate(x), 1 - torch.exp(-torch.norm(x - model.c, dim=1)))

    def test_tavil_loop_exact_epoch_training_without_local_best_reload(self):
        torch.manual_seed(42)
        x = torch.randn(12, 5)
        loader = DataLoader(TensorDataset(x, torch.zeros(12)), batch_size=5)
        model = detectors.DenseSVDD(n_in=5, n_latent=2, n_h=4)
        events = []
        curve = experiment.fit(
            model, losses.AELoss(), loader, loader, 3,
            experiment.Config(n_h=4, q=2, lr=1e-4, batch_size=5),
            torch.device("cpu"),
            lambda **event: events.append(event),
            "unit_test",
        )
        self.assertEqual(len(curve["train"]), 3)
        self.assertEqual(len(curve["val"]), 3)
        self.assertEqual([e["epoch"] for e in events], [1, 2, 3])
        self.assertTrue(np.isfinite(curve["train"]).all())
        self.assertTrue(np.isfinite(curve["val"]).all())

    def test_upstream_trainer_keeps_original_checkpoint_contract(self):
        class WeightedDetector(detectors.Detector):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(2.))

            def forward(self, x):
                return self.weight * x[:, 0]

        class SignedLoss(losses.Loss):
            def forward(self, output, target):
                return ((1 - 2 * target) * output).mean()

        train = DataLoader(TensorDataset(torch.ones(4, 1), torch.zeros(4)), batch_size=4)
        val = DataLoader(TensorDataset(torch.ones(4, 1), torch.ones(4)), batch_size=4)
        model = WeightedDetector()
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = str(Path(folder) / "best.pt")
            trainer = trainers.Trainer(torch.device("cpu"))
            trainer.fit(model, SignedLoss(), train, val, checkpoint, n_epoch=3,
                        learning_rate=.01, weight_decay=1e-3)
            self.assertEqual(len(trainer.train_losses), 3)
            self.assertEqual(len(trainer.valid_losses), 3)
            saved = torch.load(checkpoint, weights_only=True)
            torch.testing.assert_close(model.state_dict()["weight"], saved["weight"])


class ProtocolTests(unittest.TestCase):
    def test_complete_grouped_matched_protocol_and_round_robin(self):
        folds = make_protocol()
        self.assertEqual(len(folds), 5)
        labels = {r["recording_id"]: r["true_label"] for r in RECORDINGS}
        times = {r["recording_id"]: r["t"] for r in RECORDINGS}
        self.assertEqual([times[r] for r in NORMAL_GROUPS[0]],
                         [22381, 44870, 67965, 107158, 151332, 197553, 241171, 287008, 332626])
        self.assertEqual(Counter(r for f in folds for r in f["test_ids"]),
                         Counter({r["recording_id"]: 1 for r in RECORDINGS}))
        for fold in folds:
            test, refit = set(fold["test_ids"]), set(fold["refit_ids"])
            self.assertEqual((len(test), sum(labels[r] for r in test)), (10, 1))
            self.assertEqual((len(refit), sum(labels[r] for r in refit)), (40, 4))
            self.assertFalse(test & refit)
            self.assertEqual(len(fold["inner"]), 6)
            train_count, val_count = Counter(), Counter()
            for inner in fold["inner"]:
                train, val = set(inner["train_ids"]), set(inner["val_ids"])
                self.assertEqual((len(train), sum(labels[r] for r in train)), (20, 2))
                self.assertEqual((len(val), sum(labels[r] for r in val)), (20, 2))
                self.assertFalse(train & val or train & test or val & test)
                self.assertEqual(train | val, refit)
                train_count.update(train)
                val_count.update(val)
            self.assertEqual(set(train_count.values()), {3})
            self.assertEqual(set(val_count.values()), {3})

    def test_orientations_and_all_refit_partitions(self):
        self.assertEqual(A_U_ORIENTATIONS, [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertEqual(len(REFIT_A_U_ASSIGNMENTS), 6)
        self.assertEqual([labeled for labeled, _ in REFIT_A_U_ASSIGNMENTS],
                         [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        for labeled, hidden in REFIT_A_U_ASSIGNMENTS:
            self.assertEqual(set(labeled) | set(hidden), set(range(4)))
            self.assertFalse(set(labeled) & set(hidden))


class PreprocessingTests(unittest.TestCase):
    def test_floor_cuts_and_exact_nyquist_band_edge(self):
        cuts = prep.rf_cuts(107)
        np.testing.assert_array_equal(cuts, [0, 10, 21, 32, 42, 53, 64, 74, 85, 96, 107])
        self.assertEqual(int(np.diff(cuts).sum()), 107)
        for width in (74., 80., 2000., 2500.):
            edges = prep.make_band_edges(width)
            self.assertEqual(edges[0], 0.)
            self.assertEqual(edges[-1], 2000.)
            self.assertTrue((np.diff(edges) > 0).all())

    def test_native_psd_matches_scipy_and_conserves_all_mass(self):
        rng = np.random.default_rng(7)
        for n in (20, 21, 107, 108):
            signals = rng.normal(size=(3, 6, n))
            frequencies, density = periodogram(
                signals, fs=4000, window="hann", detrend="constant",
                scaling="density", nfft=None, return_onesided=True, axis=-1)
            knots, cumulative = prep.spectral_mass(signals)
            native_total = (density * 4000 / n).sum(axis=-1)
            np.testing.assert_allclose(cumulative[..., -1], native_total, rtol=2e-15)
            self.assertEqual(knots[0], 0.)
            self.assertEqual(knots[-1], 2000.)
            self.assertTrue((np.diff(knots) > 0).all())
            bands = prep.integrate_bands(knots, cumulative, prep.make_band_edges(73.5))
            self.assertTrue((bands >= 0).all())
            np.testing.assert_allclose(bands.sum(axis=-1), native_total, rtol=3e-15)

    def test_shape_is_scale_invariant_and_zero_spectrum_finite(self):
        rng = np.random.default_rng(9)
        signals = rng.normal(size=(6, 107))
        edges = prep.make_band_edges(71.25)
        knots, cumulative = prep.spectral_mass(signals)
        scaled_knots, scaled_cumulative = prep.spectral_mass(signals * 100)
        shape = prep.shape_features(knots, cumulative, edges)
        scaled = prep.shape_features(scaled_knots, scaled_cumulative, edges)
        np.testing.assert_allclose(shape, scaled, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose((10 ** (shape / 10)).sum(axis=-1), 1., rtol=1e-14)
        zero_knots, zero_cumulative = prep.spectral_mass(np.zeros((6, 107)))
        np.testing.assert_array_equal(prep.shape_features(zero_knots, zero_cumulative, edges),
                                      np.full((6, len(edges) - 1), -300.))

    @staticmethod
    def synthetic_psd(folder):
        rng = np.random.default_rng(123)
        arrays = {key: [] for key in ("recording_id", "cycle_id", "t", "true_label", "durations")}
        psd = {"recordings": {}, "fingerprint": "synthetic-source-v1", "audit": []}
        rows = np.array([(cycle, phase, segment) for cycle in range(2)
                         for phase in range(2) for segment in range(10)])
        for rid in (10, 20, 30):
            knots, cumulative = prep.spectral_mass(rng.normal(size=(40, 6, 16)))
            native = {
                "cycle_id": np.array([100 * rid, 100 * rid + 1]),
                "t": np.full(2, rid), "true_label": np.full(2, int(rid != 30)),
                "durations": np.full((2, 2), .4 if rid == 30 else .04),
                "lengths": np.array([16]), "rows_16": rows,
                "knots_16": knots, "cumulative_16": cumulative,
            }
            path = Path(folder) / f"native-{rid}.npz"
            np.savez(path, **native)
            psd["recordings"][rid] = path
            arrays["recording_id"].append(np.full(2, rid))
            for key in arrays.keys() - {"recording_id"}:
                arrays[key].append(native[key])
        psd.update({key: np.concatenate(values) for key, values in arrays.items()})
        return psd

    def test_preprocessing_fits_only_train_and_freezes_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            psd = self.synthetic_psd(folder)
            cache = Path(folder) / "fold.npz"
            train, evaluation, metadata = prep.prepare_fold(psd, [20, 10], [30], cache)
            self.assertEqual(metadata["Tref_rising"], .004)
            self.assertEqual(metadata["Tref_falling"], .004)
            self.assertEqual(metadata["band_edges_rising"], [0., 1000., 2000.])
            self.assertEqual(metadata["n_features"], 240)
            self.assertFalse(metadata["expected_3300_matches"])
            self.assertEqual(metadata["feature_order"][:3],
                             ["r1_accX_band1", "r1_accX_band2", "r1_accY_band1"])
            self.assertEqual(metadata["feature_order"][120], "f1_accX_band1")
            np.testing.assert_array_equal(train["recording_id"], [20, 20, 10, 10])
            np.testing.assert_array_equal(train["cycle_id"], [2000, 2001, 1000, 1001])
            np.testing.assert_array_equal(train["true_label"], [1, 1, 1, 1])
            raw_train = prep.transform_features(psd, [20, 10], [0., 1000., 2000.], [0., 1000., 2000.])
            raw_eval = prep.transform_features(psd, [30], [0., 1000., 2000.], [0., 1000., 2000.])
            np.testing.assert_allclose(metadata["scaler_mean"], raw_train["x"].mean(axis=0))
            np.testing.assert_allclose(metadata["scaler_scale"], raw_train["x"].std(axis=0))
            expected_eval = ((raw_eval["x"] - np.asarray(metadata["scaler_mean"]))
                             / np.asarray(metadata["scaler_scale"]))
            np.testing.assert_allclose(evaluation["x"], expected_eval, rtol=1e-6)
            np.testing.assert_allclose(train["x"].mean(axis=0), 0., atol=1e-7)
            # Cached results must not refit; different source identities invalidate cache.
            with patch.object(prep, "transform_features", side_effect=AssertionError("refit cached fold")):
                cached_train, cached_eval, cached_metadata = prep.prepare_fold(psd, [20, 10], [30], cache)
            np.testing.assert_array_equal(cached_train["x"], train["x"])
            np.testing.assert_array_equal(cached_eval["x"], evaluation["x"])
            self.assertEqual(cached_metadata, metadata)
            psd["fingerprint"] = "synthetic-source-v2"
            changed = prep.prepare_fold(psd, [20, 10], [30], cache)[2]
            self.assertEqual(changed["fingerprint"], "synthetic-source-v2")

    def test_raw_loader_last_277_complete_cycles_and_physical_calibration(self):
        rid, n = 1234567890000000, 23
        recording = {"recording_id": rid, "t": 4415, "true_label": 1}
        scales = np.array([prep.ACC_SCALE] * 3 + [prep.GYRO_SCALE] * 3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / f"record__{rid}.sqlite"
            with sqlite3.connect(path) as database:
                database.executescript(
                    "CREATE TABLE metadata (key TEXT, value TEXT);"
                    "CREATE TABLE windows (window_id INTEGER, type TEXT, start_timestamp_ns INTEGER, end_timestamp_ns INTEGER);"
                    "CREATE TABLE samples (window_id INTEGER, sample_time_ns INTEGER, acc_x INTEGER, acc_y INTEGER, acc_z INTEGER, gyr_x INTEGER, gyr_y INTEGER, gyr_z INTEGER);")
                database.executemany("INSERT INTO metadata VALUES (?, ?)", [
                    ("capture", json.dumps({"first_timestamp_ns": rid})),
                    ("imu", json.dumps({"acc_scale_g_per_count": prep.ACC_SCALE,
                                         "gyro_scale_dps_per_count": prep.GYRO_SCALE}))])
                for cycle in range(279):
                    for phase in range(2):
                        wid = 2 * cycle + phase
                        start = rid + wid * n * 250000
                        database.execute("INSERT INTO windows VALUES (?, ?, ?, ?)",
                                         (wid, prep.PHASES[phase], start, start + n * 250000))
                        counts = (np.arange(n)[:, None] + 3 * np.arange(6)[None, :] + cycle) % 23 - 11
                        stop = n - 1 if cycle == 278 and phase == 1 else n
                        database.executemany("INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                             [(wid, start + sample * 250000, *map(int, counts[sample]))
                                              for sample in range(stop)])
            database.close()
            native, audit = prep._recording_psd(path, recording)
            self.assertEqual(audit["complete_cycles"], 278)
            self.assertEqual(audit["selected_cycles"], 277)
            self.assertEqual(audit["incomplete_windows"], 1)
            self.assertEqual((native["cycle_id"][0], native["cycle_id"][-1]), (2, 554))
            np.testing.assert_array_equal(native["lengths"], [2, 3])
            np.testing.assert_allclose(native["durations"], n / 4000)
            rows = native["rows_2"]
            index = np.flatnonzero((rows == [0, 0, 0]).all(axis=1))[0]
            counts = (np.arange(2)[:, None] + 3 * np.arange(6)[None, :] + 1) % 23 - 11
            knots, expected = prep.spectral_mass((counts * scales).T)
            np.testing.assert_array_equal(native["knots_2"], knots)
            np.testing.assert_allclose(native["cumulative_2"][index], expected, rtol=1e-14)


class ExperimentTests(unittest.TestCase):
    def test_labels_preserve_features_ground_truth_and_computed_alpha(self):
        fold = make_protocol()[0]
        split = fold["inner"][0]
        table = feature_table(split["train_ids"])
        original = {key: value.copy() for key, value in table.items()}
        datasets = [experiment.FeatureDataset(table, [rid]) for rid in split["train_anomaly_ids"]]
        for data in datasets:
            np.testing.assert_array_equal(data.x.numpy(), original["x"])
            np.testing.assert_array_equal(data.true_label, original["true_label"])
            self.assertEqual(data.alpha, 1 / 19)
            mask = (data.true_label == 1) & (data.pu_label.numpy() == 0)
            self.assertEqual(int(mask.sum()), 3)
        self.assertFalse(np.array_equal(datasets[0].pu_label.numpy(), datasets[1].pu_label.numpy()))
        for key, value in table.items():
            np.testing.assert_array_equal(value, original[key])
        refit = feature_table(fold["refit_ids"])
        for labeled, _ in REFIT_A_U_ASSIGNMENTS:
            data = experiment.FeatureDataset(refit, [fold["refit_anomaly_ids"][i] for i in labeled])
            self.assertEqual(data.alpha, 1 / 19)
        normal = next(r for r in split["train_ids"] if r not in split["train_anomaly_ids"])
        with self.assertRaises(ValueError):
            experiment.FeatureDataset(table, [normal])

    def test_train_and_val_shuffle_reproducibly_without_dropping_last_batch(self):
        split = make_protocol()[0]["inner"][0]
        data = experiment.FeatureDataset(feature_table(split["train_ids"]), [split["train_anomaly_ids"][0]])
        first = experiment.make_loader(data, batch_size=13)
        second = experiment.make_loader(data, batch_size=13)
        self.assertIsInstance(first.sampler, RandomSampler)
        self.assertFalse(first.drop_last)
        batches = list(first)
        replay = list(second)
        self.assertEqual([len(x) for x, _ in batches], [13, 13, 13, 13, 8])
        for (x, y), (xr, yr) in zip(batches, replay):
            torch.testing.assert_close(x, xr)
            torch.testing.assert_close(y, yr)
        self.assertFalse(torch.equal(torch.cat([x for x, _ in batches]), data.x))

    def test_two_stage_aggregate_epoch_and_config_ties(self):
        fold = make_protocol()[0]
        inner = [(feature_table(split["train_ids"]), feature_table(split["val_ids"]), split)
                 for split in fold["inner"]]
        configs = [experiment.Config(n_h=3, q=2), experiment.Config(n_h=4, q=2)]
        curves, models, phase_counts = {}, [], Counter()
        original_new_model = experiment.new_model

        def fresh_model(*args):
            model = original_new_model(*args)
            self.assertIsNone(model.c)
            models.append(model)
            return model

        def fake_fit(model, criterion, train, val, epochs, config, device, progress, phase):
            phase_counts[phase] += 1
            key = (config.n_h, tuple(train.dataset.recording_id),
                   tuple(train.dataset.pu_label.tolist()), tuple(val.dataset.pu_label.tolist()))
            if phase == "ae_selection":
                # Local optima disagree. The aggregate chooses epoch 2.
                curve = [1., 4., 8.] if phase_counts[phase] % 2 else [9., 2., 8.]
                curves[key] = curve
                self.assertEqual(epochs, 3)
            elif phase == "ae_rerun":
                self.assertEqual(epochs, 2)
                self.assertIsNone(model.c)
                self.assertIsInstance(criterion, losses.AELoss)
                curve = curves[key][:epochs]
            else:
                self.assertIsNotNone(model.c)
                self.assertIsInstance(criterion, losses.PUSVDDLoss)
                self.assertEqual(criterion.alpha, 1 / 19)
                # Equal best SVDD epochs 1 and 3 must select epoch 1.
                curve = [3., 7., 3.]
            return {"train": curve.copy(), "val": curve.copy()}

        with tempfile.TemporaryDirectory() as folder:
            with patch.object(experiment, "fit", side_effect=fake_fit), \
                    patch.object(experiment, "new_model", side_effect=fresh_model):
                selected = experiment.select_config(inner, configs, 3, torch.device("cpu"),
                                                    Path(folder), lambda **_: None)
            self.assertEqual(selected["pretrain_epoch"], 2)
            self.assertEqual(selected["epoch"], 1)
            self.assertEqual(selected["config_index"], 0)
            self.assertEqual(selected["val_loss"], 3.)
            self.assertEqual(phase_counts, {"ae_selection": 48, "ae_rerun": 48, "svdd_selection": 48})
            self.assertEqual(len(models), 96)
            self.assertEqual(len({id(m) for m in models}), 96)
            self.assertEqual(len(list(Path(folder).glob("*_ae.json"))), 48)
            self.assertEqual(len(list(Path(folder).glob("*_svdd.json"))), 48)
            experiment.write_json(Path(folder) / "selected.json", selected)
            # The empirical verifier reads the same evidence independently.
            from tavil.verify_run import check_selection
            out = Path(folder) / "outer"
            (out / "curves").mkdir(parents=True)
            for path in Path(folder).glob("config*.json"):
                path.rename(out / "curves" / path.name)
            (Path(folder) / "selection.json").rename(out / "curves" / "selection.json")
            (Path(folder) / "selected.json").rename(out / "selected.json")
            self.assertEqual(check_selection(out, list(map(asdict, configs)), 3, fold,
                                             [(train, val) for train, val, _ in inner]), selected)
            damaged = out / "curves" / "config0_inner0_00_svdd.json"
            record = json.loads(damaged.read_text())
            record["pretrain_epoch"] = 1
            experiment.write_json(damaged, record)
            with self.assertRaises(ValueError):
                check_selection(out, list(map(asdict, configs)), 3, fold,
                                [(train, val) for train, val, _ in inner])

    def test_recording_median_metrics_and_six_statistics(self):
        table = feature_table(make_protocol()[0]["test_ids"], cycles=277)
        anomaly = np.r_[np.full(176, .6), np.ones(101)]
        normal_scores = [.7, .1, .15, .2, .25, .3, .35, .4, .45]
        scores = np.r_[anomaly, *[np.full(277, value) for value in normal_scores]]
        result = experiment.evaluate(scores, table)
        self.assertAlmostEqual(result["recording_AUROC"], 8 / 9)
        self.assertAlmostEqual(result["recording_AP"], .5)
        self.assertAlmostEqual(result["cycle_AUROC"], roc_auc_score(table["true_label"], scores))
        self.assertAlmostEqual(result["cycle_AP"], average_precision_score(table["true_label"], scores))
        self.assertEqual(len(result["recording_statistics"]), 10)
        stats = result["recording_statistics"][0]
        for key, value in {"min": .6, "q25": .6, "median": .6,
                           "q75": 1., "max": 1., "mean": anomaly.mean()}.items():
            self.assertAlmostEqual(stats[key], value)

    def test_refit_exact_epochs_all_members_and_checkpoint_replay(self):
        fold = make_protocol()[0]
        rng = np.random.default_rng(42)
        refit, test = feature_table(fold["refit_ids"]), feature_table(fold["test_ids"])
        refit["x"] = rng.normal(size=refit["x"].shape).astype(np.float32)
        test["x"] = rng.normal(size=test["x"].shape).astype(np.float32)
        selected = {"config": {"n_h": 4, "q": 2, "lr": 1e-4, "batch_size": 128},
                    "pretrain_epoch": 2, "epoch": 1}
        metadata = {"fs": 4000, "RF": 10, "M": 4, "representation": "SHAPE",
                    "Tref_rising": .004, "Tref_falling": .004,
                    "band_edges_rising": [0., 1000., 2000.],
                    "band_edges_falling": [0., 1000., 2000.],
                    "scaler_mean": [0.] * 4, "scaler_scale": [1.] * 4,
                    "n_features": 4, "feature_order": [f"column{i}" for i in range(4)],
                    "acc_scale": prep.ACC_SCALE, "gyro_scale": prep.GYRO_SCALE,
                    "preprocessing_version": prep.VERSION}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            score = experiment.run_refit(refit, test, fold, selected, metadata,
                                         torch.device("cpu"), output, lambda **_: None)
            checkpoint = torch.load(output / "ensemble.pt", weights_only=True)
            self.assertEqual(len(checkpoint["ensemble_members"]), 6)
            for m, (labeled, hidden) in zip(checkpoint["ensemble_members"], REFIT_A_U_ASSIGNMENTS):
                self.assertEqual(m["labeled_refit_ids"], [fold["refit_anomaly_ids"][i] for i in labeled])
                self.assertEqual(m["hidden_refit_ids"], [fold["refit_anomaly_ids"][i] for i in hidden])
                self.assertEqual(m["alpha"], 1 / 19)
                self.assertEqual(tuple(m["center"].shape), (2,))
            for key in metadata:
                self.assertIn(key, checkpoint["shared"])
            self.assertEqual(checkpoint["shared"]["selected_pretrain_epoch"], 2)
            self.assertEqual(checkpoint["shared"]["selected_epoch"], 1)
            with np.load(output / "test_scores.npz") as recorded:
                np.testing.assert_array_equal(score, recorded["member_scores"].mean(axis=0))
            np.testing.assert_array_equal(score, experiment.predict_ensemble(output / "ensemble.pt", test))
            for path in (output / "members").glob("*_training.json"):
                curve = json.loads(path.read_text())
                self.assertEqual(len(curve["ae"]["train"]), 2)
                self.assertEqual(len(curve["svdd"]["train"]), 1)
                self.assertEqual(curve["ae"]["val"], [])
                self.assertEqual(curve["svdd"]["val"], [])


if __name__ == "__main__":
    unittest.main()
