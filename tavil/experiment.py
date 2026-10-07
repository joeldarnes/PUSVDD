"""Two-stage inner selection, six-member refit and recording-level evaluation."""
import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from models import detectors, losses, trainers, utils
from tavil.protocol import A_U_ORIENTATIONS, REFIT_A_U_ASSIGNMENTS, make_protocol


@dataclass(frozen=True)
class Config:
    n_h: int = 500
    q: int = 128
    lr: float = 1e-4
    batch_size: int = 128

    def __post_init__(self):
        if min(self.n_h, self.q, self.batch_size) <= 0 or self.lr <= 0:
            raise ValueError("Config values must be positive")


class FeatureDataset(Dataset):
    """Labels are masks over the existing row order; features never move."""
    def __init__(self, table, labeled_ids):
        self.x = torch.from_numpy(np.asarray(table['x'], dtype=np.float32))
        self.recording_id = np.asarray(table['recording_id'])
        self.true_label = np.asarray(table['true_label'])
        self.cycle_id = np.asarray(table['cycle_id'])
        self.t = np.asarray(table['t'])
        if self.x.ndim != 2 or any(len(v) != len(self.x) for v in
                (self.recording_id, self.true_label, self.cycle_id, self.t)):
            raise ValueError("Feature rows and cycle metadata must stay aligned")
        mask = np.isin(self.recording_id, labeled_ids)
        if not set(labeled_ids).issubset(set(self.recording_id)):
            raise ValueError("Labeled recording is absent from this table")
        if np.any(self.true_label[mask] != 1):
            raise ValueError("Only anomaly recordings may be labeled A")
        self.pu_label = torch.from_numpy(mask.astype(np.int64))
        self.n_features = self.x.shape[1]

    @property
    def alpha(self):
        u = self.pu_label.numpy() == 0
        if not u.any():
            raise ValueError("An unlabeled population is required")
        return float(np.count_nonzero((self.true_label == 1) & u) / u.sum())

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.pu_label[index]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, seed=42):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def first_epoch(curves):
    values = np.asarray(curves, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite, equally long epoch curves")
    return int(values.mean(axis=0).argmin()) + 1


def write_json(path, value):
    def convert(v):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, default=convert, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(20):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05)


def new_model(n_features, config, device):
    set_seed()
    return detectors.DenseSVDD(n_in=n_features, n_h=config.n_h,
                               n_latent=config.q).to(device)


def fit(model, criterion, train_loader, valid_loader, epochs, config, device, progress, phase):
    """Exact-epoch TAVIL loop; deliberately independent of upstream Trainer."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-3)
    train_losses, valid_losses = [], []
    for epoch in range(int(epochs)):
        model.train()
        mean_train = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            mean_train += loss.item() / len(train_loader)
        train_losses.append(mean_train)

        mean_valid = None
        if valid_loader is not None:
            model.eval()
            value = 0.0
            with torch.no_grad():
                for x, y in valid_loader:
                    x, y = x.to(device), y.to(device)
                    loss = criterion(model(x), y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite validation loss")
                    value += loss.item() / len(valid_loader)
            mean_valid = value
            valid_losses.append(value)

        progress(phase=phase, epoch=epoch + 1, epochs=int(epochs),
                 train_loss=mean_train, val_loss=mean_valid)
    return dict(train=train_losses, val=valid_losses)


def inner_runs(inner_cache, config):
    for k, (train_table, val_table, split) in enumerate(inner_cache):
        for i, j in A_U_ORIENTATIONS:
            train = FeatureDataset(train_table, [split['train_anomaly_ids'][i]])
            val = FeatureDataset(val_table, [split['val_anomaly_ids'][j]])
            assert train.alpha == val.alpha == 1 / 19
            yield k, i, j, train, make_loader(train, config.batch_size), make_loader(val, config.batch_size)


def select_config(inner_cache, configs, emax, device, curve_dir, progress):
    """Every SVDD run is reinitialized and pretrained at the aggregate AE epoch."""
    results = []
    for c, config in enumerate(configs):
        ae_curves = []
        for k, i, j, train, train_loader, val_loader in inner_runs(inner_cache, config):
            path = curve_dir / f'config{c}_inner{k}_{i}{j}_ae.json'
            progress(config=c, inner=k, orientation=[i, j], phase='ae_selection')
            if path.exists():
                record = json.loads(path.read_text())
            else:
                model = new_model(train.n_features, config, device)
                curve = fit(model, losses.AELoss(), train_loader, val_loader,
                            emax, config, device, progress, 'ae_selection')
                record = dict(config=asdict(config), inner=k, orientation=[i, j],
                              alpha=train.alpha, ae=curve, load_best=False)
                write_json(path, record)
            assert len(record['ae']['val']) == emax
            ae_curves.append(record['ae']['val'])
        e_ae = first_epoch(ae_curves)
        svdd_curves = []
        for k, i, j, train, train_loader, val_loader in inner_runs(inner_cache, config):
            path = curve_dir / f'config{c}_inner{k}_{i}{j}_svdd.json'
            progress(config=c, inner=k, orientation=[i, j], phase='svdd_selection', selected_ae=e_ae)
            if path.exists():
                record = json.loads(path.read_text())
            else:
                model = new_model(train.n_features, config, device)
                pretrain = fit(model, losses.AELoss(), train_loader, val_loader,
                               e_ae, config, device, progress, 'ae_rerun')
                # With separate seeded loaders, validation does not alter TRAIN order.
                previous = json.loads((curve_dir / f'config{c}_inner{k}_{i}{j}_ae.json').read_text())
                np.testing.assert_allclose(pretrain['val'], previous['ae']['val'][:e_ae], rtol=1e-6, atol=1e-6)
                utils.set_center(model, train_loader, device, eps=0.1)
                # PU gets a fresh deterministic sampler. AE/center iterations must
                # not advance the PUSVDD minibatch generator.
                train_pu = make_loader(train, config.batch_size, seed=42)
                curve = fit(model, losses.PUSVDDLoss(train.alpha), train_pu,
                            val_loader, emax, config, device, progress, 'svdd_selection')
                record = dict(config=asdict(config), inner=k, orientation=[i, j],
                              alpha=train.alpha, pretrain_epoch=e_ae, ae=pretrain,
                              svdd=curve, center=model.c.cpu(), load_best=False,
                              ae_rerun_matches_stage1=True)
                write_json(path, record)
            assert record['pretrain_epoch'] == e_ae and len(record['svdd']['val']) == emax
            svdd_curves.append(record['svdd']['val'])
        e_svdd = first_epoch(svdd_curves)
        results.append(dict(config=asdict(config), config_index=c, pretrain_epoch=e_ae,
                            epoch=e_svdd, val_loss=float(np.mean(svdd_curves, axis=0)[e_svdd - 1]),
                            ae_mean_curve=np.mean(ae_curves, axis=0).tolist(),
                            svdd_mean_curve=np.mean(svdd_curves, axis=0).tolist()))
        write_json(curve_dir / 'selection.json', results)
    # min preserves config list order on ties.
    return min(results, key=lambda r: r['val_loss'])


def model_scores(model, table, device, batch_size=128):
    model.eval()
    x = torch.from_numpy(np.asarray(table['x'], dtype=np.float32))
    with torch.no_grad():
        return torch.cat([model.estimate(batch.to(device)).cpu()
                          for batch in x.split(batch_size)]).numpy()


def checkpoint_value(v):
    """Only weights_only-compatible tensors and primitives in .pt artifacts."""
    if isinstance(v, np.ndarray):
        return torch.from_numpy(v.copy())
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: checkpoint_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [checkpoint_value(x) for x in v]
    return v


def predict_ensemble(checkpoint, table, device=torch.device('cpu'), batch_size=128):
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
    mc = checkpoint['shared']['model_config']
    scores = []
    if table['x'].shape[1] != mc['n_in']:
        raise ValueError("Checkpoint feature dimension mismatch")
    for member in checkpoint['ensemble_members']:
        model = detectors.DenseSVDD(mc['n_in'], mc['n_latent'], mc['n_h']).to(device)
        model.load_state_dict(member['model_state_dict'])
        model.set_center(member['center'].to(device))
        scores.append(model_scores(model, table, device, batch_size))
    return np.mean(scores, axis=0)


def run_refit(refit_table, test_table, fold, selected, metadata, device, out_dir, progress):
    config = Config(**selected['config'])
    members, scores = [], []
    member_dir = out_dir / 'members'
    member_dir.mkdir(parents=True, exist_ok=True)
    for m, (labeled_idx, hidden_idx) in enumerate(REFIT_A_U_ASSIGNMENTS):
        labeled = [fold['refit_anomaly_ids'][i] for i in labeled_idx]
        hidden = [fold['refit_anomaly_ids'][i] for i in hidden_idx]
        refit = FeatureDataset(refit_table, labeled)
        assert refit.alpha == 1 / 19
        loader = make_loader(refit, config.batch_size)
        progress(phase='refit', member=m, labeled_refit_ids=labeled, hidden_refit_ids=hidden)
        path = member_dir / f'member{m}.pt'
        if path.exists():
            member = torch.load(path, map_location='cpu', weights_only=True)
            model = new_model(refit.n_features, config, device)
            model.load_state_dict(member['model_state_dict'])
            model.set_center(member['center'].to(device))
        else:
            model = new_model(refit.n_features, config, device)
            initialization_state = {k: v.detach().cpu().clone()
                                    for k, v in model.state_dict().items()}
            ae = fit(model, losses.AELoss(), loader, None, selected['pretrain_epoch'],
                     config, device, progress, 'refit_ae')
            pre_pu_state = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
            utils.set_center(model, loader, device, eps=0.1)
            pu_loader = make_loader(refit, config.batch_size, seed=42)
            svdd = fit(model, losses.PUSVDDLoss(refit.alpha), pu_loader, None,
                       selected['epoch'], config, device, progress, 'refit_svdd')
            member = dict(model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                          initialization_state_dict=initialization_state,
                          pre_pu_model_state_dict=pre_pu_state,
                          center=model.c.detach().cpu(), labeled_refit_ids=labeled,
                          hidden_refit_ids=hidden, alpha=refit.alpha,
                          sampler_policy='fresh_seed42_loader_before_pusvdd')
            torch.save(member, path)
            write_json(member_dir / f'member{m}_training.json', dict(ae=ae, svdd=svdd))
        assert member['labeled_refit_ids'] == labeled and member['hidden_refit_ids'] == hidden
        members.append(member)
        scores.append(model_scores(model, test_table, device, config.batch_size))
    shared = dict(metadata)
    shared.update(model_config=dict(n_in=refit.n_features, n_h=config.n_h,
                  n_latent=config.q, architecture='DenseSVDD', version=1),
                  selected_pretrain_epoch=selected['pretrain_epoch'], selected_epoch=selected['epoch'],
                  alpha=refit.alpha, training_config=asdict(config), seed=42,
                  weight_decay=1e-3, center_eps=0.1, checkpoint_version=2,
                  sampler_policy='AE loader -> set_center -> fresh seed-42 PUSVDD loader',
                  feature_schema_version=1, outer_fold=fold['outer_fold'],
                  refit_ids=fold['refit_ids'], test_ids=fold['test_ids'])
    checkpoint = dict(ensemble_members=members, shared=checkpoint_value(shared))
    checkpoint_path = out_dir / 'ensemble.pt'
    torch.save(checkpoint, checkpoint_path)
    score = np.mean(scores, axis=0)
    replay = predict_ensemble(checkpoint_path, test_table, device, config.batch_size)
    np.testing.assert_array_equal(score, replay)
    np.savez(out_dir / 'test_scores.npz', score=score, member_scores=np.asarray(scores),
             **{k: test_table[k] for k in ('recording_id', 'cycle_id', 't', 'true_label')})
    return score


def evaluate(scores, table):
    scores = np.asarray(scores)
    if len(scores) != len(table['recording_id']) or not np.isfinite(scores).all():
        raise ValueError("Invalid scores")
    stats = []
    for rid in np.unique(table['recording_id']):
        mask = table['recording_id'] == rid
        s = scores[mask]
        label = np.unique(table['true_label'][mask])
        assert len(s) == 277 and len(label) == 1
        stats.append(dict(recording_id=int(rid), t=int(table['t'][mask][0]),
                          true_label=int(label[0]), n_cycles=len(s), min=float(s.min()),
                          q25=float(np.quantile(s, .25)), median=float(np.median(s)),
                          q75=float(np.quantile(s, .75)), max=float(s.max()), mean=float(s.mean())))
    y, s = [r['true_label'] for r in stats], [r['median'] for r in stats]
    assert len(stats) == 10 and sum(y) == 1
    return dict(recording_AUROC=float(roc_auc_score(y, s)),
                recording_AP=float(average_precision_score(y, s)),
                cycle_AUROC=float(roc_auc_score(table['true_label'], scores)),
                cycle_AP=float(average_precision_score(table['true_label'], scores)),
                recording_statistics=stats)


def code_hash():
    root = Path(__file__).resolve().parents[1]
    paths = [root / 'tavil' / f'{name}.py' for name in
             ('__init__', 'experiment', 'preprocessing', 'protocol')]
    paths += sorted((root / 'models').glob('*.py'))
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()


def run(args):
    from tavil.preprocessing import load_psd_cache, prepare_fold
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    configs = [Config(**c) for c in json.loads(args.configs.read_text())] if args.configs else [Config()]
    if not configs:
        raise ValueError("At least one config is required")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    state = {}
    last_print = [0.0]

    def progress(**update):
        state.update(update)
        state.update(elapsed_seconds=time.perf_counter() - start, status='running')
        write_json(output / 'progress.json', state)
        if state['elapsed_seconds'] - last_print[0] >= 30 or 'orientation' in update or 'member' in update:
            print(json.dumps(state), flush=True)
            last_print[0] = state['elapsed_seconds']

    protocol = make_protocol()
    progress(phase='native_psd')
    psd = load_psd_cache(args.data, args.cache / 'native')
    manifest = dict(configs=[asdict(c) for c in configs], emax=args.emax, seed=42,
                    code_sha256=code_hash(), data_fingerprint=psd['fingerprint'],
                    device=str(device), threads=args.threads, torch=torch.__version__,
                    numpy=np.__version__, protocol=protocol, full_protocol=args.emax == 100)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Output manifest differs; use a new --output directory")
    write_json(manifest_path, manifest)
    outer_results = []
    for fold in protocol:
        o = fold['outer_fold']
        out = output / f'outer{o}'
        out.mkdir(exist_ok=True)
        cache = args.cache / 'folds' / f'outer{o}'
        cache.mkdir(parents=True, exist_ok=True)
        inner_cache = []
        for k, split in enumerate(fold['inner']):
            progress(outer=o, inner=k, phase='fold_preprocessing')
            tr, va, metadata = prepare_fold(psd, split['train_ids'], split['val_ids'], cache / f'inner{k}.npz')
            assert len(tr['x']) == len(va['x']) == 5540
            inner_cache.append((tr, va, split))
        progress(phase='refit_preprocessing')
        refit, test, metadata = prepare_fold(psd, fold['refit_ids'], fold['test_ids'], cache / 'refit.npz')
        assert len(refit['x']) == 11080 and len(test['x']) == 2770
        if args.prepare_only:
            continue
        selected = select_config(inner_cache, configs, args.emax, device, out / 'curves', progress)
        write_json(out / 'selected.json', selected)
        del inner_cache
        scores = run_refit(refit, test, fold, selected, metadata, device, out, progress)
        result = evaluate(scores, test)
        result.update(outer_fold=o, selected=selected, checkpoint=str((out / 'ensemble.pt').resolve()))
        write_json(out / 'results.json', result)
        outer_results.append(result)
        print(f'Outer {o}: recording AUROC={result["recording_AUROC"]:.6f}, AP={result["recording_AP"]:.6f}', flush=True)
    if args.prepare_only:
        write_json(output / 'progress.json', dict(status='preprocessing_complete', folds=35))
        return
    final = dict(outer_results=outer_results,
                 recording_statistics=[dict(outer_fold=r['outer_fold'], **s) for r in outer_results for s in r['recording_statistics']],
                 full_protocol=args.emax == 100)
    for level in ('recording', 'cycle'):
        for metric in ('AUROC', 'AP'):
            final[f'mean_{level}_{metric}'] = float(np.mean([r[f'{level}_{metric}'] for r in outer_results]))
    write_json(output / 'results.json', final)
    import csv
    with (output / 'recording_statistics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=final['recording_statistics'][0].keys())
        writer.writeheader()
        writer.writerows(final['recording_statistics'])
    write_json(output / 'progress.json', dict(status='complete', elapsed_seconds=time.perf_counter() - start))
    print(json.dumps({k: v for k, v in final.items() if k.startswith('mean_')}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('datasets/TAVIL'))
    parser.add_argument('--cache', type=Path, default=Path('datasets/TAVIL/.cache'))
    parser.add_argument('--output', type=Path, default=Path('results/TAVIL/full'))
    parser.add_argument('--configs', type=Path, help='JSON list of n_h, q, lr, batch_size')
    parser.add_argument('--emax', type=int, default=100, help='100 for complete protocol; smaller values are diagnostics')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.emax <= 0 or args.threads <= 0:
        parser.error('emax and threads must be positive')
    run(args)


if __name__ == '__main__':
    main()
