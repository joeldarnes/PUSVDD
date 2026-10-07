# PUSVDD sobre TAVIL

La implementació adapta les dades i l'avaluació del repositori existent al protocol de [canvis.md](reference/canvis.md), amb el context complet de [conversacio115.md](reference/conversacio115.md). Les còpies són idèntiques als originals; [context_manifest.json](reference/context_manifest.json) conserva paths d'origen, SHA256 i mida. El model DenseSVDD, AELoss, PUSVDDLoss, score i inicialització del centre es reutilitzen. La revisió de context, fonts primàries i diferències amb els papers és a [paper_review.md](paper_review.md).

## Organització i simplicitat

| Arxiu | Responsabilitat |
|---|---|
| `tavil/protocol.py` | Identitats reals dels 50 recordings, grups normals round-robin, cinc outer folds, sis INNER aparellats i assignacions A/U. |
| `tavil/preprocessing.py` | SQLite, últims 277 cicles complets, calibratge físic, RF10, PSD/massa acumulada, bandes M4/SHAPE, scaler i caches. |
| `tavil/experiment.py` | Dataset amb labels per màscara, loaders reproduïbles, dues etapes INNER, selecció, sis REFIT, checkpoints i mètriques. Entrada `python -m tavil.experiment`. |
| `tavil/verify_run.py` | Verificador separat de l'execució: recalcula preprocessament TRAIN-only, selecció agregada, scores i mètriques a partir dels artefactes. Entrada `python -m tavil.verify_run`. |
| `models/detectors.py` | Xarxes i score originals; DenseSVDD ja té l'arquitectura demanada. |
| `models/losses.py` | Pèrdues originals; cap fórmula duplicada per a TAVIL. |
| `models/utils.py` | `set_center` original, reutilitzat sobre A union U. |
| `models/trainers.py` | Trainer compartit ampliat amb validació opcional, `load_best=False` i callback. Els defaults d'imatges/toy conserven el reload del millor checkpoint local. |
| `tests/test_tavil.py` | Regressions numèriques, agrupació, preprocessament sense leakage i compatibilitat del Trainer. |

Tres mòduls principals separen dades, protocol i orquestració; un quart mòdul verifica l'experiment acabat. No hi ha una segona xarxa, una nova jerarquia de losses ni un altre training loop TAVIL: l'orchestrador crida el Trainer existent. Les metadades `recording_id`, `cycle_id`, `t`, `true_label` es mantenen lligades a cada fila de features; A/U és una màscara que no reordena X.

## Decisions explícites

**Splits reconstruïts.** Els dos adjunts no inclouen les arrays literals dels INNER que esmenten. Per cada outer, s'ordenen els quatre índexs development; les sis combinacions de dos índexs determinen TRAIN i el complement determina VAL. S'aplica la mateixa combinació a anomalies i grups normals: sis splits aparellats, amb cada grup tres vegades TRAIN i tres VAL. No es generen 36 combinacions. El manifest desa les arrays finals de recording IDs, perquè aquesta reconstrucció sigui revisable.

**Duració física.** `duration_phase = (windows.end_timestamp_ns - windows.start_timestamp_ns)/1e9`. Tref és la mediana TRAIN/REFIT d'aquesta duració dividida per 10. L'audit real dona fases d'aproximadament 0.5404 s rising i 0.5396 s falling; la referència inicial de 270 ms del context no concorda amb les dades. El recording de `t=57342` té ID `841579413593000000`, contrastat amb SQLite.

**Integració espectral.** Els adjunts demanen interpolar la massa acumulada però no especifiquen on situar els nusos. Es defineix una cel·la per bin natiu: cada ordinada aporta `PSD*fs/N`, distribuïda uniformement entre els límits de la seva cel·la. El primer límit és 0; els límits superiors són `min(f_k + fs/(2N), fs/2)` i l'últim és exactament Nyquist, també per N senar. La interpolació lineal de la massa acumulada en aquests límits conserva tota la potència, inclosos DC/Nyquist. Aquesta convenció figura a la versió `RF10-M4-SHAPE-v1-bin-cell-cumulative` i als metadades. Per un espectre nul, SHAPE es defineix a -300 dB per banda per evitar `0/0`.

**Configuració.** Sense `--configs`, es declara una única configuració: `n_h=500`, `q=128`, `lr=1e-4`, `batch_size=128`; també es proporciona a `configs/tavil.json`. L'execució completa acabada a `results/TAVIL/full_compact` usa `configs/tavil_compact.json`: `n_h=64`, `q=16`, `lr=1e-4`, batch 128, CPU amb un thread. Conserva `seed=42`, `Emax_AE=Emax_PUSVDD=100`, Adam i weight decay `1e-3`. La CLI accepta una llista JSON de configs amb aquests quatre camps; els empats respecten l'ordre de la llista. Cada configuració queda al manifest abans del training. La xarxa compacta és una configuració explícita diferent; conserva totes les etapes i cardinalitats del protocol.

**Dimensions.** No s'imposa 3300: cada preprocessing deriva el nombre de bandes de Tref, genera `feature_order` i verifica la dimensió resultant de TRAIN/evaluació. `expected_3300_matches` registra la coincidència amb l'expectativa.

## Comandos PowerShell

Executar des de `C:\Users\joeldarnes\Documents\AE\pusvdd_2`, amb la `.venv` existent.

Tests:

```powershell
& '.venv/Scripts/python.exe' -m unittest discover -s tests -v
```

Preparar els 50 caches natius i els 35 preprocessings de fold, sense training:

```powershell
& '.venv/Scripts/python.exe' -m tavil.experiment --prepare-only --threads 1 --output 'results/TAVIL/prepare'
```

Flux complet reproduïble de la configuració compacta, amb 100 epochs màxims a les dues etapes, seguit del verificador:

```powershell
& '.venv/Scripts/python.exe' -m tavil.experiment --configs 'configs/tavil_compact.json' --emax 100 --threads 1 --device cpu --output 'results/TAVIL/full_compact'
if ($LASTEXITCODE -eq 0) {
    & '.venv/Scripts/python.exe' -m tavil.verify_run --output 'results/TAVIL/full_compact'
}
```

Per llançar la configuració de referència 500/128, usar `--configs 'configs/tavil.json'`, un directori nou com `results/TAVIL/full_reference` i el threading que es vulgui declarar.

Una graella opcional es passa amb `--configs 'ruta/configs.json'`, on el contingut té forma `[ {"n_h":500,"q":128,"lr":0.0001,"batch_size":128} ]`. Fer servir un directori `--output` nou quan es modifica config, codi, dades, dispositiu, epochs o threading. Una execució amb `--emax` menor de 100 es marca `full_protocol=false` i serveix com a diagnòstic.

Reprendre usa el mateix comando i el mateix directori. Les corbes de runs i els membres REFIT acabats es reaprofiten; un run interromput abans de desar-se es reinicia amb seed 42. El manifest rebutja reprendre amb condicions diferents. S'ha de conservar el mateix codi mentre un experiment està en curs.

L'execució d'aquesta sessió distribueix outers independents entre cinc processos, cadascun amb un thread PyTorch i un directori propi. Tots criden les mateixes funcions `select_config` i `run_refit`; quan els cinc outers acaben, la CLI anterior recupera els artefactes i genera l'agregació final. L'afinitat CPU assigna processos a processadors lògics, inclosos els siblings SMT dels dos nuclis P: una comprovació amb dos epochs va confirmar pesos exactament iguals entre les afinitats provades. Les condicions i la represa consten a `execution_notes.json`.

Durant l'execució s'ha corregit exclusivament la publicació atòmica dels JSON: Windows podia denegar temporalment el canvi de nom quan un monitor llegia `progress.json`. El writer reintenta aquest error durant un màxim d'un segon; els monitors permeten compartir l'operació de delete/rename. La igualtat de l'AST de tot `experiment.py`, exclosa únicament `write_json`, comprova que no s'ha canviat cap operació numèrica ni selecció. S'han conservat els hashes anterior/nou al manifest d'execució i s'ha provat la correcció amb un bloqueig real de Windows a `results/TAVIL/windows_io_verification.json`.

Verificació independent d'un experiment complet. Desa `verification.json` amb `status=passed` si tots els controls passen; en cas d'error desa `status=failed` i el motiu. S'ha d'executar després de completar els cinc outer folds:

```powershell
& '.venv/Scripts/python.exe' -m tavil.verify_run --output 'results/TAVIL/full_compact'
```

Per verificar explícitament un diagnòstic amb Emax menor de 100, afegir `--allow-diagnostic`; el seu informe conserva `complete_protocol=false`.

Replay de l'outer 0 a partir del checkpoint i SQLite, sense tornar a ajustar bandes/scaler ni entrenar. Ajustar `run_dir` si el resultat es troba en un altre directori:

```powershell
@'
from pathlib import Path
import json
import numpy as np
import torch
from tavil.preprocessing import load_psd_cache, transform_with_metadata
from tavil.experiment import predict_ensemble

run_dir = Path('results/TAVIL/full_compact')
manifest = json.loads((run_dir / 'manifest.json').read_text())
torch.set_num_threads(manifest['threads'])
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
checkpoint = torch.load(run_dir / 'outer0/ensemble.pt', map_location='cpu', weights_only=True)
shared = checkpoint['shared']
psd = load_psd_cache(Path('datasets/TAVIL'), Path('datasets/TAVIL/.cache/native'))
table = transform_with_metadata(psd, shared['test_ids'], shared)
batch = shared['training_config']['batch_size']
score = predict_ensemble(checkpoint, table, torch.device('cpu'), batch)
with np.load(run_dir / 'outer0/test_scores.npz', allow_pickle=False) as saved:
    for key in ('recording_id', 'cycle_id', 't', 'true_label'):
        np.testing.assert_array_equal(table[key], saved[key])
    np.testing.assert_array_equal(score, saved['score'])
print('Replay exacte del checkpoint: correcte')
'@ | & '.venv/Scripts/python.exe' -
```

## Artefactes i verificació

```text
datasets/TAVIL/.cache/
  native/<recording_id>.npz + .json   # PSD nativa; SHA256 SQLite i audit
  native/audit.json                  # 50 recordings, clocks/calibratge/cicles
  folds/outer0..4/inner0..5.npz       # X TRAIN/VAL i metadades congelades
  folds/outer0..4/refit.npz           # X REFIT/TEST i metadades congelades

results/TAVIL/<experiment>/
  manifest.json                      # dades, codi, configs, protocol i entorn
  progress.json                      # estat i progrés actual
  outer0..4/curves/                   # 24 AE + 24 SVDD per config i selecció
  outer0..4/selected.json
  outer0..4/members/member0..5.pt
  outer0..4/members/member0..5_training.json
  outer0..4/ensemble.pt               # sis models+centres i preprocessament
  outer0..4/test_scores.npz           # scores per membre, ensemble i identitats
  outer0..4/results.json
  results.json                       # cinc folds i mitjanes de les mètriques
  recording_statistics.csv            # 50 files, una per recording retingut
  verification.json                   # verificació independent del run acabat

results/TAVIL/preprocessing_verification.json  # comprovació dels 35 caches reals
```

`run_refit` recarrega el checkpoint final i exigeix igualtat exacta dels scores abans de desar-los. L'avaluació exigeix 277 cicles per recording, deu recordings i una anomalia per outer. Guarda AUROC/AP de recording sobre les medians, AUROC/AP de cicle i els sis estadístics. Les quatre mitjanes finals són mitjanes dels cinc folds; no calcula una mètrica amb scores bruts concatenats. `verify_run` comprova de nou SHA256 de cada SQLite, identitats, Tref/bandes/scaler ajustats exclusivament amb TRAIN, epochs/config seleccionats sobre les corbes, sis assignacions REFIT, epochs exactes, preprocessament/score congelats i totes les mètriques desades.

## Matriu de requisits 1-8 de canvis.md

| Part | Evidència en codi/tests | Estat verificat |
|---|---|---|
| 1 RF10-M4-SHAPE | `preprocessing._recording_psd`, `rf_cuts`, `spectral_mass`, `integrate_bands`, `prepare_fold`; tests de talls, calibratge, últims cicles, potència i SHAPE; informe de preprocessament real | Fórmules i fixtures passen; 50 recordings i els 35 preprocessings reals verificats. Tots obtenen 3300 features mantenint 277 identitats de cicle per recording. |
| 2 DenseSVDD/configs | `detectors.DenseSVDD`, `generate_dense_layers`, `experiment.Config`; test d'arquitectura simètrica sense bias/BN/dropout | Nucli i arquitectura verificats; config de cada execució queda declarada al manifest. |
| 3 TRAIN/VAL/TEST | `protocol.make_protocol`, orientacions/particions, `FeatureDataset`, `make_loader`; tests de cardinalitats, disjunció, round-robin, labels alineats i sis particions | Split per recording verificat; arrays INNER reconstruïdes explícitament. A/U no reordena X ni altera `true_label`; el dataset conserva cycle_id i t. |
| 4 Resultats | `experiment.model_scores`, `run_refit`, `evaluate`, agregació final; assertions de mides i replay | Complet i verificat als cinc outers. Mitjanes recording AUROC/AP: 0.888889/0.500000; cycle AUROC/AP: 0.807936/0.207764. Totes les mètriques i els sis estadístics s'han recalculat independentment. |
| 5 Normalització | `preprocessing.prepare_fold`, `transform_with_metadata`; test TRAIN-only amb VAL de duració diferent, scaler manual i cache invalidada; verificació dels 35 caches reals | TRAIN/REFIT-only contrastat amb recomputació; màxim mòdul de mitjana TRAIN `1.294e-9`, màxim error de variància unitària `8.127e-9`; replay de features congelades amb error 0. |
| 6 Flux | `select_config`, `first_epoch`, `fit`, `run_refit`; tests de mínims agregats/empats i rerun AE; Trainer sense reload local-best; `verify_run.check_selection` | Complet: 120 corbes AE de 100 epochs i 120 reruns AE+SVDD de 100 epochs SVDD; prefix AE reproduït amb error 0. Mínims agregats i epochs exactes verificats. AE seleccionat: [97,98,98,97,100]; SVDD seleccionat: [94,99,97,94,94]. |
| 7 Checkpoint | `run_refit`, `checkpoint_value`, `predict_ensemble`, `transform_with_metadata`; test dels sis membres i replay `weights_only=True`; `verify_run` | Trenta membres reals acabats i verificats. Els cinc checkpoints d'ensemble tenen els sis membres, centres i metadades complets; replay PSD nativa → features congelades → checkpoint → scores exactament igual al desat. |
| 8 PUSVDD core | `losses.AELoss/PUSVDDLoss`, `utils.set_center`, `DeepSVDD.estimate`, `Trainer`; tests matemàtics i compatibilitat local-best original | Objectiu, epsilon, centre A union U, score, Adam i arquitectura originals preservats. |

## Estat de l'evidència

Revisió final actualitzada, 2026-10-02: la suite ampliada de 18 tests ha passat. L'audit natiu real confirma 50 recordings i 13.850 cicles seleccionats: 48 recordings tenen 278 cicles complets originals i dos en tenen 277; tots aporten els últims 277. S'han preparat i comprovat els 35 preprocessings; tots donen 3300 features, conserven les identitats dels 277 cicles de cada recording i reconstrueixen les features amb paràmetres congelats amb error 0. Sobre els TRAIN transformats, el màxim mòdul de mitjana és `1.294e-9` i l'error màxim de variància unitària és `8.127e-9`. L'informe del preprocessament es desa a `results/TAVIL/preprocessing_verification.json`.

L'execució completa és `results/TAVIL/full_compact`: config 64/16, lr `1e-4`, batch 128, Emax 100, seed 42 i un thread CPU per procés. Els cinc folds i els trenta REFIT han acabat. `progress.json` indica `complete`, `results.json` indica `full_protocol=true` i `verification.json` indica `status=passed`, `complete_protocol=true`; la verificació integral ha durat 122.812 s. Les mitjanes dels cinc folds són recording AUROC `0.888889`, recording AP `0.500000`, cycle AUROC `0.807936` i cycle AP `0.207764`. L'informe amb els epochs seleccionats, la taula per fold i la interpretació és a [TAVIL_results.md](TAVIL_results.md).

El benchmark real preliminar de `results/TAVIL/benchmark.json` mesura per epoch aproximadament AE `2.449 s` i SVDD `0.924 s` amb config 500/128 i 12 threads; config 64/16 amb un thread dona AE `0.438 s` i SVDD `0.257 s`. El flux complet pot requerir diverses hores: són mesures preliminars de cost, amb overhead de validació/artefactes, i no constitueixen una ETA garantida. Els epochs AE/SVDD finals figuren a l'informe de resultats; els mínims seleccionats dins Emax=100 no demostren convergència. El cost inclou per config 120 runs AE de selecció, 120 runs AE+SVDD i 30 REFIT.

L'avaluació estima generalització a recordings/targets retinguts. La correlació entre cicles i el confounding temporal/temperatura del context continuen sent limitacions de les dades; les comprovacions del software no demostren detecció de tipus d'anomalia externs ni les hipòtesis IID dels teoremes.
