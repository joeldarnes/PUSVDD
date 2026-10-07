# Resultats PUSVDD TAVIL

El flux complet de `canvis.md` ha acabat i ha superat el verificador independent. Informe final verificat el 2026-10-02.

Configuració fixa: `n_h=64`, `q=16`, `lr=0.0001`, `batch_size=128`, `seed=42`, `Emax_AE=Emax_PUSVDD=100`.

50 recordings; últims 277 cicles complets per recording; 13.850 cicles. RF10-M4-SHAPE produeix 3300 features en els 35 preprocessaments.

| Outer | Epoch AE seleccionat | Epoch SVDD seleccionat | AUROC recording | AP recording | AUROC cicle | AP cicle |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 97 | 94 | 0.888889 | 0.500000 | 0.776632 | 0.181638 |
| 1 | 98 | 99 | 0.888889 | 0.500000 | 0.816458 | 0.212032 |
| 2 | 98 | 97 | 0.888889 | 0.500000 | 0.812836 | 0.222639 |
| 3 | 97 | 94 | 0.888889 | 0.500000 | 0.815040 | 0.209726 |
| 4 | 100 | 94 | 0.888889 | 0.500000 | 0.818713 | 0.212784 |
| Mitjana | - | - | 0.888889 | 0.500000 | 0.807936 | 0.207764 |

El score de recording és la mediana dels 277 scores de cicle. AP és `sklearn.metrics.average_precision_score`. Les mitjanes corresponen als cinc folds; no es calcula una mètrica de scores bruts concatenats.

En tots cinc folds, el recording anòmal queda segon entre els deu recordings: supera vuit normals, però un normal rep un score superior. Això explica AUROC=8/9 i AP=1/2. Els scores, inclosos alguns normals, són molt propers a 1; `1-exp(-d)` no és una probabilitat calibrada.

## Evidència de verificació

18 tests de regressió passen. El verificador final comprova 120 corbes AE, 120 reruns AE+SVDD, 30 models REFIT, 35 preprocessaments ajustats només amb TRAIN/REFIT, els SHA256 actuals dels SQLite, labels/IDs, selecció agregada i epochs exactes. La reproducció `PSD -> preprocessament congelat -> checkpoint -> scores` és exacta als cinc ensembles.

- [Resultats JSON](../results/TAVIL/full_compact/results.json)
- [Verificació final](../results/TAVIL/full_compact/verification.json)
- [Estadístics dels 50 recordings](../results/TAVIL/full_compact/recording_statistics.csv)
- [Manifest i splits congelats](../results/TAVIL/full_compact/manifest.json)
- [Disseny, decisions i requisits](TAVIL.md)

## Límits de la interpretació

Els splits INNER literals anteriors no eren als adjunts; es van reconstruir les sis combinacions ordenades de dos grups, aparellant Ai amb Ni. Aquesta decisió queda declarada al manifest. La interpolació de massa PSD sobre cel·les natives també queda versionada. La configuració 500/128 està disponible com a referència i no és la xarxa executada aquí.

Aquest experiment avalua recordings/targets retinguts. La correlació entre cicles i el confounding temporal/temperatura documentat continuen existint. Només hi ha cinc recordings anòmals: els 13.850 cicles no són observacions independents. Totes les anomalies tenen `t≤18837` i tots els normals `t≥22381`; la separació no es pot atribuir exclusivament a la causa mecànica. Els epochs seleccionats són mínims dins Emax=100, sense demostrar convergència. No avalua encara tipus externs d'anomalia ni un model final de producció.

![Mètriques per outer fold](../results/TAVIL/full_compact/metrics.png)
