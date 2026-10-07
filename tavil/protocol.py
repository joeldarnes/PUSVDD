"""Frozen recording identities and deterministic matched nested splits.

The supplied documents omit the earlier literal inner split tables. We reconstruct
them by ordered 2-of-4 combinations, matching each anomaly Ai with normal group Ni.
The resolved recording-ID tables are written into every experiment manifest.
"""
from itertools import combinations

T = [0, 4415, 9020, 14552, 18837, 22381, 27491, 32305, 36005,
     39889, 44870, 48180, 53234, 57342, 62413, 67965, 74023, 80687,
     89376, 98403, 107158, 115459, 125314, 133616, 142979, 151332,
     160978, 169075, 178152, 187812, 197553, 205864, 214868, 222643,
     232727, 241171, 250591, 259398, 269177, 278345, 287008, 295708,
     305653, 314483, 323320, 332626, 341409, 350239, 360724, 367803]
IDS = [841573390070000000, 841573922616000000, 841574430980000000,
       841574864384000000, 841575304893000000, 841575887500000000,
       841576369458000000, 841576840909000000, 841577327166000000,
       841577773717000000, 841578186390000000, 841578615391000000,
       841579030058000000, 841579413593000000, 841583229718000000,
       841583791953000000, 841584281992000000, 841584673477000000,
       841585088631000000, 841585473299000000, 841585880084000000,
       841586275008000000, 841586718334000000, 841587179415000000,
       841587620541000000, 841590531880000000, 841588000954000000,
       841588395045000000, 841588856680000000, 841589276242000000,
       841589669678000000, 841590097410000000, 841590953810000000,
       841591335709000000, 841591710915000000, 841592077177000000,
       841592462099000000, 841592863537000000, 841593249286000000,
       841593613616000000, 841593986208000000, 841594366872000000,
       841594711272000000, 841595078390000000, 841595440212000000,
       841595801936000000, 841596157616000000, 841596542199000000,
       841596897360000000, 841597251906000000]
RECORDINGS = [dict(recording_id=r, t=t, true_label=int(t <= 18837))
              for r, t in zip(IDS, T)]
ANOMALIES = IDS[:5]
NORMAL_GROUPS = [IDS[5 + i::5] for i in range(5)]
A_U_ORIENTATIONS = [(0, 0), (0, 1), (1, 0), (1, 1)]
REFIT_A_U_ASSIGNMENTS = [(list(c), [i for i in range(4) if i not in c])
                         for c in combinations(range(4), 2)]


def make_protocol():
    """Five disjoint outer tests, each with six balanced matched inner splits."""
    folds = []
    for outer in range(5):
        development = [i for i in range(5) if i != outer]
        test = [ANOMALIES[outer]] + NORMAL_GROUPS[outer]
        refit = [ANOMALIES[i] for i in development] + [
            r for i in development for r in NORMAL_GROUPS[i]]
        inner = []
        for pair in combinations(development, 2):
            other = [i for i in development if i not in pair]
            train_a = [ANOMALIES[i] for i in pair]
            val_a = [ANOMALIES[i] for i in other]
            train = train_a + [r for i in pair for r in NORMAL_GROUPS[i]]
            val = val_a + [r for i in other for r in NORMAL_GROUPS[i]]
            assert len(train) == len(val) == 20
            assert not (set(train) & set(val) or set(train + val) & set(test))
            assert set(train + val) == set(refit)
            inner.append(dict(train_ids=train, val_ids=val,
                              train_anomaly_ids=train_a, val_anomaly_ids=val_a))
        assert all(sum(r in s['train_ids'] for s in inner) == 3 for r in refit)
        folds.append(dict(outer_fold=outer, test_ids=test, refit_ids=refit,
                          refit_anomaly_ids=[ANOMALIES[i] for i in development],
                          inner=inner))
    assert sorted(r for f in folds for r in f['test_ids']) == sorted(IDS)
    return folds
