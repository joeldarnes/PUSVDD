import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tavil import experiment
from tavil.latent_view import ViewConfig, extract_bundle, make_figure
from tavil.protocol import make_protocol


def feature_table(ids, anomaly_ids, cycles=3, features=6):
    rng=np.random.default_rng(9)
    rec=np.repeat(np.asarray(ids),cycles)
    y=np.repeat([int(i in anomaly_ids) for i in ids],cycles)
    x=rng.normal(size=(len(rec),features)).astype(np.float32)
    x[y==1]+=0.5
    return dict(x=x,recording_id=rec,cycle_id=np.tile(np.arange(cycles),len(ids)),
                t=np.arange(len(rec)),true_label=y)


class BaselineLatentViewTests(unittest.TestCase):
    def test_checkpoint_snapshots_and_orthographic_map(self):
        fold=make_protocol()[0]
        refit=feature_table(fold["refit_ids"],set(fold["refit_anomaly_ids"]))
        test=feature_table(fold["test_ids"],{fold["test_ids"][0]})
        selected={"config":{"n_h":4,"q":2,"lr":1e-3,"batch_size":16},
                  "pretrain_epoch":0,"epoch":1}
        metadata={"n_features":6}
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            experiment.run_refit(refit,test,fold,selected,metadata,torch.device("cpu"),out,lambda **_:None)
            cp=torch.load(out/"ensemble.pt",map_location="cpu",weights_only=True)
            self.assertEqual(cp["shared"]["checkpoint_version"],2)
            self.assertIn("pre_pu_model_state_dict",cp["ensemble_members"][0])
            self.assertEqual(cp["shared"]["sampler_policy"],
                             "AE loader -> set_center -> fresh seed-42 PUSVDD loader")
            bundle=extract_bundle(cp,refit,test,member_index=0)
            self.assertEqual(bundle.metadata["recovery"]["source"],"stored_snapshot")
            self.assertEqual(bundle.projection["kind"],"pad2_to_3")
            self.assertIn("initialization",set(bundle.cycles.state))
            fig=make_figure(bundle,ViewConfig(max_points=100),
                            selected_recording=str(fold["test_ids"][0]))
            self.assertEqual(fig.layout.scene.camera.projection.type,"orthographic")
            self.assertTrue(any(trace.name=="c PUSVDD" for trace in fig.data))


if __name__=="__main__":
    unittest.main()
