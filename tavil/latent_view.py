"""Interactive 3-D latent viewer for baseline TAVIL PUSVDD checkpoints."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from models import detectors, losses
from tavil import experiment
from tavil.preprocessing import load_psd_cache, transform_with_metadata

TRUTH={-1:"unknown",0:"normal",1:"anomaly"}
COLORS={"normal":"#2878B5","anomaly":"#D9534F","unknown":"#777777"}
SYMBOLS={"A":"diamond","U":"circle","TEST":"square","NEW":"cross"}

@dataclass(frozen=True)
class ViewConfig:
    max_points:int=6000
    seed:int=42
    marker_size:float=3.5
    selected_marker_size:float=6.0

@dataclass
class ViewBundle:
    cycles:pd.DataFrame
    recordings:pd.DataFrame
    center_xyz:np.ndarray
    projection:dict[str,Any]
    metadata:dict[str,Any]

def _cfg(shared):
    return experiment.Config(**shared["training_config"])

def _model(shared,state,device="cpu"):
    mc=shared["model_config"]
    m=detectors.DenseSVDD(int(mc["n_in"]),int(mc["n_latent"]),int(mc["n_h"])).to(device)
    m.load_state_dict(state)
    return m

def _encode(model,X,batch=512):
    x=torch.from_numpy(np.asarray(X,dtype=np.float32))
    out=[]; dev=next(model.parameters()).device; model.eval()
    with torch.no_grad():
        for b in x.split(batch):
            out.append(model.encode(b.to(dev)).cpu().numpy())
    return np.concatenate(out)

def _score(model,center,X,batch=512):
    model.set_center(torch.as_tensor(center,dtype=torch.float32,device=next(model.parameters()).device))
    x=torch.from_numpy(np.asarray(X,dtype=np.float32))
    out=[]; dev=next(model.parameters()).device; model.eval()
    with torch.no_grad():
        for b in x.split(batch):
            out.append(model.estimate(b.to(dev)).cpu().numpy())
    return np.concatenate(out)

def _ensemble_score(cp,X,device="cpu"):
    vals=[]
    for member in cp["ensemble_members"]:
        m=_model(cp["shared"],member["model_state_dict"],device)
        vals.append(_score(m,member["center"],X,_cfg(cp["shared"]).batch_size))
    return np.mean(np.stack(vals),axis=0)

def _pre_pu_model(cp,member_index,train_table,device="cpu"):
    member=cp["ensemble_members"][member_index]; shared=cp["shared"]
    if "pre_pu_model_state_dict" in member:
        return _model(shared,member["pre_pu_model_state_dict"],device),{"verified":True,"source":"stored_snapshot"}
    data=experiment.FeatureDataset(train_table,member["labeled_refit_ids"])
    cfg=_cfg(shared)
    model=experiment.new_model(data.n_features,cfg,torch.device(device))
    e=int(shared.get("selected_pretrain_epoch",0))
    if e:
        dl=experiment.make_loader(data,cfg.batch_size,seed=int(shared.get("seed",42)))
        experiment.fit(model,losses.AELoss(),dl,None,e,cfg,torch.device(device),lambda **_:None,"latent_recovery")
    return model,{"verified":False,"source":"deterministic_replay"}

def _rows(train,test,member,new=None):
    frames=[]; labeled={str(x) for x in member["labeled_refit_ids"]}
    for table,source in ((train,"TRAIN"),(test,"TEST")):
        rec=np.asarray(table["recording_id"]).astype(str)
        role=np.array(["A" if r in labeled else "U" for r in rec],object) if source=="TRAIN" else np.full(len(rec),"TEST",object)
        frames.append(pd.DataFrame({
            "recording_id":rec,"cycle_id":np.asarray(table["cycle_id"]),"t":np.asarray(table["t"]),
            "true_label":np.asarray(table["true_label"]).astype(int),"role":role,"source":source,
            "_X":list(np.asarray(table["x"],np.float32))
        }))
    if new is not None:
        X=np.asarray(new["x"],np.float32); n=len(X); rec=np.asarray(new["recording_id"]).astype(str)
        frames.append(pd.DataFrame({
            "recording_id":rec,"cycle_id":np.asarray(new.get("cycle_id",np.arange(n))),
            "t":np.asarray(new.get("t",np.full(n,-1))),
            "true_label":np.asarray(new.get("true_label",np.full(n,-1))).astype(int),
            "role":np.full(n,"NEW"),"source":np.full(n,"NEW"),"_X":list(X)
        }))
    return pd.concat(frames,ignore_index=True)

def _projection(zi,zf,c):
    q=zf.shape[1]
    if q>3:
        p=PCA(n_components=3,svd_solver="full").fit(np.concatenate([zi,zf]))
        fn=lambda z:p.transform(z)
        meta={"kind":"PCA3","q":q,"components":p.components_.tolist(),"mean":p.mean_.tolist(),
              "explained_variance_ratio":p.explained_variance_ratio_.tolist(),
              "fit_policy":"TRAIN initial+final equal cycle weight"}
    elif q==3:
        fn=lambda z:np.asarray(z,float); meta={"kind":"identity3","q":q}
    else:
        fn=lambda z:np.pad(np.asarray(z,float),((0,0),(0,3-q))); meta={"kind":f"pad{q}_to_3","q":q}
    return fn,fn(np.asarray(c)[None])[0],meta

def _summary(cycles):
    final=cycles[cycles.state=="final"]; rows=[]
    for rid,g in final.groupby("recording_id",sort=False):
        s=g.ensemble_score.to_numpy(float); labels=g.true_label.unique()
        rows.append({"recording_id":rid,"true_label":int(labels[0]) if len(labels)==1 else -1,
                     "source":g.source.iloc[0],"n_cycles":len(g),"median":float(np.median(s)),
                     "mean":float(s.mean()),"min":float(s.min()),"max":float(s.max())})
    return pd.DataFrame(rows)

def ranking_inversions(recordings):
    e=recordings[(recordings.source!="TRAIN")&recordings.true_label.isin([0,1])]
    rows=[]
    for _,n in e[e.true_label==0].iterrows():
        for _,a in e[e.true_label==1].iterrows():
            gap=float(n["median"]-a["median"])
            if gap>0:
                rows.append({"normal_recording":n.recording_id,"anomaly_recording":a.recording_id,
                             "normal_score":float(n["median"]),"anomaly_score":float(a["median"]),"gap":gap})
    return pd.DataFrame(rows).sort_values("gap",ascending=False,ignore_index=True) if rows else pd.DataFrame(columns=["normal_recording","anomaly_recording","normal_score","anomaly_score","gap"])

def extract_bundle(checkpoint,train_table,test_table,member_index=0,new_data=None,device="cpu"):
    cp=torch.load(checkpoint,map_location="cpu",weights_only=True) if isinstance(checkpoint,(str,Path)) else checkpoint
    member=cp["ensemble_members"][member_index]; shared=cp["shared"]
    table=_rows(train_table,test_table,member,new_data)
    X=np.stack(table.pop("_X").to_numpy()).astype(np.float32); ntr=len(train_table["x"])
    initial,recovery=_pre_pu_model(cp,member_index,train_table,device)
    final=_model(shared,member["model_state_dict"],device); c=np.asarray(member["center"],float)
    zi=_encode(initial,X); zf=_encode(final,X)
    transform,c3,meta=_projection(zi[:ntr],zf[:ntr],c)
    ens=_ensemble_score(cp,X,device)
    state0="initialization" if int(shared.get("selected_pretrain_epoch",0))==0 else "pre_pu"
    frames=[]
    for state,z in ((state0,zi),("final",zf)):
        xyz=transform(z); d=np.linalg.norm(z-c,axis=1); d3=np.linalg.norm(xyz-c3,axis=1)
        f=table.copy(); f["state"]=state; f[["x","y","z"]]=xyz
        f["distance_full"]=d; f["distance_projected"]=d3
        f["distance_discarded"]=np.sqrt(np.maximum(d*d-d3*d3,0))
        f["member_center_score"]=1-np.exp(-d); f["ensemble_score"]=ens
        f["truth"]=[TRUTH.get(int(v),"unknown") for v in f.true_label]
        frames.append(f)
    cycles=pd.concat(frames,ignore_index=True); recordings=_summary(cycles)
    meta["center_xyz"]=c3.tolist()
    metadata={"member_index":member_index,"run_type":"baseline_pusvdd","recovery":recovery,
              "selected_pretrain_epoch":int(shared.get("selected_pretrain_epoch",0)),
              "selected_epoch":int(shared.get("selected_epoch",0)),
              "ranking_inversions":ranking_inversions(recordings).to_dict(orient="records")}
    return ViewBundle(cycles,recordings,c3,meta,metadata)

def balanced_sample(cycles,max_points=6000,seed=42,selected_recording=None):
    if len(cycles)<=max_points:return cycles.copy()
    rng=np.random.default_rng(seed); groups=list(cycles.groupby(["state","recording_id"],sort=True))
    quota=max(1,max_points//len(groups)); keep=[]
    for (_,rid),g in groups:
        idx=g.index.to_numpy()
        chosen=idx if (selected_recording is not None and str(rid)==str(selected_recording)) or len(idx)<=quota else rng.choice(idx,quota,replace=False)
        keep.extend(map(int,chosen))
    if len(keep)>max_points and selected_recording is None:
        keep=list(rng.choice(np.asarray(keep),max_points,replace=False))
    return cycles.loc[sorted(set(keep))].copy()

def make_figure(bundle,config=ViewConfig(),selected_recording=None,visible_states=None):
    import plotly.graph_objects as go
    states=set(visible_states or bundle.cycles.state.unique())
    full=bundle.cycles[bundle.cycles.state.isin(states)]
    shown=balanced_sample(full,config.max_points,config.seed,selected_recording)
    fig=go.Figure()
    for (state,truth,role),g in shown.groupby(["state","truth","role"],sort=True):
        sel=(g.recording_id.astype(str)==str(selected_recording)) if selected_recording is not None else np.zeros(len(g),bool)
        hover=[f"recording={r.recording_id}<br>cycle={r.cycle_id}<br>state={r.state}<br>role={r.role}<br>truth={r.truth}<br>d(full)={r.distance_full:.6g}<br>d(projected)={r.distance_projected:.6g}<br>d(discarded)={r.distance_discarded:.6g}<br>ensemble score={r.ensemble_score:.6g}" for r in g.itertuples(index=False)]
        fig.add_trace(go.Scatter3d(x=g.x,y=g.y,z=g.z,mode="markers",name=f"{state} · {truth} · {role}",
            marker={"size":np.where(sel,config.selected_marker_size,config.marker_size),"opacity":.32 if state!="final" else .88,"color":COLORS[truth],"symbol":SYMBOLS[role]},
            text=hover,hovertemplate="%{text}<extra></extra>"))
    c=bundle.center_xyz
    fig.add_trace(go.Scatter3d(x=[c[0]],y=[c[1]],z=[c[2]],mode="markers+text",name="c PUSVDD",marker={"size":8,"symbol":"diamond"},text=["c"]))
    if selected_recording is not None:
        sel=full[full.recording_id.astype(str)==str(selected_recording)]; xx=[];yy=[];zz=[]
        for r in sel.itertuples(index=False):
            xx += [r.x,c[0],None]; yy += [r.y,c[1],None]; zz += [r.z,c[2],None]
        fig.add_trace(go.Scatter3d(x=xx,y=yy,z=zz,mode="lines",name="selected → c",opacity=.18,line={"width":1},hoverinfo="skip"))
    fig.update_layout(template="plotly_white",height=760,scene={"aspectmode":"data","camera":{"projection":{"type":"orthographic"}}},margin={"l":0,"r":0,"t":45,"b":0},title=f"TAVIL baseline · member {bundle.metadata['member_index']} · {bundle.projection['kind']}")
    return fig

def dashboard_from_results(runs:Mapping[str,str|Path],data=Path("datasets/TAVIL"),native_cache=Path("datasets/TAVIL/.cache/native"),outers:Iterable[int]=range(5),config=ViewConfig(),new_loader:Callable|None=None,device="cpu"):
    import ipywidgets as widgets
    from IPython.display import display,clear_output
    psd=load_psd_cache(data,native_cache)
    run_dd=widgets.Dropdown(options=list(runs),description="Run")
    outer_dd=widgets.Dropdown(options=list(outers),description="Outer")
    member_dd=widgets.Dropdown(options=list(range(6)),description="Member")
    recording_dd=widgets.Dropdown(options=[("all",None)],description="Recording")
    states=widgets.SelectMultiple(options=[],description="States")
    inv=widgets.Checkbox(False,description="Inversions")
    out=widgets.Output()
    @lru_cache(maxsize=64)
    def get_bundle(run,outer,member):
        cp_path=Path(str(runs[run]).format(outer=outer))
        cp=torch.load(cp_path,map_location="cpu",weights_only=True); shared=cp["shared"]
        train=transform_with_metadata(psd,shared["refit_ids"],shared)
        test=transform_with_metadata(psd,shared["test_ids"],shared)
        new=new_loader(run,outer) if new_loader else None
        return extract_bundle(cp,train,test,member,new,device)
    def redraw(*_):
        b=get_bundle(run_dd.value,int(outer_dd.value),int(member_dd.value))
        opts=[("all",None)]+[(str(r),str(r)) for r in b.recordings.recording_id]
        cur=recording_dd.value; recording_dd.options=opts
        recording_dd.value=cur if cur in [v for _,v in opts] else None
        st=b.cycles.state.unique().tolist(); states.options=st
        if not states.value: states.value=tuple(st)
        fig=make_figure(b,config,recording_dd.value,states.value)
        with out:
            clear_output(wait=True); display(fig)
            print("recovery:",b.metadata["recovery"],"| full cycles:",len(b.cycles)//2,"| render cap:",config.max_points)
            display(ranking_inversions(b.recordings) if inv.value else b.recordings)
    for w in (run_dd,outer_dd,member_dd,recording_dd,states,inv):
        w.observe(redraw,names="value")
    redraw()
    return widgets.VBox([widgets.HBox([run_dd,outer_dd,member_dd,recording_dd,states,inv]),out])

__all__=["ViewConfig","ViewBundle","extract_bundle","make_figure","dashboard_from_results","ranking_inversions","balanced_sample"]
