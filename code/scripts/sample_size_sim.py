import numpy as np, pandas as pd
g=pd.read_parquet('data/processed/vnforest/grid500_tn.parquet').drop(columns='geometry')
e=pd.read_parquet('data/processed/vnforest/grid500_tn_extra.parquet')
g=pd.concat([g,e],axis=1); g=g[g.in_lots==1].copy()
g['gf']=g.gfcV3>0
g['h']=g.off_cls+'|'+np.where(g.gf,'F','N')
m=g.off_mdsd.fillna('NA')
g['type']=np.select([m.str.startswith('PH'),m.isin(['VQG','BTTN','NCKH','VHCQ']),m.str.startswith('SX')],['phong_ho','dac_dung','san_xuat'],'khac')
agree={'natural|F','plantation|F','nonforest|N'}
# assumed share of points where record != reality: 0.10 in agreeing strata, 0.50 in disagreeing strata
g['p']=np.where(g.h.isin(agree),0.10,0.50)
W=g.h.value_counts(normalize=True)
print('strata shares',W.round(3).to_dict()); print('type shares',g.type.value_counts(normalize=True).round(3).to_dict())
rng=np.random.default_rng(0); R=400
def run(nh):
    res={'overall':[],**{t:[] for t in ['phong_ho','dac_dung','san_xuat']}}; cnt={t:[] for t in ['phong_ho','dac_dung','san_xuat']}
    grp={h:d for h,d in g.groupby('h')}
    truth_overall=(g.p).mean()
    for _ in range(R):
        S=[]
        for h,d in grp.items():
            s=d.sample(nh,replace=False,random_state=int(rng.integers(1e9))).copy()
            s['y']=rng.random(nh)<s.p; s['w']=W[h]/nh; S.append(s)
        S=pd.concat(S)
        res['overall'].append((S.w*S.y).sum())
        for t in cnt:
            d=S[S.type==t]; res[t].append((d.w*d.y).sum()/d.w.sum()); cnt[t].append(len(d))
    return {k:round(1.96*np.std(v)*100,1) for k,v in res.items()},{k:int(np.mean(v)) for k,v in cnt.items()}
for nh in [50,100,150,200]:
    ci,c=run(nh); print(f'n/stratum={nh} total={6*nh}: 95% CI half-width (pp) {ci}; mean points per forest type {c}')
print('--- strata = record class x GFC x forest type (24 cells); equal per forest type, equal per cell within type')
g['hh']=g.h+'|'+g.type
WW=g.hh.value_counts(normalize=True); grp={h:d for h,d in g.groupby('hh')}
print('cells',len(grp),'smallest cell share',WW.min().round(4))
def run2(N):
    per=N//(4*6); res={'overall':[],'phong_ho':[],'dac_dung':[],'san_xuat':[],'khac':[]}
    for _ in range(R):
        S=[]
        for h,d in grp.items():
            k=min(per,len(d)); s=d.sample(k,random_state=int(rng.integers(1e9))).copy()
            s['y']=rng.random(k)<s.p; s['w']=WW[h]/k; S.append(s)
        S=pd.concat(S); res['overall'].append((S.w*S.y).sum())
        for t in ['phong_ho','dac_dung','san_xuat','khac']:
            d=S[S.type==t]; res[t].append((d.w*d.y).sum()/d.w.sum())
    return per,{k:round(1.96*np.std(v)*100,1) for k,v in res.items()}
for N in [600,960,1200]:
    per,ci=run2(N); print(f'N={N} ({per}/cell): 95% CI half-width (pp) {ci}')
