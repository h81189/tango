from features import *
import DataStruct as ds
import visualization as vis
import interface as itf
import numpy as np
import pandas as pd
from embed import Watershed
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import random

pstruct = ds.DataStruct(config_path = '/hpc/group/tdunn/hk276/tangosave/configs/path_configs/NUCLA_datastruct_config.yaml')
# pstruct = ds.DataStruct(config_path = '/hpc/group/tdunn/hk276/tangosave/configs/path_configs/NTU60_datastruct_config.yaml')
# pstruct = ds.DataStruct(config_path = '/hpc/group/tdunn/hk276/tangosave/configs/path_configs/NTU120_datastruct_config.yaml')
pstruct.load_connectivity()
pstruct.load_pose()

pc_feats, pc_labels = read_h5(path = '/hpc/group/tdunn/hk276/tangosave/src/NUCLAcapture/pca_feats.h5')
# pc_feats, pc_labels = read_h5(path = '/hpc/group/tdunn/hk276/tangosave/src/NTU60view/pca_feats.h5')
# pc_feats, pc_labels = read_h5(path = '/hpc/group/tdunn/hk276/tangosave/src/NTU120view/pca_feats.h5')


params = itf.read_params_config(config_path = '../configs/fitsne.yaml')

print("downsampling")

pstruct.features = pc_feats[::params['downsample']]
pstruct.frame_id = np.arange(0,pc_feats.shape[0],params['downsample'])
# print(len(pstruct.exp_id))
print(pstruct.exp_ids_full[::params['downsample']])
pstruct.exp_id = pstruct.exp_ids_full[::params['downsample']]
pstruct.downsample = params['downsample']
pstruct.load_meta()

print("done downsampling")

params_config = '../configs/fitsne.yaml'
params = itf.params_process(params_config)
pstruct, embedder = itf.embed_pipe(pstruct, params, save_embedder = params['save_embedder'])

rf_avg=[[],[],[]]
rf_std=[[],[],[]]

meta=pd.read_csv("/hpc/group/tdunn/hk276/CAPTURE_demo/Python_analysis/engine/NUCLAmeta.csv")
# meta=pd.read_csv("/hpc/group/tdunn/hk276/CAPTURE_demo/Python_analysis/engine/NTU60ViewFullMeta.csv")
# meta=pd.read_csv("/hpc/group/tdunn/hk276/CAPTURE_demo/Python_analysis/engine/NTU120ViewFullMeta.csv")

for sig in np.arange(20,7,-1):
    pstruct = itf.watershed_cluster(pstruct,
                            sigma = sig,
                            column = params['density_by_column'],
                            plots_label = "final",
                            save_ws = True)

    vis.density(pstruct.ws.density, pstruct.ws.borders, show=True,
                filepath = ''.join([pstruct.out_path,'sigma_'+str(sig)+'_density.png']))

    act_covs=[[] for i in range(12)]
    # act_covs=[[] for i in range(60)]
    # act_covs=[[] for i in range(120)]

    person_covs=[[] for i in range(10)]
    # person_covs=[[] for i in range(40)]
    # person_covs=[[] for i in range(106)]

    view_covs=[[] for i in range(3)]

    scene_covs=[[] for i in range(5)]
    # scene_covs=[[] for i in range(17)]
    # scene_covs=[[] for i in range(32)]

    cmax=max(pstruct.data.Cluster)
    vidstarts=np.append(np.unique(pstruct.data.exp_id,return_index=True)[1],-1)
    leng=len(pstruct.data.exp_id)
    for i in range(len(vidstarts)-1):
        curr=np.unique(pstruct.data.Cluster[vidstarts[i]:vidstarts[i+1]], return_counts=True)
        freqs=curr[1]
        pos=curr[0]
        freqs=freqs/sum(freqs)
        t=np.zeros(cmax)
        t.put(pos-1,freqs)
        act_covs[pstruct.data.action[vidstarts[i]]-1].append(t)
        person_covs[meta['person'][i]-1].append(i)
        scene_covs[meta['scene'][i]-1].append(i)
        # view_covs[meta['view'][i]-1].append(i)
        # view_covs[pstruct.data.action[vidstarts[i]]-1].append(meta['view'][i])


    acts=np.concatenate([np.full(len(act_covs[i]),i) for i in range(len(act_covs))])
    cov=np.concatenate(act_covs)
    # persons=np.concatenate(person_covs)
    # scenes=np.concatenate(scene_covs)
    # views=np.concatenate(view_covs)
    comb=[[[i] for i in range(len(acts))],persons,scenes,views]
    num=[len(acts),106,32,3]
    cvfold=[3,3,3,3]
    folds=[]
    for i in range(len(num)):
        fold=np.arange(num[i])
        random.shuffle(fold)
        folds.append(np.array_split(fold,cvfold[i]))
    rfacc=[]
    knnacc=[]
    for k in range(4):
        for i in range(cvfold[k]):
            inds=[]
            for l in range(folds[k]):
                inds.append(comb[k][l])
            X_test=cov[inds]
            y_test=acts[inds]
            X_train=np.delete(cov,inds)
            y_train=np.delete(acts,inds)

            rf = RandomForestClassifier(n_estimators = 100)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            rfacc.append(metrics.accuracy_score(y_test, y_pred))

        rf_avg[k].append(np.average(rfacc))
        rf_std[k].append(np.std(rfacc))
    print(rf_avg)
    print(rf_std)