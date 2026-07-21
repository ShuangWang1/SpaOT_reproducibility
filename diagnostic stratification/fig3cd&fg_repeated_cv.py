import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pickle as pkl
import os
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
import pickle

dataset = 'bz'  #crc,metabric, bz


if dataset == 'crc':
    # #patient
    # files = os.listdir('crc_bary_res')

    # dist = np.zeros((35,35))
    # for file in files:
    #     with open(f'crc_bary_res/{file}','rb') as f:
    #         data = pkl.load(f)
    #     for d in data:
    #         dist[d[0],d[1]] = d[2]
    #         dist[d[1],d[0]] = d[2]


    # #region
    # files = os.listdir('crc_bary_region_res')

    # dist = np.zeros((70,70))
    # for file in files:
    #     with open(f'crc_bary_region_res/{file}','rb') as f:
    #         data = pkl.load(f)
    #     for d in data:
    #         dist[d[0],d[1]] = d[2]
    #         dist[d[1],d[0]] = d[2]
            
    #spot
    files = os.listdir('crc_res')

    dist = np.zeros((140,140))
    for file in files:
        with open(f'crc_res/{file}','rb') as f:
            data = pkl.load(f)
        for d in data:
            dist[d[0],d[1]] = d[2]
            dist[d[1],d[0]] = d[2]

    groups = []
    import pandas as pd
    df = pd.read_csv('data/CRC_clusters_neighborhoods_markers.csv')
    for spot, group in df.groupby("spots"):
        print(f"=== Spot {spot} ===,patient {group['patients'].iloc[0]}") 
        groups.append(group['groups'].iloc[0])
    y = np.array(groups) - 1

    # ### patient level
    # df1 = df[df['groups']==1]
    # df2 = df[df['groups']==2]
    # group1 = df1['patients'].unique() - 1
    # group2 = df2['patients'].unique() - 1

    # ## region level 
    # group1 = np.array([int(i.lstrip("reg")) for i in df1['Region'].unique()]) - 1
    # group2 = np.array([int(i.lstrip("reg")) for i in df2['Region'].unique()]) - 1

    # spots = sorted(df['spots'].unique())


    # y = np.zeros(70, dtype=int)

    # y[group2] = 1

    D = dist
    X = D
    
if dataset == 'bz':
            
    #spot
    files = os.listdir('BZ_res')

    dist = np.zeros((559,559))
    for file in files:
        with open(f'BZ_res/{file}','rb') as f:
            data = pkl.load(f)
        for d in data:
            dist[d[0],d[1]] = d[2]
            dist[d[1],d[0]] = d[2]

    groups = []
    
    with open('BZ.pickle','rb') as f:
        data = pickle.load(f)
    celldata = data.img_celldata
    celldata = dict(sorted(celldata.items()))
    
    
    # import scanpy as sc
    # adata = sc.read_h5ad('data/BZ.h5ad')
    
    # adata_dict = {
    #     img_id: adata[adata.obs['core'] == img_id].copy()
    #     for img_id in sorted(adata.obs['core'].unique())
    # }
    
    # idx_dict = {j:i for i,j in enumerate(adata_dict.keys())}
        
    # bad_idx = list(set(adata_dict.keys()) - set(data.keys()))
    
    y = np.zeros(559, dtype=int)
    for i,ad in enumerate(celldata.values()):
        x = ad.uns['graph_covariates']['label_tensors']['grade']
        grade = np.argmax(x)
        if grade == 2:
            y[i] = 1
    
    #remove bad index
    D = dist
    X = D
    
    
if dataset == 'metabric':
            
    #spot
    files = os.listdir('metabric_res')

    dist = np.zeros((515,515))
    for file in files:
        with open(f'metabric_res/{file}','rb') as f:
            data = pkl.load(f)
        for d in data:
            dist[d[0],d[1]] = d[2]
            dist[d[1],d[0]] = d[2]

    groups = []
    
    with open('img_celldata_metabric.pkl','rb') as f:
        data = pickle.load(f)
    data = {int(k): v for k, v in data.items()}
    
    
    import scanpy as sc
    adata = sc.read_h5ad('data/metabric.h5ad')
    
    adata_dict = {
        img_id: adata[adata.obs['ImageNumber'] == img_id].copy()
        for img_id in sorted(adata.obs['ImageNumber'].unique())
    }
    
    idx_dict = {j:i for i,j in enumerate(adata_dict.keys())}
        
    bad_idx = list(set(adata_dict.keys()) - set(data.keys()))
    
    y = np.zeros(515, dtype=int)
    for num,ad in data.items():
        x = ad.uns['graph_covariates']['label_tensors']['grade']
        grade = np.argmax(x)
        if grade == 2:
            y[idx_dict[num]] = 1
    
    #remove bad index
    
    bad_idx = np.array(bad_idx)

    dist = np.delete(dist, bad_idx, axis=0) 
    dist = np.delete(dist, bad_idx, axis=1)
    y = np.delete(y,bad_idx)

    D = dist
    X = D

# Optional: scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


from sklearn.manifold import MDS
mds = MDS(n_components=20, dissimilarity='precomputed', random_state=42)
#X = mds.fit_transform(X)


import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    accuracy_score     
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import torch
from torchmetrics.functional import dice as torch_dice

mds_dim = 20

def dice_score(y_true, y_pred):
    #return f1_score(y_true, y_pred)

    yt = torch.tensor(y_true, dtype=torch.int)
    yp = torch.tensor(y_pred, dtype=torch.int)
    return float(torch_dice(yp, yt))

models = {
    "LR_unreg": {
        "clf": LogisticRegression(
            penalty="l2",     # old sklearn requires this
            C=1e12,           # approximates no regularization
            solver="lbfgs",
            max_iter=5000
        ),
        "param_grid": {}      # no tuning
    },
    

    "LR": {
        "clf": LogisticRegression(max_iter=2000, solver="lbfgs",class_weight="balanced"),
        "param_grid": {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__penalty": ["l2"]
        }
    },
    "RF": {
        "clf": RandomForestClassifier(random_state=42),
        "param_grid": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 5, 10]
        }
    },
    "MLP": {
        "clf": MLPClassifier(max_iter=1000,random_state=42),
        "param_grid": {
            "clf__hidden_layer_sizes": [(32,), (64,), (64, 32)],
            "clf__alpha": [1e-4, 1e-3]
        }
    },
    "SVM": {
        "clf":  SVC(kernel="precomputed",probability=True,random_state=42),
        "param_grid": {
            # Tuning the SVC regularization parameter
            "svc__C": [0.1, 1, 10, 100],
            
            # Tuning the kernel coefficient (gamma) for RBF kernel
            # 'scale' uses 1 / (n_features * X.var()) as gamma
            "svc__gamma": ["scale", 0.001, 0.01, 0.1], 
            
            # Specifying the kernel (RBF is the most common and powerful non-linear kernel)
            "svc__kernel": ["rbf"] 
        }
    }
}

# ---------------------------
# Nested CV settings
# ---------------------------
from sklearn.decomposition import KernelPCA
### repeated nested CV

repeat_times = 5 
outer_splits = 3
inner_splits = 3

results = {'val':{m: [] for m in models.keys()}, 'test': {m: [] for m in models.keys()}}

for repeat in range(repeat_times):
    print(f"\n================ Repeat {repeat+1}/{repeat_times} ================")

    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True,
                               random_state=100 + repeat)

    for model_name, info in models.items():
        print(f"\n----- {model_name} -----")

        pipe = Pipeline([
            # ("mds", MDS(
            #     n_components=mds_dim,
            #     dissimilarity="precomputed",
            #     random_state=0,
            #     normalized_stress="auto"
            # )),
            #("scaler", StandardScaler()),
            ("clf", info["clf"]),
        ])

        #param_grid = info["param_grid"]

        fold_metrics = {
            "Accuracy":[],"AUC": [], "AUPR": [], "F1": [],
            "Dice": [], "Precision": [], "Recall": [], "MCC": []
        }
        val_metrics = {
            "Accuracy":[],"AUC": [], "AUPR": [], "F1": [],
            "Dice": [], "Precision": [], "Recall": [], "MCC": []
        }
        for train_idx, test_idx in outer_cv.split(X, y):
            X_train, X_test = X[np.ix_(train_idx, train_idx)], X[np.ix_(test_idx,train_idx)]
            
            if model_name == 'SVM':
                    
                X_train = np.exp(-1 * X[train_idx].T[train_idx].T)
                X_test = np.exp(-1 * X[test_idx].T[train_idx].T)
            X_test_copy = X_test.copy()
            
            y_train, y_test = y[train_idx], y[test_idx]

            inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True,
                                       random_state=200 + repeat)

            # gs = GridSearchCV(
            #     estimator=pipe,
            #     param_grid=param_grid,
            #     scoring="average_precision",
            #     cv=inner_cv,
            #     n_jobs=-1
            # )

            # gs.fit(X_train, y_train)
            # best_model = gs.best_estimator_

            # y_prob = best_model.predict_proba(X_test)[:, 1]
            # y_pred = (y_prob > 0.5).astype(int)

            # # Metrics
            # fold_metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
            # fold_metrics["AUC"].append(roc_auc_score(y_test, y_prob))
            # fold_metrics["AUPR"].append(average_precision_score(y_test, y_prob))
            # fold_metrics["F1"].append(f1_score(y_test, y_pred))
            # fold_metrics["Precision"].append(precision_score(y_test, y_pred))
            # fold_metrics["Recall"].append(recall_score(y_test, y_pred))
            # fold_metrics["MCC"].append(matthews_corrcoef(y_test, y_pred))

            # # Dice (torchmetrics)
            # y_test_t = torch.tensor(y_test)
            # y_pred_t = torch.tensor(y_pred)
            # fold_metrics["Dice"].append(float(dice_score(y_pred_t, y_test_t)))
                        
            
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train):
                X_inner_train, X_inner_val = X_train[np.ix_(inner_train_idx,inner_train_idx)], X_train[np.ix_(inner_val_idx,inner_train_idx)]
                X_test = X_test_copy.T[inner_train_idx].T
                y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]
                
                if model_name == 'SVM':
                        
                    X_inner_train = (X_train[inner_train_idx].T[inner_train_idx].T)
                    X_inner_val = (X_train[inner_val_idx].T[inner_train_idx].T)
                    X_test = X_test_copy.T[inner_train_idx].T

                # Fit the pipeline on inner train
                pipe.fit(X_inner_train, y_inner_train)
                
                
                
            
                # Evaluate on validation set first
                y_pred = pipe.predict(X_inner_val)
                y_val_prob = pipe.predict_proba(X_inner_val)[:, 1]
                y_val_pred = (y_val_prob > 0.5).astype(int)

                
                val_metrics["Accuracy"].append(accuracy_score(y_inner_val, y_val_pred))
                val_metrics["AUC"].append(roc_auc_score(y_inner_val, y_val_prob))
                val_metrics["AUPR"].append(average_precision_score(y_inner_val, y_val_prob))
                val_metrics["F1"].append(f1_score(y_inner_val, y_val_pred))
                val_metrics["Precision"].append(precision_score(y_inner_val, y_val_pred))
                val_metrics["Recall"].append(recall_score(y_inner_val, y_val_pred))
                val_metrics["MCC"].append(matthews_corrcoef(y_inner_val, y_val_pred))

                # Dice score (torchmetrics)
                y_val_t = torch.tensor(y_inner_val)
                y_val_pred_t = torch.tensor(y_val_pred)
                val_metrics["Dice"].append(float(dice_score(y_val_pred_t, y_val_t)))



                # After validation, evaluate on test set
                y_prob = pipe.predict_proba(X_test)[:, 1]
                y_pred = (y_prob > 0.5).astype(int)

                fold_metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
                fold_metrics["AUC"].append(roc_auc_score(y_test, y_prob))
                fold_metrics["AUPR"].append(average_precision_score(y_test, y_prob))
                fold_metrics["F1"].append(f1_score(y_test, y_pred))
                fold_metrics["Precision"].append(precision_score(y_test, y_pred))
                fold_metrics["Recall"].append(recall_score(y_test, y_pred))
                fold_metrics["MCC"].append(matthews_corrcoef(y_test, y_pred))

                # Dice (torchmetrics)
                y_test_t = torch.tensor(y_test)
                y_pred_t = torch.tensor(y_pred)
                fold_metrics["Dice"].append(float(dice_score(y_pred_t, y_test_t)))
     

        results['test'][model_name].append({k: v for k,v in fold_metrics.items()})
        
        results['val'][model_name].append({k: v for k,v in val_metrics.items()})

import pickle

with open(f'{dataset}_rcv_res/repeated_cv_new_533.pkl', 'wb') as f:
    pickle.dump(results,f)


def plot(data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Prepare AUPR dataframe
    aupr_values = {
        model: [float(entry["AUPR"]) for entry in results]
        for model, results in data.items()
    }

    df = pd.DataFrame([
        {"Model": model, "AUPR": auprs[i]}
        for model, auprs in aupr_values.items()
        for i in range(len(auprs))
    ])

    plt.figure(figsize=(8, 5))
    sns.set(style="whitegrid", font_scale=1.2)

    # Boxen plot (very clean, less noisy than violin)
    sns.boxenplot(
        data=df,
        x="Model",
        y="AUPR",
        palette="Set2"  # nice soft color palette
    )

    plt.title("AUPR Comparison Across Models", fontsize=16)
    plt.xlabel("")
    plt.ylabel("AUPR", fontsize=14)
    plt.tight_layout()
    plt.savefig(f'aupr/aupr_patient_533.png')
#plot(results)

# Final averaging across repeats
# final_results = {
#     m: {metric: np.mean([r[metric] for r in res])
#         for metric in res[0].keys()}
#     for m, res in results.items()
# }

# print("\n\n================ FINAL RESULTS ================\n")
# for m, res in final_results.items():
#     print(f"{m}: {res}")
