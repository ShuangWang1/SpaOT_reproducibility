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


#use metabric to predict bz, if dataset = 'bz', use bz to predict metabric
dataset = 'metabric' 

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
    bad_idx = [idx_dict[i] for i in bad_idx]
    
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


#load inter, use metabric to predict BZ

files = os.listdir('inter_res')

inter_dist = np.zeros((500,559))
for file in files:
    with open(f'inter_res/{file}','rb') as f:
        data = pkl.load(f)
    for d in data:
        inter_dist[d[0],d[1]] = d[2]
            
with open('BZ.pickle','rb') as f:
    data = pickle.load(f)
celldata = data.img_celldata
celldata = dict(sorted(celldata.items()))

y_test = np.zeros(559, dtype=int)
for i,ad in enumerate(celldata.values()):
    x = ad.uns['graph_covariates']['label_tensors']['grade']
    grade = np.argmax(x)
    if grade == 2:
        y_test[i] = 1

# Optional: scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#use z-score
# mean = dist.mean()
# std  = dist.std() + 1e-8
# D_train_norm = (dist - mean) / std

# mean = inter_dist.mean()
# std  = inter_dist.std() + 1e-8
# D_test_norm = (inter_dist - mean) / std


D_train_norm = (dist - dist.min(axis=1, keepdims=True)) / \
               (dist.max(axis=1, keepdims=True) - dist.min(axis=1, keepdims=True) + 1e-8)

D_test_norm  = (inter_dist - inter_dist.min(axis=0, keepdims=True)) / \
               (inter_dist.max(axis=0, keepdims=True) - inter_dist.min(axis=0, keepdims=True) + 1e-8)


from sklearn.decomposition import PCA

pca = PCA(
    n_components=50,      # try 20, 50, 100
    whiten=True,          # important for LR/SVM
    random_state=42
)

X_train_pca = pca.fit_transform(D_train_norm)
X_test_pca  = pca.transform(D_test_norm)


from sklearn.manifold import MDS
mds = MDS(n_components=20, dissimilarity='precomputed', random_state=42)
#X = mds.fit_transform(X)


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    make_scorer,
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def dice_score(y_true, y_pred):
    #return f1_score(y_true, y_pred)

    yt = torch.tensor(y_true, dtype=torch.int)
    yp = torch.tensor(y_pred, dtype=torch.int)
    return float(torch_dice(yp, yt))

pos_num = len(y[y==1])
neg_num = len(y[y==0])

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
        "clf": LogisticRegression(max_iter=2000),
        "param_grid": {
            'C': [0.01, 0.1, 1, 10],          # Regularization strength
            'penalty': ['l1', 'l2'],          # L1 = Lasso, L2 = Ridge
            'solver': ["lbfgs",'liblinear'],           # liblinear supports l1 & l2
            'class_weight': ['balanced', None] # handle class imbalance
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
    # "SVM": {
    #     "clf":  SVC(kernel="precomputed",probability=True,random_state=42),
    #     "param_grid": {
    #         # Tuning the SVC regularization parameter
    #         "svc__C": [0.1, 1, 10, 100],
            
    #         # Tuning the kernel coefficient (gamma) for RBF kernel
    #         # 'scale' uses 1 / (n_features * X.var()) as gamma
    #         "svc__gamma": ["scale", 0.001, 0.01, 0.1], 
            
    #         # Specifying the kernel (RBF is the most common and powerful non-linear kernel)
    #         "svc__kernel": ["rbf"] 
    #     }
    # },
    "xgboost":{
        "clf": xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42
                )
    }
    
}

#best model

models = {    

    "LR": {
        "clf": LogisticRegression(max_iter=2000),
        "param_grid": {
            'C': [0.01, 0.1, 1, 10],          # Regularization strength
            'penalty': ['l1', 'l2'],          # L1 = Lasso, L2 = Ridge
            'solver': ["lbfgs",'liblinear'],           # liblinear supports l1 & l2
            'class_weight': ['balanced', None] # handle class imbalance
        }
    }
    
}



# ---------------------------
# Nested CV settings
# ---------------------------
### repeated nested CV

results = {'test': {m: [] for m in models.keys()}}

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

    fold_metrics = {
        "Accuracy":[],"AUC": [], "AUPR": [], "F1": [],
        "Dice": [], "Precision": [], "Recall": [], "MCC": [],'cm':[]
    }


    X_train, X_test = X_train_pca, X_test_pca
    
    if model_name == 'SVM':
            
        X_train = np.exp(-1 * X_train)
        X_test = np.exp(-1 * X_test)
        
    
    y_train, y_test = y, y_test
    
    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test.T)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # scorer = make_scorer(f1_score)

    # # 4️⃣ Grid Search
    # grid = GridSearchCV(
    #     estimator=info["clf"],
    #     param_grid=info["param_grid"],
    #     scoring=scorer,
    #     cv=cv,
    #     n_jobs=-1,
    #     verbose=2
    # )

    # # 5️⃣ Fit
    # grid.fit(X_train, y_train)
    
    # best_model = grid.best_estimator_
    # y_prob = best_model.predict_proba(X_test)[:, 1]
    # y_pred = (y_prob > 0.5).astype(int)
    
    
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)
    fold_metrics["cm"].append(cm)

    # Metrics
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
    # print(fold_metrics)
                
    results['test'][model_name].append({k: v for k,v in fold_metrics.items()})
    

import pickle

with open('inter_cm.pkl', 'wb') as f:
    pickle.dump(results,f)

