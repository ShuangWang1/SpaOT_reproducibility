import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

import os,sys
resolved_path = os.path.realpath('..')
sys.path.append(resolved_path+'/moscot-framework_reproducibility')
from lib.fused_pgw_barycenter import fmpgw_barycenters

def normaization(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    normalized = (arr - arr_min) / (arr_max - arr_min)
    return normalized

adj_flag = False
label_feature = False

df = pd.read_csv('data/CRC_clusters_neighborhoods_markers.csv')

df["cluster_id"] = df["ClusterName"].astype("category").cat.codes
one_hot_df = pd.get_dummies(df['cluster_id'])
df['onehot_label'] = one_hot_df.values.tolist()
df['onehot_label'] = df['onehot_label'].apply(lambda x: [int(i) for i in x])

protein_df = df.filter(regex='Cyc_\\d+_ch_\\d+$')
df['expression'] = protein_df.values.tolist()

patients = list(df['patients'].unique())

types = list(df['groups'].unique())

G_list1 = []
G_list2 = []

for spot, group in df.groupby("spots"):
    print(f"=== spot {spot} ===") 
    group = group.reset_index(drop=True)
    points = group[["X:X", "Y:Y"]].values
    tri = Delaunay(points)

    # 3. Extract edges from Delaunay simplices (triangles)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                edges.add((simplex[i], simplex[j]))

    # 4. Build graph
    G = nx.Graph()
    for idx, row in group.iterrows():
        G.add_node(idx, pos=(row["X:X"], row["Y:Y"]), cluster=row["onehot_label"], expression=row["expression"])

    G.add_edges_from(edges)  
    
    if group['groups'].iloc[0] == types[0]:
        G_list1.append(G)
    elif group['groups'].iloc[0] == types[1]:
        G_list2.append(G)

import pickle
for t, G_list in zip(types,[G_list1,G_list2]):
    mean_size1 = int(np.mean([G.number_of_nodes() for G in G_list]) )

    num_x = len(G_list)
    graph_weights =  np.ones(num_x)/num_x    
    internal_weights = [np.ones(len(x.nodes())) / x.number_of_nodes() for x in G_list] 
    mass_list = [None for i in range(num_x)]    

    if label_feature:
        features = [np.array(list(nx.get_node_attributes(x, 'cluster').values())) for x in G_list]
    else:
        features = [np.array(list(nx.get_node_attributes(x, 'expression').values())) for x in G_list]

    if adj_flag:
        structures = [nx.to_numpy_array(x) for x in G_list]  

    else:
        ####Euclidean distance
        from scipy.spatial.distance import cdist

        coords = [np.array(list(nx.get_node_attributes(x, 'pos').values())) for x in G_list]
        structures = [cdist(coord, coord, metric='euclidean') for coord in coords]


    features = [normaization(f) for f in features]
    structures = [normaization(f) for f in structures]

    feature, structure = fmpgw_barycenters(N=mean_size1,
                            Ys=features,
                            Cs=structures,
                            ps=internal_weights,
                            lambdas=graph_weights,
                            alpha=0.5,
                            mass_list=mass_list,
                            max_iter=30,
                            fixed_structure=False,
                            fixed_features=False,
                            verbose=False,
                            init_C=None,
                            init_X=None,
                            tol = 0.01)


    with open(f"bary/barycenter_crc{t}.pkl", "wb") as f:
        pickle.dump({"feature": feature, "structure": structure}, f)
        
    

