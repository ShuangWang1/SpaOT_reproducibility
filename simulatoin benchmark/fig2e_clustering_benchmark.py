import os, sys
resolved_path = os.path.realpath('..')
sys.path.append(resolved_path)
import time
import numpy as np

# ── Global seeds — set before any library that draws random numbers ───────────
SEED = 42
np.random.seed(SEED)
import torch
torch.manual_seed(SEED)
from lib.fgw.graph import Graph
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import adjusted_mutual_info_score
from lib.graph_clustering import *
import copy
import networkx as nx

X, y, V = build_dataset(5, type_list=[1, 2, 3],seed1=42,seed2=42)
for x in X:
    x.N = len(x.values())
X = np.array(X)

X1 = [copy.deepcopy(x) for x in X]
per = 0.5
num_graphs = int(len(X) * per)
eta = 0.3
for i in range(num_graphs):
    x1 = X1[i]
    add_outliers(x1, eta=eta, gap=4, seed=SEED + i)
print(x1.nodes().keys())
X1 = np.array(X1)

# Precompute all shortest-path distances
for x in X:
    C = x.distance_matrix(method='shortest_path', force_recompute=True)
for x in X1:
    x.distance_matrix(method='shortest_path', force_recompute=True)
X1 = np.array(X1)
connected = []
for x in X1:
    connected.append(nx.is_connected(x.nx_graph))

alpha = 0.50

torch.save(X1, 'data.pt')

colors = {
    'node': '#465A7D',
    'node_outline': '#000000',
    'edge': '#465A7D',
    'outlier_node': '#E98B7F',
    'label': '#000000',
    'title': '#000000'
}

parameter_set = {}
parameter_set['moscot'] = {'i_max': 2, 'row_num': 3}
parameter_set['paste2'] = {'i_max': 2, 'row_num': 3}


def enlarge_pos(pos, scale=10):
    pos2 = {}
    for key in pos:
        pos2[key] = pos[key] * scale
    return pos2


def run_and_visualize(measure):
    print(f'\n=== {measure} ===')
    st_time = time.time()
    kmeans = FusedGromovWassersteinGraphKMeans1(
        N=None, Nc=3, n_clusters=3, max_iter=200, max_iter_barycenter=200, n_init=2,
        metric_params={"alpha": 0.5}, verbose=1, measure=measure, random_state=SEED
    )
    kmeans.fit(X1)
    run_time = time.time() - st_time

    print(kmeans.labels_)
    ami = adjusted_mutual_info_score(y, kmeans.labels_)

    print(f'run_time: {run_time:.1f}s   AMI: {ami:.4f}')

    result = {
        'method': measure,
        'time': run_time,
        '_cluster_centers_features': kmeans._cluster_centers_features,
        '_cluster_centers_structure': kmeans._cluster_centers_structure,
        'labels': kmeans.labels_,
        'ami': ami,
    }
    torch.save(result, f'result/{measure}_{alpha}_adjust_s.pt')

    # ── visualisation (mirrors clustering.py) ─────────────────────────────────
    labels = result['labels']
    centers_features = result['_cluster_centers_features']
    centers_structure = result['_cluster_centers_structure']

    rows_per_label = [int(np.ceil(np.sum(labels == label) / 5)) for label in [0, 1, 2]]
    row_num = sum(rows_per_label)
    fig, axes = plt.subplots(row_num, 6, figsize=(3 * 6, 3 * row_num))

    centroid_rows = [0, rows_per_label[0], rows_per_label[0] + rows_per_label[1]]
    for i, (feature, structure) in enumerate(zip(centers_features, centers_structure)):
        ax = axes[centroid_rows[i], 0]
        print('i is', i)

        A = sp_to_adjency(structure, threshinf=0, threshsup=find_thresh(structure, sup=1.5, step=150)[0])
        bary = nx.from_numpy_array(A)
        for j in range(len(feature)):
            bary.add_node(j, attr_name=float(feature[j][0]))

        pos = nx.kamada_kawai_layout(bary)
        nx.draw(bary, pos=pos, node_color=feature.ravel(),
                vmin=feature.min(), vmax=feature.max(),
                with_labels=False, node_size=150, ax=ax)

    row = 0
    margin = 0.1
    for label in [0, 1, 2]:
        graphs = X1[labels == label]
        col = 1

        for j, g in enumerate(graphs):
            if col >= 6 and len(graphs) - col >= 0:
                print('here')
                row += 1
                col -= 5

            ax = axes[row, col]
            pos = nx.kamada_kawai_layout(g.nx_graph)
            regular_nodes = [n for n in g.nx_graph.nodes() if n <= 45]
            outlier_nodes = [n for n in g.nx_graph.nodes() if n > 45]
            v = g.values()
            v = np.array(v)

            ax.set_xlim(-1.5 - margin, 1.5 + margin)
            ax.set_ylim(-1.5 - margin, 1.5 + margin)

            regular_nodes = [n for n in g.nx_graph.nodes() if n <= 45]
            outlier_nodes = [n for n in g.nx_graph.nodes() if n > 45]
            pos = nx.kamada_kawai_layout(g.nx_graph.subgraph(regular_nodes))
            if outlier_nodes:
                pos2 = nx.kamada_kawai_layout(g.nx_graph.subgraph(outlier_nodes))
                pos2 = enlarge_pos(pos2, scale=1.5)
                pos.update(pos2)

            v = g.values()
            v = np.array(v)

            nx.draw_networkx_nodes(g.nx_graph, pos=pos, ax=ax,
                                   nodelist=regular_nodes,
                                   node_color=v[0:g.N],
                                   node_size=60, edgecolors='none')

            if len(v) > g.N:
                v[g.N + 1:] = -10
                nx.draw_networkx_nodes(g.nx_graph, pos=pos, ax=ax,
                                       nodelist=outlier_nodes,
                                       node_color=colors['node_outline'],
                                       node_size=60, edgecolors='none',
                                       node_shape='s')

            nx.draw_networkx_edges(g.nx_graph, pos=pos, ax=ax,
                                   edge_color=colors['edge'],
                                   width=0.5, alpha=0.5)
            col += 1
        row += 1

    for ax in axes.flat:
        ax.axis('off')

    line_positions = [rows_per_label[0] / row_num, (rows_per_label[0] + rows_per_label[1]) / row_num]
    line_positions = [1 - p for p in line_positions]
    for pos in line_positions:
        fig.add_artist(plt.Line2D([0, 1], [pos, pos], color='grey',
                                  linestyle='--', linewidth=2,
                                  transform=fig.transFigure))

    line_positions = [1 / 6]
    for pos in line_positions:
        fig.add_artist(plt.Line2D([pos, pos], [0, 1], color='black',
                                  linestyle='-', linewidth=2,
                                  transform=fig.transFigure))

    fig.text(0.1, -0.05, 'centroid', fontsize=35, ha='center', va='center',
             transform=fig.transFigure)
    fig.text(0.5, -0.05, 'data', fontsize=35, color='grey', ha='center',
             va='center', transform=fig.transFigure)
    fig.suptitle(f'{measure}  α={alpha}  AMI={ami:.3f}', fontsize=14)

    plt.tight_layout()
    plt.savefig(f'result/{measure}_{alpha}_adjust_s.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'result/{measure}_{alpha}_adjust_s.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved result/{measure}_{alpha}_adjust_s.png / .pdf')
    return result


result_moscot = run_and_visualize('moscot')
result_paste2 = run_and_visualize('paste2')
result_fpgw = run_and_visualize('fmpgw')
result_fgw = run_and_visualize('fgw')

print('\n=== Summary ===')
print(f"moscot  AMI={result_moscot['ami']:.4f}  time={result_moscot['time']:.1f}s")
print(f"paste2  AMI={result_paste2['ami']:.4f}  time={result_paste2['time']:.1f}s")
print(f"fpgw    AMI={result_fpgw['ami']:.4f}  time={result_fpgw['time']:.1f}s")
print(f"fgw     AMI={result_fgw['ami']:.4f}  time={result_fgw['time']:.1f}s")
