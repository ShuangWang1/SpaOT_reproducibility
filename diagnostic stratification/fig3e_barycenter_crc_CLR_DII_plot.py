import pickle
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

# D = your (n x n) distance matrix
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)

with open('bary_CLR_DII/barycenter_crc2_zeroinit.pkl','rb') as f:
     data2 = pickle.load(f)
with open('bary_CLR_DII/barycenter_crc1_zeroinit.pkl','rb') as f:
     data1 = pickle.load(f)
     
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
coords = mds.fit_transform(data2['structure'])
plt.close()
plt.scatter(coords[:,0], coords[:,1],s=5)
plt.savefig('bary_CLR_DII/barycenter_crc3.png')


import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd


for i in range(len(data2['feature'][0])):
    values = data2['feature'][:,i]
    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())


    cmap = cm.viridis  

    colors = cmap(norm(values))
    plt.scatter(coords[:,0], coords[:,1],s=5, c=values, cmap='viridis')
    plt.colorbar()
    plt.savefig(f'bary_CLR_DII/barycenter_crc2_feature{i}.png')
    plt.close()

for i in range(len(data1['feature'][0])):
    values = data1['feature'][:,i]
    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())


    cmap = cm.viridis  

    colors = cmap(norm(values))
    plt.scatter(coords[:,0], coords[:,1],s=5, c=values, cmap='viridis')
    plt.colorbar()
    plt.savefig(f'bary_CLR_DII/barycenter_crc1_feature{i}.png')
    plt.close()

df = pd.read_csv('data/CRC_clusters_neighborhoods_markers.csv')


protein_df = df.filter(regex='Cyc_\\d+_ch_\\d+$')
df['expression'] = protein_df.values.tolist()