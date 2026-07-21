
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt


def rotation_2d(theta=0):
    return np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
    ])

flip_M=np.array([[0,1],[1,0]])

def make_moon_2d(N=200, eta=0.10, ns=0.2,theta=0,beta=0,low=-3,high=-2,random_state=0):
    X, y = make_moons(n_samples=2*N, noise=ns,random_state=random_state)
    
    X1=X[y==0]
    X1_extra = X1.copy()
    X1_extra[:,1] += 0.2

    X2=X[y==1]
    X2=X2.dot(rotation_2d(theta)).dot(flip_M)
    X2=X2+beta
    N=X1.shape[0]
    
    X2_extra = X2.copy()
    X2_extra[:,1] += 0.2
    
    
    #outlier_idx = np.random.choice(np.where(y == 0)[0], int(N*eta))
    #outliers = np.random.uniform(low=low, high=high, size=(int(N*eta), 2))
    
    seed_value = 42
    rng = np.random.default_rng(seed=seed_value)

    # 2. Use the Generator's uniform method
    outliers = rng.uniform(low=low, high=high, size=(int(N*eta), 2))

    noise_len = int(len(outliers)/2)
    return np.vstack((X1,X1_extra)), np.vstack((X2,X2_extra)),outliers, [0]*len(X1) + [1]*len(X1),[0]*len(X2) + [1]*len(X2),[0]*(noise_len) + [1]*(noise_len)

def make_3d_sphere(N=200, eta=0.2, d_min=0.5,low=-3,high=-2,random_state=0):
    circle = (np.linspace(0,2*np.pi,N+1)[0:N]).reshape((N,1)) #np.random.uniform(low=0, high=2*np.pi, size=(N, 1))
    circle = np.hstack((np.cos(circle), np.sin(circle)))
    
    circle2 = (np.linspace(0,3*np.pi,N+1)[0:N]).reshape((N,1)) #np.random.uniform(low=0, high=2*np.pi, size=(N, 1))
    circle2 = np.hstack((np.cos(circle2), np.sin(circle2)))
    
    circle = np.hstack((circle, circle2))
    
    np.random.seed(random_state)
    k=int(np.sqrt(N))
    n_remains=N-k**2
    
    angles =np.random.uniform([0,0],[np.pi,2*np.pi],(N,2))
    phi = angles[:,0]  #np.concatenate((phi,phi_1))
    
    theta = angles[:,1] # np.concatenate((theta,theta_1))
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    sphere = 2 * np.stack((x, y, z), axis=1) 
    
    sphere += np.array([0, 0, 4])
    
    center = np.array([0, 0, 4])
    radii = np.linalg.norm(sphere - center, axis=1)
    
    labels = np.zeros(len(radii), dtype=int)
    labels[(radii >= 1) & (radii <= 2)] = 1
    
    n_outliers = int(N*eta)
    #outliers = sphere[np.random.choice(sphere.shape[0], n_outliers, replace=True)]
    outliers = np.random.uniform(low=low, high=high, size=(n_outliers,2))
    
    
    return sphere, circle, outliers  

# def plot_2d(X1, X2):
#     plt.figure(figsize=(8, 6))
#     plt.scatter(X[y == 0, 0], X[y == 0, 1], c='r', s=20)
#     plt.scatter(X[y == 1, 0], X[y == 1, 1], c='b', s=20)
#     plt.scatter(X[y == 2, 0], X[y == 2, 1], c='c', s=20)
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()

#@nb.njit(fastmath=True)
def generate_d_square(d,n):
    x_list=[]
    for i in range(d):
        x = np.linspace(-1, 1, n)
        x_list.append(x)
    xx= np.meshgrid(*x_list,indexing='ij')
    data=np.zeros((n**d,d))
    for i in range(d):
        data[:,i]=xx[i].reshape(-1)
    return data



def make_3d_square(k=2, eta=0.2, d_min=0.5,low=-3,high=-2):
    N=k**(2*3)
    square=generate_d_square(2,k**3)
    cubic=generate_d_square(3,k**2)
    
    cubic += np.array([0, 0, 4])
    
    n_outliers = int(N*eta)

    outliers = np.random.uniform(low=low, high=high, size=(n_outliers,2))
    
    
    return cubic,square, outliers  


def plot_2d(X1, X2, X1_label, X2_label):
    plt.figure(figsize=(8, 6))
    plt.scatter(X1[:,0], X1[:,1], c=X1_label, s=20)
    plt.scatter(X2[:,0], X2[:, 1], c=X2_label, s=20)
    #plt.scatter(X[y == 2, 0], X[y == 2, 1], c='c', s=20)
    #plt.xticks([])
    #plt.yticks([])
    plt.savefig('demo.png')
    
    
import numpy as np
import matplotlib.pyplot as plt

def plot_plan_2d(X, Y, gamma, threshold=1e-8, 
                 X1_label=None, X2_label=None,name = None):

    plt.figure(figsize=(8, 6))

    # --- Auto-detect if X is mirrored relative to Y ---
    # Compare average transport distance with and without flipping
    def transport_cost(Xa, Y, gamma):
        cost = 0.0
        for i in range(Xa.shape[0]):
            for j in range(Y.shape[0]):
                if gamma[i, j] > threshold:
                    dx = Xa[i, 0] - Y[j, 0]
                    dy = Xa[i, 1] - Y[j, 1]
                    cost += gamma[i, j] * (dx*dx + dy*dy)
        return cost

    # Original cost
    cost_original = transport_cost(X, Y, gamma)

    # Flipped version (horizontal mirror)
    X_flipped = X.copy()
    X_flipped[:, 0] = -X_flipped[:, 0]
    cost_flipped = transport_cost(X_flipped, Y, gamma)

    # Choose better alignment
    if cost_flipped < cost_original:
        X_used = X_flipped
    else:
        X_used = X

    # --- Plot points ---
    plt.scatter(X_used[:, 0], X_used[:, 1], c=X1_label)
    plt.scatter(Y[:, 0], Y[:, 1], c=X2_label)

    # --- Plot transport plan ---
    for i in range(X_used.shape[0]):
        for j in range(Y.shape[0]):
            if gamma[i, j] > threshold:
                plt.plot(
                    [X_used[i, 0], Y[j, 0]],
                    [X_used[i, 1], Y[j, 1]],
                    'grey',
                    lw=gamma[i, j] * 150,
                    alpha=0.4
                )

    #plt.savefig(f'demo_{alpha}_new.png')
    plt.savefig(f'demo_{name}_.png',dpi=300)
    plt.close()
    
 
def moon_to_halfdisk(X, y, alpha, seed=42, eps=1e-6):
    rng = np.random.default_rng(seed)
    N = len(X)

    # ---- center ----
    center = X.mean(axis=0)
    Xc = X - center

    # ---- radius from x-span ----
    x_min, x_max = Xc[:,0].min(), Xc[:,0].max()
    R = (x_max - x_min) / 2.0

    # ---- container ----
    X_circle = np.zeros_like(Xc)

    # ---- map each class ----
    for label in [0, 1]:
        idx = (y == label)
        n = idx.sum()

        # stable angle sampling
        if label == 1:
            theta = rng.uniform(0, np.pi, n)
        else:
            theta = rng.uniform(np.pi, 2*np.pi, n)

        # stable radial sampling (avoid exact zero)
        r = np.sqrt(rng.uniform(eps, 1.0, n))

        X_circle[idx] = np.stack([
            R * r * np.cos(theta),
            R * r * np.sin(theta)
        ], axis=1)

    # ---- smooth interpolation ----
    X_new = (1 - alpha) * Xc + alpha * X_circle

    # ---- soft radial normalization (instead of hard clip) ----
    norms = np.linalg.norm(X_new, axis=1, keepdims=True)
    scale = np.minimum(1.0, R / (norms + eps))
    X_new = X_new * scale

    return X_new + center, y


# benchmarking(spaOT, moscot, paste2, pythonOT) with moon dataset with outlier noise    
X1,X2,outliers,l1,l2,l3 = make_moon_2d(N=200,eta=0.20,ns=0.08,theta=-1/2*np.pi,beta=[-1,-2.5],low=[-2,-3.5],high=[-1.5,-3],random_state=4)

import numpy as np
# X2_noise,l2_noise = moon_to_halfdisk(X2, np.array(l2), 1.0)

# X2=np.vstack((X2,outliers))

# l2 = l2+l3

# plot_2d(X1,X2,l1,l2)



import os,sys
resolved_path = os.path.realpath('..')
sys.path.append(resolved_path+'/moscot-framework_reproducibility')

from scipy.spatial.distance import cdist

from lib.fused_pgw import fused_partial_gromov_wasserstein, fused_partial_gromov_wasserstein_mass


M = cdist([[i] for i in l1], [[i] for i in l2], metric='hamming')


n,m = M.shape
p=np.ones(n)/max(n,m)
q=np.ones(m)/max(n,m)
# p=np.ones(n)/n
# q=np.ones(m)/m

ref_structure = cdist(X1, X1, metric='euclidean')
tgt_structure = cdist(X2, X2, metric='euclidean')

G0=fused_partial_gromov_wasserstein(M,ref_structure,tgt_structure,p=p,q=q,omega2=0.5, Lambda = 1.0)



import pandas as pd
import anndata as ad
import scanpy as sc

data1 = ad.AnnData(
    X=np.array([[i] for i in l1]),
    var=pd.DataFrame(index=[f"feat_{i}" for i in range(1)])
)

data1.obsm["spatial"] = X1

data2=  ad.AnnData(
    X=np.array([[i] for i in l2]),
    var=pd.DataFrame(index=[f"feat_{i}" for i in range(1)])
)

data2.obsm["spatial"] = X2

adata = sc.concat(
    {"xenium": data1, "visium": data2},
    label="dataset",
    join="inner",   # already inner, but safe
    merge="same"
)
from moscot.problems.space import AlignmentProblem


ap = AlignmentProblem(adata=adata)
ap = ap.prepare(batch_key="dataset", policy="sequential")

ap = ap.solve()

gamma=ap[('xenium','visium')].solution

gamma = np.array(gamma.transport_matrix)

import anndata as ad

from paste2 import PASTE2
data1 = ad.AnnData(
    X=np.array([[i] for i in l1]),
    var=pd.DataFrame(index=[f"feat_{i}" for i in range(1)])
)

data1.obsm["spatial"] = X1

data2=  ad.AnnData(
    X=np.array([[i] for i in l2]),
    var=pd.DataFrame(index=[f"feat_{i}" for i in range(1)])
)

data2.obsm["spatial"] = X2

pi_AB,cost = PASTE2.partial_pairwise_align(data1, data2, s=0.8,return_obj = True,dissimilarity = 'euclidean')



import ot

gamma_pot=ot.gromov.entropic_fused_gromov_wasserstein(M,ref_structure,tgt_structure, p=p, q=q, loss_fun='square_loss', alpha=0.5, max_iter=10000.0, tol_rel=1e-09, tol_abs=1e-09)

plot_plan_2d(X1,X2, G0,X1_label = l1, X2_label = l2,name = 'SpaOT')
plot_plan_2d(X1,X2, gamma,X1_label = l1, X2_label = l2,name = 'moscot')
plot_plan_2d(X1,X2, pi_AB,X1_label = l1, X2_label = l2,name = 'PASTE2')
plot_plan_2d(X1,X2, gamma_pot,X1_label = l1, X2_label = l2,name = 'pythonOT')


#add structural noise to the moon dataset and see how the mapping changes with gradually increasing the noise level
X2_copy = X2.copy()
l2_copy = l2.copy()
for alpha in [0.0,0.1,0.3,0.5,0.8,1.0]:
    
    X2_noise,l2_noise = moon_to_halfdisk(X2_copy, np.array(l2_copy), alpha)

    X2=np.vstack((X2_noise,outliers))

    l2 = list(l2_noise)+l3
    
    
    M = cdist([[i] for i in l1], [[i] for i in l2], metric='hamming')


    n,m = M.shape
    p=np.ones(n)/max(n,m)
    q=np.ones(m)/max(n,m)
    # p=np.ones(n)/n
    # q=np.ones(m)/m

    ref_structure = cdist(X1, X1, metric='euclidean')
    tgt_structure = cdist(X2, X2, metric='euclidean')

    G0=fused_partial_gromov_wasserstein(M,ref_structure,tgt_structure,p=p,q=q,omega2=0.5, Lambda = 1.0)
    
    n_out = len(outliers)

    # columns corresponding to outliers
    G_out = G0[:, -n_out:]

    # total transported mass
    total_mass = G0.sum()

    # mass mapped to outliers
    mass_to_outliers = G_out.sum()

    proportion = mass_to_outliers / total_mass

    print(f"Alpha={alpha}")
    print(f"Mass to outliers: {mass_to_outliers:.4f}")
    print(f"Proportion mapped to outliers: {proportion:.4%}")

    plot_plan_2d(X1,X2, G0,X1_label = l1, X2_label = l2,alpha = alpha)