
import time
import torch
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
import networkx as nx
import ot
from scipy.spatial.distance import cdist
from tqdm import tqdm
import multiprocessing
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed
import re

import os,sys
resolved_path = os.path.realpath('..')
sys.path.append(resolved_path+'/moscot-framework_reproducibility')

from lib.fused_pgw import fused_partial_gromov_wasserstein, fused_partial_gromov_lambda,fused_pgw_cost
from lib.fgw.ot_distances import Fused_Gromov_Wasserstein_distance
from lib.sinkhorn_pgw import entropic_semirelaxed_partial_fused_gromov_wasserstein_numba
from lib.fgw.graph import Graph
import pickle

import sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
args = parser.parse_args()

def global_calc_fgw(M,C1,C2,t1masses,t2masses,mass,Lambda,alpha,amijo,verbose,symmetric,measure='fpgw'):
        
    if measure =='fmpgw':
        gamma=fused_partial_gromov_wasserstein(M,C1,C2,t1masses,t2masses,mass=mass,omega2=alpha,loss_fun='square_loss',amijo=amijo,verbose=verbose,symmetric=symmetric)
        cost=fused_pgw_cost(M,C1,C2,gamma,alpha,loss_fun='square_loss')
        return gamma,cost
    elif measure =='mpgw':
        gamma=fused_partial_gromov_wasserstein(M,C1,C2,t1masses,t2masses,mass=mass,omega2=1.0,loss_fun='square_loss',amijo=amijo,verbose=verbose,symmetric=symmetric)
        cost=fused_pgw_cost(M,C1,C2,gamma,1.0,loss_fun='square_loss')
        return gamma,cost
        
        
    elif measure=='fpgw':
        gamma=fused_partial_gromov_lambda(M,C1,C2,t1masses,t2masses,Lambda=Lambda,omega2=alpha,loss_fun='square_loss',amijo=amijo,
                                            verbose=verbose,symmetric=symmetric)
        cost=fused_pgw_cost(M,C1,C2,gamma,alpha,loss_fun='square_loss')
        return gamma,cost

    elif measure=='pgw':
        gamma=fused_partial_gromov_lambda(M,C1,C2,t1masses,t2masses,Lambda=Lambda,omega2=1.0,loss_fun='square_loss',amijo=amijo,
                                            verbose=verbose,symmetric=symmetric)
        cost=fused_pgw_cost(M,C1,C2,gamma,alpha,loss_fun='square_loss')
        return gamma,cost
        
    elif measure=='sink-fpgw':  
        M_normal = M / M.sum()
        C1_nor = C1/C1.sum()     
        C2_nor = C2/C2.sum()
        
        coup_pgw=entropic_semirelaxed_partial_fused_gromov_wasserstein_numba(M_normal,C1_nor,C2_nor,p=t1masses,q=t2masses,Lambda1=Lambda,omega2=alpha,reg=0.01,F1='PTV',F2='PTV',stable=True)

        cost=fused_pgw_cost(M,C1,C2,coup_pgw,alpha,loss_fun='square_loss')
        return coup_pgw,cost
        

    elif measure=='fugw':
        rho1=0.1
        rho2=0.1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        C1_torch = torch.tensor(C1, dtype=torch.float32, device=device)
        C2_torch = torch.tensor(C2, dtype=torch.float32, device=device)
        M_torch = torch.tensor(M, dtype=torch.float32, device=device)
        p_torch = torch.tensor(t1masses, dtype=torch.float32, device=device)
        q_torch = torch.tensor(t2masses, dtype=torch.float32, device=device)
        result=self.fugw.solve(alpha=self.alpha,rho_s=rho1,rho_t=rho2,eps=1e-2,reg_mode="joint",divergence="kl",F=M_torch,Ds=C1_torch,Dt=C2_torch,
                                ws=p_torch,wt=q_torch,init_plan=None,solver="sinkhorn",callback_bcd=None,verbose=False)
        gamma=result['gamma']
        cost=result['loss']['total'][-1]
        return gamma,cost


 
def global_graph_d(graph1,graph2,method,features_metric,measure,mass,Lambda,alpha,amijo,verbose,symmetric):
    """ Compute the Fused Gromov-Wasserstein distance between two graphs. Uniform weights are used.        
    Parameters
    ----------
    graph1 : a Graph object
    graph2 : a Graph object
    Returns
    -------
    The Fused Gromov-Wasserstein distance between the features of graph1 and graph2
    """
    gofeature=True
    nodes1=graph1.nodes()
    nodes2=graph2.nodes()
    startstruct=time.time()
    C1=graph1.distance_matrix(method=method)
    C2=graph2.distance_matrix(method=method)
    
    C1 = (C1 - C1.min()) / (C1.max() - C1.min())
    C2 = (C2 - C2.min()) / (C2.max() - C2.min())
    
    end2=time.time()
    

    try :
        x1=graph1.all_matrix_attr()
        x2=graph2.all_matrix_attr()
        #print('grapha1.node_std',graph1.node_std)
        if graph1.node_std is not None:
            #print('here')
            x1/=graph1.node_std
            x2/=graph2.node_std
        
    except:
        gofeature=False
    #print('gofeature',gofeature)
    if gofeature : 
        if features_metric=='dirac':
            f=lambda x,y: x!=y
            M=ot.dist(x1,x2,metric=f)
        elif features_metric=='hamming_dist': #see experimental setup in the original paper
            f=lambda x,y: hamming_dist(x,y)
            M=ot.dist(x1,x2,metric=f)
        else:
            M=ot.dist(x1,x2,metric=features_metric)
    else:
        M=np.zeros((C1.shape[0],C2.shape[0]))
    M = (M - M.min()) / (M.max() - M.min())
        
    startdist=time.time()
    if measure in ['fgw']:
        t1masses,t2masses=np.ones(len(nodes1))/len(nodes1),np.ones(len(nodes2))/len(nodes2)
        transpwgw,log=global_calc_fgw(M,C1,C2,t1masses,t2masses,mass,Lambda,alpha,amijo,verbose,symmetric,measure=measure)
        enddist=time.time()
        log['struct_time']=(end2-startstruct)
        log['dist_time']=(enddist-startdist)
        return log['loss'][::-1][0]
    elif measure in ['fpgw','fmpgw','fugw', 'sink-fpgw']:
        N1,N2=graph1.N,graph2.N
        
        if N1 is None or N2 is None:
            print('error in data loading')
        
        t1masses = np.ones(len(nodes1))/N1
        t2masses = np.ones(len(nodes2))/N2
                    
        gamma,cost = global_calc_fgw(M,C1,C2,t1masses,t2masses,mass,Lambda,alpha,amijo,verbose,symmetric,measure=measure)
        return cost
    
    elif measure in ['rgw']:
        t1masses,t2masses=np.ones([len(nodes1),1])/len(nodes1),np.ones([len(nodes2),1])/len(nodes2)
        transpwgw,cost=global_calc_fgw(M,C1,C2,t1masses,t2masses,mass,Lambda,alpha,amijo,verbose,symmetric,measure=measure)
        enddist=time.time()
        return cost

def compute_pair(pair,method,features_metric,measure,mass,Lambda,alpha,amijo,verbose,symmetric):
    i, j, s,t = pair
    cost = global_graph_d(s,t,method,features_metric,measure,mass,Lambda,alpha,amijo,verbose,symmetric)
    return i, j, cost  


def compute_batch(batch,method,features_metric,measure,mass,Lambda,alpha,amijo,verbose,symmetric):
    result = []
    for pair in batch:
        i, j, s,t = pair
        cost = global_graph_d(s,t,method,features_metric,measure,mass,Lambda,alpha,amijo,verbose,symmetric)
        result.append((i,j,cost))
        print(i,j,cost)
    return result

class Fused_Partial_Gromov_Wasserstein_distance(Fused_Gromov_Wasserstein_distance):
     def __init__(self,alpha=0.5,method='shortest_path',features_metric='sqeuclidean',measure='fgw',mass=0,Lambda=0,max_iter=None,verbose=False,symmetric=True,amijo=False): #remplacer method par distance_method
        Fused_Gromov_Wasserstein_distance.__init__(self,alpha=alpha,method=method,features_metric=features_metric,max_iter=max_iter,verbose=verbose,amijo=amijo)
        self.symmetric=symmetric
        self.mass=mass
        self.Lambda=Lambda
        self.measure = measure
        self.dist_matrix=None
    
     def calc_fgw(self,M,C1,C2,t1masses,t2masses):
            
        if self.measure =='fmpgw':
            #transpwgw,log= fgw_lp((1-self.alpha)*M,C1,C2,t1masses,t2masses,'square_loss',G0=None,alpha=self.alpha,verbose=self.verbose,amijo=self.amijo,log=True)
            gamma=fused_partial_gromov_wasserstein(M,C1,C2,t1masses,t2masses,mass=self.mass,omega2=self.alpha,loss_fun='square_loss',amijo=self.amijo,verbose=self.verbose,symmetric=self.symmetric)
            cost=fused_pgw_cost(M,C1,C2,gamma,self.alpha,loss_fun='square_loss')
            return gamma,cost
        elif self.measure =='mpgw':
            gamma=fused_partial_gromov_wasserstein(M,C1,C2,t1masses,t2masses,mass=self.mass,omega2=1.0,loss_fun='square_loss',amijo=self.amijo,verbose=self.verbose,symmetric=self.symmetric)
            cost=fused_pgw_cost(M,C1,C2,gamma,1.0,loss_fun='square_loss')
            return gamma,cost
         
            
        elif self.measure=='fpgw':
            gamma=fused_partial_gromov_lambda(M,C1,C2,t1masses,t2masses,Lambda=self.Lambda,omega2=self.alpha,loss_fun='square_loss',amijo=self.amijo,
                                              verbose=self.verbose,symmetric=self.symmetric)
            cost=fused_pgw_cost(M,C1,C2,gamma,self.alpha,loss_fun='square_loss')
            return gamma,cost

        elif self.measure=='pgw':
            gamma=fused_partial_gromov_lambda(M,C1,C2,t1masses,t2masses,Lambda=self.Lambda,omega2=1.0,loss_fun='square_loss',amijo=self.amijo,
                                              verbose=self.verbose,symmetric=self.symmetric)
            cost=fused_pgw_cost(M,C1,C2,gamma,self.alpha,loss_fun='square_loss')
            return gamma,cost
         
        elif self.measure=='sink-fpgw':  
            M_normal = M / M.sum()
            C1_nor = C1/C1.sum()     
            C2_nor = C2/C2.sum()
            
            coup_pgw=entropic_semirelaxed_partial_fused_gromov_wasserstein_numba(M_normal,C1_nor,C2_nor,p=t1masses,q=t2masses,Lambda1=self.Lambda,omega2=self.alpha,reg=0.01,F1='PTV',F2='PTV',stable=True)

            cost=fused_pgw_cost(M,C1,C2,coup_pgw,self.alpha,loss_fun='square_loss')
            return coup_pgw,cost
            

        elif self.measure=='fugw':
            rho1=0.1
            rho2=0.1
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            C1_torch = torch.tensor(C1, dtype=torch.float32, device=device)
            C2_torch = torch.tensor(C2, dtype=torch.float32, device=device)
            M_torch = torch.tensor(M, dtype=torch.float32, device=device)
            p_torch = torch.tensor(t1masses, dtype=torch.float32, device=device)
            q_torch = torch.tensor(t2masses, dtype=torch.float32, device=device)
            result=self.fugw.solve(alpha=self.alpha,rho_s=rho1,rho_t=rho2,eps=1e-2,reg_mode="joint",divergence="kl",F=M_torch,Ds=C1_torch,Dt=C2_torch,
                                   ws=p_torch,wt=q_torch,init_plan=None,solver="sinkhorn",callback_bcd=None,verbose=False)
            gamma=result['gamma']
            cost=result['loss']['total'][-1]
            return gamma,cost


        

    
    
        
     def graph_d(self,graph1,graph2,beta=0.001):
        """ Compute the Fused Gromov-Wasserstein distance between two graphs. Uniform weights are used.        
        Parameters
        ----------
        graph1 : a Graph object
        graph2 : a Graph object
        Returns
        -------
        The Fused Gromov-Wasserstein distance between the features of graph1 and graph2
        """
        gofeature=True
        nodes1=graph1.nodes()
        nodes2=graph2.nodes()
        startstruct=time.time()
        C1=graph1.distance_matrix(method=self.method)
        C2=graph2.distance_matrix(method=self.method)
        
        C1 = (C1 - C1.min()) / (C1.max() - C1.min())
        C2 = (C2 - C2.min()) / (C2.max() - C2.min())
        
        end2=time.time()
       

        try :
            x1=self.reshaper(graph1.all_matrix_attr())
            x2=self.reshaper(graph2.all_matrix_attr())
            #print('grapha1.node_std',graph1.node_std)
            if graph1.node_std is not None:
                #print('here')
                x1/=graph1.node_std
                x2/=graph2.node_std
            self.x1=x1
            self.x2=x2
            
        except:
            gofeature=False
        #print('gofeature',gofeature)
        if gofeature : 
            if self.features_metric=='dirac':
                f=lambda x,y: x!=y
                M=ot.dist(x1,x2,metric=f)
            elif self.features_metric=='hamming_dist': #see experimental setup in the original paper
                f=lambda x,y: hamming_dist(x,y)
                M=ot.dist(x1,x2,metric=f)
            else:
                M=ot.dist(x1,x2,metric=self.features_metric)
            self.M=M
        else:
            M=np.zeros((C1.shape[0],C2.shape[0]))
        
        M = (M - M.min()) / (M.max() - M.min())
        self.M=M
        self.C1=C1
        self.C2=C2
            
        startdist=time.time()
        if self.measure in ['fgw']:
            t1masses,t2masses=np.ones(len(nodes1))/len(nodes1),np.ones(len(nodes2))/len(nodes2)
            transpwgw,log=self.calc_fgw(M,C1,C2,t1masses,t2masses)
            enddist=time.time()
            log['struct_time']=(end2-startstruct)
            log['dist_time']=(enddist-startdist)
            self.transp=transpwgw
            self.log=log
            return log['loss'][::-1][0]
        elif self.measure in ['fpgw','fmpgw','fugw', 'sink-fpgw']:
            N1,N2=graph1.N,graph2.N
            
            if N1 is None or N2 is None:
                print('error in data loading')
            
            t1masses = np.ones(len(nodes1))/N1
            t2masses = np.ones(len(nodes2))/N2
                        
            gamma,cost=self.calc_fgw(M,C1,C2,t1masses,t2masses)
            self.transp=gamma
            print(cost)
            return cost
        
        elif self.measure in ['rgw']:
            t1masses,t2masses=np.ones([len(nodes1),1])/len(nodes1),np.ones([len(nodes2),1])/len(nodes2)
            transpwgw,cost=self.calc_fgw(M,C1,C2,t1masses,t2masses)
            enddist=time.time()
            self.transp=transpwgw
            return cost

     def compute_all_distance(self, X, Y, matrix=None): 
        """ 
        Compute all similarities f(x, y) for x, y in X and Y and f the similarity measure.
        
        Parameters
        ----------
        X : array-like
            Array of abstract objects.
        Y : array-like
            Array of abstract objects.
        matrix : ndarray, optional
            If specified, used to compute the similarity matrix instead of calculating all the similarities.
        
        Returns
        -------
        None
            Sets the similarity matrix to self.D.
        """
        if matrix is not None:
            self.D = matrix
        else:
            # Ensure X and Y are 1D arrays
            X = X.reshape(X.shape[0],)
            Y = Y.reshape(Y.shape[0],)
        
            if X.shape[0] == Y.shape[0] and np.all(X == Y):
                # Symmetric case: compute upper triangle only and mirror

                # --- Setup ---
                
                pairs = [(i, j, X[i], Y[j]) for i in range(len(X)) for j in range(i + 1, len(Y))]
                            
                batch_size = 1
                pattern_batches = [pairs[i:i+batch_size]
                           for i in range(0, len(pairs), batch_size)]

                D = np.zeros((X.shape[0], Y.shape[0]))
                results_buffer = []
                
                
                # for i, x1 in enumerate(tqdm(X, desc="Processing")):
                #     for j, x2 in enumerate(Y):
                #         if j >= i + 1:  # Compute only upper triangle
                #             dist = self.graph_d(x1, x2)
                #             D[i, j] = dist
                #             print(i,j,dist)
                            
                pairs_selected = pairs[args.start : args.end]
                for i,j, x1,x2 in tqdm(pairs_selected, desc="Processing"):
                    dist = self.graph_d(x1, x2)
                    #D[i, j] = dist
                    results_buffer.append((i,j,dist))
                    
                with open(f'crc_bary_region_res/results_buffer_{args.start}.pkl', 'wb') as f:
                    pickle.dump(results_buffer, f) 

            else:
                # Non-symmetric case: compute full matrix
                D = np.zeros((X.shape[0], Y.shape[0]))
                for i, x1 in enumerate(tqdm(X, desc="Processing")): #for i, x1 in enumerate(X):
                    row = [self.graph_d(x1, x2) for x2 in Y]
                    D[i, :] = row
        
            # Set very small values (numerical noise) to zero
            D[np.abs(D) <= 1e-15] = 0
        
            # Assign the computed distance matrix to self.D
            self.dist_matrix = D

def load_CRC_bary(path):
    FGW_graph_dict = {(i+1):None for i in range(70)}
    FGW_graph_list = []
    for file in os.listdir(path):
        
        if 'reg' not in file:
            continue
     
        num = int(re.search(r'reg(\d+)', file).group(1))
        
        print(file)
        with open(f'{path}/{file}','rb') as f:
            data = pickle.load(f)
                    
        g = Graph()
        g.C = data['structure']
        g.N = g.C.shape[0]
        
        X_sub = data['feature']
        X_sub_dict = {i:v for i,v in enumerate(X_sub)}
        g.add_attibutes(X_sub_dict)
        
        FGW_graph_dict[num] = g
            
    FGW_graph_list = list(FGW_graph_dict.values())
       
    return FGW_graph_list
        
method='fpgw'
alpha=0.5
feature_metric ='sqeuclidean'
FPGW=Fused_Partial_Gromov_Wasserstein_distance(alpha=alpha,method='shortest_path',features_metric=feature_metric,measure=method,mass=1-1e-15,amijo=False,Lambda=10.0)


#load CRC data
df = pd.read_csv('data/CRC_clusters_neighborhoods_markers.csv')
X = load_CRC_bary('bary')
X = np.array(X)
FPGW.compute_all_distance(X,X)
dist=FPGW.dist_matrix

