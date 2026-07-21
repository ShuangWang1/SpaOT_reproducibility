
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile
import pickle

# -----------------------------
# LOAD VISIUM
# -----------------------------
adata_vis_all = sc.read_h5ad("outs_04/segmented_outputs/seg.h5ad")
img_vis = np.array(Image.open("outs_04/spatial/tissue_hires_image.png"))
vis_scale = 0.27254146

# -----------------------------
# LOAD XENIUM
# -----------------------------
adata_xen_all = sc.read_h5ad("Xenium_adata/IU04.h5ad")
img_xen_full = np.array(tifffile.imread("Xenium/HE_images/0015383__IU04.TIF"))

# -----------------------------
# PREPARE XENIUM COORDINATES (aligned like your code)
# -----------------------------
img_h, img_w = img_xen_full.shape[:2]

x = adata_xen_all.obsm["spatial"][:, 0]
y = adata_xen_all.obsm["spatial"][:, 1]

x = x - x.min()
y = y - y.min()

scale_x = img_w / max(x)
scale_y = img_h / max(y)

x = x * scale_x
y = y * scale_y

x = img_w - x

#for sample 0
x -= 250
x -= 10
y -= 10
y -= 100
y-= 50

#for sample 1
x -= 100
y += 50

xen_x = x
xen_y = y

adata_xen_all.obsm['spatial'][:,0] = xen_x
adata_xen_all.obsm['spatial'][:,1] = xen_y


adata_vis = adata_vis_all[adata_vis_all.obs["spatial_region"] == 0].copy()
adata_xen = adata_xen_all[adata_xen_all.obs["spatial_region"] == 2].copy()

# -----------------------------
# LOAD TRANSPORT PLAN
# -----------------------------
with open('xen_hd_map/2_spaot_seg.pickle', 'rb') as f:
    G0 = pickle.load(f)

T = G0

random_numbers = [1860]
for random_number in random_numbers:
    xen_mask = np.zeros(adata_xen.n_obs, dtype=bool)
    lst = [i for i,j in enumerate(G0[:,random_number]) if j >= 1e-10]
    xen_mask[lst] = True
    vis_mask = np.zeros(adata_vis.n_obs, dtype=bool)
    vis_mask[[random_number]] = True

    xen_idx = np.where(xen_mask)[0]
    vis_idx = np.where(vis_mask)[0]
    xen_mass = T[:, vis_idx].sum(axis=1)

    xen_mapped_idx = np.where(xen_mass > 1e-10)[0]

    x_map = adata_xen.obsm['spatial'][xen_mapped_idx,0]
    y_map = adata_xen.obsm['spatial'][xen_mapped_idx,1]

    pad = 2000

    xmin2 = int(max(x_map.min() - pad-600, 0))
    xmax2 = int(min(x_map.max() + pad-600, img_w))
    ymin2 = int(max(y_map.min() - pad-200, 0))
    ymax2 = int(min(y_map.max() + pad-200, img_h))

    img_xen_crop2 = img_xen_full[ymin2:ymax2, xmin2:xmax2]

    xen_x_crop2 = x_map - xmin2
    xen_y_crop2 = y_map - ymin2

    # ---------------------------------
    # 3️⃣ Crop Visium to G3 (source region)
    # ---------------------------------
    vis_coords = adata_vis.obsm["spatial"] * vis_scale

    vis_x_g3 = vis_coords[vis_idx, 0]
    vis_y_g3 = vis_coords[vis_idx, 1]

    pad_vis = 600

    xmin_v2 = int(max(vis_x_g3.min() - pad_vis, 0))
    xmax_v2 = int(min(vis_x_g3.max() + pad_vis, img_vis.shape[1]))
    ymin_v2 = int(max(vis_y_g3.min() - pad_vis-300, 0))
    ymax_v2 = int(min(vis_y_g3.max() + pad_vis-300, img_vis.shape[0]))

    img_vis_crop2 = img_vis[ymin_v2:ymax_v2, xmin_v2:xmax_v2]

    vis_x_crop2 = vis_x_g3 - xmin_v2
    vis_y_crop2 = vis_y_g3 - ymin_v2


    xen_h2, xen_w2 = img_xen_crop2.shape[:2]
    vis_h2, vis_w2 = img_vis_crop2.shape[:2]

    scale_factor2 = xen_h2 / vis_h2
    new_vis_w2 = int(vis_w2 * scale_factor2)

    img_vis_resized2 = np.array(
        Image.fromarray(img_vis_crop2).resize((new_vis_w2, xen_h2))
    )

    vis_x_rescaled2 = vis_x_crop2 * scale_factor2
    vis_y_rescaled2 = vis_y_crop2 * scale_factor2

    gap = 200
    vis_x_shifted2 = vis_x_rescaled2 + xen_w2 + gap

    # store original dimensions
    vis_h2, vis_w2 = img_vis_resized2.shape[:2]

    # rotate image
    img_vis_resized2 = np.rot90(img_vis_resized2, k=1)

    # rotate coordinates (90° CCW)
    # remove gap first
    vis_x_no_shift = vis_x_shifted2 - (xen_w2 + gap)

    vis_x_rot = vis_y_rescaled2
    vis_y_rot = vis_w2 - vis_x_no_shift

    # update width/height after rotation
    new_vis_w2_rot = vis_h2
    new_vis_h2_rot = vis_w2

    # re-apply gap shift
    vis_x_shifted2 = vis_x_rot + xen_w2 + gap
    vis_y_rescaled2 = vis_y_rot




    # Visium coords (no gap shift — already 0-based after rotation)                                                                                                                                            
    vis_x_plot = vis_x_rot                                                                                                                                                                                     
    vis_y_plot = vis_y_rot                                                                                                                                                                                     
                                                                                                                                                                                                                
    # Xenium coords shifted to the right of Visium                                                                                                                                                             
    xen_x_plot = xen_x_crop2 + new_vis_w2_rot + gap
    xen_y_plot = xen_y_crop2                                                                                                                                                                                   
                                                                                                                                                                                                                
    fig, ax = plt.subplots(figsize=(14, 8))
                                                                                                                                                                                                                
    # Images — Visium left, Xenium right                      
    ax.imshow(
        img_vis_resized2,                                                                                                                                                                                      
        extent=[0, new_vis_w2_rot, new_vis_h2_rot, 0]
    )                                                                                                                                                                                                          
    ax.imshow(                                                
        img_xen_crop2,
        extent=[new_vis_w2_rot + gap, new_vis_w2_rot + gap + xen_w2, xen_h2, 0]                                                                                                                                
    )                                                                                                                                                                                                          
                                                                                                                                                                                                                               
    # # Scatter                                                                                                                                                                                                  
    # ax.scatter(vis_x_plot, vis_y_plot, s=30, c="blue")         
    # ax.scatter(xen_x_plot, xen_y_plot, s=30, c="red")
                                                                                                                                                                                                                
    # Lines
    Tmax = T[:,vis_idx].max()
    for i, xi in enumerate(xen_mapped_idx):                                                                                                                                                                    
        for j, vj in enumerate(vis_idx):                      
            w = T[xi, vj]
            if w >= 1e-10 and w/Tmax > 0.1:
                                                                                                                                                                                          
                ax.scatter(vis_x_plot[j], vis_y_plot[j], s=80, c="blue")         
                ax.scatter(xen_x_plot[i], xen_y_plot[i], s=30, c="red")
                
                ax.plot(                                                                                                                                                                                       
                    [vis_x_plot[j], xen_x_plot[i]],
                    [vis_y_plot[j], xen_y_plot[i]],                                                                                                                                                            
                    linewidth=3,                            
                    alpha=w/Tmax,                                                                                                                                                                                 
                    color="blue"
                )                                                                                                                                                                                              
                                                            
    ax.set_xlim(0, new_vis_w2_rot + gap + xen_w2)
    ax.set_ylim(new_vis_h2_rot if new_vis_h2_rot > xen_h2 else xen_h2, 0)
                                                                                                                                                                                                                
    ax.axis("off")
                                                                                                                                                          
    plt.savefig(f"p_demo_{random_number}_fpgw!.png", dpi=300)                                                                                                                                          
    plt.close()
    

with open('xen_hd_map/2_spaot_seg.pickle', 'rb') as f:
    mos = pickle.load(f)

T = np.array(mos.transport_matrix)

random_numbers = [1860]
for random_number in random_numbers:
    xen_mask = np.zeros(adata_xen.n_obs, dtype=bool)
    lst = [i for i,j in enumerate(G0[:,random_number]) if j >= 1e-10]
    xen_mask[lst] = True
    vis_mask = np.zeros(adata_vis.n_obs, dtype=bool)
    vis_mask[[random_number]] = True

    xen_idx = np.where(xen_mask)[0]
    vis_idx = np.where(vis_mask)[0]
    xen_mass = T[:, vis_idx].sum(axis=1)

    xen_mapped_idx = np.where(xen_mass > 1e-10)[0]

    x_map = adata_xen.obsm['spatial'][xen_mapped_idx,0]
    y_map = adata_xen.obsm['spatial'][xen_mapped_idx,1]

    pad = 1000

    xmin2 = int(max(x_map.min() - pad+400, 0))
    xmax2 = int(min(x_map.max() + pad-400, img_w))
    ymin2 = int(max(y_map.min() - pad-200, 0))
    ymax2 = int(min(y_map.max() + pad, img_h))

    img_xen_crop2 = img_xen_full[ymin2:ymax2, xmin2:xmax2]

    xen_x_crop2 = x_map - xmin2
    xen_y_crop2 = y_map - ymin2

    # ---------------------------------
    # 3️⃣ Crop Visium to G3 (source region)
    # ---------------------------------
    vis_coords = adata_vis.obsm["spatial"] * vis_scale

    vis_x_g3 = vis_coords[vis_idx, 0]
    vis_y_g3 = vis_coords[vis_idx, 1]

    pad_vis = 600

    xmin_v2 = int(max(vis_x_g3.min() - pad_vis, 0))
    xmax_v2 = int(min(vis_x_g3.max() + pad_vis, img_vis.shape[1]))
    ymin_v2 = int(max(vis_y_g3.min() - pad_vis-300, 0))
    ymax_v2 = int(min(vis_y_g3.max() + pad_vis-300, img_vis.shape[0]))

    img_vis_crop2 = img_vis[ymin_v2:ymax_v2, xmin_v2:xmax_v2]

    vis_x_crop2 = vis_x_g3 - xmin_v2
    vis_y_crop2 = vis_y_g3 - ymin_v2


    xen_h2, xen_w2 = img_xen_crop2.shape[:2]
    vis_h2, vis_w2 = img_vis_crop2.shape[:2]

    scale_factor2 = xen_h2 / vis_h2
    new_vis_w2 = int(vis_w2 * scale_factor2)

    img_vis_resized2 = np.array(
        Image.fromarray(img_vis_crop2).resize((new_vis_w2, xen_h2))
    )

    vis_x_rescaled2 = vis_x_crop2 * scale_factor2
    vis_y_rescaled2 = vis_y_crop2 * scale_factor2

    gap = 200
    vis_x_shifted2 = vis_x_rescaled2 + xen_w2 + gap

    # store original dimensions
    vis_h2, vis_w2 = img_vis_resized2.shape[:2]

    # rotate image
    img_vis_resized2 = np.rot90(img_vis_resized2, k=1)

    # rotate coordinates (90° CCW)
    # remove gap first
    vis_x_no_shift = vis_x_shifted2 - (xen_w2 + gap)

    vis_x_rot = vis_y_rescaled2
    vis_y_rot = vis_w2 - vis_x_no_shift

    # update width/height after rotation
    new_vis_w2_rot = vis_h2
    new_vis_h2_rot = vis_w2

    # re-apply gap shift
    vis_x_shifted2 = vis_x_rot + xen_w2 + gap
    vis_y_rescaled2 = vis_y_rot




    # Visium coords (no gap shift — already 0-based after rotation)                                                                                                                                            
    vis_x_plot = vis_x_rot                                                                                                                                                                                     
    vis_y_plot = vis_y_rot                                                                                                                                                                                     
                                                                                                                                                                                                                
    # Xenium coords shifted to the right of Visium                                                                                                                                                             
    xen_x_plot = xen_x_crop2 + new_vis_w2_rot + gap
    xen_y_plot = xen_y_crop2                                                                                                                                                                                   
                                                                                                                                                                                                                
    fig, ax = plt.subplots(figsize=(14, 8))
                                                                                                                                                                                                                
    # Images — Visium left, Xenium right                      
    ax.imshow(
        img_vis_resized2,                                                                                                                                                                                      
        extent=[0, new_vis_w2_rot, new_vis_h2_rot, 0]
    )                                                                                                                                                                                                          
    ax.imshow(                                                
        img_xen_crop2,
        extent=[new_vis_w2_rot + gap, new_vis_w2_rot + gap + xen_w2, xen_h2, 0]                                                                                                                                
    )                                                                                                                                                                                                          
                                                                                                                                                                                                                               
    # # Scatter                                                                                                                                                                                                  
    # ax.scatter(vis_x_plot, vis_y_plot, s=30, c="blue")         
    # ax.scatter(xen_x_plot, xen_y_plot, s=30, c="red")
                                                                                                                                                                                                                
    # Lines
    Tmax = T[:,vis_idx].max()
    for i, xi in enumerate(xen_mapped_idx):                                                                                                                                                                    
        for j, vj in enumerate(vis_idx):                      
            w = T[xi, vj]
            if w >= 1e-10 and w/Tmax > 0.1:
                                                                                                                                                                                          
                ax.scatter(vis_x_plot[j], vis_y_plot[j], s=80, c="blue")         
                ax.scatter(xen_x_plot[i], xen_y_plot[i], s=30, c="red")
                
                ax.plot(                                                                                                                                                                                       
                    [vis_x_plot[j], xen_x_plot[i]],
                    [vis_y_plot[j], xen_y_plot[i]],                                                                                                                                                            
                    linewidth=3,                            
                    alpha=w/Tmax,                                                                                                                                                                                 
                    color="blue"
                )                                                                                                                                                                                              
                                                            
    ax.set_xlim(0, new_vis_w2_rot + gap + xen_w2)
    ax.set_ylim(new_vis_h2_rot if new_vis_h2_rot > xen_h2 else xen_h2, 0)
                                                                                                                                                                                                                
    ax.axis("off")
                                                                                                                                                          
    plt.savefig(f"p_demo_{random_number}_moscot!.png", dpi=300)                                                                                                                                          
    plt.close()