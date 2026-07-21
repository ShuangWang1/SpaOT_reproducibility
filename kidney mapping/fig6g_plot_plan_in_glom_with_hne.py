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
# LOAD TRANSPORT PLAN
# -----------------------------
with open('xen_hd_map/1_spaot_seg.pickle', 'rb') as f:
    G0 = pickle.load(f)

T = G0

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


adata_vis = adata_vis_all[adata_vis_all.obs["spatial_region"] == 2].copy()
adata_xen = adata_xen_all[adata_xen_all.obs["spatial_region"] == 1].copy()


# -----------------------------
# REGION MASKS
# -----------------------------
xen_mask = adata_xen.obs['glomerulus'] == 'Selection 14'
vis_mask = adata_vis.obs["glomerulus"] == "G14"

xen_idx = np.where(xen_mask)[0]
vis_idx = np.where(vis_mask)[0]



# =====================================================
# PLOT 2: VISIUM → XENIUM  
# =====================================================

# ---------------------------------
# 1️⃣ Compute Xenium mapped from Visium G14
# ---------------------------------

def max_sum_projection(arr, axis='row'):
    arr = np.asarray(arr)
    out = np.zeros_like(arr)

    if axis == 'row':
        # sum of each row
        sums = arr.sum(axis=1)
        # index of max element in each row
        idx = arr.argmax(axis=1)
        # assign row sum to the max column
        out[np.arange(arr.shape[0]), idx] = sums

    elif axis == 'col':
        # sum of each column
        sums = arr.sum(axis=0)
        # index of max element in each column
        idx = arr.argmax(axis=0)
        # assign col sum to the max row
        out[idx, np.arange(arr.shape[1])] = sums

    else:
        raise ValueError("axis must be 'row' or 'col'")

    return out

T_col_sum = max_sum_projection(T, axis='col')

xen_mass = T_col_sum[:, vis_idx].sum(axis=1)
xen_mapped_idx = np.where(xen_mass > 0)[0]

if len(xen_mapped_idx) == 0:
    raise ValueError("No Xenium cells receive mass from Visium G3")

def plot_14_region():
        
    xen_mass = T_col_sum[:, vis_idx].sum(axis=1)
    xen_mapped_idx = np.where(xen_mass > 0)[0]

    if len(xen_mapped_idx) == 0:
        raise ValueError("No Xenium cells receive mass from Visium G14")

    # ---------------------------------
    # 2️⃣ Crop Xenium based on mapped cells
    # ---------------------------------
    x_map = adata_xen.obsm['spatial'][xen_mapped_idx,0]
    y_map = adata_xen.obsm['spatial'][xen_mapped_idx,1]

    pad = 50

    xmin2 = int(max(x_map.min() - pad, 0))
    xmax2 = int(min(x_map.max() + pad, img_w))
    ymin2 = int(max(y_map.min() - 520, 0))
    ymax2 = int(min(y_map.max() + 250, img_h))

    img_xen_crop2 = img_xen_full[ymin2:ymax2, xmin2:xmax2]

    xen_x_crop2 = x_map - xmin2
    xen_y_crop2 = y_map - ymin2

    # ---------------------------------
    # 3️⃣ Crop Visium to G14 (source region)
    # ---------------------------------
    vis_coords = adata_vis.obsm["spatial"] * vis_scale

    vis_x_g3 = vis_coords[vis_idx, 0]
    vis_y_g3 = vis_coords[vis_idx, 1]

    pad_vis = 20

    xmin_v2 = int(max(vis_x_g3.min() - 220, 0))
    xmax_v2 = int(min(vis_x_g3.max() + 300, img_vis.shape[1]))
    ymin_v2 = int(max(vis_y_g3.min() - pad_vis, 0))
    ymax_v2 = int(min(vis_y_g3.max() + 310, img_vis.shape[0]))

    img_vis_crop2 = img_vis[ymin_v2:ymax_v2, xmin_v2:xmax_v2]

    vis_x_crop2 = vis_x_g3 - xmin_v2
    vis_y_crop2 = vis_y_g3 - ymin_v2

    # ---------------------------------
    # 4️⃣ Normalize heights
    # ---------------------------------
    xen_h2, xen_w2 = img_xen_crop2.shape[:2]
    vis_h2, vis_w2 = img_vis_crop2.shape[:2]

    scale_factor2 = xen_h2 / vis_w2
    new_vis_w2 = int(vis_w2 * scale_factor2)

    img_vis_resized2 = np.array(
        Image.fromarray(img_vis_crop2).resize((new_vis_w2, xen_h2))
    )

    vis_x_rescaled2 = vis_x_crop2 * scale_factor2
    vis_y_rescaled2 = vis_y_crop2 * scale_factor2

    gap = 200
    vis_x_shifted2 = vis_x_rescaled2 + xen_w2 + gap

    # ---------------------------------
    # 5️⃣ Draw transport lines
    # ---------------------------------

    # ---------------------------------
    # Rotate Visium 90° CCW (at last step)
    # ---------------------------------

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
        extent=[0, new_vis_w2_rot*0.85, xen_h2, 0]
    )                                                                                                                                                                                                          
    ax.imshow(                                                
        img_xen_crop2,
        extent=[new_vis_w2_rot + gap, new_vis_w2_rot + gap + xen_w2, xen_h2, 0]                                                                                                                                
    ) 
    vis_x_plot =1.13*     vis_x_plot                                                                                                                                                                                                    
    ax.scatter(vis_x_plot, vis_y_plot, s=3, c="blue")         
    ax.scatter(xen_x_plot, xen_y_plot, s=3, c="red")

                                                                                                                                                                                                    
    # Lines
    for i, xi in enumerate(xen_mapped_idx):                                                                                                                                                                    
        for j, vj in enumerate(vis_idx):                      
            w = T_col_sum[xi, vj]
            if w > 0:
                ax.plot(                                                                                                                                                                                       
                    [vis_x_plot[j], xen_x_plot[i]],
                    [vis_y_plot[j], xen_y_plot[i]],                                                                                                                                                            
                    linewidth=0.4,                            
                    alpha=0.3,                                                                                                                                                                                 
                    color="blue"
                )   
                                
    ax.set_xlim(0, new_vis_w2_rot + gap + xen_w2)
    ax.set_ylim(new_vis_h2_rot if new_vis_h2_rot > xen_h2 else xen_h2, 0)
                                                                                                                                                                                                                
    ax.axis("off")
    plt.title("Region 14: Visium → Xenium Mapping")                                                                                                                                                            
    plt.savefig("Region14_Visium_to_Xenium_IMG_LINES_new_demo.png", dpi=300)                                                                                                                                          
    plt.close()

plot_14_region()