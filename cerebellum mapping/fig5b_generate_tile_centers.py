"""
Generate tile_centers.pickle for the cerebellum image2seq benchmark.

Tiles a regular 50-pixel-stride grid over the region spanned by the
sequencing spot coordinates (rounded outward to the nearest grid point),
then filters to tiles where at least 40% of the 50×50 check-patch is tissue.

"Tissue" = pixels with mean(R,G,B) < 220  (background H&E is near-white ≥ 220).

Grid parameters (recovered from the tile_centers.pickle data pattern):
  - STRIDE = 50 px  (consecutive x/y values differ by 50)
  - Grid starts at STRIDE//2 = 25 px  (25, 75, 125, …)
  - Restricted to the bounding box of seq spots, rounded to nearest grid point:
      x: round(seq_x_min/50)*50+25  to  round(seq_x_max/50)*50+25
      y: floor(seq_y_min/50)*50+25  to  floor(seq_y_max/50)*50+25
    → for Cerebellum-MAGIC-seq.h5ad: x=[1575, 3675], y=[1375, 3425]

Output: cerebellum_image2seq_from_MAGIC/tile_centers.pickle
  np.ndarray shape (1251, 2), dtype float32, columns = (col/x, row/y)
  in the same hires-pixel coordinate system as adata_seq.obsm['spatial'].
"""

import os
import pickle
import numpy as np
import scanpy as sc

os.makedirs("cerebellum_image2seq_from_MAGIC", exist_ok=True)

# ── Load H&E image ─────────────────────────────────────────────────────────────
adata_seq = sc.read_h5ad("cerebellum/Cerebellum-MAGIC-seq.h5ad")
library_id = list(adata_seq.uns['spatial'].keys())[0]
img = adata_seq.uns['spatial'][library_id]['images']['hires']   # (H, W, 3) uint8
H, W = img.shape[:2]

# ── Derive grid bounds from seq spot bounding box ──────────────────────────────
seq_coords = adata_seq.obsm['spatial'].astype(float)   # (n, 2)  col, row in hires px

STRIDE     = 50
HALF       = STRIDE // 2    # = 25

# Round to nearest grid point for x; floor for y (matches empirical data pattern)
x0 = int(round(seq_coords[:, 0].min() / STRIDE)) * STRIDE + HALF   # 1575
x1 = int(round(seq_coords[:, 0].max() / STRIDE)) * STRIDE + HALF   # 3675
y0 = int(np.floor(seq_coords[:, 1].min() / STRIDE)) * STRIDE + HALF   # 1375
y1 = int(np.floor(seq_coords[:, 1].max() / STRIDE)) * STRIDE + HALF   # 3425

xs = np.arange(x0, x1 + 1, STRIDE)   # 43 values
ys = np.arange(y0, y1 + 1, STRIDE)   # 42 values  →  1806 candidate tiles

# ── Tissue mask ────────────────────────────────────────────────────────────────
CHECK_HALF    = HALF           # 25-px neighbourhood around each centre
TISSUE_THRESH = 220            # pixels with mean(R,G,B) < 220 are tissue (uint8)
MIN_TISSUE_FRAC = 0.40         # at least 40 % of check-patch must be tissue

centers = []
for y in ys:
    for x in xs:
        r0, r1 = max(0, y - CHECK_HALF), min(H, y + CHECK_HALF)
        c0, c1 = max(0, x - CHECK_HALF), min(W, x + CHECK_HALF)
        patch   = img[r0:r1, c0:c1].astype(np.float32)   # (≤50, ≤50, 3)
        tissue_frac = np.mean(patch.mean(axis=2) < TISSUE_THRESH)
        if tissue_frac >= MIN_TISSUE_FRAC:
            centers.append([float(x), float(y)])

tile_centers = np.array(centers, dtype=np.float32)   # (n, 2)

print(f"Generated {len(tile_centers)} tile centres  (expected 1251)")
print(f"  x range: {tile_centers[:,0].min():.0f} – {tile_centers[:,0].max():.0f}  (expected 1575 – 3675)")
print(f"  y range: {tile_centers[:,1].min():.0f} – {tile_centers[:,1].max():.0f}  (expected 1375 – 3425)")

out_path = "cerebellum_image2seq_from_MAGIC/tile_centers.pickle"
with open(out_path, "wb") as f:
    pickle.dump(tile_centers, f)
print(f"Saved → {out_path}")
