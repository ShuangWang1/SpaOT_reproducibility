import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


def to_rgb_tensor(img):
    """
    Convert image to torch tensor [3, H, W] in [0, 1].
    Accepts numpy or torch, grayscale or RGB.
    """
    if isinstance(img, np.ndarray):
        x = torch.from_numpy(img)
    else:
        x = img.detach().clone()

    if x.ndim == 2:
        x = x.unsqueeze(-1)

    if x.ndim == 3 and x.shape[-1] in (1, 3, 4):
        if x.shape[-1] == 4:
            x = x[..., :3]
        if x.shape[-1] == 1:
            x = x.repeat(1, 1, 3)
        x = x.permute(2, 0, 1).contiguous()
    elif x.ndim == 3 and x.shape[0] in (1, 3, 4):
        if x.shape[0] == 4:
            x = x[:3]
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
    else:
        raise ValueError(f"Unsupported image shape: {tuple(x.shape)}")

    x = x.float()
    if x.max() > 1.0:
        x = x / 255.0
    return x


def crop_square(img_chw, center_xy, size):
    """
    Crop a square patch centered at (x, y) from [3, H, W].
    Pads by replication if needed.
    Returns [3, size, size].
    """
    c, h, w = img_chw.shape
    cx, cy = float(center_xy[0]), float(center_xy[1])

    half = size // 2
    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = x1 + size
    y2 = y1 + size

    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, x2 - w)
    pad_b = max(0, y2 - h)

    if pad_l or pad_r or pad_t or pad_b:
        img_chw = F.pad(img_chw, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
        x1 += pad_l
        x2 += pad_l
        y1 += pad_t
        y2 += pad_t

    patch = img_chw[:, y1:y2, x1:x2]

    if patch.shape[1] != size or patch.shape[2] != size:
        patch = F.interpolate(
            patch.unsqueeze(0),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return patch


def patches_from_tile_centers(img, tile_centers, crop_size=256):
    """
    Build a list of HWC uint8 RGB patches for HF processor.
    """
    img_t = to_rgb_tensor(img)
    patches = []

    for center in tile_centers:
        patch = crop_square(img_t, center, crop_size)  # [3, S, S]
        patch = patch.clamp(0, 1)
        patch = (patch * 255.0).round().byte()
        patch = patch.permute(1, 2, 0).cpu().numpy()    # HWC uint8
        patches.append(patch)

    return patches


@torch.no_grad()
def extract_dinov3_tile_embeddings(
    img,
    tile_centers,
    #model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
    model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
    crop_size=256,
    batch_size=32,
    use_patch_token_mean=False,
    device=None,
):
    """
    Return tile embeddings [N, D] using pretrained DINOv3.

    Args:
        img: HxW, HxWx3, or [3,H,W]
        tile_centers: array-like [N, 2] in (x, y) pixel coordinates
        crop_size: square crop size around each tile center, should be a multiple of 16
        use_patch_token_mean: if True, average patch tokens instead of pooler_output
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # DINOv3 model card shows AutoImageProcessor + AutoModel usage.
    # The official repo says released Transformers support starts at 4.56.0.
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    tile_centers = np.asarray(tile_centers)
    patches = patches_from_tile_centers(img, tile_centers, crop_size=crop_size)

    all_embs = []
    for start in range(0, len(patches), batch_size):
        batch_patches = patches[start:start + batch_size]
        inputs = processor(images=batch_patches, return_tensors="pt").to(device)

        outputs = model(**inputs)

        if use_patch_token_mean:
            # outputs.last_hidden_state: [B, 1 + num_patches, D]
            # drop CLS token and mean-pool patch tokens
            emb = outputs.last_hidden_state[:, 1:, :].mean(dim=1)
        else:
            # recommended default from the model card example
            emb = outputs.pooler_output

        all_embs.append(emb.detach().cpu())

    return torch.cat(all_embs, dim=0)


import scanpy as sc
import pickle


adata_seq = sc.read_h5ad("cerebellum/Cerebellum-MAGIC-seq.h5ad")
clusters   = adata_seq.obs['cluster'].astype(str).values
seq_coords = adata_seq.obsm['spatial'].astype(float)

library_id = list(adata_seq.uns["spatial"].keys())[0]
img        = adata_seq.uns["spatial"][library_id]["images"]["hires"]
with open("cerebellum_image2seq_from_MAGIC/tile_centers.pickle", "rb") as f: 
    tile_centers = pickle.load(f)

embeddings = extract_dinov3_tile_embeddings(
    img=img,
    tile_centers=tile_centers,
    model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
    crop_size=256,
    batch_size=32,
    use_patch_token_mean=False,
)
print(embeddings.shape)  # [N, D]

        
with open("cerebellum_image2seq_from_MAGIC/tile_embeddings_dino.pickle", "wb") as f:
    pickle.dump(embeddings.cpu().numpy(), f)