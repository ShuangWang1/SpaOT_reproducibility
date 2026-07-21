import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import pickle

adata_hd = sc.read_h5ad('outs_04/segmented_outputs/seg.h5ad')

adata_xen = sc.read_h5ad('Xenium_adata/IU04.h5ad')

import scipy.sparse as sp

def compute_glomeruli_mapping_rate(
    source_adata,
    target_adata,
    transport_plan,
    source_key="glomerulus",
    target_key="glomerulus",
    ignore_label="None"
):
    """
    Compute glomeruli mapping rate from source->target and target->source.

    Parameters
    ----------
    source_adata : AnnData
    target_adata : AnnData
    transport_plan : np.ndarray
        Transport matrix of shape (n_source, n_target)
    source_key : str
        Column name in source_adata.obs
    target_key : str
        Column name in target_adata.obs
    ignore_label : str
        Label to ignore (e.g., 'None')

    Returns
    -------
    dict with:
        - source_to_target_rate (per glomerulus)
        - target_to_source_rate (per glomerulus)
        - global_source_to_target_rate
        - global_target_to_source_rate
    """

    # --------- helper: normalize labels ----------
    def normalize_source_label(label):
        if label == ignore_label:
            return None
        match = re.search(r"Selection\s*(\d+)", str(label))
        return f"G{match.group(1)}" if match else None

    def normalize_target_label(label):
        if label == ignore_label:
            return None
        match = re.search(r"G\s*(\d+)", str(label))
        return f"G{match.group(1)}" if match else None

    # Normalize labels
    source_labels = source_adata.obs[source_key].apply(normalize_source_label)
    target_labels = target_adata.obs[target_key].apply(normalize_target_label)

    T = np.asarray(transport_plan)

    # Mask valid cells (exclude None)
    source_valid = source_labels.notna().values
    target_valid = target_labels.notna().values

    source_labels = source_labels.values
    target_labels = target_labels.values

    # Unique glomeruli
    glomeruli = sorted(set(source_labels[source_valid]))

    source_to_target = {}
    target_to_source = {}

    # ---------- Source -> Target ----------
    for g in glomeruli:
        src_idx = np.where(source_labels == g)[0]
        tgt_idx = np.where(target_labels == g)[0]

        if len(src_idx) == 0:
            continue

        total_mass = T[src_idx, :].sum()
        correct_mass = T[np.ix_(src_idx, tgt_idx)].sum()

        rate = correct_mass / total_mass if total_mass > 0 else np.nan
        source_to_target[g] = rate

    # ---------- Target -> Source ----------
    for g in glomeruli:
        src_idx = np.where(source_labels == g)[0]
        tgt_idx = np.where(target_labels == g)[0]

        if len(tgt_idx) == 0:
            continue

        total_mass = T[:, tgt_idx].sum()
        correct_mass = T[np.ix_(src_idx, tgt_idx)].sum()

        rate = correct_mass / total_mass if total_mass > 0 else np.nan
        target_to_source[g] = rate

    # ---------- Global scores ----------
    valid_source_idx = np.where(source_valid)[0]
    valid_target_idx = np.where(target_valid)[0]

    total_mass_global = T[np.ix_(valid_source_idx, valid_target_idx)].sum()

    correct_mass_global = 0
    for g in glomeruli:
        src_idx = np.where(source_labels == g)[0]
        tgt_idx = np.where(target_labels == g)[0]
        correct_mass_global += T[np.ix_(src_idx, tgt_idx)].sum()

    global_s2t = correct_mass_global / T[valid_source_idx, :].sum()
    global_t2s = correct_mass_global / T[:, valid_target_idx].sum()

    return {
        "source_to_target_rate": source_to_target,
        "target_to_source_rate": target_to_source,
        "global_source_to_target_rate": global_s2t,
        "global_target_to_source_rate": global_t2s
    }

def plot_glomeruli_mapping_scatter(
    our_results,
    competitor_results,
    our_label="Our Method",
    competitor_label="Competitor"
):
    """
    Scatter plot of source→target vs target→source mapping rates
    for two methods.
    """

    # Convert dictionaries to DataFrame
    def dict_to_df(results, label):
        s2t = results["source_to_target_rate"]
        t2s = results["target_to_source_rate"]

        df = pd.DataFrame({
            "glomerulus": list(s2t.keys()),
            "source_to_target": list(s2t.values()),
            "target_to_source": [t2s[g] for g in s2t.keys()],
        })
        df["method"] = label
        return df

    df_ours = dict_to_df(our_results, our_label)
    df_comp = dict_to_df(competitor_results, competitor_label)

    df = pd.concat([df_ours, df_comp], ignore_index=True)

    # Plot
    plt.figure(figsize=(6, 6))

    for method, color in zip(
        df["method"].unique(),
        ["blue", "red"]
    ):
        subset = df[df["method"] == method]
        plt.scatter(
            subset["source_to_target"],
            subset["target_to_source"],
            label=method,
            alpha=0.8
        )

    # Diagonal reference line
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("Source → Target Mapping Rate")
    plt.ylabel("Target → Source Mapping Rate")
    plt.title("Glomeruli Mapping Accuracy")
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('glom_acc.png')


def plot_glomeruli_separate_regions(
    our_results,
    competitor_results,
    our_label="FPGW",
    competitor_label="moscot"
):
    s2t_ours = our_results["source_to_target_accuracy"]
    t2s_ours = our_results["target_to_source_accuracy"]

    s2t_comp = competitor_results["source_to_target_accuracy"]
    t2s_comp = competitor_results["target_to_source_accuracy"]

    glomeruli = sorted(s2t_ours.keys())

    n = len(glomeruli)
    ncols = 4
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = axes.flatten()

    for i, g in enumerate(glomeruli):
        ax = axes[i]

        # Ours
        ax.scatter(
            s2t_ours[g],
            t2s_ours[g],
            label=our_label
        )

        # Competitor
        ax.scatter(
            s2t_comp[g],
            t2s_comp[g],
            label=competitor_label
        )

        ax.set_title(g)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.plot([0,1],[0,1], linestyle="--")
        ax.set_xlabel("S→T")
        ax.set_ylabel("T→S")

    # Remove empty panels
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    plt.savefig('glom_acc_reg.png')


def plot_glomeruli_accuracy_difference(our_results, competitor_results, our_label="FPGW", competitor_label="moscot"):
    glomeruli = sorted(set(our_results["source_to_target_accuracy"]) & set(competitor_results["source_to_target_accuracy"]))

    diff_s2t = [our_results["source_to_target_accuracy"][g] - competitor_results["source_to_target_accuracy"][g] for g in glomeruli]
    diff_t2s = [our_results["target_to_source_accuracy"][g] - competitor_results["target_to_source_accuracy"][g] for g in glomeruli]

    plt.figure(figsize=(6, 6))
    plt.scatter(diff_s2t, diff_t2s, alpha=0.8)
    for g, xv, yv in zip(glomeruli, diff_s2t, diff_t2s):
        plt.annotate(g, (xv, yv), fontsize=7, ha="left", va="bottom")
    plt.axhline(0, color="gray", linestyle="--")
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel(f"Source→Target Difference ({our_label} − {competitor_label})")
    plt.ylabel(f"Target→Source Difference ({our_label} − {competitor_label})")
    plt.title(f"{our_label} vs {competitor_label} per Glomerulus\n(1st quadrant: {our_label} better, 3rd: {competitor_label} better)")
    plt.tight_layout()
    plt.savefig("glom_acc_difference.png", dpi=150)


def compute_glomeruli_argmax_mapping_rate(
    source_adata,
    target_adata,
    transport_plan,
    source_key="glomerulus",
    target_key="glomerulus",
    ignore_label="None"
):
    """
    Compute hard (argmax-based) glomeruli mapping accuracy
    from source→target and target→source.

    Returns:
        dict with:
            - source_to_target_accuracy (per glomerulus)
            - target_to_source_accuracy (per glomerulus)
            - global_source_to_target_accuracy
            - global_target_to_source_accuracy
    """

    # --------- label normalization ----------
    def normalize_source_label(label):
        if label == ignore_label:
            return None
        match = re.search(r"Selection\s*(\d+)", str(label))
        return f"G{match.group(1)}" if match else None

    def normalize_target_label(label):
        if label == ignore_label:
            return None
        match = re.search(r"G\s*(\d+)", str(label))
        return f"G{match.group(1)}" if match else None

    source_labels = source_adata.obs[source_key].apply(normalize_source_label).values
    target_labels = target_adata.obs[target_key].apply(normalize_target_label).values

    T = np.asarray(transport_plan)

    # Hard assignments
    source_to_target_match = T.argmax(axis=1)  # each source → best target
    target_to_source_match = T.argmax(axis=0)  # each target → best source

    glomeruli = sorted(set(source_labels) - {None})

    s2t_acc = {}
    t2s_acc = {}

    # -------- Source → Target accuracy --------
    for g in glomeruli:
        src_idx = np.where(source_labels == g)[0]
        if len(src_idx) == 0:
            continue

        matched_targets = source_to_target_match[src_idx]
        matched_labels = target_labels[matched_targets]

        correct = np.sum(matched_labels == g)
        s2t_acc[g] = correct / len(src_idx)

    # -------- Target → Source accuracy --------
    for g in glomeruli:
        tgt_idx = np.where(target_labels == g)[0]
        if len(tgt_idx) == 0:
            continue

        matched_sources = target_to_source_match[tgt_idx]
        matched_labels = source_labels[matched_sources]

        correct = np.sum(matched_labels == g)
        t2s_acc[g] = correct / len(tgt_idx)

    # -------- Global accuracy --------
    valid_source = np.where(source_labels != None)[0]
    valid_target = np.where(target_labels != None)[0]

    global_s2t = np.mean(
        target_labels[source_to_target_match[valid_source]] == source_labels[valid_source]
    )

    global_t2s = np.mean(
        source_labels[target_to_source_match[valid_target]] == target_labels[valid_target]
    )

    return {
        "source_to_target_accuracy": s2t_acc,
        "target_to_source_accuracy": t2s_acc,
        "global_source_to_target_accuracy": global_s2t,
        "global_target_to_source_accuracy": global_t2s
    }


# glom for seg
region_dict = {0:3,1:2,2:0,3:1}

for region in [0,1,2,3]:
    
    with open(f'xen_hd_map/{region}_moscot_seg.pickle','rb') as f:
        mos = pickle.load(f)
    G_mos = np.array(mos.transport_matrix)
    
    
    with open(f'xen_hd_map/{region}_spaot_seg_normalized.pickle','rb') as f:
        G0 = pickle.load(f)
    
    region_src = region_dict[region]    
    adata_src = adata_hd[adata_hd.obs["spatial_region"] == region_src].copy()
    adata_tgt = adata_xen[adata_xen.obs["spatial_region"] == region].copy()
    
        
    res=compute_glomeruli_mapping_rate(
            adata_tgt,
            adata_src,
            G0,
            source_key="glomerulus",
            target_key="glomerulus",
            ignore_label="None"
        ) 
    
    res_mos=compute_glomeruli_mapping_rate(
            adata_tgt,
            adata_src,
            G_mos,
            source_key="glomerulus",
            target_key="glomerulus",
            ignore_label="None"
        ) 
    
    print(res)
    print(res_mos)
    
    
    
    with open(f'glom_res/fpgw_{region}_normalized.pickle','wb') as f:
        pickle.dump(res,f)
        
    res=compute_glomeruli_argmax_mapping_rate(
            adata_tgt,
            adata_src,
            G0,
            source_key="glomerulus",
            target_key="glomerulus",
            ignore_label="None"
        ) 
    
    
    with open(f'glom_res/fpgw_{region}_argmax_normalized.pickle','wb') as f:
        pickle.dump(res,f)
        

mos_acc1 = {}
mos_acc2 = {}
for i in [0,1,3]:
    with open(f'glom_res/moscot_{i}_argmax.pickle','rb') as f:
        res_mos1 = pickle.load(f)
        mos_acc1 = mos_acc1 | res_mos1['source_to_target_accuracy']
        mos_acc2 = mos_acc2 | res_mos1['target_to_source_accuracy']
res_mos = {'source_to_target_accuracy':mos_acc1,'target_to_source_accuracy':mos_acc2}

fpgw_acc1 = {}
fpgw_acc2 = {}
for i in [0,1,3]:
    with open(f'glom_res/fpgw_{i}_argmax.pickle','rb') as f:
        res_mos1 = pickle.load(f)
        fpgw_acc1 = fpgw_acc1 | res_mos1['source_to_target_accuracy']
        fpgw_acc2 = fpgw_acc2 | res_mos1['target_to_source_accuracy']
        
res = {'source_to_target_accuracy':fpgw_acc1,'target_to_source_accuracy':fpgw_acc2}

plot_glomeruli_mapping_scatter(
    res,
    res_mos,
    our_label="FPGW",
    competitor_label="moscot"
)
np.mean(list(res['source_to_target_accuracy'].values()))
np.mean(list(res['target_to_source_accuracy'].values()))
plot_glomeruli_accuracy_difference(res, res_mos, our_label="FPGW", competitor_label="moscot")
