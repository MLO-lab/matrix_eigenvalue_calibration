import re
import time
import numpy as np
from threadpoolctl import threadpool_limits
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import qutip as qt
import pandas as pd
import seaborn as sns
import sklearn as sk
import scipy as sp
import scipy.cluster.hierarchy as spc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from qutip import Bloch, basis
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from datasets import load_dataset
from tqdm import tqdm
import pickle, os
import gzip
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy
import hdbscan

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
from webencodings import lookup

model_names = {'phi_4': 'Phi 4', 'phi_4_mini': 'Phi 4 Mini', 'llama4_maverick': 'Llama4 Maverick'}
dataset_names = {'TriviaQA': 'TriviaQA', 'OpenNQ': 'Natural Questions'}
DATASET_TO_NAME = {
    'TriviaQA': 'TriviaQA',
    'OpenNQ': 'Natural Questions',
}

ALL_TEMPS = np.logspace(np.log10(0.1), np.log10(3.0), num=20) #Linear search space over sampling temperatures for the Lamb et al. sampling temperature calibration baseline

def plot_risk_TS_curves(results, file_name=None, font_size=10, figsize=(10, 4), n_size=300):
    results = pd.DataFrame(results)
    datasets = ['TriviaQA', 'OpenNQ']
    models = ['phi_4_mini', 'phi_4', 'llama4_maverick']
    n_rows = len(datasets)
    n_cols = len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    for dataset_id, dataset in enumerate(datasets):
        for model_id, model in enumerate(models):
            selected_instances = results[(results['model']==model)&(results['dataset']==dataset)].reset_index(drop=True)
            temps = np.array(selected_instances['temps'][0])
            #temps = np.round(1/temps, 2)
            risks = np.array(selected_instances['risks'][0])
            risks_stds = np.array(selected_instances['risks_stds'][0])
            risks_ub = risks + risks_stds/n_size
            risks_lb = risks - risks_stds/n_size
            entropies = np.array(selected_instances['avg_entropies'][0])
            entropies_stds = np.array(selected_instances['avg_entropies_stds'][0])
            entropies_ub = entropies + entropies_stds/n_size
            entropies_lb = entropies - entropies_stds/n_size

            axes[dataset_id][model_id].axvline(x=1, color='grey', linestyle='--', linewidth=2)

            axes[dataset_id][model_id].plot(temps, risks, color='red', linewidth=2, label='Risk')
            axes[dataset_id][model_id].fill_between(
                temps, risks_lb, risks_ub, alpha=0.2, color='red'
            )
            axes[dataset_id][model_id].plot(temps, entropies, color='blue', linewidth=2, label='Average \n Entropy')
            axes[dataset_id][model_id].fill_between(
                temps, entropies_lb, entropies_ub, alpha=0.2, color='blue'
            )
            axes[dataset_id][model_id].set_xscale("log")
            if dataset_id < 1:
                axes[dataset_id][model_id].set_title(model_names[model])
                axes[dataset_id][model_id].set_xticks([], [])
                if model_id==2:
                     # top right subfigure
                    axes[dataset_id][model_id].legend(bbox_to_anchor=(1.05, 1))
            else:
                axes[dataset_id][model_id].set_xticks([0.5, 1.0, 2.0, 4.0], [0.5, 1.0, 2.0, 4.0])
            if dataset_id==len(datasets)-1:
                axes[dataset_id][model_id].set_xlabel('Temperature', fontsize=font_size)
            if model_id < 1:
                axes[dataset_id][model_id].set_ylabel(dataset_names[dataset], fontsize=font_size)
    # plt.xticks(fontsize=font_size)
    # plt.yticks(fontsize=font_size)                    
    plt.tight_layout()
    if file_name is not None:
        plt.savefig('figures/'+file_name, dpi=300, bbox_inches="tight")
    plt.show();

def balanced_kmeans(corr, k, random_state=0):
    """
    Balanced clustering into k equal-sized clusters
    from a correlation matrix using Hungarian assignment.
    """
    np.random.seed(random_state)
    N = corr.shape[0]
    assert N % k == 0, "N must be divisible by k for equal clusters"
    cluster_size = N // k

    # Use correlation distance
    dist = 1 - corr

    # Initialize k random centers
    centers = np.random.choice(N, k, replace=False)

    for _ in range(10):  # a few refinement iterations
        # Compute distances to cluster centers
        D = dist[:, centers]  # shape (N, k)

        # Expand cost matrix to enforce equal cluster sizes
        cost = np.repeat(D, cluster_size, axis=1)  # (N, k*cluster_size)

        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        assignments = col_ind // cluster_size

        # Update centers as mean of assigned points (like k-means)
        new_centers = []
        for c in range(k):
            idx = np.where(assignments == c)[0]
            # compute centroid in correlation space
            sub_corr = corr[np.ix_(idx, idx)]
            center = idx[np.argmax(np.sum(sub_corr, axis=1))]
            new_centers.append(center)
        centers = new_centers

    return assignments
# # Example
# np.random.seed(0)
# N = 2000
# X = np.random.randn(100, N)
# corr = np.corrcoef(X, rowvar=False)

# labels = balanced_kmeans(corr, k=4)
# print("Cluster labels:", labels)
# print("Cluster sizes:", np.bincount(labels))

def load_data(dataset, answer_model, split, embedding_model='all_mpnet_base_v2', baseline = False, baseline_temperature_index=0):
    """
    Available datasets:
    - TriviaQA
    - OpenNQ
    Available answer models:
    - phi_4
    - phi_4_mini
    - llama4_maverick
    Possible splits:
    - validation
    - test
    Note: Validation dataset has 100 answers per question. Test has only 20 answers per question.
    """

    # Added block to correctly fetch baseline data
    if baseline:
        log_dir = 'baseline_logs'
        temperature_str = f'-t{baseline_temperature_index}'
    else:
        log_dir = 'logs'
        temperature_str = ''

    if dataset not in ['TriviaQA', 'OpenNQ']:
        raise ValueError(f"Dataset {dataset} not recognized. Only 'TriviaQA' and 'OpenNQ' are supported.")

    #Get ground truth data
    if split not in ['validation', 'test']:
        raise ValueError("split must be 'validation' or 'test'")

    elif split == 'test':
        dataset = dataset + '_2k'

    dataset_path = f'data/eval/{dataset}.json'
    dataset_df = pd.read_json(dataset_path)

    if split == 'test': #If we want the test set, we take the 2k set and remove the overlapping points between it and the validation set
        base_dataset = dataset.replace('_2k', '')
        base_path = f'data/eval/{base_dataset}.json'
        base_df = pd.read_json(base_path)
        base_ids = set(base_df['datapoint_id'])
        dataset_df = dataset_df[~dataset_df['datapoint_id'].isin(base_ids)].reset_index(drop=True)


    #Get answer embeddings
    pickle_path = f'data/{log_dir}/{dataset}/no_ensembling/no_model-variations/{answer_model}-embeddings{temperature_str}-{embedding_model}.pkl.gz'
    with gzip.open(pickle_path, 'rb') as f:
        embedding_data = pickle.load(f)
    dataset_df = dataset_df.merge(embedding_data, on='question_id', how='left')

    if not baseline:
        #Get answer correctness (fuzzy and model/judge-based labels, one per question)
        standard_answers_path = f'data/logs/{dataset}/standard_answers/{answer_model}-answers.json'
        standard_df = pd.read_json(standard_answers_path)
        standard_df = standard_df.drop(columns=['generated_answer'])
        # some embedding pickles already carry fuzzy_correctness; drop before merge to avoid _x/_y suffix collision
        conflict_cols = [c for c in standard_df.columns if c != 'question_id' and c in dataset_df.columns]
        if conflict_cols:
            dataset_df = dataset_df.drop(columns=conflict_cols)
        dataset_df = dataset_df.merge(standard_df, on='question_id', how='left')
    
    #Get ground truth embeddings
    gt_pickle_path = f'data/logs/{dataset}/gt-embeddings-{embedding_model}.pkl.gz'
    with gzip.open(gt_pickle_path, 'rb') as f:
        gt_embeddings_data = pickle.load(f)
    gt_embeddings_data_df = pd.DataFrame(gt_embeddings_data)
    gt_embeddings_data_df['gt_embedding'] = gt_embeddings_data_df['embedding']
    gt_embeddings_data_df= gt_embeddings_data_df[['question_id', 'gt_embedding']]
    dataset_df = dataset_df.merge(gt_embeddings_data_df, on='question_id', how = 'left')

    return dataset_df


def cross_entropy_loss(EV_decomp, target_vec, eps=1e-10):
    eigenvals, eigenvecs = EV_decomp #np.linalg.eigh(psd_matrix)
    eigenvals = np.maximum(eigenvals, eps)
    log_matrix = eigenvecs @ np.diag(np.log(eigenvals)) @ eigenvecs.T
    return (-1) * target_vec.T @ log_matrix @ target_vec

def kernel_score(psd_matrix, target_vec):
    matrix_length = np.linalg.norm(psd_matrix, 'fro')**2
    cross_term = target_vec.T @ psd_matrix @ target_vec
    return matrix_length - 2*cross_term

# Load GloVe embeddings (e.g., glove.6B.100d.txt)
def load_glove_embeddings(filepath):
    embeddings = {}
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.array(values[1:], dtype='float32')
            embeddings[word] = vec
    return embeddings

# Preprocess text (simple tokenization)
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def argsort_based_on_max_EV(eigenvals, eigenvecs):
    eigenvals_max = eigenvals.copy()
    eigenvals_max[eigenvals < np.max(eigenvals)] = 0
    reduced_gram_matrix = eigenvecs @ np.diag(eigenvals_max) @ eigenvecs.T
    reduced_probs = np.diag(reduced_gram_matrix)
    ranks = np.argsort(reduced_probs)
    return ranks[::-1]

# Average word embeddings for a sentence
def sentence_embedding(sentence, glove):
    tokens = tokenize(sentence)
    vectors = [glove[word] for word in tokens if word in glove]
    if not vectors:
        return np.zeros(100)  # Fallback for unknown tokens
    return np.mean(vectors, axis=0)

def normalise_rows(data):
    return data / np.linalg.norm(data, axis=1, keepdims=True)

def spectral_decomp_gram_matrix(vec):
    vec = vec.astype(np.float32)
    gram_matrix = vec @ vec.T/vec.shape[0]
    eigenvals, eigenvecs = np.linalg.eigh(gram_matrix)
    # eigenvalues are expected to be normalised
    eigenvals = np.clip(eigenvals, 0, 1)
    return eigenvals, eigenvecs

def TS_EV(eigenvals, temp=1): #Now temperature is inversed to match the standard temperature definition.
    eigenvals = np.abs(eigenvals)
    try:
        scaled_EVs = eigenvals**(1/temp) / np.sum(eigenvals**(1/temp))
        return scaled_EVs
    except ZeroDivisionError:
        print(eigenvals)
        print(temp)
        #print(eigenvals**(1/temp))
        return np.ones_like(eigenvals)/len(eigenvals)

def TS_matrix(psd_matrix, temp=1):
    eigenvals, eigenvecs = np.linalg.eigh(psd_matrix)
    eigenvals = np.clip(eigenvals, 0, 1)
    scaled_EVs = TS_EV(eigenvals, temp=temp)
    return scaled_EVs, eigenvecs

def plot_eigenvals(eigenvals, file_name=None, font_size=12, figsize=(6,6)):
    outcomes = ['#{}'.format(i+1) for i, _ in enumerate(eigenvals)]

    # Create a bar plot
    plt.figure(figsize=figsize)
    colors = ['red'] + ['blue'] * (len(outcomes) - 1)
    plt.bar(outcomes, eigenvals[::-1], color=colors)
    
    # Add labels and title
    plt.xlabel('Latent Outcome', fontsize=font_size+4)
    plt.ylabel('Probability', fontsize=font_size+4)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if file_name is not None:
        plt.savefig('figures/'+file_name, dpi=300, bbox_inches="tight")
    plt.show();

def plot_bloch(data, color_gradient=None, view=None, file_name=None, point_size=100, font_size=15, figsize=(6,6)):
    """
    data shape: (3, n_samples)
    color_gradient: (n_samples,)
    view: (2,)
    """
    # Instantiate the Bloch sphere
    b = Bloch()
    x = (basis(2,0)+(1+0j)*basis(2,1)).unit()
    y = (basis(2,0)+(0+1j)*basis(2,1)).unit()
    z = (basis(2,0)+(0+0j)*basis(2,1)).unit()
    b.add_states([x,y,z])

    if color_gradient is not None:
        norm = colors.Normalize(vmin=np.min(color_gradient), vmax=np.max(color_gradient))
        cmap = plt.get_cmap('coolwarm')
        mapped_colors = cmap(norm(color_gradient))
        b.point_color = mapped_colors
        b.add_points(data, 'm')
    else:
        b.add_points(data)

    if view is not None:
        b.view = view
    b.zlabel = ["", ""]
    b.xlabel = ["", ""]
    b.ylabel = ["", ""]
    b.point_size = [point_size]
    b.vector_color = ["black", "black", "black"]
    b.font_size = font_size
    b.fig = plt.figure(figsize=figsize)
    # ax = b.axes    
    # ax.set_ylabel("Dim 3", labelpad=20)
    # ax.yaxis.set_label_coords(-0.2, 0.5)    
    if file_name is not None:
        b.render()
        b.fig.savefig('figures/'+file_name, dpi=300, bbox_inches="tight")
    else:
        b.show()

def sim_estimator(emb_answers):
    n_val = emb_answers.shape[0]
    n_reps = 1000
    results = []
    for n_subset in tqdm([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75][::-1]):
        for seed in np.arange(n_reps):    
            np.random.seed(seed)
            row_indices = np.random.choice(n_val, size=n_subset, replace=False)
            sampled_embs = emb_answers[row_indices]
            sampled_embs = normalise_rows(sampled_embs)
            eigenvals, _ = spectral_decomp_gram_matrix(sampled_embs)
            results += [{
                'seed': seed.item(),
                'n_size': n_subset,
                'max EV': eigenvals[-1].item(),
            }]

    return pd.DataFrame(results)

def ECE_conf_bin(
    confs,
    pred_covs,
    target_embs,
    n_bins: int = 15,
    num_clusters: int = 1,
    strategy: str = "uniform",  # "uniform" or "quantile" (aka "adaptive")
    min_n = 2,
):
    """
    Compute Expected Calibration Error (ECE).

    Parameters
    ----------
    confs_clusters : array-like, shape (N,2)
        Predicted confidences with clusters.
    labels : array-like, shape (N, D)
        True embeddings.
    n_bins : int, optional (default=15)
        Number of bins for calibration.
    strategy : {"uniform", "quantile"}, optional (default="uniform")
        - "uniform": equal-width bins over [0,1].
        - "quantile": bins chosen by quantiles of confidence (adaptive binning).

    Returns
    -------
    ece : float
        Expected Calibration Error using L1 distance: sum_b ( (|acc_b - conf_b|) * (n_b / N) ).
    details : dict
        Useful per-bin details:
        {
            "bin_edges": np.ndarray, shape (B+1,),
            "bin_count": np.ndarray, shape (B,),
            "avg_conf": np.ndarray, shape (B,),
            "avg_acc": np.ndarray, shape (B,)
        }
    """
    if num_clusters > 1:
        cov_corr_matrix = pairwise_matrix_cos(pred_covs)
        clusters = get_clusters(cov_corr_matrix, num_clusters)
    else:
        clusters = np.repeat(0, confs.shape[0])

    target_embs = normalise_rows(target_embs)
    N = confs.shape[0]
    
    # Build bin edges
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy in ("quantile", "adaptive"):
        # Quantile edges; ensure coverage of [0,1]
        q = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(confs, q)
        edges = np.unique(edges)  # guard against repeated edges if conf has ties
        # Guarantee exact bounds
        edges[0] = 0.0
        edges[-1] = 1.0
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile' (alias: 'adaptive').")

    # Assign each sample to a bin: digitize with right-closed bins except the last
    # np.digitize returns indices in [0, len(edges)-1]; we use edges[1:-1] as split points -> [0, B-1]
    bin_ids = np.digitize(confs, edges[1:-1], right=True)

    # Per-bin counts and sums
    B = len(edges) - 1
    bin_count = np.bincount(bin_ids, minlength=B)
    sum_conf = np.bincount(bin_ids, weights=confs, minlength=B)
    max_EVs = np.zeros((B, 2))
    for bin_id in np.unique(bin_ids):
        for cluster_id in np.unique(clusters):
            mask = (bin_ids==bin_id) & (clusters==cluster_id)
            # ignore all target EVs based on less than min_n instances
            if np.sum(mask) >= min_n:
                selected_embs = target_embs[mask]
                eigenvals, _ = spectral_decomp_gram_matrix(selected_embs)
                max_EVs[bin_id, 0] += eigenvals[-1].item()
                max_EVs[bin_id, 1] += 1

    # avoid div by 0
    max_EVs = max_EVs[:,0] / np.maximum(max_EVs[:,1], 1)
    nonzero = bin_count > 0
    avg_conf = np.zeros(B, dtype=float)
    avg_conf[nonzero] = sum_conf[nonzero] / bin_count[nonzero]
    
    weights = bin_count.astype(float) / float(N)
    ece = np.sum(weights * np.abs(max_EVs - avg_conf))

    return ece, {
        "bin_edges": edges,
        "bin_count": bin_count,
        "avg_conf": avg_conf, #Average model confidence within bin
        "max_EVs": max_EVs, #Target max EV within bin
    }
    
def get_max_EV(emb_matrix):
    """
    emb_matrix shape (samples x dimension)
    return: scalar max EV
    """
    emb_matrix = normalise_rows(emb_matrix)
    eigenvals, _ = spectral_decomp_gram_matrix(emb_matrix)
    return eigenvals[-1].item()


def dataset_to_confs_targets(dataset, temp=None):
    df_wide = dataset.pivot(index='question_id', columns='answer_id', values='embedding')
    cols = df_wide.keys()
    df_wide['emb_matrix'] = df_wide.apply(lambda row: np.vstack([row[c] for c in cols]), axis=1)
    df_wide['emb_matrix'] = df_wide.apply(lambda row: normalise_rows(row['emb_matrix']), axis=1)
    df_wide['EVs'] = df_wide.apply(lambda row: spectral_decomp_gram_matrix(row['emb_matrix'])[0], axis=1)
    if temp is not None:
        df_wide['EVs'] = df_wide.apply(lambda row: TS_EV(row['EVs'], temp=temp), axis=1)    

    df_wide['max_EV'] = df_wide.apply(lambda row: row['EVs'][-1].item(), axis=1)
    confs = df_wide['max_EV']
    n_uniques = np.unique(dataset['question_id']).shape[0]
    reps = dataset.shape[0] // n_uniques
    target_embs = dataset['gt_embedding'].iloc[::reps]
    target_embs = np.stack(target_embs.to_numpy())

    return confs, target_embs

def dataset_to_evs_uncertainties(dataset, temp=None):
    df_wide = dataset.pivot(index='question_id', columns='answer_id', values='embedding')
    cols = df_wide.keys()
    n_answers = len(cols)
    df_wide['emb_matrix'] = df_wide.apply(lambda row: np.vstack([row[c] for c in cols]), axis=1)
    df_wide['emb_matrix'] = df_wide.apply(lambda row: normalise_rows(row['emb_matrix']), axis=1)
    df_wide['emb_list'] = df_wide.apply(lambda row: [row[c]/np.linalg.norm(row[c]) for c in cols], axis=1)
    df_wide['EVs'] = df_wide.apply(lambda row: spectral_decomp_gram_matrix(row['emb_matrix'])[0], axis=1)
    df_wide['EVects'] = df_wide.apply(lambda row: spectral_decomp_gram_matrix(row['emb_matrix'])[1], axis=1)
    if temp is not None:
        df_wide['EVs'] = df_wide.apply(lambda row: TS_EV(row['EVs'], temp=temp), axis=1)
    df_wide['max_EV'] = df_wide.apply(lambda row: row['EVs'][-1].item(), axis=1)
    df_wide['VNE'] = df_wide.apply(lambda row: entropy(row['EVs']), axis=1)

    def compute_pke_from_eigenvalues_eigenvectors(eigenvals, eigenvecs):
        n=len(eigenvals)
        matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        return float((matrix.diagonal().sum() - matrix.sum())/(n*(n-1)))
    
    df_wide['PKE'] = df_wide.apply(lambda row: compute_pke_from_eigenvalues_eigenvectors(row['EVs'], row['EVects']), axis=1)
    
    df_wide['question_id'] = df_wide.index.map(int)
    df_wide = df_wide.reset_index(drop=True).sort_values('question_id', ascending=True).reset_index(drop=True)
    df_evs = df_wide[['question_id', 'EVs', 'max_EV', 'VNE', 'PKE']]
    return df_evs, n_answers

def emb_to_cov_matrix(emb_matrix):
    emb_matrix = normalise_rows(emb_matrix.astype(np.float32))
    cov_matrix = emb_matrix.T @ emb_matrix/emb_matrix.shape[0]
    return cov_matrix

def _compute_one_temp(ev_decomps, target_embs, fuzzy_correctness, temp):
    ts_ev_decomps = [(TS_EV(ev[0], temp=temp), ev[1]) for ev in ev_decomps]
    xent_losses = np.array([cross_entropy_loss(ts_ev, t_emb) for ts_ev, t_emb in zip(ts_ev_decomps, target_embs)])
    entropies = np.array([entropy(ts_ev[0]) for ts_ev in ts_ev_decomps])
    confs = np.array([ts_ev[0][-1].item() for ts_ev in ts_ev_decomps])
    return {
        'risk': xent_losses.mean(),
        'risk_std': xent_losses.std(),
        'avg_entropy': entropies.mean(),
        'avg_entropy_std': entropies.std(),
        'ev_auroc': roc_auc_score(fuzzy_correctness, confs),
        'ent_auroc': roc_auc_score(fuzzy_correctness, -entropies),
    }


def dataset_to_risks(dataset, temps, progress_bar=None):
    df_wide = dataset.pivot(index='question_id', columns='answer_id', values='embedding')
    cols = df_wide.keys()
    n_uniques = np.unique(dataset['question_id']).shape[0]
    reps = dataset.shape[0] // n_uniques

    df_wide['emb_matrix'] = df_wide.apply(lambda row: np.vstack([row[c] for c in cols]), axis=1)
    df_wide['pred_matrix'] = df_wide.apply(lambda row: emb_to_cov_matrix(row['emb_matrix']), axis=1)
    df_wide['EV_decomp'] = df_wide.apply(lambda row: np.linalg.eigh(row['pred_matrix']), axis=1)

    target_embs = dataset['gt_embedding'].iloc[::reps].reset_index(drop=True).tolist()
    df_wide['target_embs'] = target_embs
    fuzzy_correctness = dataset['fuzzy_correctness'].iloc[::reps].to_numpy()
    ev_decomps = df_wide['EV_decomp'].tolist()

    results = {
        'risks': [None] * len(temps),
        'risks_stds': [None] * len(temps),
        'ev_aurocs': [None] * len(temps),
        'ent_aurocs': [None] * len(temps),
        'avg_entropies': [None] * len(temps),
        'avg_entropies_stds': [None] * len(temps),
    }
    for i, temp in enumerate(temps):
        r = _compute_one_temp(ev_decomps, target_embs, fuzzy_correctness, temp)
        results['risks'][i] = r['risk']
        results['risks_stds'][i] = r['risk_std']
        results['avg_entropies'][i] = r['avg_entropy']
        results['avg_entropies_stds'][i] = r['avg_entropy_std']
        results['ev_aurocs'][i] = r['ev_auroc']
        results['ent_aurocs'][i] = r['ent_auroc']
        if progress_bar is not None:
            progress_bar.update(1)
    return results

def dataset_to_cov_matrix(dataset):
    df_wide = dataset.pivot(index='question_id', columns='answer_id', values='embedding')
    cols = df_wide.keys()
    df_wide['emb_matrix'] = df_wide.apply(lambda row: np.vstack([row[c] for c in cols]), axis=1)
    pred_matrix = np.array(df_wide.apply(lambda row: emb_to_cov_matrix(row['emb_matrix']), axis=1))
    pred_matrix = np.stack(pred_matrix)
    return pred_matrix

def load_all_settings(datasets, models, split):
    data = {}
    for dataset_id, dataset in enumerate(datasets):
        data[dataset] = {}
        for model_id, model in enumerate(models):
            dataframe = load_data(dataset, model, split='validation')
            data[dataset][model] = dataframe
    return data

def get_clusters(corr_matrix, num_clusters, min_cluster_size=None): #min_n here does not really work.
    pdist = spc.distance.pdist(corr_matrix) #Compute pairwise euclidian distances between cosine similarity vectors: Similar similarity vectors <-> both matrices correlate similarly to all other matrices <-> low distance
    linkage = spc.linkage(pdist, method='average')
    if num_clusters is None:
        idx = spc.fcluster(linkage, 0.5 * pdist.max(), criterion='distance')
    else:
        idx = spc.fcluster(linkage, criterion='maxclust', t=num_clusters)
    return idx

def _compute_TS_pair(dataset, model, temps, embedding_model):
    dataframe = load_data(dataset, model, split='validation', embedding_model=embedding_model)
    risks_dict = dataset_to_risks(dataframe, temps, progress_bar=None)
    risks_dict['temps'] = temps
    risks_dict['model'] = model
    risks_dict['dataset'] = dataset
    return risks_dict


def compute_TS_curves(datasets, models, temps, embedding_model='all_mpnet_base_v2', save_results='results/TS_results.json'):
    pairs = [(dataset, model) for dataset in datasets for model in models]
    results = []

    _log(f"compute_TS_curves [{embedding_model}]: {len(pairs)} pairs, {len(temps)} temps")
    t0 = time.perf_counter()

    n_workers = len(pairs)
    blas_threads = max(1, (os.cpu_count() or 1) // n_workers)
    _log(f"  [{embedding_model}] using {n_workers} workers × {blas_threads} BLAS threads")
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(blas_threads,)) as executor:
        futures = {
            executor.submit(_compute_TS_pair, dataset, model, temps, embedding_model): (dataset, model)
            for dataset, model in pairs
        }
        with tqdm(total=len(pairs), desc='dataset/model pairs') as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                _log(f"  [{embedding_model}] finished {result['dataset']}/{result['model']}  ({time.perf_counter()-t0:.1f}s elapsed)")
                pbar.update(1)

    _log(f"compute_TS_curves [{embedding_model}] done in {time.perf_counter()-t0:.1f}s")

    if save_results:
        pd.DataFrame(results).to_json(save_results)
    return results

def plot_auroc_TS_curves(results, unc_type='EV', file_name=None, figsize=(10, 4)):
    results = pd.DataFrame(results)
    datasets = ['TriviaQA', 'OpenNQ']
    models = ['phi_4_mini', 'phi_4', 'llama4_maverick']
    n_rows = len(datasets)
    n_cols = len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    for dataset_id, dataset in enumerate(datasets):
        for model_id, model in enumerate(models):
            selected_instances = results[(results['model']==model)&(results['dataset']==dataset)].reset_index(drop=True)
            temps = selected_instances['temps'][0]
            if unc_type=='EV':
                aurocs = selected_instances['ev_aurocs'][0]
                axes[dataset_id][model_id].plot(temps, aurocs, color='blue', linewidth=2)
            elif unc_type=='ENT':
                aurocs = selected_instances['ent_aurocs'][0]
                axes[dataset_id][model_id].plot(temps, aurocs, color='blue', linewidth=2)
            axes[dataset_id][model_id].set_xscale("log")
            if dataset_id < 1:
                axes[dataset_id][model_id].set_title(model)
            if dataset_id==len(datasets)-1:
                axes[dataset_id][model_id].set_xlabel('Temperature')
            if model_id < 1:
                axes[dataset_id][model_id].set_ylabel(DATASET_TO_NAME[dataset])
                    
    plt.tight_layout()
    if file_name is not None:
        plt.savefig('figures/'+file_name, dpi=300, bbox_inches="tight")
    plt.show();

def matrix_cos_sim(A, B):
    """only used for unit testing the next function"""
    AB_prod = np.trace(A @ B)
    l2_A = np.sqrt(np.trace(A@A))
    l2_B = np.sqrt(np.trace(B@B))
    return AB_prod / (l2_A * l2_B)

# # unit test
# A = np.array([[0., 1., 3.], [1., 0., 2.], [3., 2., 0.]])
# B = np.array([[1., 1., 0.], [1., 1., 2.], [0., 2., 1.]])
# D = np.array([A,B])
# sim1 = matrix_cos_sim(A,B)
# sim2 = pairwise_matrix_cos(D)
# print(sim1, sim2)

def pairwise_matrix_cos(mats):
    N = mats.shape[0]
    # Reshape matrices into vectors of length (m*n)
    vecs = mats.reshape(N, -1)   # shape (N, m*n)    
    # Pairwise Frobenius inner products = Gram matrix
    G = vecs @ vecs.T
    # Norms of each matrix (Frobenius norm)
    norms = np.linalg.norm(vecs, axis=1)
    # Normalized Frobenius inner product (cosine similarity)
    cosine_sim = G / np.outer(norms, norms)
    return cosine_sim

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

def cluster_by_cosine_similarity(embeddings, n_clusters=5):
    """
    Cluster data instances based on cosine similarity.

    Parameters
    ----------
    embeddings : ndarray, shape (N, D)
        Rows are data instances, columns are embedding dimensions.
    n_clusters : int
        Desired number of clusters.

    Returns
    -------
    labels : ndarray, shape (N,)
        Cluster assignment for each instance.
    similarity_matrix : ndarray, shape (N, N)
        Pairwise cosine similarity matrix.
    """
    embeddings = np.asarray(embeddings)

    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # Convert similarity to distance (AgglomerativeClustering uses distance)
    dist_matrix = 1.0 - sim_matrix

    # Use "metric" instead of "affinity" in new versions
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average"
    )
    labels = clustering.fit_predict(dist_matrix)

    return labels, sim_matrix

# # -------------------
# # Example usage:
# # Suppose we have 6 samples with 3D embeddings
# embeddings = np.random.randn(6, 3)
# labels, sim = cluster_by_cosine_similarity(embeddings, n_clusters=2)

# print("Cluster labels:", labels)
# print("Cosine similarity matrix:\n", sim)

# from functions import dataset_to_confs_targets, dataset_to_cov_matrix, pairwise_matrix_cos, get_clusters, balanced_kmeans

def ECE_conf_bin_alt(
    confs,
    pred_covs,
    target_embs,
    n_bins: int = 15,
    num_clusters: int = 10,
    min_n = 5,
    cluster_func=get_clusters
):
    """
    Compute Expected Calibration Error (ECE).

    Parameters
    ----------
    confs_clusters : array-like, shape (N,2)
        Predicted confidences with clusters.
    labels : array-like, shape (N, D)
        True embeddings.
    n_bins : int, optional (default=15)
        Number of bins for calibration.

    Returns
    -------
    ece : float
        Expected Calibration Error using L1 distance: sum_b ( (|acc_b - conf_b|) * (n_b / N) ).
    details : dict
        Useful per-bin details:
        {
            "bin_edges": np.ndarray, shape (B+1,),
            "bin_count": np.ndarray, shape (B,),
            "avg_conf": np.ndarray, shape (B,),
            "avg_acc": np.ndarray, shape (B,)
        }
    """
    target_embs = normalise_rows(target_embs)
    N = confs.shape[0]
    
    # Quantile edges; ensure coverage of [0,1]
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(confs, q)
    edges = np.unique(edges)  # guard against repeated edges if conf has ties
    # Guarantee exact bounds
    edges[0] = 0.0
    edges[-1] = 1.0

    # Assign each sample to a bin: digitize with right-closed bins except the last
    # np.digitize returns indices in [0, len(edges)-1]; we use edges[1:-1] as split points -> [0, B-1]
    bin_ids = np.digitize(confs, edges[1:-1], right=True)

    # Per-bin counts and sums
    B = len(edges) - 1
    bin_count = np.bincount(bin_ids, minlength=B)
    sum_conf = np.bincount(bin_ids, weights=confs, minlength=B)
    max_EVs = np.zeros((B, 2))
    for bin_id in np.unique(bin_ids): #Bin then cluster
        bin_covs = pred_covs[bin_ids==bin_id]
        bin_confs = confs[bin_ids==bin_id]
        bin_targets = target_embs[bin_ids==bin_id]
        cov_corr_matrix = pairwise_matrix_cos(bin_covs)
        clusters = cluster_func(cov_corr_matrix, num_clusters, min_n)
        for cluster_id in np.unique(clusters):
            if np.sum(clusters==cluster_id) >= min_n: #If cluster does not have enough instances, ignore
                selected_targets = bin_targets[clusters==cluster_id]
                eigenvals, _ = spectral_decomp_gram_matrix(selected_targets)
                max_EVs[bin_id, 0] += eigenvals[-1].item() #Running sum of max EVs
                max_EVs[bin_id, 1] += 1 #Running number of clusters considered within bin

    # avoid div by 0
    max_EVs = max_EVs[:,0] / np.maximum(max_EVs[:,1], 1) #Bins with no clusters considered will have max_EV=0 (this usually doesn't happen with quantile binning, but depends also on the clustering algorithm)
    nonzero = bin_count > 0
    avg_conf = np.zeros(B, dtype=float)
    avg_conf[nonzero] = sum_conf[nonzero] / bin_count[nonzero] #Average model confidence within bin
    
    weights = bin_count.astype(float) / float(N)
    ece = np.sum(weights * np.abs(max_EVs - avg_conf))

    return ece, {
        "bin_edges": edges,
        "bin_count": bin_count,
        "avg_conf": avg_conf, #Average model confidence within bin
        "max_EVs": max_EVs, #Target max EV within bin
    }

def plot_single_rel_diag(confs, max_EVs, freq, ece_results, figsize=(2.5, 2), font_size=10, title=None, file_name='rel_diagram_bincluster_trivia_phi4mini.png'):
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    plot_rel_diag(confs, max_EVs, freq, ece_results, axes, bin_type='bin2cluster', legend=False)
    plt.xlabel('Predicted eigenvalue', fontsize=font_size)
    plt.ylabel('Target eigenvalue', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('figures/'+file_name, dpi=300, bbox_inches="tight")
    plt.show();

def get_hdbscan_cluster(corr, num_cluster, min_cluster_size):
    # Convert to distance matrix
    corr = np.minimum(corr, 1)
    dist = np.sqrt(4 * (1 - corr))
    dist = dist.astype(np.float64)    
    # Run HDBSCAN (minimum cluster size = 3)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed')
    labels = clusterer.fit_predict(dist)
    return labels

def _init_worker(num_threads=1):
    threadpool_limits(num_threads)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _preprocess_pair(dataset, model, TS_dict, split, embedding_model, correctness_type):
    dataframe = load_data(dataset, model, split=split, embedding_model=embedding_model)
    df_wide = dataframe.pivot(index='question_id', columns='answer_id', values='embedding')
    cols = df_wide.keys()
    n_uniques = np.unique(dataframe['question_id']).shape[0]
    reps = dataframe.shape[0] // n_uniques

    df_wide['emb_matrix'] = df_wide.apply(lambda row: np.vstack([row[c] for c in cols]), axis=1)
    df_wide['pred_matrix'] = df_wide.apply(lambda row: emb_to_cov_matrix(row['emb_matrix']), axis=1)
    df_wide['EV_decomp'] = df_wide.apply(lambda row: np.linalg.eigh(row['pred_matrix']), axis=1)

    if correctness_type == 'fuzzy_correctness':
        if 'fuzzy_correctness_y' in dataframe.keys():
            correctness_key = 'fuzzy_correctness_y'
        elif 'fuzzy_correctness' in dataframe.keys():
            correctness_key = 'fuzzy_correctness'
    elif correctness_type == 'model_correctness':
        correctness_key = 'model_correctness'
    else:
        raise ValueError("correctness_type must be 'fuzzy_correctness' or 'model_correctness'")

    correctness = dataframe[correctness_key].iloc[::reps].to_numpy()
    target_embs = dataframe['gt_embedding'].iloc[::reps]
    df_wide['target_embs'] = target_embs.reset_index(drop=True)

    temp = TS_dict[dataset][model] if TS_dict is not None else None

    df_wide['TS_EV_decomp'] = df_wide.apply(
        lambda row: (TS_EV(row['EV_decomp'][0], temp=temp), row['EV_decomp'][1]), axis=1
    )
    df_wide['entropy'] = df_wide.apply(lambda row: entropy(np.maximum(row['EV_decomp'][0], 0)), axis=1)
    df_wide['TS_entropy'] = df_wide.apply(
        lambda row: entropy(np.maximum(row['TS_EV_decomp'][0], 0)), axis=1
    )

    confs = df_wide.apply(lambda row: row['EV_decomp'][0][-1].item(), axis=1).to_numpy()
    TS_confs = df_wide.apply(lambda row: row['TS_EV_decomp'][0][-1].item(), axis=1).to_numpy()
    ents = (-1) * df_wide['entropy'].to_numpy()
    TS_ents = (-1) * df_wide['TS_entropy'].to_numpy()
    return dataset, model, correctness, confs, TS_confs, ents, TS_ents


def _auroc_one_seed(dataset, model, seed, correctness, confs, TS_confs, ents, TS_ents):
    np.random.seed(seed)
    sample_idx = np.random.choice(confs.shape[0], size=confs.shape[0], replace=True)
    sampled_corrects = correctness[sample_idx]
    return {
        'dataset': dataset, 'model': model, 'seed': seed,
        'ev_auroc': roc_auc_score(sampled_corrects, confs[sample_idx]),
        'TS_ev_auroc': roc_auc_score(sampled_corrects, TS_confs[sample_idx]),
        'ent_auroc': roc_auc_score(sampled_corrects, ents[sample_idx]),
        'TS_ent_auroc': roc_auc_score(sampled_corrects, TS_ents[sample_idx]),
    }


def auroc_experiments(datasets, models, TS_dict, n_bootstrap, split='test', embedding_model='all_mpnet_base_v2', correctness_type='fuzzy_correctness'):
    pairs = [(dataset, model) for dataset in datasets for model in models]
    total_tasks = len(pairs) * n_bootstrap

    _log(f"auroc_experiments: {len(pairs)} pairs, {n_bootstrap} bootstrap seeds, split={split}")
    t_total = time.perf_counter()

    with ProcessPoolExecutor(max_workers=len(pairs), initializer=_init_worker) as executor:
        # Phase 1: preprocess all (dataset, model) pairs in parallel
        _log(f"Phase 1 start — preprocessing {len(pairs)} pairs in parallel")
        t1 = time.perf_counter()
        prep_futures = {
            executor.submit(_preprocess_pair, dataset, model, TS_dict, split, embedding_model, correctness_type): (dataset, model)
            for dataset, model in pairs
        }
        preprocessed = {}
        with tqdm(total=len(pairs), desc='preprocessing pairs') as pbar:
            for future in as_completed(prep_futures):
                dataset, model, correctness, confs, TS_confs, ents, TS_ents = future.result()
                preprocessed[(dataset, model)] = (correctness, confs, TS_confs, ents, TS_ents)
                _log(f"  finished {dataset}/{model}  ({time.perf_counter()-t1:.1f}s elapsed)")
                pbar.update(1)
        _log(f"Phase 1 done in {time.perf_counter()-t1:.1f}s")

        # Phase 2: run all (pair, seed) combinations in parallel
        _log(f"Phase 2 start — {total_tasks} bootstrap tasks ({len(pairs)} pairs × {n_bootstrap} seeds)")
        t2 = time.perf_counter()
        seed_futures = [
            executor.submit(_auroc_one_seed, dataset, model, seed, *preprocessed[(dataset, model)])
            for dataset, model in pairs
            for seed in range(n_bootstrap)
        ]
        results = []
        with tqdm(total=total_tasks, desc='bootstrap seeds') as pbar:
            for future in as_completed(seed_futures):
                results.append(future.result())
                pbar.update(1)
        _log(f"Phase 2 done in {time.perf_counter()-t2:.1f}s")

    _log(f"Total time: {time.perf_counter()-t_total:.1f}s")
    return pd.DataFrame(results)

def compute_rel_diag(dataset, model, temp, bin_type='uniform', min_n=2, num_clusters=1,
    num_bins=15, split='test', embedding_model='all_mpnet_base_v2', baseline = False, baseline_temperature_index=0):
    dataframe = load_data(dataset, model, split=split, embedding_model=embedding_model, baseline=baseline, baseline_temperature_index=baseline_temperature_index)
    confs, target_embs = dataset_to_confs_targets(dataframe, temp=temp)
    cov_mats = dataset_to_cov_matrix(dataframe)

    if bin_type=='bin2cluster':
        ece_result = ECE_conf_bin_alt(
            confs, cov_mats, target_embs, min_n=min_n, num_clusters=num_clusters, n_bins=num_bins,
            cluster_func=get_clusters
        )
        counts = ece_result[1]['bin_count']
    elif bin_type=='bin2hdbscan':
        ece_result = ECE_conf_bin_alt(
            confs, cov_mats, target_embs, min_n=min_n, num_clusters=num_clusters, n_bins=num_bins,
            cluster_func=get_hdbscan_cluster
        )
        counts = ece_result[1]['bin_count']
    else:
        ece_result = ECE_conf_bin(
            confs, cov_mats, target_embs, strategy=bin_type, min_n=min_n, num_clusters=num_clusters,
            n_bins=num_bins
        )
        counts = ece_result[1]['bin_count']
    confs = ece_result[1]['avg_conf']
    max_EVs = ece_result[1]['max_EVs']
    
    max_EVs[counts<min_n] = None
    freq = counts/counts.sum()
    freq[counts<min_n] = None
    return confs, max_EVs, freq, ece_result

def plot_rel_diag(confs, max_EVs, freq, ece_result, plot_axis, legend, bin_type='uniform'):
    plot_axis.plot(confs, max_EVs, marker='o', label="Target Max EV", color='red')
    if bin_type=='uniform' or bin_type=='cluster':
        plot_axis.plot(confs, freq, marker='o', label="Bin Freq.", color='orange')
    plot_axis.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfectly calibrated")
    plot_axis.set_xlabel("")
    plot_axis.set_ylabel("")
    plot_axis.grid(True)
    plot_axis.text(
        0.05, 0.95, f'ECE:{ece_result[0]:.2f}', transform=plot_axis.transAxes, ha="left", va="top",
    )
    if legend:
        plot_axis.legend(loc="center left", bbox_to_anchor=(1, 0.5))

def get_rel_diag(
    dataset, model, temp, plot_axis, legend, bin_type='uniform', min_n=2, num_clusters=1,
    num_bins=15, embedding_model='all_mpnet_base_v2'
):
    confs, max_EVs, freq, ece_result = compute_rel_diag(
        dataset=dataset, model=model, temp=temp, bin_type=bin_type,
        min_n=min_n, num_clusters=num_clusters, num_bins=num_bins, embedding_model=embedding_model
    )
    plot_rel_diag(
        confs=confs, max_EVs=max_EVs, freq=freq, ece_result=ece_result, plot_axis=plot_axis,
        legend=legend, bin_type=bin_type
    )

def plot_all_rel_diag(
    datasets, models, bin_type='uniform', TS_dict=None, num_clusters=1,
    min_n=2, file_name=None, num_bins=15, figsize=(10, 4), embedding_model='all_mpnet_base_v2'
):
    n_rows = len(datasets)
    n_cols = len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    for dataset_id, dataset in enumerate(datasets):
        for model_id, model in enumerate(models):
            if n_rows > 1:
                selected_axis = axes[dataset_id][model_id]
            else:
                selected_axis = axes[model_id]
            # only add legend to the last subfigure in the first row
            add_legend = (model_id==n_cols-1) and (dataset_id==0)
            if TS_dict is not None:
                temp = TS_dict[dataset][model]
            else:
                temp = None
                
            get_rel_diag(
                dataset, model, temp, selected_axis, legend=add_legend,
                bin_type=bin_type, min_n=min_n, num_clusters=num_clusters, num_bins=num_bins, embedding_model=embedding_model
            )
            if dataset_id < 1:
                selected_axis.set_title(model_names[model])
            if dataset_id==len(datasets)-1:
                selected_axis.set_xlabel('Mean predicted latent probability')
            if model_id < 1:
                selected_axis.set_ylabel(DATASET_TO_NAME[dataset])

    plt.tight_layout()
    if file_name is not None:
        plt.savefig('figures/'+file_name, dpi=300, bbox_inches="tight")
    plt.show();

def compute_multiple_rel_diag(
    dataset, models, bin_type='bin2cluster', min_n=2, num_clusters=5, num_bins=8, TS_dict=None
):
    results = {'confs': {}, 'max_EVs': {}, 'freq': {}, 'ece_result': {}}
    for model_id, model in enumerate(models):
        if TS_dict is not None:
            temp = TS_dict[dataset][model]
        else:
            temp = None
    
        confs, max_EVs, freq, ece_result = compute_rel_diag(
            dataset=dataset, model=model, temp=temp, bin_type=bin_type,
            min_n=min_n, num_clusters=num_clusters, num_bins=num_bins
        )
        results['confs'][model] = confs
        results['max_EVs'][model] = max_EVs
        results['freq'][model] = freq
        results['ece_result'][model] = ece_result
    return results

def plot_multiple_rel_diag(
    results, dataset, models, bin_type='bin2cluster', TS_dict=None, num_clusters=1,
    min_n=2, file_name=None, num_bins=15, figsize=(10, 3)
):
    n_cols = len(models)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    for model_id, model in enumerate(models):
        if TS_dict is not None:
            temp = TS_dict[dataset][model]
        else:
            temp = None
        plot_rel_diag(
            confs=results['confs'][model], max_EVs=results['max_EVs'][model], freq=results['freq'][model],
            ece_result=results['ece_result'][model], plot_axis=axes[model_id], bin_type=bin_type, legend=False
        )
        axes[model_id].set_title(model_names[model])
        axes[model_id].set_xlabel('Predicted eigenvalue')
        if model_id < 1:
            axes[model_id].set_ylabel('Target eigenvalue')

    plt.tight_layout()
    if file_name is not None:
        plt.savefig('figures/'+file_name, dpi=300, bbox_inches="tight")
    plt.show();

def generate_auroc_scores(
    datasets, models, split, confidence_measure, correctness_label, TS_dict=None,
):
    if correctness_label not in ['model_correctness', 'fuzzy_correctness']:
        raise ValueError("correctness_label must be 'model_correctness' or 'fuzzy_correctness'")
    results = {'dataset': [], 'model': [], 'AUROC_no_TS': []}
    if TS_dict is not None:
        results['AUROC_with_TS'] = []

    for dataset in datasets:
        for model in models:
            dataframe = load_data(dataset, model, split)
            correctness_df = load_standard_answers_and_correctness(dataset, model, split)
            correctness_df = correctness_df[['question_id', correctness_label]]
            correctness_df = correctness_df.sort_values('question_id', ascending=True).reset_index(drop=True)
            correctness = correctness_df[correctness_label].to_numpy()

            def compute_auroc_for_temp(temp):
                df_evs_unc, _ = dataset_to_evs_uncertainties(dataframe, temp=temp)
                df_evs_unc = df_evs_unc.sort_values('question_id', ascending=True).reset_index(drop=True)
                if confidence_measure=='max_EV':
                    confs = df_evs_unc['max_EV'].to_numpy()
                elif confidence_measure=='VNE':
                    confs = - df_evs_unc['VNE'].to_numpy()
                elif confidence_measure=='PKE':
                    confs = - df_evs_unc['PKE'].to_numpy()
                else:
                    raise ValueError("confidence_measure must be 'max_EV' or 'VNE' or 'PKE'")

                # print(f"Dataset: {dataset}, Model: {model}, Temp: {temp}")
                # print(df_evs_unc['PKE'].describe())
                # df_evs_unc['PKE'].hist()
                # plt.show()
                return roc_auc_score(correctness, confs)
            
            results['dataset'].append(dataset) 
            results['model'].append(model)
            auroc_no_TS = compute_auroc_for_temp(None)
            results['AUROC_no_TS'].append(auroc_no_TS)
            if TS_dict is not None:
                temp = TS_dict[dataset][model]
                auroc_with_TS = compute_auroc_for_temp(temp)
                results['AUROC_with_TS'].append(auroc_with_TS)

    return pd.DataFrame(results)

def load_standard_answers_and_correctness(dataset, model, split):
    if dataset not in ['TriviaQA', 'OpenNQ']:
        raise ValueError(f"Dataset {dataset} not recognized. Only 'TriviaQA' and 'OpenNQ' are supported.")

    #Get ground truth data
    if split not in ['validation', 'test']:
        raise ValueError("split must be 'validation' or 'test'")

    elif split == 'test':
        reference_data = load_data(dataset, model, 'test', embedding_model='all_mpnet_base_v2')
        dataset = dataset + '_2k'

    dataset_path = f'data/logs/{dataset}/standard_answers/{model}-answers.json'
    dataset_df = pd.read_json(dataset_path)

    if split == 'test': #If we want the test set, we take the 2k set and remove the overlapping points between it and the validation set
        reference_ids = set(reference_data['question_id'])
        dataset_df = dataset_df[dataset_df['question_id'].isin(reference_ids)].sort_values('question_id', ascending=True).reset_index(drop=True)
    return dataset_df

def ECE_conf_bin_against_correctness(confs, correctness, strategy='uniform', n_bins=15):
    """
    Compute Expected Calibration Error (ECE).

    Parameters
    ----------
    confs : array-like, shape (N,)
        Predicted confidences.
    correctness : array-like, shape (N,)
        Correctness labels (0 or 1).
    n_bins : int, optional (default=15)
        Number of bins for calibration.
    strategy : {"uniform", "quantile"}, optional (default="uniform")
        - "uniform": equal-width bins over [0,1].
        - "quantile": bins chosen by quantiles of confidence (adaptive binning).

    Returns
    -------
    ece : float
        Expected Calibration Error using L1 distance: sum_b ( (|acc_b - conf_b|) * (n_b / N) ).
    details : dict
        Useful per-bin details:
        {
            "bin_edges": np.ndarray, shape (B+1,),
            "bin_count": np.ndarray, shape (B,),
            "avg_conf": np.ndarray, shape (B,),
            "avg_acc": np.ndarray, shape (B,)
        }
    """
    N = confs.shape[0]
    
    # Build bin edges
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy in ("quantile", "adaptive"):
        # Quantile edges; ensure coverage of [0,1]
        q = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(confs, q)
        edges = np.unique(edges)  # guard against repeated edges if conf has ties
        # Guarantee exact bounds
        edges[0] = 0.0
        edges[-1] = 1.0
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile' (alias: 'adaptive').")

    # Assign each sample to a bin: digitize with right-closed bins except the last
    # np.digitize returns indices in [0, len(edges)-1]; we use edges[1:-1] as split points -> [0, B-1]
    bin_ids = np.digitize(confs, edges[1:-1], right=True)

    # Per-bin counts and sums
    B = len(edges) - 1
    bin_count = np.bincount(bin_ids, minlength=B)
    sum_conf = np.bincount(bin_ids, weights=confs, minlength=B)
    sum_acc = np.bincount(bin_ids, weights=correctness, minlength=B)
    nonzero = bin_count > 0
    avg_conf = np.zeros(B, dtype=float)
    avg_acc = np.zeros(B, dtype=float)
    avg_conf[nonzero] = sum_conf[nonzero] / bin_count[nonzero]
    avg_acc[nonzero] = sum_acc[nonzero] / bin_count[nonzero]
    weights = bin_count.astype(float) / float(N)
    ece = np.sum(weights * np.abs(avg_acc - avg_conf))
    return ece, {
        "bin_edges": edges,
        "bin_count": bin_count,
        "avg_conf": avg_conf, #Average model confidence within bin
        "avg_acc": avg_acc, #Average accuracy within bin
    }

def plot_rel_diag_against_correctness(
    dataset, model, split, confidence_measure, correctness_label, temp, plot_axis, legend, bin_type='uniform', 
    num_bins=15
):
    if correctness_label not in ['model_correctness', 'fuzzy_correctness']:
        raise ValueError("correctness_label must be 'model_correctness' or 'fuzzy_correctness'")
    dataframe = load_data(dataset, model, split)
    correctness_df = load_standard_answers_and_correctness(dataset, model, split)
    correctness_df = correctness_df[['question_id', correctness_label]]
    correctness_df = correctness_df.sort_values('question_id', ascending=True).reset_index(drop=True)
    df_evs_unc, n_answers_per_question = dataset_to_evs_uncertainties(dataframe, temp=temp)
    df_evs_unc = df_evs_unc.sort_values('question_id', ascending=True).reset_index(drop=True)
    correctness = correctness_df[correctness_label].to_numpy()
    if confidence_measure=='max_EV':
        confs = df_evs_unc['max_EV'].to_numpy()
    elif confidence_measure=='scaled_VNE':
        confs = 1-(df_evs_unc['VNE']/np.log(n_answers_per_question)).to_numpy()
    elif confidence_measure=='scaled_PKE':
        confs = (-df_evs_unc['PKE'])* n_answers_per_question
    else:
        raise ValueError("confidence_measure must be 'max_EV', 'scaled_VNE', or 'scaled_PKE'")

    ece_result = ECE_conf_bin_against_correctness(
        confs,  correctness, strategy=bin_type, n_bins=num_bins
    )
    counts = ece_result[1]['bin_count']
    confs = ece_result[1]['avg_conf']
    acc = ece_result[1]['avg_acc'] #todo change

    freq = counts/counts.sum()
    plot_axis.plot(confs, acc, marker='o', label="Accuracy", color='red')
    if bin_type=='uniform':
        plot_axis.plot(confs, freq, marker='o', label="Bin Freq.", color='orange')
    plot_axis.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfectly calibrated")
    plot_axis.set_xlabel("")
    plot_axis.set_ylabel("")
    plot_axis.grid(True)
    plot_axis.text(
        0.05, 0.95, f'ECE:{ece_result[0]:.2f}', transform=plot_axis.transAxes, ha="left", va="top",
    )
    if legend:
        plot_axis.legend(loc="center left", bbox_to_anchor=(1, 0.5))


def plot_all_rel_diag_against_correctness(
    datasets, models, split, confidence_measure, correctness_label, bin_type='uniform', 
    TS_dict=None, file_name=None, num_bins=15
):
    n_rows = len(datasets)
    n_cols = len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4))

    for dataset_id, dataset in enumerate(datasets):
        for model_id, model in enumerate(models):
            # only add legend to the last subfigure in the first row
            add_legend = (model_id==n_cols-1) and (dataset_id==0)
            if TS_dict is not None:
                temp = TS_dict[dataset][model]
            else:
                temp = None
                
            plot_rel_diag_against_correctness(
                dataset, model, split, confidence_measure, correctness_label, temp, axes[dataset_id][model_id], legend=add_legend,
                bin_type=bin_type, num_bins=num_bins
            )
            if dataset_id < 1:
                axes[dataset_id][model_id].set_title(model_names[model])
            if dataset_id==len(datasets)-1:
                axes[dataset_id][model_id].set_xlabel(f'Mean predicted {confidence_measure}')
            if model_id < 1:
                axes[dataset_id][model_id].set_ylabel(DATASET_TO_NAME[dataset])

    plt.tight_layout()
    if file_name is not None:
        plt.savefig('figures/'+file_name, dpi=300, bbox_inches="tight")
    plt.show();


def compute_correctness_ece_at_temp(dataset, model, split, temp_idx, correctness_label='fuzzy_correctness', bin_type='uniform', 
    num_bins=15):
    dataframe = load_data(dataset, model, split, baseline=True, baseline_temperature_index=temp_idx)
    correctness_df = load_standard_answers_and_correctness(dataset, model, split)
    correctness_df = correctness_df[['question_id', correctness_label]]
    correctness_df = correctness_df.sort_values('question_id', ascending=True).reset_index(drop=True)
    df_evs_unc, _ = dataset_to_evs_uncertainties(dataframe)
    df_evs_unc = df_evs_unc.sort_values('question_id', ascending=True).reset_index(drop=True)
    correctness = correctness_df[correctness_label].to_numpy()
    confs = df_evs_unc['max_EV'].to_numpy()
    ece_result = ECE_conf_bin_against_correctness(confs, correctness, strategy=bin_type, n_bins=num_bins)
    ece = ece_result[0]
    counts = ece_result[1]['bin_count']
    confs = ece_result[1]['avg_conf']
    acc = ece_result[1]['avg_acc'] 

    avg_conf = np.sum(confs * counts) / np.sum(counts)
    avg_acc = np.sum(acc * counts) / np.sum(counts)
    return avg_conf, avg_acc, ece


def plot_temperature_ECE_curve_baseline(temp_start, temp_stop, temp_num, dataset, model, split, correctness_label='fuzzy_correctness', bin_type='uniform', 
    num_bins=15, file_name=None, font_size=10, figsize=(10, 4)):
    temperatures = np.logspace(start=np.log10(temp_start), stop=np.log10(temp_stop), num=temp_num)
    avg_confs, eces = [], []
    for temp_idx, _ in enumerate(temperatures):
        avg_conf, _, ece = compute_correctness_ece_at_temp(
            dataset, model, split, temp_idx, correctness_label=correctness_label,
            bin_type=bin_type, num_bins=num_bins
        )
        avg_confs += [avg_conf]
        eces += [ece]
    fig = plt.figure(figsize=figsize)
    plt.plot(temperatures, eces, color='red', linewidth=2, label='ECE')

    plt.plot(temperatures, avg_confs, color='blue', linewidth=2, label='Average \n Confidence')

    plt.xscale("log")
    # plt.xticks(fontsize=font_size)
    # plt.yticks(fontsize=font_size)                    
    plt.tight_layout()
    if file_name is not None:
        plt.savefig('figures/'+file_name, dpi=300, bbox_inches="tight")
    plt.show();
    print(f'Minimum ECE is achieved for temperature index: {np.argmin(eces)}, with temperature: {temperatures[np.argmin(eces)]} and ECE: {min(eces)}')

def compute_risk_at_sampling_temp(dataset, model, split, temp_idx, ev_scaling_temp=1.0):
    df = load_data(dataset, model, split, baseline=True, baseline_temperature_index=temp_idx)
    df_wide = df.pivot(index='question_id', columns='answer_id', values='embedding')
    cols = df_wide.keys()
    n_uniques = np.unique(df['question_id']).shape[0]
    reps = df.shape[0] // n_uniques

    df_wide['emb_matrix'] = df_wide.apply(lambda row: np.vstack([row[c] for c in cols]), axis=1)
    df_wide['pred_matrix'] = df_wide.apply(lambda row: emb_to_cov_matrix(row['emb_matrix']), axis=1)
    df_wide['EV_decomp'] = df_wide.apply(lambda row: np.linalg.eigh(row['pred_matrix']), axis=1)
    
    target_embs = df['gt_embedding'].iloc[::reps]
    df_wide['target_embs'] = target_embs.reset_index(drop=True)

    df_wide['TS_EV_decomp'] = df_wide.apply(
        lambda row: (TS_EV(row['EV_decomp'][0], temp=ev_scaling_temp), row['EV_decomp'][1]),
        axis=1
    )
    df_wide['xent_loss'] = df_wide.apply(lambda row: cross_entropy_loss(row['TS_EV_decomp'], row['target_embs']), axis=1)
    risk = df_wide['xent_loss'].mean().item()

    df_wide['entropy'] = df_wide.apply(lambda row: entropy(row['TS_EV_decomp'][0]), axis=1)
    ent = df_wide['entropy'].mean().item()
    return risk, ent

def plot_temperature_risk_curve_baseline(temp_start, temp_stop, temp_num, dataset, model, split,
                                          file_name=None, font_size=10, figsize=(10, 4)):
    temperatures = np.logspace(start=np.log10(temp_start), stop=np.log10(temp_stop), num=temp_num)
    risks, ents = [], []
    for temp_idx, _ in tqdm(enumerate(temperatures)):
        risk, ent = compute_risk_at_sampling_temp(
            dataset, model, split, temp_idx
        )
        risks += [risk]
        ents += [ent]
    fig = plt.figure(figsize=figsize)
    plt.plot(temperatures, risks, color='red', linewidth=2, label='Risk')

    plt.plot(temperatures, ents, color='blue', linewidth=2, label='Entropy')

    plt.xscale("log")
    # plt.xticks(fontsize=font_size)
    # plt.yticks(fontsize=font_size)                    
    plt.tight_layout()
    if file_name is not None:
        plt.savefig('figures/'+file_name, dpi=300, bbox_inches="tight")
    plt.show();
    print(f'Minimum Risk is achieved for temperature index: {np.argmin(risks)}, with temperature: {temperatures[np.argmin(risks)]} and Risk: {min(risks)}')

def plot_temperature_risk_curve_baseline_as_function_of_ev_TS(dataset, model, split, sampling_temp_idx, 
                                                              ev_scaling_temps, file_name=None, font_size=10, figsize=(10, 4)):
    risks, ents = [], []
    for ev_scaling_temp in tqdm(ev_scaling_temps):
        risk, ent = compute_risk_at_sampling_temp(
            dataset, model, split, sampling_temp_idx, ev_scaling_temp
        )
        risks += [risk]
        ents += [ent]
    fig = plt.figure(figsize=figsize)
    plt.plot(ev_scaling_temps, risks, color='red', linewidth=2, label='Risk')

    plt.plot(ev_scaling_temps, ents, color='blue', linewidth=2, label='Entropy')

    plt.xscale("log")
    # plt.xticks(fontsize=font_size)
    # plt.yticks(fontsize=font_size)                    
    plt.tight_layout()
    if file_name is not None:
        plt.savefig('figures/'+file_name, dpi=300, bbox_inches="tight")
    plt.show();
    print(f'Minimum Risk is achieved for ev scaling temperature index: {np.argmin(risks)}, with temperature: {ev_scaling_temps[np.argmin(risks)]} and Risk: {min(risks)}')


def load_data_other_uncertainties(dataset, model, uncertainty_method):
    """
    Load uncertainty values and standard-answer correctness labels for a given
    (dataset, model, uncertainty_method) combination.

    Available datasets:    TriviaQA, OpenNQ, TriviaQA_2k, OpenNQ_2k
    Available models:      phi_4, phi_4_mini, llama4_maverick
    Available methods:     kle, se

    Returns a DataFrame with columns:
        question_id, <method>_total, <method>_confidence,
        generated_answer, fuzzy_correctness, model_correctness
    """
    unc_path = (
        f'data/logs/{dataset}/no_ensembling/no_model-variations/'
        f'{model}-{uncertainty_method}-uncertainties.json'
    )
    answers_path = f'data/logs/{dataset}/standard_answers/{model}-answers.json'
    unc_df = pd.read_json(unc_path)
    answers_df = pd.read_json(answers_path)
    return unc_df.merge(answers_df, on='question_id', how='left')


def auroc_experiments_other_uncertainties(datasets, models, uncertainty_methods, correctness_label, n_bootstrap):
    """
    Compute AUROC of -uncertainty as a predictor of correctness with bootstrap confidence bounds.

    Parameters
    ----------
    datasets            : list of str
    models              : list of str
    uncertainty_methods : list of str  ('kle' or 'se')
    correctness_label   : str          ('fuzzy_correctness' or 'model_correctness')
    n_bootstrap         : int

    Returns a DataFrame with columns:
        dataset, model, uncertainty_method, seed, auroc
    """
    combinations = [
        (d, m, u)
        for d in datasets
        for m in models
        for u in uncertainty_methods
    ]
    results = []
    with tqdm(total=len(combinations) * n_bootstrap) as progress_bar:
        for dataset, model, unc_method in combinations:
            df = load_data_other_uncertainties(dataset, model, unc_method)
            uncertainties = df[f'{unc_method}_total'].to_numpy()
            correctness = df[correctness_label].to_numpy()

            for seed in range(n_bootstrap):
                np.random.seed(seed)
                idx = np.random.choice(len(uncertainties), size=len(uncertainties), replace=True)
                auroc = roc_auc_score(correctness[idx], -uncertainties[idx])
                results.append({
                    'dataset': dataset,
                    'model': model,
                    'uncertainty_method': unc_method,
                    'seed': seed,
                    'auroc': auroc,
                })
                progress_bar.update(1)
    return pd.DataFrame(results)


def get_temp_indices(dataset, model, uncertainty_method):
    """Return sorted list of available temperature indices for a (dataset, model, method) combo."""
    d = f'data/baseline_logs/{dataset}/no_ensembling/no_model-variations'
    pattern = re.compile(
        rf'^{re.escape(model)}-{re.escape(uncertainty_method)}-uncertainties-t(\d+)\.json$'
    )
    return sorted(
        int(m.group(1)) for f in os.listdir(d) if (m := pattern.match(f))
    )


def load_baseline_data(dataset, model, uncertainty_method, temp_idx):
    """
    Load uncertainties at a specific sampling temperature index + standard answers.

    Returns a DataFrame with columns:
        question_id, <method>_total, <method>_confidence,
        generated_answer, fuzzy_correctness, model_correctness
    """
    unc_path = (
        f'data/baseline_logs/{dataset}/no_ensembling/no_model-variations/'
        f'{model}-{uncertainty_method}-uncertainties-t{temp_idx}.json'
    )
    answers_path = f'data/logs/{dataset}/standard_answers/{model}-answers.json'
    unc_df = pd.read_json(unc_path)
    answers_df = pd.read_json(answers_path)
    return unc_df.merge(answers_df, on='question_id', how='left')


def compute_ece(confidence, correctness, n_bins=15):
    """
    Equal-width Expected Calibration Error for scalar confidence values in [0, 1].

    Parameters
    ----------
    confidence  : 1-D array of predicted confidence values
    correctness : 1-D array of binary correctness labels (bool or 0/1)
    n_bins      : number of equal-width bins

    Returns
    -------
    ece : float
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidence)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(confidence[mask].mean() - correctness[mask].mean())
    return ece




def plot_ece_vs_temperature(
    datasets, models, uncertainty_methods, correctness_label, n_bins=15
):
    """
    For every available (dataset, model, uncertainty_method) combination, compute ECE
    at each sampling temperature index and plot the curves.

    Subplots are arranged as: rows = uncertainty_methods, cols = models.
    One figure is produced per dataset.
    A red dashed vertical line marks the temperature that minimises ECE.

    Parameters
    ----------
    datasets            : list of str
    models              : list of str
    uncertainty_methods : list of str  ('kle' or 'se')
    correctness_label   : str          ('fuzzy_correctness' or 'model_correctness')
    n_bins              : int  (equal-width ECE bins)

    Returns
    -------
    optimal_temps : dict  {(dataset, model, method): temp_index}
    """
    optimal_temps = {}

    for dataset in datasets:
        baseline_dir = f'data/baseline_logs/{dataset}/no_ensembling/no_model-variations'
        if not os.path.isdir(baseline_dir):
            print(f'No baseline data for {dataset}, skipping.')
            continue

        n_rows = len(uncertainty_methods)
        n_cols = len(models)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False
        )
        fig.suptitle(
            f'{dataset}  —  ECE vs Sampling Temperature  ({correctness_label})',
            fontsize=13, y=1.01
        )

        for row, method in enumerate(uncertainty_methods):
            for col, model in enumerate(models):
                ax = axes[row][col]
                temp_indices = get_temp_indices(dataset, model, method)

                if not temp_indices:
                    ax.set_visible(False)
                    continue

                eces = []
                for t_idx in temp_indices:
                    df = load_baseline_data(dataset, model, method, t_idx)
                    conf = df[f'{method}_confidence'].to_numpy(dtype=float)
                    corr = df[correctness_label].to_numpy(dtype=float)
                    eces.append(compute_ece(conf, corr, n_bins=n_bins))

                temp_values = ALL_TEMPS[temp_indices]
                best_pos = int(np.argmin(eces))
                best_idx = temp_indices[best_pos]
                optimal_temps[(dataset, model, method)] = best_idx

                ax.plot(temp_values, eces, marker='o', linewidth=2)
                ax.axvline(
                    temp_values[best_pos], color='red', linestyle='--', linewidth=1.5,
                    label=f'min ECE  t={best_idx}  ({temp_values[best_pos]:.3f})'
                )
                ax.set_title(f'{model}  /  {method}', fontsize=11)
                ax.set_xlabel('Sampling temperature', fontsize=10)
                ax.set_ylabel('ECE', fontsize=10)
                ax.legend(fontsize=8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()

    print('\nOptimal temperature indices (minimising ECE):')
    print(f'  {"dataset":15s}  {"model":20s}  {"method"}  ->  index   temperature')
    for (dataset, model, method), best_idx in optimal_temps.items():
        print(
            f'  {dataset:15s}  {model:20s}  {method}    ->  '
            f'idx={best_idx:2d}   T={ALL_TEMPS[best_idx]:.4f}'
        )

    return optimal_temps


def ECE_conf_bin_alt_with_dX_similarities(
    dataset,
    model,
    confs,
    pred_covs,
    target_embs,
    n_bins: int = 15,
    num_clusters: int = 10,
    min_n = 5,
    cluster_func=get_clusters
):
    """
    Compute a variant of Expected Calibration Error (ECE) that replaces the usual
    per-bin accuracy with a per-bin diversity statistic derived from target
    embeddings: within each confidence bin, samples are clustered by the
    similarity of their predicted covariance matrices, and for each sufficiently
    large cluster the largest eigenvalue of the (normalised) target embeddings'
    Gram matrix is computed and averaged across clusters.

    Parameters
    ----------
    dataset : str
        Name of the dataset being processed; used only for logging.
    model : str
        Name of the model being processed; used only for logging.
    confs : array-like, shape (N,)
        Predicted confidence scores in [0, 1].
    pred_covs : array-like, shape (N, ...)
        Predicted covariance matrices per sample, used to build a pairwise
        similarity matrix (via `pairwise_matrix_cos`) for clustering within each bin.
    target_embs : array-like, shape (N, D)
        Target embeddings; rows are L2-normalised before use.
    n_bins : int, optional (default=15)
        Number of quantile bins for confidence.
    num_clusters : int, optional (default=10)
        Number of clusters to form within each bin via `cluster_func`.
    min_n : int, optional (default=5)
        Minimum number of samples a cluster must have to be included in the
        per-bin aggregation; smaller clusters are ignored.
    cluster_func : callable, optional (default=get_clusters)
        Function used to cluster samples within a bin given the pairwise
        covariance-similarity matrix and (num_clusters, min_n).

    Returns
    -------
    ece : float
        Weighted absolute difference between per-bin average confidence and
        per-bin max-eigenvalue statistic: sum_b ( |max_EV_b - conf_b| * (n_b / N) ).
    details : dict
        Useful per-bin details:
        {
            "bin_edges": np.ndarray, shape (B+1,),
            "bin_count": np.ndarray, shape (B,),
            "avg_conf": np.ndarray, shape (B,), average confidence within bin,
            "max_EVs": np.ndarray, shape (B,), average largest Gram-matrix
                eigenvalue across qualifying clusters within bin,
            "within_cluster_pairwise_sims": np.ndarray, shape (B,), average
                within-cluster pairwise covariance similarity within bin,
            "within_bin_pairwise_sims": list, length B, average pairwise
                covariance similarity across all samples within bin,
        }
    """
    print("Dataset:", dataset, "Model:", model)
    target_embs = normalise_rows(target_embs)
    N = confs.shape[0]
    
    # Quantile edges; ensure coverage of [0,1]
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(confs, q)
    edges = np.unique(edges)  # guard against repeated edges if conf has ties
    # Guarantee exact bounds
    edges[0] = 0.0
    edges[-1] = 1.0

    # Assign each sample to a bin: digitize with right-closed bins except the last
    # np.digitize returns indices in [0, len(edges)-1]; we use edges[1:-1] as split points -> [0, B-1]
    bin_ids = np.digitize(confs, edges[1:-1], right=True)

    # Per-bin counts and sums
    B = len(edges) - 1
    bin_count = np.bincount(bin_ids, minlength=B)
    sum_conf = np.bincount(bin_ids, weights=confs, minlength=B)
    max_EVs = np.zeros((B, 2))
    within_bin_pairwise_sims = [None] * B
    within_cluster_pairwise_sims = np.zeros((B, 2)) #Running sum of within-cluster pairwise sims and count of clusters considered within bin
    for bin_id in np.unique(bin_ids): #Bin then cluster
        bin_covs = pred_covs[bin_ids==bin_id]
        bin_confs = confs[bin_ids==bin_id]
        bin_targets = target_embs[bin_ids==bin_id]
        cov_corr_matrix = pairwise_matrix_cos(bin_covs)
        cov_sim_upper_triangle = cov_corr_matrix[np.triu_indices_from(cov_corr_matrix, k=1)]
        within_bin_pairwise_sims[bin_id] = cov_sim_upper_triangle.mean()
        clusters = cluster_func(cov_corr_matrix, num_clusters, min_n)
        for cluster_id in np.unique(clusters):
            if np.sum(clusters==cluster_id) >= min_n: #If cluster does not have enough instances, ignore
                selected_targets = bin_targets[clusters==cluster_id]
                eigenvals, _ = spectral_decomp_gram_matrix(selected_targets)
                max_EVs[bin_id, 0] += eigenvals[-1].item() #Running sum of max EVs
                max_EVs[bin_id, 1] += 1 #Running number of clusters considered within bin
                cluster_covs = bin_covs[clusters==cluster_id]
                cluster_corr_matrix = pairwise_matrix_cos(cluster_covs)
                cluster_corr_upper_triangle = cluster_corr_matrix[np.triu_indices_from(cluster_corr_matrix, k=1)]
                within_cluster_pairwise_sim = cluster_corr_upper_triangle.mean()

                within_cluster_pairwise_sims[bin_id, 0] += within_cluster_pairwise_sim
                within_cluster_pairwise_sims[bin_id, 1] += 1 #Running count of clusters considered within bin

    # avoid div by 0
    max_EVs = max_EVs[:,0] / np.maximum(max_EVs[:,1], 1) #Bins with no clusters considered will have max_EV=0 (this usually doesn't happen with quantile binning, but depends also on the clustering algorithm)
    within_cluster_pairwise_sims = within_cluster_pairwise_sims[:,0] / np.maximum(within_cluster_pairwise_sims[:,1], 1) #Bins with no clusters considered will have within-cluster pairwise sim=0
    nonzero = bin_count > 0
    avg_conf = np.zeros(B, dtype=float)
    avg_conf[nonzero] = sum_conf[nonzero] / bin_count[nonzero] #Average model confidence within bin
    
    weights = bin_count.astype(float) / float(N)
    ece = np.sum(weights * np.abs(max_EVs - avg_conf))

    return ece, {
        "bin_edges": edges,
        "bin_count": bin_count,
        "avg_conf": avg_conf, #Average model confidence within bin
        "max_EVs": max_EVs, #Target max EV within bin
        "within_cluster_pairwise_sims": within_cluster_pairwise_sims, #Average within-cluster pairwise similarity within bin
        "within_bin_pairwise_sims": within_bin_pairwise_sims, #Average within-bin pairwise similarity within bin
    }


def compute_average_pairwise_dX_similarities(datasets, models):
    """
    For each (dataset, model) pair, run `ECE_conf_bin_alt_with_dX_similarities`
    on the test split and summarise the resulting within-cluster and within-bin
    pairwise covariance-similarity statistics into a single average per pair.

    Parameters
    ----------
    datasets : iterable of str
        Dataset names to load via `load_data`.
    models : iterable of str
        Model names to load via `load_data`.

    Returns
    -------
    similarities_df : pd.DataFrame
        One row per (dataset, model) with columns:
        {
            "dataset": str,
            "model": str,
            "within_cluster_pairwise_sim": float, mean of per-bin
                within-cluster pairwise covariance similarity,
            "within_bin_pairwise_sim": float, mean of per-bin
                within-bin pairwise covariance similarity,
        }
    """
    dfs_dict = dict()
    similarities = []
    for dataset in datasets:
        dfs_dict[dataset] = dict()
        for model in models:
            dataframe = load_data(dataset, model, split='test')
            confs, target_embs = dataset_to_confs_targets(dataframe)
            cov_mats = dataset_to_cov_matrix(dataframe)
            ece_result = ECE_conf_bin_alt_with_dX_similarities(
                dataset, model, confs, cov_mats, target_embs, min_n=2, num_clusters=5, n_bins=8,
                cluster_func=get_clusters
            )
            print(f"Dataset: {dataset}, Model: {model}, ECE: {ece_result[0]:.4f}")
            ece_result[1].pop('bin_edges', None)
            dfs_dict[dataset][model] = pd.DataFrame(ece_result[1])

            similarities.append({
                'dataset': dataset,
                'model': model,
                'within_cluster_pairwise_sim': dfs_dict[dataset][model]["within_cluster_pairwise_sims"].mean(),
                'within_bin_pairwise_sim': dfs_dict[dataset][model]["within_bin_pairwise_sims"].mean(),
            })
    similarities_df = pd.DataFrame(similarities)
    return similarities_df


def log_score(M, eps=1e-10):
    eigenvals, eigenvecs = np.linalg.eigh(M)
    eigenvals = np.maximum(eigenvals, eps)
    log_matrix = - eigenvecs @ np.diag(np.log(eigenvals)) @ eigenvecs.T
    return log_matrix

def divergence_function(M, N, score_function=log_score):
    score_M = score_function(M)
    score_N = score_function(N)
    return np.trace(score_N @ M - score_M @ M)

def expected_matrix_calibration_error(scaled_pred_matrices, target_embs, num_clusters = 30, cluster_func=get_clusters):
    cov_corr_matrix = pairwise_matrix_cos(scaled_pred_matrices)
    clusters = cluster_func(cov_corr_matrix, num_clusters)
    unique_clusters = np.unique(clusters)
    cluster_scores = []
    for cluster_id in unique_clusters:
        cluster_targets = target_embs[clusters == cluster_id]
        cluster_pred_matrices = scaled_pred_matrices[clusters == cluster_id]
        cluster_avg_pred_matrix = cluster_pred_matrices.mean(axis=0)
        cluster_target_cov = cluster_targets.T @ cluster_targets / cluster_targets.shape[0]
        score = divergence_function(cluster_target_cov, cluster_avg_pred_matrix)
        cluster_scores.append(score)
    return np.mean(cluster_scores)



def _compute_one_temp_risks_and_eces(pred_matrices, ev_decomps, target_embs, temp):
    ts_ev_decomps = [(TS_EV(ev[0], temp=temp), ev[1]) for ev in ev_decomps]
    xent_losses = np.array([cross_entropy_loss(ts_ev, t_emb) for ts_ev, t_emb in zip(ts_ev_decomps, target_embs)])
    confs = np.array([ts_ev[0][-1].item() for ts_ev in ts_ev_decomps])
    ece, _ = ECE_conf_bin_alt(confs, pred_matrices, target_embs, n_bins=8, num_clusters=5, min_n = 2, cluster_func=get_clusters)
    scaled_pred_matrices = np.array([ts_ev[1] @ np.diag(ts_ev[0]) @ ts_ev[1].T for ts_ev in ts_ev_decomps])
    expected_matrix_calibration_error_value = expected_matrix_calibration_error(scaled_pred_matrices, target_embs)
    return {
        'risk': xent_losses.mean(),
        'risk_std': xent_losses.std(),
        'ece': ece,
        'expected_matrix_calibration_error': expected_matrix_calibration_error_value
    }


def dataset_to_risks_and_eces(dataset, temps, progress_bar=None):
    df_wide = dataset.pivot(index='question_id', columns='answer_id', values='embedding')
    cols = df_wide.keys()
    n_uniques = np.unique(dataset['question_id']).shape[0]
    reps = dataset.shape[0] // n_uniques

    df_wide['emb_matrix'] = df_wide.apply(lambda row: np.vstack([row[c] for c in cols]), axis=1)
    df_wide['pred_matrix'] = df_wide.apply(lambda row: emb_to_cov_matrix(row['emb_matrix']), axis=1)
    df_wide['EV_decomp'] = df_wide.apply(lambda row: np.linalg.eigh(row['pred_matrix']), axis=1)

    target_embs = dataset['gt_embedding'].iloc[::reps].reset_index(drop=True).tolist()
    df_wide['target_embs'] = target_embs
    ev_decomps = df_wide['EV_decomp'].tolist()
    pred_matrices = df_wide['pred_matrix'].tolist()

    results = {
        'risks': [None] * len(temps),
        'risks_stds': [None] * len(temps),
        'ece_values': [None] * len(temps),
        'expected_matrix_calibration_errors': [None] * len(temps),
    }
    for i, temp in enumerate(temps):
        r = _compute_one_temp_risks_and_eces(np.array(pred_matrices), ev_decomps, np.array(target_embs), temp)
        results['risks'][i] = r['risk']
        results['risks_stds'][i] = r['risk_std']
        results['ece_values'][i] = r['ece']
        results['expected_matrix_calibration_errors'][i] = r['expected_matrix_calibration_error']
        if progress_bar is not None:
            progress_bar.update(1)
    return results


def _compute_TS_pair_risks_and_eces(dataset, model, temps, embedding_model):
    dataframe = load_data(dataset, model, split='validation', embedding_model=embedding_model)
    risks_dict = dataset_to_risks_and_eces(dataframe, temps, progress_bar=None)
    risks_dict['temps'] = temps
    risks_dict['model'] = model
    risks_dict['dataset'] = dataset
    return risks_dict


def compute_TS_curves_risks_and_eces(datasets, models, temps, embedding_model='all_mpnet_base_v2', save_results=None):
    pairs = [(dataset, model) for dataset in datasets for model in models]
    results = []

    n_workers = len(pairs)
    blas_threads = max(1, (os.cpu_count() or 1) // n_workers)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(blas_threads,)) as executor:
        futures = {
            executor.submit(_compute_TS_pair_risks_and_eces, dataset, model, temps, embedding_model): (dataset, model)
            for dataset, model in pairs
        }
        with tqdm(total=len(pairs), desc='dataset/model pairs') as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)

    if save_results:
        pd.DataFrame(results).to_json(save_results, orient='records')
    return pd.DataFrame(results)


def plot_risk_ece_curves(results, file_name=None):
    import textwrap

    datasets = ['TriviaQA', 'OpenNQ']
    models = ['phi_4_mini', 'phi_4', 'llama4_maverick']
    model_names = {'phi_4': 'Phi 4', 'phi_4_mini': 'Phi 4 Mini', 'llama4_maverick': 'Llama4 Maverick'}
    dataset_names = {'TriviaQA': 'TriviaQA', 'OpenNQ': 'Natural Questions'}

    fontsize = 14
    legend_wrap_width = 14

    # Convert results to dataframe if it isn't already
    if not isinstance(results, pd.DataFrame):
        results = pd.DataFrame(results)

    def minmax(v):
        a = np.array([float(x) for x in v])
        return (a - a.min()) / (a.max() - a.min())

    fig, axes = plt.subplots(2, 3, figsize=(13, 6))

    for row, dataset in enumerate(datasets):
        for col, model in enumerate(models):
            ax = axes[row, col]
            entry = results[(results['dataset'] == dataset) & (results['model'] == model)].iloc[0].to_dict() if len(results[(results['dataset'] == dataset) & (results['model'] == model)]) > 0 else None
            if entry is None:
                ax.set_visible(False)
                continue

            temps = entry['temps']
            ax.plot(temps, minmax(entry['risks']), label='Risk', marker='o', markersize=3, linewidth=1.5)
            ax.plot(temps, minmax(entry['ece_values']), label='Eigenvalue ECE (Algorithm 1)', marker='s', markersize=3, linewidth=1.5)
            ax.plot(temps, minmax(entry['expected_matrix_calibration_errors']), label='Matrix calib. error (Eq 22)', marker='^', markersize=3, linewidth=1.5)

            ax.set_xscale('log')
            ax.minorticks_off()
            ax.set_xticks([0.5, 1.0, 2.0, 4.0], [0.5, 1.0, 2.0, 4.0])
            ax.tick_params(axis='both', labelsize=fontsize)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if row == 0:
                ax.set_title(model_names[model], fontsize=fontsize)
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Temperature', fontsize=fontsize)

            if col == 0:
                ax.set_ylabel(f'{dataset_names[dataset]}\n(min-max scaled)', fontsize=fontsize)

            if row == 0 and col == 2:
                handles, labels = ax.get_legend_handles_labels()
                wrapped_labels = ['\n'.join(textwrap.wrap(label, legend_wrap_width)) for label in labels]
                ax.legend(handles, wrapped_labels, bbox_to_anchor=(1.02, 1), loc='upper left',
                          frameon=False, fontsize=fontsize, labelspacing=1.5, handletextpad=0.5)

    plt.tight_layout()
    if file_name is not None:
        plt.savefig('figures/'+file_name, dpi=300, bbox_inches="tight")
    plt.show()
