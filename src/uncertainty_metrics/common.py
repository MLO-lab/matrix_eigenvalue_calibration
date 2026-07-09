import scipy.linalg
from .semantic_entropy import cluster_assignment_entropy, cluster_assignment_entropy_and_confidence, get_semantic_ids, BaseEntailment
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import numpy as np
import sklearn.metrics
import scipy
from .kernel_language_entropy import vn_entropy, heat_kernel, get_entailment_graph, vn_entropy_and_confidence

def compute_discrete_semantic_entropy(texts: List[str], model: BaseEntailment) -> float:
    semantic_ids = get_semantic_ids(texts, model, strict_entailment=True)
    return float(cluster_assignment_entropy(semantic_ids))

def compute_discrete_semantic_entropy_and_confidence(texts: List[str], model: BaseEntailment) -> Tuple[float, float]:
    semantic_ids = get_semantic_ids(texts, model, strict_entailment=True)
    entropy, confidence = cluster_assignment_entropy_and_confidence(semantic_ids)
    return float(entropy), float(confidence)

def compute_kernel_language_entropy(texts: List[str], model: BaseEntailment, kernel: str= "heat") -> float:
    weighted_graph = get_entailment_graph(texts, model, is_weighted=True)
    if kernel=="heat":
        return float(vn_entropy(heat_kernel(weighted_graph), scale = False))
    else:
        raise NotImplementedError("Only heat kernel supported for now.")
def compute_kernel_language_entropy_and_confidence(texts: List[str], model: BaseEntailment, kernel: str= "heat") -> Tuple[float, float]:
    weighted_graph = get_entailment_graph(texts, model, is_weighted=True)
    if kernel=="heat":
        entropy, confidence = vn_entropy_and_confidence(heat_kernel(weighted_graph), scale = False)
        return float(entropy), float(confidence)
    else:
        raise NotImplementedError("Only heat kernel supported for now.")

def compute_predictive_kernel_entropy(texts: List[str], model: SentenceTransformer, kernel: str= "rbf") -> float:
    if kernel=="rbf":
        kernel_function = lambda x: sklearn.metrics.pairwise.rbf_kernel(x, gamma = 1.0)
    elif kernel == "laplacian":
        kernel_function = lambda x: sklearn.metrics.pairwise.laplacian_kernel(x, gamma = 1.0)
    elif kernel == "cosine":
        kernel_function = sklearn.metrics.pairwise.linear_kernel

    embeddings = model.encode(texts, batch_size=256, convert_to_numpy = True, convert_to_tensor =False)
    kernel_matrix = kernel_function(embeddings)
    n = len(texts)
    return float((kernel_matrix.diagonal().sum() - kernel_matrix.sum())/(n*(n-1)))

def compute_von_neuman_entropy(texts: List[str], model: SentenceTransformer, kernel: str= "rbf") -> float:
    if kernel=="rbf":
        kernel_function = lambda x: sklearn.metrics.pairwise.rbf_kernel(x, gamma = 1.0)
    elif kernel == "laplacian":
        kernel_function = lambda x: sklearn.metrics.pairwise.laplacian_kernel(x, gamma = 1.0)
    elif kernel == "cosine":
        kernel_function = sklearn.metrics.pairwise.linear_kernel
    
    embeddings = model.encode(texts, batch_size=256, convert_to_numpy = True, convert_to_tensor =False)
    kernel_matrix = kernel_function(embeddings)
    kernel_matrix /= len(texts)
    try:
        eigenvalues, _ = scipy.linalg.eigh(kernel_matrix)
    except np.linalg.LinAlgError:
        print(f"Linalg error happened because of condition {np.linalg.cond(kernel_matrix)}. Trying numpy function.")
        try:
            eigenvalues, _ = np.linalg.eigh(kernel_matrix)
        except np.linalg.LinAlgError:
            epsilon = 1e-6  # You can tune this
            kernel_matrix += epsilon * np.eye(kernel_matrix.shape[0])
            print(f"Numy function failed. Adding {epsilon} to the diagonal. New condition is {np.linalg.cond(kernel_matrix)}")
            eigenvalues, _ = scipy.linalg.eigh(kernel_matrix)
    #Due to numerical issues, sometimes negative eigenvalues appear (although theoretically impossible because PSD). These are very marginal and are clipped to 0.
    eigenvalues = np.array([(ev if ev>=0 else 0.0) for ev in eigenvalues])
    return float(scipy.stats.entropy(eigenvalues))


def compute_predictive_kernel_entropy_no_model(embeddings: List[np.ndarray], kernel: str= "rbf") -> float:
    if kernel=="rbf":
        kernel_function = lambda x: sklearn.metrics.pairwise.rbf_kernel(x, gamma = 1.0)
    elif kernel == "laplacian":
        kernel_function = lambda x: sklearn.metrics.pairwise.laplacian_kernel(x, gamma = 1.0)
    elif kernel == "cosine":
        kernel_function = sklearn.metrics.pairwise.linear_kernel
    kernel_matrix = kernel_function(embeddings)
    n = len(embeddings)
    return float((kernel_matrix.diagonal().sum() - kernel_matrix.sum())/(n*(n-1)))

def compute_von_neuman_entropy_no_model(embeddings: List[np.ndarray], kernel: str= "rbf") -> float:
    if kernel=="rbf":
        kernel_function = lambda x: sklearn.metrics.pairwise.rbf_kernel(x, gamma = 1.0)
    elif kernel == "laplacian":
        kernel_function = lambda x: sklearn.metrics.pairwise.laplacian_kernel(x, gamma = 1.0)
    elif kernel == "cosine":
        kernel_function = sklearn.metrics.pairwise.linear_kernel
    kernel_matrix = kernel_function(embeddings)
    kernel_matrix /= len(embeddings)
    try:
        eigenvalues, _ = scipy.linalg.eigh(kernel_matrix)
    except np.linalg.LinAlgError:
        print(f"Linalg error happened because of condition {np.linalg.cond(kernel_matrix)}. Trying numpy function.")
        try:
            eigenvalues, _ = np.linalg.eigh(kernel_matrix)
        except np.linalg.LinAlgError:
            epsilon = 1e-6  # You can tune this
            kernel_matrix += epsilon * np.eye(kernel_matrix.shape[0])
            print(f"Numy function failed. Adding {epsilon} to the diagonal. New condition is {np.linalg.cond(kernel_matrix)}")
            eigenvalues, _ = scipy.linalg.eigh(kernel_matrix)
    #Due to numerical issues, sometimes negative eigenvalues appear (although theoretically impossible because PSD). These are very marginal and are clipped to 0.
    eigenvalues = np.array([(ev if ev>=0 else 0.0) for ev in eigenvalues])
    return float(scipy.stats.entropy(eigenvalues))


def compute_cluster_average_entropy(texts: List[List[str]], model, entropy_function, **params) -> float:
    per_cluster_entropy = [entropy_function(cluster_texts, model, **params) for cluster_texts in texts]
    return float(np.mean(per_cluster_entropy))
    
def compute_disagreement_measure(texts: List[List[str]], model, entropy_function, **params) -> Tuple[float, float, float]:
    cluster_average_entropy = compute_cluster_average_entropy(texts, model, entropy_function, **params)
    flattened_texts = [text for cluster_texts in texts for text in cluster_texts]
    total_entropy = entropy_function(flattened_texts, model, **params)
    disagreement_measure = total_entropy - cluster_average_entropy
    return total_entropy, cluster_average_entropy, disagreement_measure

def compute_cluster_average_entropy_no_model(texts: List[List[str]], entropy_function, **params) -> float:
    per_cluster_entropy = [entropy_function(cluster_texts, **params) for cluster_texts in texts]
    return float(np.mean(per_cluster_entropy))
    
def compute_disagreement_measure_no_model(texts: List[List[str]], entropy_function, **params) -> Tuple[float, float, float]:
    cluster_average_entropy = compute_cluster_average_entropy_no_model(texts, entropy_function, **params)
    flattened_texts = [text for cluster_texts in texts for text in cluster_texts]
    total_entropy = entropy_function(flattened_texts, **params)
    disagreement_measure = total_entropy - cluster_average_entropy
    return total_entropy, cluster_average_entropy, disagreement_measure