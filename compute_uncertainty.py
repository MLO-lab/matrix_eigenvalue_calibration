from tqdm import tqdm
from src.uncertainty_metrics.common import compute_discrete_semantic_entropy, compute_discrete_semantic_entropy_and_confidence, compute_kernel_language_entropy, compute_kernel_language_entropy_and_confidence, compute_von_neuman_entropy, compute_predictive_kernel_entropy
from sentence_transformers import SentenceTransformer
from src.uncertainty_metrics.semantic_entropy import EntailmentDeberta
from src.uncertainty_metrics.kernel_language_entropy import scale_entropy
import torch, yaml, argparse, os, logging, json
import pandas as pd
import numpy as np
from typing import List
from src.util.misc import setup_output_and_logging
logger = logging.getLogger(__name__)


UNCERTAINTY_METRICS = {"discrete_semantic_entropy": compute_discrete_semantic_entropy,
                        "kernel_language_entropy": compute_kernel_language_entropy, 
                       "predictive_kernel_entropy": compute_predictive_kernel_entropy, 
                       "von_neuman_entropy": compute_von_neuman_entropy,
                       "se": compute_discrete_semantic_entropy_and_confidence,
                       "kle": compute_kernel_language_entropy_and_confidence,}

def compute_uncertainty_values(input_df: pd.DataFrame, metric: str, config: dict) -> List[dict]:
    assert metric in UNCERTAINTY_METRICS, f"Uncertainty metric should be one of {UNCERTAINTY_METRICS.keys()}"
    print(f"Computing uncertainty metric: {metric}")
    compute_function = UNCERTAINTY_METRICS[metric]
    if metric in ["discrete_semantic_entropy", "kernel_language_entropy", "se", "kle" ]:
        model = EntailmentDeberta()
    else:
        model = SentenceTransformer(config['embedding_model']['model_name'],
                                    cache_folder= config['embedding_model']['cache_folder'],
                                    device = torch.device(config['embedding_model']['device']))
    output_data = []
    question_ids = np.unique(input_df["question_id"]) #Sorted unique values
    for question_id in tqdm(question_ids):
        question_df = input_df[input_df["question_id"] == question_id]
        #Both scaled and unscaled
        answers = list(question_df["generated_answer"])
        uncertainty , confidence = compute_function(answers, model)
        output_data.append({"question_id": int(question_id),
                            f"{metric}_total": uncertainty,
                            f"{metric}_confidence": confidence})

    return output_data

# First parse config
if __name__=='__main__':
    setup_output_and_logging('experiment_logs/compute_uncertainty')
    parser = argparse.ArgumentParser(description="Compute uncertainty script.")
    parser.add_argument(
        "--config_path", 
        type=str, 
        required=True, 
        help="Path to the config file")
    args = parser.parse_args()
    logger.info(f"Using config file: {args.config_path}")
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"Using config:\n{yaml.dump(config)}")
    root_path = os.path.dirname(__file__)
    variation_model_name=config.get('variation_model', {'model_safename': 'no_model'})['model_safename']
    sampling_temperature_index =config.get('sampling_temperature_index') #TODO add in config
    target_model_name=config['target_model']['model_safename']
    if sampling_temperature_index is not None:
        input_directory = os.path.join(root_path, 
                                    f"data/baseline_logs/{config['dataset']}/{config['ensembling_method']}/{variation_model_name}-variations")
        input_path = os.path.join(input_directory, f'{target_model_name}-answers-t{sampling_temperature_index}.json')
    else:
        input_directory = os.path.join(root_path, 
                                    f"data/logs/{config['dataset']}/{config['ensembling_method']}/{variation_model_name}-variations")
        input_path = os.path.join(input_directory, f'{target_model_name}-answers.json')
    
    
    input_df = pd.read_json(input_path)
    metric = config["uncertainty_metric"]
    output_data = compute_uncertainty_values(input_df, metric, config)
    output_directory = input_directory
    if sampling_temperature_index is not None:
        with open(os.path.join(output_directory, f'{target_model_name}-{metric}-uncertainties-t{sampling_temperature_index}.json'), 'w') as file:
            json.dump(output_data, file, indent=4)
    else:
        with open(os.path.join(output_directory, f'{target_model_name}-{metric}-uncertainties.json'), 'w') as file:
            json.dump(output_data, file, indent=4)

