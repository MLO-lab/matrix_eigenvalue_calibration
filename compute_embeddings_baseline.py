from tqdm import tqdm
import pickle
from sentence_transformers import SentenceTransformer
from src.util.embeddings_util import embedding_factory
import torch, yaml, argparse, os, logging, json
import pandas as pd
import numpy as np
from typing import List
from src.util.misc import setup_output_and_logging
import gzip
logger = logging.getLogger(__name__)



def compute_embeddings(input_df: pd.DataFrame, config: dict) -> List[np.ndarray]:
    embedding_model = embedding_factory(config['embedding_model'])
    answers = list(input_df["generated_answer"])
    embeddings = []
    print("Generating embeddings..")
    for i in tqdm(range(0, len(answers), config['embedding_model']['batch_size'])):
        batch = answers[i:i+config['embedding_model']['batch_size']]
        batch_embeddings = embedding_model.encode(batch, config['embedding_model']['encoding_arguments'])
        embeddings = embeddings + list(batch_embeddings)


    return embeddings



# First parse config
if __name__=='__main__':
    setup_output_and_logging('experiment_logs/compute_embeddings')
    parser = argparse.ArgumentParser(description="Compute embeddings script.")
    parser.add_argument(
        "--config_path", 
        type=str, 
        required=True, 
        help="Path to the config file")
    parser.add_argument(
        "--temperature", 
        type=int, 
        required=True, 
        help="Temperature index for sampling")
    args = parser.parse_args()
    logger.info(f"Using config file: {args.config_path}")
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"Using config:\n{yaml.dump(config)}")
    temperatures = np.logspace(start=np.log10(config['temperature_values']['start']), 
                               stop=np.log10(config['temperature_values']['stop']), 
                               num=config['temperature_values']['num_temperatures'])
    assert args.temperature < len(temperatures) and args.temperature >= 0, "Temperature index out of range"
    temperature = temperatures[args.temperature]
    logger.info(f"Using temperature: {temperature}")
    root_path = os.path.dirname(__file__)
    variation_model_name=config.get('variation_model', {'model_safename': 'no_model'})['model_safename']
    target_model_name=config['target_model']['model_safename']
    input_directory = os.path.join(root_path, 
                                        f"data/baseline_logs/{config['dataset']}/{config['ensembling_method']}/{variation_model_name}-variations")
    input_path = os.path.join(input_directory, f'{target_model_name}-answers-t{args.temperature}.json')
    input_df = pd.read_json(input_path)

    #First we compute the embeddings
    embeddings = compute_embeddings(input_df, config)
    input_df["embedding"] = embeddings
    output_directory = input_directory
    embedding_model_safename = config['embedding_model']['model_safename']

    with gzip.open(os.path.join(output_directory, f'{target_model_name}-embeddings-t{args.temperature}-{embedding_model_safename}.pkl.gz'), "wb") as f:
        pickle.dump(input_df, f)

