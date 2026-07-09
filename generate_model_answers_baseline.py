from src.util.misc import setup_output_and_logging, set_random_seed
import logging
import yaml
import argparse
import pandas as pd
import numpy as np
import os, json
from typing import List
from src.util.model_util import model_factory
import src.prompt_generation.answer_generation as answer_generation_util
logger = logging.getLogger(__name__)

EXTRA_COLUMNS_PER_DATASET = {"AmbigQA": [],
                                "AmbigInst": ["input"],
                                "AmbigInst2": ["input"],
                                "Mentat": ["situation", "answer_a", "answer_b", "answer_c", "answer_d", "answer_e"], 
                                "test": [],
                                "CoQA2": ["story", "history"],
                                "TriviaQA": [],
                                "OpenNQ": [],
                                "TriviaQA_2k": [],
                                "OpenNQ_2k": []} 
def generate_model_answer(input_df: pd.DataFrame, config: dict) -> List[dict]:
    seed = config["target_model"].get("random_seed", None)
    if seed is not None:
        set_random_seed(config["target_model"]["random_seed"])
    dataset = config["dataset"]
    extra_columns = EXTRA_COLUMNS_PER_DATASET[dataset]

    messages = []
    answer_data = []
    for _, row in input_df.iterrows():
        extra_info = {column: row[column] for column in extra_columns}
        new_messages = [[{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", 
                         "content": answer_generation_util.
                         format_query(row["generated_variant"], dataset, **extra_info)}] 
                         for _ in range(config["answer_per_variant"])]
        new_answer_lines =[{"question_id": row["question_id"], "variant_id": row["variant_id"], "answer_id": i}
            for i in range(config["answer_per_variant"])
        ]
        messages = messages + new_messages
        answer_data = answer_data + new_answer_lines

    target_model = model_factory(config["target_model"]["config"])
    llm_outputs = target_model.generate(messages, 
                                               generation_config=config['target_model']['generation_config'],
                                               **config['target_model']['generate_function_parameters'],
                                               **config['target_model']['generate_function_kwargs'])
    
    for llm_output, answer_line in zip (llm_outputs, answer_data):
        generated_answer = answer_generation_util.parse_generated_answer(llm_output, dataset)
        
        answer_line["generated_answer"] = generated_answer

        if len(generated_answer)==0:
            logger.debug(f"LLM generated answer {answer_line['answer_id']} for variant {answer_line['variant_id']} of question {answer_line['question_id']} did not parse into a non-empty string. LLM output: {llm_output}.")
    
    return answer_data

    

if __name__ == "__main__":
    setup_output_and_logging('experiment_logs/generate_model_answers')
    parser = argparse.ArgumentParser(description="Model answers generation script.")
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
    config['target_model']['generation_config']['temperature'] = float(temperature)
    root_path = os.path.dirname(__file__)
    variation_model_name=config.get('variation_model', {'model_safename': 'no_model'})['model_safename']
    target_model_name=config['target_model']['model_safename']
    input_directory = os.path.join(root_path, 
                                    f"data/logs/{config['dataset']}/{config['ensembling_method']}/{variation_model_name}-variations")
    input_path = os.path.join(input_directory, 'question_variants.json')
    input_df = pd.read_json(input_path)

    output_data = generate_model_answer(input_df, config)
    output_directory = os.path.join(root_path, 
                                    f"data/baseline_logs/{config['dataset']}/{config['ensembling_method']}/{variation_model_name}-variations")
    os.makedirs(output_directory, exist_ok=True)
    with open(os.path.join(output_directory, f'{target_model_name}-answers-t{args.temperature}.json'), 'w') as file:
        json.dump(output_data, file, indent=4)

