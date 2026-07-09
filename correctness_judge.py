from src.util.misc import setup_output_and_logging
import logging
import yaml
import argparse
import pandas as pd
import os, evaluate
from copy import deepcopy
from src.util.model_util import model_factory
import src.prompt_generation.correctness_judge as correctness_judge_util
from typing import List
logger = logging.getLogger(__name__)
#Columns other than "question_id", "gt_answer", "generated_answer" that are needed to generate the correctness judge query
EXTRA_COLUMNS_PER_DATASET = {"CoQA2": ["story", "history"],
                                "TriviaQA": [],
                                "OpenNQ": [],
                                "TriviaQA_2k": [],
                                "OpenNQ_2k": []} 
 
    
def determine_model_correctness(input_df: pd.DataFrame, output_df: pd.DataFrame, config: dict) -> List[str]:
    dataset = config["dataset"]
    extra_columns = EXTRA_COLUMNS_PER_DATASET[dataset]
    merged_df = output_df.merge(
        input_df,
        on="question_id",
        how="left"
    )

    messages = []
    for _, row in merged_df.iterrows():
        extra_info = {column: row[column] for column in extra_columns}
        messages.append([{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", 
                         "content": correctness_judge_util.
                         format_correctness_judge_query(row["question"], row["gt_answer"], row["generated_answer"], dataset, **extra_info)}])
        

    correctness_judge_model = model_factory(config["correctness_judge_model"]["config"])
    llm_outputs = correctness_judge_model.generate(messages, 
                                               generation_config=config['correctness_judge_model']['generation_config'],
                                               **config['correctness_judge_model']['generate_function_parameters'],
                                               **config['correctness_judge_model']['generate_function_kwargs'])
    
    correctness_data = []
    for i, row in merged_df.iterrows():
        correctness = correctness_judge_util.parse_correctness_judge_output(llm_outputs[i])
        correctness_data.append(correctness)
    return correctness_data

def determine_fuzzy_correctness(input_df: pd.DataFrame, output_df: pd.DataFrame, config: dict) -> List[str]:
    merged_df = output_df.merge(
            input_df,
            on="question_id",
            how="left"
    )
    rouge = evaluate.load("rouge", keep_in_memory=True)
    return merged_df.apply(lambda row: 
                           rouge.compute(predictions = [row['generated_answer']], references = [row['gt_answer']])['rougeL']>0.3,
                             axis=1).to_list()



if __name__ == "__main__":
    setup_output_and_logging('experiment_logs/correctness_judge')
    parser = argparse.ArgumentParser(description="Correctness judge script.")
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
    target_model_name=config['target_model']['model_safename']
    input_path = os.path.join(root_path, f"data/eval/{config['dataset']}.json")
    input_df = pd.read_json(input_path)
    if config.get('standard_answers', False):
        output_path = os.path.join(root_path, f"data/logs/{config['dataset']}/standard_answers/{config['target_model']['model_safename']}-answers.json")
    else:
        output_path = os.path.join(root_path, f"data/logs/{config['dataset']}/{config['ensembling_method']}/{variation_model_name}-variations/{config['target_model']['model_safename']}-answers.json")
    output_df = pd.read_json(output_path)

    if config['correctness_judge'] == "fuzzy_correctness":
        output_df["fuzzy_correctness"] = determine_fuzzy_correctness(deepcopy(input_df), deepcopy(output_df), config)
    elif config['correctness_judge'] == "model_correctness":
        output_df["model_correctness"] = determine_model_correctness(deepcopy(input_df), deepcopy(output_df), config)
    elif config['correctness_judge'] == "all":
        output_df["fuzzy_correctness"] = determine_fuzzy_correctness(deepcopy(input_df), deepcopy(output_df), config)
        output_df["model_correctness"] = determine_model_correctness(deepcopy(input_df), deepcopy(output_df), config)
    else:
        raise NotImplementedError(f"Correctness judge {config['correctness_judge']} not implemented.")


    output_df.to_json(output_path, orient='records', indent=4)

