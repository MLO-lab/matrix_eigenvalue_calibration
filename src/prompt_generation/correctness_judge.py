import os
def format_correctness_judge_query(question: str, gt_answer: str, generated_answer: str, dataset: str, **kwargs)-> str:
    template_name = dataset
    template_path = os.path.join(os.path.dirname(__file__), f'../../prompt_templates/{template_name}/correctness_judge.txt')
    with open(template_path, 'r') as file:
        template = file.read()
    
    if template_name=="CoQA2":
        prompt_full = template + "**Input**"
        prompt_full = prompt_full + f"\n## Story:\n{kwargs['story']}"
        prompt_full = prompt_full +  f"\n\n## Questions and Answers History:\n{kwargs['history']}"
        prompt_full = prompt_full + f"\n\n### Final Question:\n{question}"
        prompt_full = prompt_full + f"\n\n### Ground Truth Answer:\n{gt_answer}"
        prompt_full = prompt_full + f"\n\n### Model Generated Answer:\n{generated_answer}"
        prompt_full = prompt_full + f"\n\nIs the model generated answer correct? Answer with yes or no."
        return prompt_full
    if template_name in ["TriviaQA", "TriviaQA_2k"]:
        prompt_full = template + "**Input**"
        prompt_full = prompt_full + f"\n### Question:\n{question}"
        prompt_full = prompt_full + f"\n\n### Ground Truth Answer:\n{gt_answer}"
        prompt_full = prompt_full + f"\n\n### Model Generated Answer:\n{generated_answer}"
        prompt_full = prompt_full + f"\n\nIs the model generated answer correct? Answer with yes or no."
        return prompt_full
    if template_name in ["OpenNQ", "OpenNQ_2k"]:
        prompt_full = template + "**Input**"
        prompt_full = prompt_full + f"\n### Question:\n{question}"
        prompt_full = prompt_full + f"\n\n### Ground Truth Answer:\n{gt_answer}"
        prompt_full = prompt_full + f"\n\n### Model Generated Answer:\n{generated_answer}"
        prompt_full = prompt_full + f"\n\nIs the model generated answer correct? Answer with yes or no."
        return prompt_full
    raise NotImplementedError(f"Correctness judge query formatting is not implemented for dataset {dataset}.")

def parse_correctness_judge_output(llm_output: str)-> bool: 
    llm_output = llm_output.strip().lower()
    no = "no" in llm_output
    yes = "yes" in llm_output
    if (not no) and yes:
        return True
    elif (not yes) and no:
        return False
    else:
        raise ValueError(f"LLM output {llm_output} is not a valid response for correctness judge. Should be 'yes' or 'no'.")
