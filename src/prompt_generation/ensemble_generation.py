import os, re
from typing import List, Tuple




def extract_clarification_ambigqa(model_ans: str) -> Tuple[List[str], List[str]]:
    lines = model_ans.split('\n')
    extract_list = []
    others = []
    pattern = r'\d+\.\s*(.*)'
    for line in lines:
        if line.startswith('Clarifications'):
            continue
        match = re.match(pattern, line)
        if match:
            ext = match.group(1)
            if 'No clarification needed'.lower() not in ext.lower():
                extract_list.append(ext.strip())
            else:
                others.append(line.strip())
        else:
            others.append(line.strip())
    return extract_list, others

def format_zero_shot_clarification_query(question: str, dataset: str, **kwargs) -> str:
    if dataset=="test": #Quick fix to allow for testing with a small subset of AmbigQA
        template_name = "AmbigQA"
    else:
        template_name = dataset
    template_path = os.path.join(os.path.dirname(__file__), f'../../prompt_templates/{template_name}/zero_shot_clarification.txt')
    with open(template_path, 'r') as file:
        template = file.read()

    if template_name=="AmbigQA":
        prompt_full = template + "\n\n" + "**Question to Analyse**"
        prompt_full = prompt_full + f"\n### Question:\n{question}"
        return prompt_full
    if template_name=="AmbigInst" or template_name=="AmbigInst2":
        prompt_full = template + "\n\n" + "**Task Description and Input**"
        prompt_full = prompt_full + f"\n### Task description:\n{question}"
        prompt_full = prompt_full + f"\n\n### Task input:\n{kwargs['input']}"
        return prompt_full
    if template_name=="Mentat":
        prompt_full = template + "\n\n" + "**Patient situation, Question and Answer Choices**"
        prompt_full = prompt_full + (f"\n### Patient situation:\n{kwargs['situation']}" if len(kwargs['situation'].strip())>0 else "\n### Patient situation:\n No patient situation provided. This is a general question.")
        prompt_full = prompt_full + f"\n\n### Question:\n{question}"
        prompt_full = prompt_full + f"\n\n### Answer choices:\n"
        for letter in ['a', 'b', 'c', 'd', 'e']:
            prompt_full = prompt_full + f"({letter}) {kwargs[f'answer_{letter}']}\n"
        return prompt_full
    if template_name=="CoQA2":
        prompt_full = template + "\n\n" + "**Story, Questions and Answers History, and Final Question**"
        prompt_full = prompt_full + f"\n## Story:\n{kwargs['story']}"
        prompt_full = prompt_full +  f"\n\n## Questions and Answers History:\n{kwargs['history']}"
        prompt_full = prompt_full + f"\n\n### Final Question:\n{question}"
        return prompt_full
    if template_name in ["TriviaQA", "TriviaQA_2k"]:
        prompt_full = template + "**Task Input**"
        prompt_full = prompt_full + f"\n### Original Question:\n{question}"
        return prompt_full
    if template_name in ["OpenNQ", "OpenNQ_2k"]:
        prompt_full = template + "**Task Input**"
        prompt_full = prompt_full + f"\n### Original Question:\n{question}"
        return prompt_full
    raise NotImplementedError(f"Zero shot clarification query formatting is not implemented for dataset {dataset}")


def parse_zero_shot_clarification_output(model_answer: str, dataset: str) -> Tuple[str, List[str], List[str]]:
    if dataset in ["test", "AmbigQA", "AmbigInst", "AmbigInst2", "Mentat"]:
        reasoning = ""
        clarifications = []
        other_outputs = []

        # Extract the Analyses section
        analyses_match = re.search(r"### Analyses:\s*(.*?)(?=### Clarifications)", model_answer, re.DOTALL)
        if analyses_match:
            reasoning = analyses_match.group(1).strip()

        # Extract the Clarifications section
        clarifications_match = re.search(r"### Clarifications:\s*(.*)", model_answer, re.DOTALL)
        if clarifications_match:
            clarifications_text = clarifications_match.group(1).strip()
            if "No clarification needed".lower() in clarifications_text.lower():
                clarifications = []
            else:
                clarifications = re.findall(r"#\d+\s+(.*?)(?=(?:\-\-\-)|#|\Z)", clarifications_text, re.DOTALL)
                clarifications = [clarification.strip() for clarification in clarifications]

        # Extract anything else that doesn't match either section (extra/hallucinated content)
        expected_sections = re.findall(r"(### Analyses:.*?)(?=### Clarifications)", model_answer, re.DOTALL)
        expected_sections += re.findall(r"(### Clarifications:.*)", model_answer, re.DOTALL)
        combined_expected = "\n".join(expected_sections)
        
        extra_lines = [line for line in model_answer.strip().splitlines()
                    if line.strip() and line not in combined_expected]
        
        other_outputs = [line.strip() for line in extra_lines]

        return reasoning, clarifications, other_outputs
    if dataset in ["CoQA2", "TriviaQA", "OpenNQ", "TriviaQA_2k", "OpenNQ_2k"]:
        reasoning = ""
        clarifications = []
        other_outputs = []

        # Extract the Clarifications section
        clarifications_match = re.search(r"### Rephrasings:\s*(.*)", model_answer, re.DOTALL)
        if clarifications_match:
            clarifications_text = clarifications_match.group(1).strip()
            clarifications = re.findall(r"#\d+\s+(.*?)(?=(?:\-\-\-)|#|\Z)", clarifications_text, re.DOTALL)
            clarifications = [clarification.strip() for clarification in clarifications]

        # Extract anything else that doesn't match that section (extra/hallucinated content)
        expected_sections = re.findall(r"(### Rephrasings:.*)", model_answer, re.DOTALL)
        combined_expected = "\n".join(expected_sections)
        
        extra_lines = [line for line in model_answer.strip().splitlines()
                    if line.strip() and line not in combined_expected]
        
        other_outputs = [line.strip() for line in extra_lines]

        return reasoning, clarifications, other_outputs
    raise NotImplementedError(f"Zero shot clarification response parsing is not implemented for dataset {dataset}")

