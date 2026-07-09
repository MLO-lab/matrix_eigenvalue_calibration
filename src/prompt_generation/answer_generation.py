import os
import re 
import logging
logger = logging.getLogger(__name__)

def format_query(question: str, dataset: str, **kwargs) -> str:
    template_path = os.path.join(os.path.dirname(__file__), f'../../prompt_templates/{dataset}/generate_answer.txt')
    with open(template_path, 'r') as file:
        template = file.read()

    if dataset=="AmbigQA":
        prompt_full = template + question
        return prompt_full
    if dataset=="AmbigInst" or dataset=="AmbigInst2":
        prompt_full = template + f"Task description: {question}\nInput: {kwargs['input']}"
        return prompt_full
    if dataset=="Mentat":
        situation = kwargs['situation'] if len(kwargs['situation'].strip())>0 else "No patient situation provided. This is a general question."
        prompt_full = template + f"Patient situation: {situation}\nQuestion: {question}\nAnswer choices:\n" 
        for letter in ['a', 'b', 'c', 'd', 'e']:
            prompt_full = prompt_full + f"({letter}) {kwargs[f'answer_{letter}']}\n"
        return prompt_full
    if dataset=="CoQA2":
        prompt_full = template + f"Story:\n{kwargs['story']}\n\nQuestion History:\n{kwargs['history']}\n\nFinal Question:\nQ: {question}"
        return prompt_full
    if dataset in ["TriviaQA", "TriviaQA_2k"]:
        prompt_full = template + f"Question:\nQ: {question}"
        return prompt_full
    if dataset in ["OpenNQ", "OpenNQ_2k"]:
        prompt_full = template + f"Question:\nQ: {question}"
        return prompt_full
    raise NotImplementedError(f"Answer generation query formatting is not implemented for dataset {dataset}")

def generate_task_description(question: str, dataset: str, **kwargs) -> str:
    '''
    Only for embeddings containing task descriptions.
    '''
    if dataset=="AmbigQA":
        return f"Question: {question}"
    if dataset=="AmbigInst":
        return f"Task description: {question}\nInput: {kwargs['input']}"
    if dataset=="Mentat":
        situation = kwargs['situation'] if len(kwargs['situation'].strip())>0 else "No patient situation provided. This is a general question."
        description = f"Patient situation: {situation}\nQuestion: {question}\nAnswer choices:\n" 
        for letter in ['a', 'b', 'c', 'd', 'e']:
            description = description + f"({letter}) {kwargs[f'answer_{letter}']}\n"
        return description
    raise NotImplementedError(f"Answer generation query formatting is not implemented for dataset {dataset}")

def parse_generated_answer(llm_output: str, dataset: str) -> str:
    if dataset=="AmbigQA":
        match = re.search(r"Answer:\s*(.*)", llm_output, re.DOTALL)
        if match: 
            return match.group(1).strip()
        else: 
            if "Answer:".lower() in llm_output.lower():
                return ''
            else:
                return llm_output
    if dataset=="AmbigInst" or dataset=="AmbigInst2":
        pattern = r"Reasoning:\s*(.*?)\s*Answer:\s*(.*)"
        match = re.search(pattern, llm_output, re.DOTALL)

        if not match:
            pattern = r"Answer:\s*(.*)"
            match = re.search(pattern, llm_output, re.DOTALL)
            if not match:
                pattern = r"Answer(.*)"
                match = re.search(pattern, llm_output, re.DOTALL)
                if not match:
                    return ''
                
            return match.group(1).strip()
            
        answer = match.group(2).strip()
        return answer
    
    if dataset=="Mentat":
        match = re.search(r"Answer:\s*(.*)", llm_output, re.DOTALL)
        if match: 
            return match.group(1).strip()
        else: 
            logger.warning(f"Answer generation output does not contain 'Answer:'\n{llm_output}")
            if "Answer:".lower() in llm_output.lower():
                return llm_output.lower().replace("answer:", "").strip()
            else:
                return llm_output
    if dataset in ["CoQA2", "TriviaQA", "OpenNQ", "TriviaQA_2k", "OpenNQ_2k"]:
        answer = llm_output
        patterns = [
            re.compile(r'The question is unclear\. Random guess:\s*', re.IGNORECASE),
            re.compile(r'Random guess:\s*', re.IGNORECASE),
            re.compile(r'^A:\s*', re.IGNORECASE)
        ]
        #Try flexible patterns (anywhere in the string)
        for pattern in patterns:
            match = pattern.search(answer)
            if match:
                answer = answer[match.end():].strip()
                break

        # Remove surrounding quotes
        answer = answer.strip('"“”')
         # Remove trailing punctuation
        answer = re.sub(r'[.?!]+$', '', answer)
        return answer

    raise NotImplementedError(f"Parsing model generated answer is not implemented for dataset {dataset}")