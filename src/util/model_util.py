from abc import ABC, abstractmethod
from typing import List, Union, Tuple
import logging, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
import openai
import openrouter
import time, json, uuid
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from transformers import LogitsProcessor, LogitsProcessorList

class MyCustomLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        # scores = logits at current step (1D tensor over vocab)
        
        # example: clamp logits to avoid huge numbers
        #scores = scores.clamp(-1e4, 1e4)
        print(scores.size())
        print(scores.max(dim=1))
        print(scores.min(dim=1))
        return scores


logger = logging.getLogger(__name__)
class BaseGenerativeModel(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def generate(self, messages: List[List], verbose_inputs: bool=False, verbose_outputs: bool=False, return_logprobs: bool = False, generation_config: dict= dict(), **kwargs) -> Union[List[str], Tuple[List[str], List[List[Tuple[str, float]]]]]:
        pass

class OpenAIBaseModel(BaseGenerativeModel, ABC):
    def __init__(self, config: dict):
        config = deepcopy(config)
        #Get model name
        self.model_name = config["model_name"]
        self.config = config.get("model_config", dict())
        
        #Default value
        self.config["max_retries"] = self.config.get("max_retries", 0) #No retries happen within openaiAPI SDK. All retries are implemented here in the generate function.
        #Default max_retries value in SDK is 2.

    @property
    @abstractmethod
    def client(self) -> openai.OpenAI: 
        pass

    def generate(self, messages: List[List], verbose_inputs: bool=False, verbose_outputs: bool=False, return_logprobs: bool = False, generation_config: dict= dict(), **kwargs) -> Union[List[str], Tuple[List[str], List[List[Tuple[str, float]]]]]:
        generation_config = deepcopy(generation_config)
        max_retries = kwargs.get("max_retries", 18)
        backoff_base = kwargs.get("backoff_base", 2)
        max_workers = kwargs.get("max_workers", None)

        # If we want logprobs, request logprobs from OpenAI. This overwrites the generation_config
        generation_config["logprobs"] = return_logprobs

        def call_openai(msg, idx):
            if verbose_inputs:
                print(f"[INPUT {idx}] [{msg}]")
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=msg,
                        **generation_config
                    )
                    text = response.choices[0].message.content

                    if verbose_outputs:
                        print(f"[OUTPUT {idx}] [{text}]")

                    if return_logprobs:
                        logprobs = response.choices[0].logprobs
                        # Generate pairs of token, logprob
                        sequence_token_logprobs = [(token.token, token.logprob) for token in logprobs.content]
                        return text, sequence_token_logprobs
                    else:
                        return text

                except openai.APIStatusError as e:
                    if isinstance(e, openai.RateLimitError):
                        print(f"OpenAI API error (attempt {attempt + 1}/{max_retries}): {e}")
                        if attempt == max_retries - 1:
                            raise RuntimeError("Maximum retry attempts exceeded") from e
                        time.sleep((backoff_base ** attempt) * (1 + random.uniform(0, 0.2)))  # Sleeptime in seconds, with random jitter to avoid collutions
                    else:
                        print(F"Got APIStatusError different from ReateLimitError. Code: {e.status_code}. Response: {e.response}")
                        raise e

        results = [None] * len(messages)
        token_logprob_pairs = [None] * len(messages)

        with ThreadPoolExecutor(max_workers = max_workers) as executor: 
            futures = {executor.submit(call_openai, msg, idx): idx for idx, msg in enumerate(messages)}
            for future in tqdm(as_completed(futures)):
                idx = futures[future]
                result = future.result()
                if return_logprobs:
                    text, logprobs = result
                    results[idx]=text
                    token_logprob_pairs[idx]=logprobs
                else:
                    results[idx]=result
            
            

        if return_logprobs:
            return results, token_logprob_pairs
        else:
            return results
        

class OpenAIModel(OpenAIBaseModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self._client = openai.OpenAI(api_key = os.environ["OPENAI_API_KEY"],
                                    base_url = 'https://api.openai.com/v1',
                                    **self.config) 
    
    @property
    def client(self) -> openai.OpenAI: 
        return self._client


class GroqModel(OpenAIBaseModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self._client = openai.OpenAI(api_key = os.environ["GROQ_API_KEY"],
                                    base_url = 'https://api.groq.com/openai/v1',
                                    **self.config)
    @property
    def client(self) -> openai.OpenAI: 
        return self._client
    

        
# class OpenRouterModel(OpenAIBaseModel):
#     def __init__(self, config: dict):
#         super().__init__(config)
#         self._client = openai.OpenAI(api_key = os.environ["OPENROUTER_API_KEY"],
#                                     base_url = 'https://openrouter.ai/api/v1',
#                                     **self.config)
#     @property
#     def client(self) -> openai.OpenAI: 
#         return self._client

class OpenRouterModel(BaseGenerativeModel):
    def __init__(self, config: dict):
        config = deepcopy(config)
        #Get model name
        self.model_name = config["model_name"]
        self.config = config.get("model_config", dict())
        
        #Default value
        self.config["max_retries"] = self.config.get("max_retries", 0) #No retries happen within openaiAPI SDK. All retries are implemented here in the generate function.
        #Default max_retries value in SDK is 2.


    def generate(self, messages: List[List], verbose_inputs: bool=False, verbose_outputs: bool=False, return_logprobs: bool = False, generation_config: dict= dict(), **kwargs) -> Union[List[str], Tuple[List[str], List[List[Tuple[str, float]]]]]:
        generation_config = deepcopy(generation_config)
        max_retries = kwargs.get("max_retries", 18)
        backoff_base = kwargs.get("backoff_base", 2)
        max_workers = kwargs.get("max_workers", None)

        # If we want logprobs, request logprobs from OpenAI. This overwrites the generation_config
        generation_config["logprobs"] = return_logprobs

        def call_openrouter(msg, idx):
            if verbose_inputs:
                print(f"[INPUT {idx}] [{msg}]")
            for attempt in range(max_retries):
                try:
                    with openrouter.OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as client:
                        response = client.chat.send(
                            model=self.model_name,
                            messages=msg,
                            **generation_config
                        )
                    text = response.choices[0].message.content

                    if verbose_outputs:
                        print(f"[OUTPUT {idx}] [{text}]")

                    if return_logprobs:
                        logprobs = response.choices[0].logprobs
                        # Generate pairs of token, logprob
                        sequence_token_logprobs = [(token.token, token.logprob) for token in logprobs.content]
                        return text, sequence_token_logprobs
                    else:
                        return text

                except openrouter.errors.OpenRouterError as e:
                    if isinstance(e, openrouter.errors.TooManyRequestsResponseError):
                        print(f"OpenRouter API error (attempt {attempt + 1}/{max_retries}): {e}")
                        if attempt == max_retries - 1:
                            raise RuntimeError("Maximum retry attempts exceeded") from e
                        time.sleep((backoff_base ** attempt) * (1 + random.uniform(0, 0.2)))  # Sleeptime in seconds, with random jitter to avoid collutions
                    else:
                        print(f"Got OpenRouterError different from TooManyRequestsResponseError. Code: {e.status_code}. Response: {e.response}")
                        raise e

        results = [None] * len(messages)
        token_logprob_pairs = [None] * len(messages)

        with ThreadPoolExecutor(max_workers = max_workers) as executor: 
            futures = {executor.submit(call_openrouter, msg, idx): idx for idx, msg in enumerate(messages)}
            for future in tqdm(as_completed(futures)):
                idx = futures[future]
                result = future.result()
                if return_logprobs:
                    text, logprobs = result
                    results[idx]=text
                    token_logprob_pairs[idx]=logprobs
                else:
                    results[idx]=result
            
            

        if return_logprobs:
            return results, token_logprob_pairs
        else:
            return results
        

class DeepSeekModel(OpenAIBaseModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self._client = openai.OpenAI(api_key = os.environ["DEEPSEEK_API_KEY"],
                                    base_url = 'https://api.deepseek.com/v1',
                                    **self.config)
    @property
    def client(self) -> openai.OpenAI: 
        return self._client

class OpenAIBatchBaseModel(BaseGenerativeModel, ABC):
    def __init__(self, config: dict):
        self.model_name = config["model_name"]
        self.config = config.get("model_config", dict())
        self.poll_interval = config.get("poll_interval", 60)  
        

    @property
    @abstractmethod
    def client(self) -> openai.OpenAI:
        pass

    def _submit_batch_job(self, batch: List[List], custom_ids: List[str], generation_config: dict) -> str:
        instructions = [
                {
                    "custom_id": custom_ids[i],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model_name,
                        "messages": msg,
                        **generation_config,
                    }
                }
                for i, msg in enumerate(batch)
            ]

        response = self.client.batches.create(
            input_file_id=self._upload_input_file(instructions),
            endpoint="/v1/chat/completions",
            completion_window=self.completion_window
        )

        return response.id


    def _upload_input_file(self, instructions: List[dict]):
        file_content = ""
        for entry in instructions:
            file_content =  file_content + json.dumps(entry) + "\n"

        tmp_file_path = f"/tmp/openai_batch_{uuid.uuid4().hex}.jsonl"

        with open(tmp_file_path, "w") as f:
            f.write(file_content)

        uploaded_file = self.client.files.create(
            file=open(tmp_file_path, "rb"),
            purpose="batch"
        )
        # if os.path.exists(tmp_file_path):
        #     os.remove(tmp_file_path)


        return uploaded_file.id

    def _poll_batch_completion(self, batch_id: str) -> Tuple[str, Union[str, None]]:
        start_time = time.time()
        while True:
            batch = self.client.batches.retrieve(batch_id)
            if batch.status == "completed":
                return batch.output_file_id, batch.error_file_id
            elif batch.status in ("failed", "expired", "canceled"):
                raise RuntimeError(f"Batch {batch_id} failed with status: {batch.status}")
            elif time.time() - start_time > self.max_poll_time:
                raise TimeoutError("Polling time exceeded maximum allowed duration.")
            time.sleep(self.poll_interval)

    def _read_file_lines(self, file_id: str) -> List[str]:
        content = self.client.files.content(file_id)
        return [line for line in content.iter_lines() if line]

    
    def generate(
        self,
        messages: List[List],
        verbose_inputs: bool = False,
        verbose_outputs: bool = False,
        return_logprobs: bool = False,
        generation_config: dict = dict(),
        **kwargs
    ) -> Union[List[str], Tuple[List[str], List[List[Tuple[str, float]]]]]:
        generation_config = deepcopy(generation_config)
        # If we want logprobs, request logprobs from OpenAI. This overwrites the generation_config
        generation_config["logprobs"] = return_logprobs

        batch_size = kwargs.get("batch_size", 1000)

        # Split into batches
        batches = [
            (i, messages[i:i + batch_size])
            for i in range(0, len(messages), batch_size)
        ]

        response_ids = []

        # Step 1: Submit all batches
        for batch_start_idx, batch in batches:
            custom_ids = [str(batch_start_idx + i) for i in range(len(batch))]

            if verbose_inputs:
                for idx, msg in enumerate(batch):
                    print(f"[INPUT {batch_start_idx + idx}] [{msg}]")
                
                try:
                    response_id = self._submit_batch_job(batch, custom_ids, generation_config)
                    response_ids.append(response_id)
                except Exception as e:
                    print(f"Error submitting batch: {str(e)}")
                    raise

        # Step 2: Poll all batches
        results_map = {}
        for batch_id in tqdm(response_ids, desc="Polling batches"):
            try:
                output_file_id, error_file_id = self._poll_batch_completion(batch_id)

                if error_file_id:
                    error_lines = self._read_file_lines(error_file_id)
                    raise RuntimeError(f"Batch {batch_id} had errors:\n" + "\n".join(error_lines))

                output_lines = self._read_file_lines(output_file_id)

                for line in output_lines:
                    record = json.loads(line)
                    custom_id = record.get("custom_id")
                    text = record["response"]["body"]["choices"][0]["message"]["content"]
                    results_map[int(custom_id)] = text

                    if verbose_outputs:
                        print(f"[OUTPUT {custom_id}] [{text}]")

            except Exception as e:
                print(f"Error in batch {batch_id}: {str(e)}")
                raise

        # Step 3: Collect results in order
        final_results = [results_map[i] for i in range(len(messages))]

        return final_results

class OpenAIBatchModel(OpenAIBatchBaseModel):
    def __init__(self, config: dict):
        config = deepcopy(config)
        super().__init__(config)
        self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                                     base_url='https://api.openai.com/v1',
                                     **self.config)
        self.max_poll_time = config.get("max_poll_time", 60 * 60 * 25)
        self.completion_window = config.get("completion_window", "24h")

    @property
    def client(self) -> openai.OpenAI:
        return self._client
    
class GroqBatchModel(OpenAIBatchBaseModel):
    def __init__(self, config: dict):
        config = deepcopy(config)
        super().__init__(config)
        self._client = openai.OpenAI(api_key = os.environ["GROQ_API_KEY"],
                                    base_url = 'https://api.groq.com/openai/v1',
                                    **self.config)
        self.max_poll_time = config.get("max_poll_time", 60 * 60 * 24 * 8)
        self.completion_window = config.get("completion_window", "24h")

    @property
    def client(self) -> openai.OpenAI:
        return self._client


    

class HFModel(BaseGenerativeModel):
    def __init__(self, config: dict):
        config = deepcopy(config)

        #Get model name
        self.model_name = config["model_name"]
        #Get model config
        self.config = config.get("model_config", dict())
        #Get tokenizer config
        self.tokenizer_config = config.get("tokenizer_config", dict())

        #Default values for parameters
        self.config['torch_dtype'] = self.config.get("torch_dtype", torch.float16)  #Default value for us
        if self.config['torch_dtype']=='float16': 
                self.config['torch_dtype'] = torch.float16
            #TODO Allow for other data types 

        self.config['device_map'] = self.config.get("device_map", 'auto') 

        
        #Default values for tokenizer_config parameters
        self.tokenizer_config["padding_side"] = self.tokenizer_config.get("padding_side", 'left')

        #Default value for cache_dir
        self.config["cache_dir"] = self.config.get("cache_dir", os.environ['HF_CACHE'])
        self.tokenizer_config["cache_dir"] = self.tokenizer_config.get("cache_dir", os.environ['HF_CACHE'])

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                          **self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, 
                                                       **self.tokenizer_config)

    def generate(self, messages: List[List], verbose_inputs: bool=False, verbose_outputs: bool=False, return_logprobs: bool = False, generation_config: dict= dict(), **kwargs) -> Union[List[str], Tuple[List[str], List[List[Tuple[str, float]]]]]:
        generation_config = deepcopy(generation_config)
        # Overwrite the output parameters in the generation config
        generation_config['return_dict_in_generate'] = return_logprobs
        generation_config['output_logits'] = return_logprobs

        if generation_config.get('temperature', 0.0)< 0.2:
            logits_processor = LogitsProcessorList()
            logits_processor.append(MyCustomLogitsProcessor())
            generation_config['logits_processor'] = logits_processor

        #Batch size is a kwarg for hf models
        batch_size = kwargs.get("batch_size", len(messages)) #By default all the messages in one batch

        #Setting default values for important tokenization config params
        tokenization_config = kwargs.get("tokenization_config", dict())
        default_tokenization_config = {"add_generation_prompt": True,
                                    "padding": True,
                                    "return_dict": True,
                                    "return_tensors": "pt"}
        default_tokenization_config.update(tokenization_config)
        tokenization_config = default_tokenization_config

        def generate_for_batch(batch_idx, batch_messages):
            inputs = self.tokenizer.apply_chat_template(batch_messages, **tokenization_config).to(self.model.device)
            decoded_inputs = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
            if verbose_inputs:
                print(f"Batch {batch_idx} input:")
                for idx, msg in enumerate(decoded_inputs):
                    print(f"[INPUT {idx+batch_idx*batch_size}] [{msg}]")
            input_length = len(inputs["input_ids"][0])
            if verbose_inputs:
                print(f"Inputs have length: {input_length}")

            output = self.model.generate(**inputs, **generation_config) #Is a tensor if not return_logprobs, otherwise is a GenerateDecoderOnlyOutput(ModelOutput)

            token_logprob_pairs = [] # For each sequence, a list of (token, logprob) pairs, excluding padding tokens. Nonempty only if return_logprobs.
            if return_logprobs:
                decoded_outputs = self.tokenizer.batch_decode(output.sequences[:,input_length:], skip_special_tokens=True)
                logits_tensor = torch.stack(output.logits).transpose(0,1).to("cpu") # shape from (slength, batch, vocab) to (batch, slength, vocab)
                generated_sequences_tensor = output.sequences[:,input_length:].to("cpu")

                #Optimization to avoid out of memory
                del output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache() 
                
                logprobs = torch.nn.functional.log_softmax(logits_tensor, dim=-1) #Compute logprobs based on logits
                output_logprobs = torch.gather(
                    logprobs,
                    dim=2,
                    index=generated_sequences_tensor.unsqueeze(-1)
                ).squeeze(-1) #Need only logprobs of the tokens that are actually generated

                #Mask to avoid including special token logprobs in the final output since they are not relevant.
                special_token_ids = torch.tensor(self.tokenizer.all_special_ids).to("cpu")
                non_special_mask = ~((generated_sequences_tensor.unsqueeze(-1)==special_token_ids).
                                any(dim=-1))
                non_special_mask = non_special_mask.tolist()

                token_lists = [self.tokenizer.convert_ids_to_tokens(seq) for seq in generated_sequences_tensor]
                logprob_lists = output_logprobs.tolist()

                for sequence_tokens, sequence_logprobs, sequence_mask in zip(token_lists, logprob_lists, non_special_mask):
                    filtered = [
                        (t, lp) for t, lp, m in zip(sequence_tokens, sequence_logprobs, sequence_mask) if m
                    ]
                    token_logprob_pairs.append(filtered) 

            else:
                decoded_outputs = self.tokenizer.batch_decode(output[:,input_length:], skip_special_tokens=True)

            if verbose_outputs:
                print(f"Batch {batch_idx} output:")
                for idx, text in enumerate(decoded_outputs):
                    print(f"[OUTPUT {idx+batch_idx*batch_size}] [{text}]")
            
            
            return decoded_outputs, token_logprob_pairs

            
        
        n_batches = len(messages) // batch_size + (0 if (len(messages)%batch_size==0) else 1)

        outputs, logprobs = [], []
        for batch_idx in tqdm(range(n_batches)):
            batch_messages = messages[batch_idx*batch_size:(batch_idx+1)*batch_size]
            batch_outputs, batch_logprobs = generate_for_batch(batch_idx, batch_messages)
            outputs += batch_outputs
            logprobs += batch_logprobs
        
        if return_logprobs:
            return outputs, logprobs
        else: 
            return outputs

def model_factory(config: dict) -> BaseGenerativeModel:
    model_type = config["model_type"]
    if model_type == "openai":
        return OpenAIModel(config)
    elif model_type == "hf":
        return HFModel(config)
    elif model_type == "deepseek":
        return DeepSeekModel(config)
    elif model_type == "groq":
        return GroqModel(config)
    elif model_type == "openrouter":
        return OpenRouterModel(config)
    elif model_type == "openai-batch":
        return OpenAIBatchModel(config)
    elif model_type == "groq-batch":
        return GroqBatchModel(config)
    else:
        raise NotImplementedError

