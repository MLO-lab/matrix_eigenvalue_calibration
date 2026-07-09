import logging, os
import numpy as np
from typing import List
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import gc
from torch import Tensor


logger = logging.getLogger(__name__)

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def encode(self, texts: List[str], encoding_arguments: dict) -> np.ndarray:
        pass

class AllMpnet(BaseEmbeddingModel):
    def __init__(self):
        self.model = SentenceTransformer("all-mpnet-base-v2",
                                    cache_folder= os.environ['HF_CACHE'],
                                    device = torch.device("cuda"))
    def encode(self, texts: List[str], encoding_arguments: dict) -> np.ndarray:
        return self.model.encode(texts, **encoding_arguments)

class JinaEmbedding(BaseEmbeddingModel):
    def __init__(self):
        self.model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True, cache_dir = os.environ["HF_CACHE"], torch_dtype=torch.float16).to("cuda")
        
    def encode(self, texts: List[str], encoding_arguments: dict) -> np.ndarray:
        return self.model.encode(texts, **encoding_arguments)

class AlibabaGTEQwen(BaseEmbeddingModel):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True, cache_dir=os.environ["HF_CACHE"])
        self.model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True, cache_dir=os.environ["HF_CACHE"], device_map="auto",
                                               torch_dtype=torch.float16)
        self._default_task = 'Given two sentences, determine whether they are semantically similar or not'
    @staticmethod
    def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'
    
    def encode(self, texts: List[str], encoding_arguments: dict, task_description = None) -> np.ndarray:
        if task_description is None:
            task_description = self._default_task
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        texts = [self.get_detailed_instruct(task_description, text) for text in texts]
        # Tokenize the input texts
        batch_dict = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}
        outputs = self.model(**batch_dict)
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1).detach().cpu().numpy()
        return embeddings



class SFREMbeddingMistral(BaseEmbeddingModel):

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral',  cache_dir=os.environ["HF_CACHE"])
        self.model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral', cache_dir=os.environ["HF_CACHE"], device_map="auto",
                                               torch_dtype=torch.float16)
        self._default_task = 'Given two sentences, determine whether they are semantically similar or not'
    @staticmethod
    def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'
    
    def encode(self, texts: List[str], encoding_arguments: dict, task_description = None) -> np.ndarray:
        if task_description is None:
            task_description = self._default_task
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        texts = [self.get_detailed_instruct(task_description, text) for text in texts]
        # Tokenize the input texts
        batch_dict = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}
        outputs = self.model(**batch_dict)
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1).detach().cpu().numpy()
        return embeddings

class MultilingualE5(BaseEmbeddingModel):

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct',  cache_dir=os.environ["HF_CACHE"])
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct', cache_dir=os.environ["HF_CACHE"], device_map="auto",
                                               torch_dtype=torch.float16)
        self._default_task = 'Given two sentences, determine whether they are semantically similar or not'
    @staticmethod
    def average_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'
    
    def encode(self, texts: List[str], encoding_arguments: dict, task_description = None) -> np.ndarray:
        if task_description is None:
            task_description = self._default_task
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        texts = [self.get_detailed_instruct(task_description, text) for text in texts]
        # Tokenize the input texts
        batch_dict = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}
        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1).detach().cpu().numpy()
        return embeddings

class LinqEmbedMistral(BaseEmbeddingModel):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('Linq-AI-Research/Linq-Embed-Mistral',  cache_dir=os.environ["HF_CACHE"])
        self.model = AutoModel.from_pretrained('Linq-AI-Research/Linq-Embed-Mistral', cache_dir=os.environ["HF_CACHE"], device_map="auto",
                                               torch_dtype=torch.float16)
        self._default_task = 'Given two sentences, determine whether they are semantically similar or not'
    @staticmethod
    def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'
    
    def encode(self, texts: List[str], encoding_arguments: dict, task_description = None) -> np.ndarray:
        if task_description is None:
            task_description = self._default_task
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        texts = [self.get_detailed_instruct(task_description, text) for text in texts]
        # Tokenize the input texts
        batch_dict = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}
        outputs = self.model(**batch_dict)
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1).detach().cpu().numpy()
        return embeddings
    
class LajavanessBilingual(BaseEmbeddingModel):
    def __init__(self):
        self.model = SentenceTransformer('Lajavaness/bilingual-embedding-large', trust_remote_code=True,
                                    cache_folder= os.environ['HF_CACHE'],
                                    device = torch.device("cuda"))
    def encode(self, texts: List[str], encoding_arguments: dict) -> np.ndarray:
        return self.model.encode(texts, **encoding_arguments)

class E5Mistral(BaseEmbeddingModel):

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct',  cache_dir=os.environ["HF_CACHE"])
        self.model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct', cache_dir=os.environ["HF_CACHE"], device_map="auto",
                                               torch_dtype=torch.float16)
        self._default_task = 'Given two sentences, determine whether they are semantically similar or not'
    @staticmethod
    def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'
    
    def encode(self, texts: List[str], encoding_arguments: dict, task_description = None) -> np.ndarray:
        if task_description is None:
            task_description = self._default_task
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        texts = [self.get_detailed_instruct(task_description, text) for text in texts]
        # Tokenize the input texts
        batch_dict = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}
        outputs = self.model(**batch_dict)
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1).detach().cpu().numpy()
        return embeddings
    
class Jasper(BaseEmbeddingModel):
    def __init__(self):
        self.model = SentenceTransformer(
            "infgrad/jasper_en_vision_language_v1",
            trust_remote_code=True,
            device=torch.device("cuda"),
            cache_folder= os.environ['HF_CACHE'],
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "sdpa"
            },
            config_kwargs={"is_text_encoder": True, "vector_dim": 1024},
        )
    def encode(self, texts: List[str], encoding_arguments: dict) -> np.ndarray:
        return self.model.encode(texts, **encoding_arguments)

class JinaV5Nano(BaseEmbeddingModel):
    def __init__(self):
        self.model = SentenceTransformer(
            "jinaai/jina-embeddings-v5-text-nano",
            trust_remote_code=True,
            cache_folder=os.environ['HF_CACHE'],
            device=torch.device("cuda"),
        )

    def encode(self, texts: List[str], encoding_arguments: dict) -> np.ndarray:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return self.model.encode(texts, task="text-matching", normalize_embeddings=True, **encoding_arguments).astype(np.float16)


class HarrierOSS(BaseEmbeddingModel):
    def __init__(self):
        self.model = SentenceTransformer(
            "microsoft/harrier-oss-v1-270m",
            cache_folder=os.environ['HF_CACHE'],
            device=torch.device("cuda"),
        )

    def encode(self, texts: List[str], encoding_arguments: dict) -> np.ndarray:
        return self.model.encode(texts, prompt_name="sts_query", normalize_embeddings=True, **encoding_arguments)


class SpeedEmbedding(BaseEmbeddingModel):

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('Haon-Chen/speed-embedding-7b-instruct',  cache_dir=os.environ["HF_CACHE"])
        self.model = AutoModel.from_pretrained('Haon-Chen/speed-embedding-7b-instruct', cache_dir=os.environ["HF_CACHE"], device_map="auto",
                                               torch_dtype=torch.float16)
        self._default_task = 'Retrieve semantically similar text.'
    @staticmethod
    def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'
    
    def encode(self, texts: List[str], encoding_arguments: dict, task_description = None) -> np.ndarray:
        if task_description is None:
            task_description = self._default_task
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        texts = [self.get_detailed_instruct(task_description, text) for text in texts]
        # Tokenize the input texts
        batch_dict = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}
        outputs = self.model(**batch_dict)
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1).detach().cpu().numpy()
        return embeddings
    
def embedding_factory(config: dict) -> BaseEmbeddingModel:
    model_name = config["model_name"]
    if model_name == "all-mpnet-base-v2":
        return AllMpnet()
    elif model_name == "jina-embeddings-v3":
        return JinaEmbedding()
    elif model_name == "gte-Qwen2-7B-instruct":
        return AlibabaGTEQwen()
    elif model_name == "SFR-Embedding-Mistral":
        return SFREMbeddingMistral()
    elif model_name == "multilingual-e5-large-instruct":
        return MultilingualE5()
    elif model_name == "Lajavaness/bilingual-embedding-large":
        return LajavanessBilingual()
    elif model_name == "Linq-Embed-Mistral":
        return LinqEmbedMistral()
    elif model_name == "e5-mistral-7b-instruct":
        return E5Mistral()
    elif model_name == "jasper_en_vision_language_v1":
        return Jasper()
    elif model_name == "speed-embedding-7b-instruct":
        return SpeedEmbedding()
    elif model_name == "jina-embeddings-v5-text-nano":
        return JinaV5Nano()
    elif model_name == "harrier-oss-v1-270m":
        return HarrierOSS()
    else:
        raise NotImplementedError

