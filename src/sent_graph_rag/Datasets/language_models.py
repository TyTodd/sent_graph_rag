import spacy
from spacy.language import Language
import torch
import gc
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union, Optional
import time
from sentence_transformers import SentenceTransformer
from datasets.utils.logging import disable_progress_bar
    
class EmbeddingModel(ABC):
    @abstractmethod
    def get_embeddings(self, texts):
        """
        gets embeddings for a list of texts
        must return embeddings as a torch tensor of shape (dim, len(texts))
        """
        pass
    
    
    def __enter__(self):
        self.load()
        return self
    
    @abstractmethod
    def load(self):
        pass
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.unload()
        return exc_type is None
    
    @abstractmethod
    def unload(self):
        pass

    @abstractmethod
    def get_dim(self):
        pass

class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        model_name: Name of the sentence transformer model to use (defaults to 'sentence-transformers/all-MiniLM-L6-v2')
        device: Device to use for embedding model (optional: chooses cuda if available) 
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model_name = model_name
        self.load()
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.unload()

    def get_embeddings(self, texts):
        """
        gets embeddings for a list of texts
        returns embeddings as a torch tensor of shape (dim, len(texts))
        """
        
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=True)
        embeddings = embeddings.cpu().T
        # print("embeddings",embeddings)
        return embeddings
    
    def load(self):
        self.embedding_model = SentenceTransformer(self.model_name, device=self.device)

        
    def unload(self):
        del self.embedding_model
        gc.collect()
        torch.cuda.empty_cache()
    
    def get_dim(self):
        return self.embedding_dim


class TransformersEmbeddingModel(EmbeddingModel):
    def __init__(self, create_model_fn, create_tokenizer_fn, device=None):
        """
        Intializes Transformer style Embedding model
        create_model_fn: Function to create embedding model
        create_tokenizer_fn: Function to create tokenizer
        device: Device to use for embedding model
        """
        self.create_model_fn = create_model_fn
        self.create_tokenizer_fn = create_tokenizer_fn
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
    
    def load(self):
        self.embedding_model = self.create_model_fn()
        self.tokenizer = self.create_tokenizer_fn()
    
    def unload(self):
        pass
    
    def get_dim(self):
        return self.embedding_model.config.hidden_size

    def get_embeddings(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs.to(self.device)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            token_embeddings = (
                outputs.last_hidden_state
            )  # Shape: (batch_size, seq_len, hidden_dim)
            embeddings = token_embeddings[:, 0, :].T  # Shape: (hidden_dim, batch_size)
            embeddings = embeddings.cpu()
        return embeddings

class SpacyModel():
    def __init__(self, spacy_model: str = "en_core_web_sm", device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        print("Initializing spacy model...")
        try:
            disable_progress_bar()
            self.nlp = spacy.load(spacy_model)
            self.nlp.add_pipe("fastcoref",  config={'device': self.device, "enable_progress_bar": False})
            print("Spacy model loaded successfully.")
        except:
            raise ValueError(f"""Error loading spacy model: '{spacy_model}' on device: '{device}'.
                             Please ensure: 
                             - '{spacy_model}' is installed 
                             - '{device}' is available
                             - 'fastcoref' is installed""")


class LanguageModel:
    def __init__(self, spacy_model: str = "en_core_web_sm", embedding_model: Optional[EmbeddingModel] = None, device: Optional[str] = None):
        """
        spacy_model: Name of the spacy model to use (defaults to 'en_core_web_sm')
        embedding_model: Embedding model to use (defaults to 'sentence-transformers/all-MiniLM-L6-v2')
        device: Device to use for embedding model (optional: chooses cuda if available) 
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        if embedding_model is None:
            self.embedding_model = SentenceTransformerEmbeddingModel('sentence-transformers/all-MiniLM-L6-v2', device=self.device)
        else:
            self.embedding_model = embedding_model
        self.spacy_model = SpacyModel(spacy_model, device)
        

    def get_embeddings(self, texts: List[str]):
        return self.embedding_model.get_embeddings(texts)