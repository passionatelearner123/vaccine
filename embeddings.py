"""
Embedding utilities for ELMo and GloVe integration
"""
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
import tensorflow_hub as hub
import tensorflow as tf
import gensim.downloader as api
from config import Config

class ELMoEmbedding(nn.Module):
    """ELMo embedding extractor"""
    
    def __init__(self, elmo_model_path=None):
        super(ELMoEmbedding, self).__init__()
        self.device = torch.device(Config.DEVICE)
        
        # Try to load ELMo model
        try:
            if elmo_model_path:
                self.elmo = hub.load(elmo_model_path)
            else:
                # Use TensorFlow Hub ELMo
                self.elmo = hub.load("https://tfhub.dev/google/elmo/3")
            self.use_tf = True
        except Exception as e:
            print(f"Warning: Could not load ELMo model: {e}")
            print("Using BERT as ELMo alternative...")
            self.use_tf = False
            self.bert_model = AutoModel.from_pretrained("bert-base-uncased")
            self.bert_model.eval()
    
    def forward(self, texts):
        """Extract ELMo embeddings"""
        if self.use_tf:
            # TensorFlow ELMo
            embeddings = self.elmo(texts)
            # Convert to numpy then to torch
            if isinstance(embeddings, dict):
                embeddings = embeddings['default']
            embeddings = tf.reduce_mean(embeddings, axis=1)
            embeddings = embeddings.numpy()
            return torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        else:
            # Use BERT as alternative
            with torch.no_grad():
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                inputs = tokenizer(texts, return_tensors="pt", padding=True, 
                                 truncation=True, max_length=128).to(self.device)
                outputs = self.bert_model(**inputs)
                # Use mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings

class GloVeEmbedding:
    """GloVe embedding loader"""
    
    def __init__(self, glove_dim=300):
        self.glove_dim = glove_dim
        self.device = torch.device(Config.DEVICE)
        self.word_to_idx = {}
        self.embeddings = None
        self._load_glove()
    
    def _load_glove(self):
        """Load GloVe embeddings"""
        try:
            # Try to load pre-trained GloVe from gensim
            print("Loading GloVe embeddings...")
            self.glove_model = api.load(f"glove-wiki-gigaword-{self.glove_dim}")
            print("GloVe embeddings loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load GloVe model: {e}")
            print("Initializing random embeddings...")
            self.glove_model = None
    
    def get_embedding(self, word):
        """Get embedding for a word"""
        if self.glove_model and word in self.glove_model:
            return self.glove_model[word]
        else:
            # Return zero vector if word not found
            return np.zeros(self.glove_dim)
    
    def get_sentence_embedding(self, tokens):
        """Get sentence embedding by averaging word embeddings"""
        if not tokens:
            return np.zeros(self.glove_dim)
        
        embeddings = []
        for token in tokens:
            emb = self.get_embedding(token.lower())
            embeddings.append(emb)
        
        if embeddings:
            return np.mean(embeddings, axis=0)
        return np.zeros(self.glove_dim)

class HybridEmbedding(nn.Module):
    """Weighted average of ELMo and GloVe embeddings"""
    
    def __init__(self, elmo_weight=0.6, glove_weight=0.4, glove_dim=300):
        super(HybridEmbedding, self).__init__()
        self.elmo_weight = elmo_weight
        self.glove_weight = glove_weight
        self.glove_dim = glove_dim
        
        self.elmo = ELMoEmbedding()
        self.glove = GloVeEmbedding(glove_dim=glove_dim)
        
        # Projection layer to align dimensions
        self.elmo_proj = nn.Linear(768, 1024)  # BERT output to 1024
        self.glove_proj = nn.Linear(glove_dim, 1024)
        
        # Final projection to embedding dimension
        self.final_proj = nn.Linear(1024, Config.EMBEDDING_DIM)
    
    def forward(self, texts, tokens_list=None):
        """Get hybrid embeddings"""
        # Get ELMo embeddings
        elmo_emb = self.elmo(texts)
        elmo_emb = self.elmo_proj(elmo_emb)
        
        # Get GloVe embeddings
        if tokens_list is None:
            # Extract tokens from texts
            import nltk
            from nltk.tokenize import word_tokenize
            tokens_list = [word_tokenize(str(text).lower()) for text in texts]
        
        glove_embs = []
        for tokens in tokens_list:
            glove_emb = self.glove.get_sentence_embedding(tokens)
            glove_embs.append(glove_emb)
        
        glove_emb = torch.tensor(np.array(glove_embs), dtype=torch.float32).to(elmo_emb.device)
        glove_emb = self.glove_proj(glove_emb)
        
        # Weighted average
        hybrid_emb = self.elmo_weight * elmo_emb + self.glove_weight * glove_emb
        
        # Final projection
        hybrid_emb = self.final_proj(hybrid_emb)
        
        return hybrid_emb

