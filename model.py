"""
Hybrid Deep Learning Model: Bi-LSTM + BERT + Multi-Head Attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from embeddings import HybridEmbedding
from config import Config

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        
        # Apply layer norm
        x_norm = self.layer_norm(x)
        
        # Linear projections
        Q = self.query(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # Output projection
        output = self.out_proj(attn_output)
        
        # Residual connection
        output = output + x
        
        return output, attn_weights

class HybridSentimentModel(nn.Module):
    """Hybrid Deep Learning Model for Sentiment Analysis"""
    
    def __init__(self, num_classes=3, elmo_weight=0.6, glove_weight=0.4,
                 lstm_hidden_size=256, lstm_num_layers=2, lstm_dropout=0.3,
                 num_heads=8, attention_dropout=0.1):
        super(HybridSentimentModel, self).__init__()
        
        self.num_classes = num_classes
        self.lstm_hidden_size = lstm_hidden_size
        
        # Hybrid embedding layer
        self.hybrid_embedding = HybridEmbedding(
            elmo_weight=elmo_weight,
            glove_weight=glove_weight
        )
        
        # BERT model
        self.bert_model = AutoModel.from_pretrained(Config.BERT_MODEL_NAME)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        self.bert_hidden_size = Config.BERT_HIDDEN_SIZE
        
        # Bi-LSTM layer
        self.bi_lstm = nn.LSTM(
            input_size=Config.EMBEDDING_DIM,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Multi-Head Attention
        self.multi_head_attention = MultiHeadAttention(
            embed_dim=lstm_hidden_size * 2,  # Bidirectional
            num_heads=num_heads,
            dropout=attention_dropout
        )
        
        # Feature fusion layer
        fusion_input_size = (lstm_hidden_size * 2) + self.bert_hidden_size
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classification head
        self.classifier = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, input_ids, attention_mask, texts=None):
        batch_size = input_ids.size(0)
        
        # Get BERT embeddings
        bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        bert_pooled = bert_outputs.pooler_output  # [batch_size, 768]
        
        # Get hybrid embeddings (ELMo + GloVe)
        if texts is None:
            # Decode texts from input_ids
            texts = [self.bert_tokenizer.decode(ids, skip_special_tokens=True) 
                    for ids in input_ids]
        
        # Extract tokens for GloVe
        import nltk
        from nltk.tokenize import word_tokenize
        tokens_list = [word_tokenize(str(text).lower()) for text in texts]
        
        hybrid_emb = self.hybrid_embedding(texts, tokens_list)  # [batch_size, embedding_dim]
        
        # Reshape for LSTM (add sequence dimension)
        # For simplicity, we'll use the embedding as a single timestep
        # In practice, you might want to use word-level embeddings
        hybrid_emb = hybrid_emb.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        
        # Bi-LSTM
        lstm_out, (hidden, cell) = self.bi_lstm(hybrid_emb)
        # lstm_out: [batch_size, seq_len, hidden_size * 2]
        
        # Multi-Head Attention
        attn_output, attn_weights = self.multi_head_attention(lstm_out, attention_mask.unsqueeze(1))
        
        # Pooling: take the last timestep or mean pooling
        lstm_pooled = attn_output.squeeze(1)  # [batch_size, hidden_size * 2]
        
        # Feature fusion: concatenate BERT and Bi-LSTM features
        fused_features = torch.cat([bert_pooled, lstm_pooled], dim=1)
        fused_features = self.fusion_layer(fused_features)
        fused_features = self.dropout(fused_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits, attn_weights

