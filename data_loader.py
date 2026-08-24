"""
Data loading and preprocessing utilities
"""
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class TweetDataset(Dataset):
    """Custom Dataset for tweet sentiment analysis"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'text': text
        }

class DataPreprocessor:
    """Preprocesses tweets for sentiment analysis"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.label_encoder = LabelEncoder()
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenize text"""
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        return tokens
    
    def preprocess_dataframe(self, df, text_column='text', label_column='label'):
        """Preprocess entire dataframe"""
        df = df.copy()
        df[text_column] = df[text_column].apply(self.clean_text)
        df = df[df[text_column].str.len() > 0]  # Remove empty texts
        
        if label_column in df.columns:
            df[label_column] = self.label_encoder.fit_transform(df[label_column])
        
        return df
    
    def load_data(self, file_path, text_column='text', label_column='label'):
        """Load and preprocess data from file"""
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            print("Creating sample data for demonstration...")
            return self._create_sample_data()
        
        df = self.preprocess_dataframe(df, text_column, label_column)
        return df
    
    def _create_sample_data(self):
        """Create sample data for demonstration"""
        sample_texts = [
            "Great vaccine rollout! Feeling safe and protected.",
            "Not sure about this vaccine, need more information.",
            "Vaccination is essential for public health.",
            "Concerned about vaccine side effects.",
            "Thankful for the vaccine program!"
        ]
        sample_labels = ['positive', 'neutral', 'positive', 'negative', 'positive']
        
        df = pd.DataFrame({
            'text': sample_texts,
            'label': sample_labels
        })
        
        return self.preprocess_dataframe(df)
    
    def create_data_loaders(self, texts, labels, tokenizer, batch_size=32, 
                           test_size=0.2, val_size=0.1, random_state=42):
        """Create train, validation, and test data loaders"""
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size/(1-test_size), 
            random_state=random_state, stratify=y_train
        )
        
        # Create datasets
        train_dataset = TweetDataset(X_train, y_train, tokenizer, Config.MAX_SEQUENCE_LENGTH)
        val_dataset = TweetDataset(X_val, y_val, tokenizer, Config.MAX_SEQUENCE_LENGTH)
        test_dataset = TweetDataset(X_test, y_test, tokenizer, Config.MAX_SEQUENCE_LENGTH)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader

