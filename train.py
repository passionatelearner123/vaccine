"""
Training script for the Hybrid Deep Learning Model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from config import Config
from model import HybridSentimentModel
from data_loader import DataPreprocessor
from transformers import AutoTokenizer

class Trainer:
    """Trainer class for the hybrid model"""
    
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            texts = batch['text']
            
            optimizer.zero_grad()
            
            logits, _ = self.model(input_ids, attention_mask, texts)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                texts = batch['text']
                
                logits, _ = self.model(input_ids, attention_mask, texts)
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs, learning_rate, 
              patience=5, save_path=None):
        """Full training loop"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Model saved to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        return best_val_acc
    
    def evaluate(self, test_loader):
        """Evaluate the model on test set"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                texts = batch['text']
                
                logits, _ = self.model(input_ids, attention_mask, texts)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        cm = confusion_matrix(all_labels, all_preds)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.train_accuracies, label='Train Acc')
        axes[1].plot(self.val_accuracies, label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training history saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_confusion_matrix(self, cm, class_names=None, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

def train_model(hyperparameters=None, dataset_name="indian"):
    """Main training function"""
    # Set random seeds
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load hyperparameters
    if hyperparameters is None:
        hyperparameters = {
            'learning_rate': Config.LEARNING_RATE,
            'batch_size': Config.BATCH_SIZE,
            'lstm_hidden_size': Config.LSTM_HIDDEN_SIZE,
            'lstm_num_layers': Config.LSTM_NUM_LAYERS,
            'lstm_dropout': Config.LSTM_DROPOUT,
            'num_heads': Config.NUM_HEADS,
            'attention_dropout': Config.ATTENTION_DROPOUT,
            'elmo_weight': Config.ELMO_WEIGHT,
            'glove_weight': Config.GLOVE_WEIGHT
        }
    
    # Load data
    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    
    if dataset_name == "indian":
        data_path = Config.INDIAN_TWEETS_PATH
    elif dataset_name == "worldwide":
        data_path = Config.WORLDWIDE_TWEETS_PATH
    elif dataset_name == "semeval":
        data_path = Config.SEMEVAL_PATH
    else:
        data_path = Config.INDIAN_TWEETS_PATH
    
    df = preprocessor.load_data(data_path)
    
    # Get number of classes
    num_classes = len(df['label'].unique()) if 'label' in df.columns else 3
    
    # Create data loaders
    tokenizer = AutoTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
    train_loader, val_loader, test_loader = preprocessor.create_data_loaders(
        df['text'].tolist(),
        df['label'].tolist(),
        tokenizer,
        batch_size=int(hyperparameters['batch_size']),
        random_state=Config.RANDOM_SEED
    )
    
    # Initialize model
    print("Initializing model...")
    model = HybridSentimentModel(
        num_classes=num_classes,
        elmo_weight=hyperparameters['elmo_weight'],
        glove_weight=hyperparameters['glove_weight'],
        lstm_hidden_size=int(hyperparameters['lstm_hidden_size']),
        lstm_num_layers=int(hyperparameters['lstm_num_layers']),
        lstm_dropout=hyperparameters['lstm_dropout'],
        num_heads=int(hyperparameters['num_heads']),
        attention_dropout=hyperparameters['attention_dropout']
    ).to(device)
    
    # Create trainer
    trainer = Trainer(model, device, Config)
    
    # Train
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    model_path = os.path.join(Config.MODELS_DIR, f"hybrid_model_{dataset_name}.pth")
    
    print("Starting training...")
    best_val_acc = trainer.train(
        train_loader,
        val_loader,
        num_epochs=Config.NUM_EPOCHS,
        learning_rate=hyperparameters['learning_rate'],
        patience=Config.EARLY_STOPPING_PATIENCE,
        save_path=model_path
    )
    
    # Load best model
    model.load_state_dict(torch.load(model_path))
    
    # Evaluate
    print("\nEvaluating on test set...")
    results = trainer.evaluate(test_loader)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    
    # Plot results
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    trainer.plot_training_history(
        os.path.join(Config.RESULTS_DIR, f"training_history_{dataset_name}.png")
    )
    trainer.plot_confusion_matrix(
        results['confusion_matrix'],
        class_names=[f'Class {i}' for i in range(num_classes)],
        save_path=os.path.join(Config.RESULTS_DIR, f"confusion_matrix_{dataset_name}.png")
    )
    
    return results['accuracy'], model

