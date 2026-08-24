"""
Main script for training and evaluating the Hybrid Deep Learning Framework
"""
import argparse
import os
from config import Config
from train import train_model
from emoa import EMOA
import torch

def objective_function(hyperparameters):
    """
    Objective function for EMOA optimization
    Returns validation accuracy (fitness)
    """
    try:
        # Normalize weights
        total_weight = hyperparameters['elmo_weight'] + hyperparameters['glove_weight']
        hyperparameters['elmo_weight'] = hyperparameters['elmo_weight'] / total_weight
        hyperparameters['glove_weight'] = hyperparameters['glove_weight'] / total_weight
        
        # Train model with given hyperparameters
        val_acc, _ = train_model(hyperparameters, dataset_name="indian")
        
        return val_acc
    except Exception as e:
        print(f"Error in objective function: {e}")
        return 0.0

def main():
    parser = argparse.ArgumentParser(description='Hybrid Deep Learning Framework for Sentiment Analysis')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'optimize', 'evaluate'],
                       help='Mode: train, optimize, or evaluate')
    parser.add_argument('--dataset', type=str, default='indian',
                       choices=['indian', 'worldwide', 'semeval'],
                       help='Dataset to use')
    parser.add_argument('--use_emoa', action='store_true',
                       help='Use EMOA for hyperparameter optimization')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    if args.mode == 'optimize' or args.use_emoa:
        print("=" * 60)
        print("Enhanced Meerkat Optimization Algorithm (EMOA)")
        print("=" * 60)
        
        # Initialize EMOA
        emoa = EMOA(
            objective_function=objective_function,
            search_space=Config.HYPERPARAMETER_SPACE,
            population_size=Config.EMOA_POPULATION_SIZE,
            max_iterations=Config.EMOA_MAX_ITERATIONS,
            crossover_rate=Config.EMOA_CROSSOVER_RATE,
            mutation_rate=Config.EMOA_MUTATION_RATE,
            reinitialization_threshold=Config.EMOA_REINITIALIZATION_THRESHOLD
        )
        
        # Run optimization
        best_hyperparameters = emoa.optimize()
        
        print("\n" + "=" * 60)
        print("Training with optimized hyperparameters...")
        print("=" * 60)
        
        # Train with best hyperparameters
        accuracy, model = train_model(best_hyperparameters, dataset_name=args.dataset)
        
        print(f"\nFinal Model Accuracy: {accuracy:.4f}")
        
    elif args.mode == 'train':
        print("=" * 60)
        print("Training Hybrid Deep Learning Model")
        print("=" * 60)
        
        accuracy, model = train_model(dataset_name=args.dataset)
        
        print(f"\nFinal Model Accuracy: {accuracy:.4f}")
        
    elif args.mode == 'evaluate':
        print("=" * 60)
        print("Evaluating Model")
        print("=" * 60)
        
        # Load model and evaluate
        from train import Trainer
        from model import HybridSentimentModel
        from data_loader import DataPreprocessor
        from transformers import AutoTokenizer
        
        device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
        
        # Load data
        preprocessor = DataPreprocessor()
        if args.dataset == "indian":
            data_path = Config.INDIAN_TWEETS_PATH
        elif args.dataset == "worldwide":
            data_path = Config.WORLDWIDE_TWEETS_PATH
        else:
            data_path = Config.SEMEVAL_PATH
        
        df = preprocessor.load_data(data_path)
        num_classes = len(df['label'].unique()) if 'label' in df.columns else 3
        
        tokenizer = AutoTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        _, _, test_loader = preprocessor.create_data_loaders(
            df['text'].tolist(),
            df['label'].tolist(),
            tokenizer,
            batch_size=Config.BATCH_SIZE
        )
        
        # Load model
        model = HybridSentimentModel(num_classes=num_classes).to(device)
        model_path = os.path.join(Config.MODELS_DIR, f"hybrid_model_{args.dataset}.pth")
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model not found at {model_path}. Please train first.")
            return
        
        # Evaluate
        trainer = Trainer(model, device, Config)
        results = trainer.evaluate(test_loader)
        
        print(f"\nTest Results:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")

if __name__ == "__main__":
    main()

