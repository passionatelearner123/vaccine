HYBRID DEEP LEARNING FRAMEWORK FOR SENTIMENT ANALYSIS
======================================================

A hybrid deep learning framework for tweet sentiment analysis that combines
BERT, a Bi-LSTM encoder, Multi-Head Attention, and hybrid ELMo/GloVe
embeddings, with hyperparameters tuned automatically via a custom Enhanced
Meerkat Optimization Algorithm (EMOA).


OVERVIEW
--------

The model fuses two complementary representations of each tweet:

  1. Contextual BERT embeddings - the pooled [CLS] output of a pretrained
     BERT model.
  2. Hybrid ELMo + GloVe embeddings - a weighted combination of contextual
     ELMo (or BERT-based fallback) embeddings and static GloVe word
     embeddings, projected into a shared space and passed through a
     Bi-LSTM and Multi-Head Attention block.

These two feature sets are concatenated and passed through a fusion MLP to
produce the final sentiment classification (positive / neutral / negative,
or however many classes are present in the data).

Hyperparameters (learning rate, LSTM size, attention heads, embedding
weights, etc.) can be tuned automatically with EMOA, a nature-inspired
metaheuristic that models meerkat sentinel, foraging, and alert behaviors
on top of a genetic-algorithm-style crossover/mutation loop.


PROJECT STRUCTURE
------------------

  .
  |-- main.py           Entry point - train, optimize, or evaluate
  |-- config.py         Central configuration (paths, hyperparameters,
  |                     search space)
  |-- data_loader.py    Text cleaning, tokenization, PyTorch
  |                     Dataset/DataLoader creation
  |-- embeddings.py     ELMo, GloVe, and Hybrid embedding modules
  |-- model.py          HybridSentimentModel (BERT + Bi-LSTM +
  |                     Multi-Head Attention)
  |-- train.py          Trainer class: training loop, validation,
  |                     evaluation, plotting
  |-- emoa.py           Enhanced Meerkat Optimization Algorithm for
  |                     hyperparameter search
  `-- requirements.txt

  NOTE: config.py is imported by every module but was not included in the
  provided files. See the CONFIGURATION section below for the values it
  must define.


INSTALLATION
------------

  git clone https://github.com/passionatelearner123/vaccine
  cd mrnavaccine
  pip install -r requirements.txt

The first run will automatically download required NLTK resources
(punkt, stopwords).

If you plan to use GPU acceleration, install a CUDA-enabled build of
PyTorch and TensorFlow matching your driver version. See:
  - PyTorch:     https://pytorch.org/get-started/locally/
  - TensorFlow:  https://www.tensorflow.org/install


CONFIGURATION
-------------

Create a config.py in the project root with (at minimum) the following
attributes:

  class Config:
      # Data
      DATA_DIR = "data"
      MODELS_DIR = "models"
      RESULTS_DIR = "results"
      INDIAN_TWEETS_PATH = "data/indian_tweets.csv"
      WORLDWIDE_TWEETS_PATH = "data/worldwide_tweets.csv"
      SEMEVAL_PATH = "data/semeval.csv"

      # Model / embeddings
      DEVICE = "cuda"                 # or "cpu"
      MAX_SEQUENCE_LENGTH = 128
      EMBEDDING_DIM = 300
      BERT_MODEL_NAME = "bert-base-uncased"
      BERT_HIDDEN_SIZE = 768

      # Training defaults
      LEARNING_RATE = 2e-5
      BATCH_SIZE = 32
      NUM_EPOCHS = 20
      EARLY_STOPPING_PATIENCE = 5
      RANDOM_SEED = 42

      # HybridSentimentModel defaults
      LSTM_HIDDEN_SIZE = 256
      LSTM_NUM_LAYERS = 2
      LSTM_DROPOUT = 0.3
      NUM_HEADS = 8
      ATTENTION_DROPOUT = 0.1
      ELMO_WEIGHT = 0.6
      GLOVE_WEIGHT = 0.4

      # EMOA
      EMOA_POPULATION_SIZE = 30
      EMOA_MAX_ITERATIONS = 50
      EMOA_CROSSOVER_RATE = 0.8
      EMOA_MUTATION_RATE = 0.2
      EMOA_REINITIALIZATION_THRESHOLD = 0.1

      HYPERPARAMETER_SPACE = {
          "learning_rate": (1e-5, 5e-5),
          "batch_size": [16, 32, 64],
          "lstm_hidden_size": [128, 256, 512],
          "lstm_num_layers": [1, 2, 3],
          "lstm_dropout": (0.1, 0.5),
          "num_heads": [4, 8, 16],
          "attention_dropout": (0.0, 0.3),
          "elmo_weight": (0.0, 1.0),
          "glove_weight": (0.0, 1.0),
      }

Input CSVs are expected to have a "text" column and a "label" column
(string class names are label-encoded automatically). If a file can't be
loaded, DataPreprocessor falls back to a small built-in sample dataset for
demonstration.


USAGE
-----

Train with default hyperparameters:

  python main.py --mode train --dataset indian

Optimize hyperparameters with EMOA, then train the final model:

  python main.py --mode optimize --dataset indian

  or equivalently:

  python main.py --dataset indian --use_emoa

Evaluate a previously trained model:

  python main.py --mode evaluate --dataset indian

--dataset accepts: indian, worldwide, or semeval.

Trained model weights are saved to:
  Config.MODELS_DIR/hybrid_model_<dataset>.pth

Training curves and confusion matrices are saved to:
  Config.RESULTS_DIR/


HOW IT WORKS
------------

Data pipeline (data_loader.py)
  Lowercases text, strips URLs/mentions/hashtags/punctuation, tokenizes
  with NLTK, removes stopwords, label-encodes classes, and produces
  stratified train/val/test DataLoaders.

Embeddings (embeddings.py)
  - ELMoEmbedding tries to load ELMo from TensorFlow Hub; if unavailable,
    it falls back to mean-pooled BERT embeddings.
  - GloVeEmbedding loads pretrained GloVe vectors via gensim.downloader
    and averages word vectors per sentence (zero vector for
    out-of-vocabulary words or if the download fails).
  - HybridEmbedding projects both into a shared 1024-d space, combines
    them with configurable weights, and projects to
    Config.EMBEDDING_DIM.

Model (model.py)
  HybridSentimentModel runs BERT on the tokenized input for a pooled
  representation, feeds the hybrid embedding through a Bi-LSTM and a
  custom MultiHeadAttention block (pre-norm, residual connection),
  concatenates both representations, and classifies through a fusion
  MLP.

Training (train.py)
  Standard AdamW + cross-entropy training loop with gradient clipping,
  ReduceLROnPlateau scheduling, early stopping, and
  accuracy/precision/recall/F1 evaluation. Also plots loss/accuracy
  curves and a confusion matrix heatmap.

EMOA (emoa.py)
  A population-based optimizer over Config.HYPERPARAMETER_SPACE:
  tournament selection, crossover, Gaussian mutation, meerkat-inspired
  sentinel/foraging moves, and periodic re-initialization of stagnant
  individuals when convergence stalls. The objective function in main.py
  trains a full model per candidate and returns validation accuracy as
  fitness - expect this to be computationally expensive for large
  population sizes / iteration counts.


REQUIREMENTS
------------

See requirements.txt. Core dependencies: PyTorch, Transformers,
TensorFlow + TensorFlow Hub, Gensim, scikit-learn, NLTK, pandas,
matplotlib, seaborn, tqdm.


KNOWN CONSIDERATIONS
---------------------

  - ELMo via TF Hub may fail to load depending on your TensorFlow version
    (the elmo/3 module is TF1-style); the code transparently falls back
    to BERT-based embeddings in that case.
  - GloVe download via gensim.downloader fetches a large pretrained model
    on first use and requires internet access; embeddings fall back to
    zero vectors if it can't be loaded.
  - Running both BERT and a second embedding model (ELMo/BERT-fallback)
    simultaneously is memory-intensive - a GPU with sufficient VRAM is
    recommended for anything beyond small batch sizes.
  - EMOA's objective function fully trains a model per candidate per
    generation, so EMOA_POPULATION_SIZE x EMOA_MAX_ITERATIONS full
    training runs will be executed - tune these down for experimentation.
