# Perception Analysis of mRNA Vaccine Tweets (Tr-PCS)

This repository contains the official implementation of the **Transformer-based Perception Capturing System (Tr-PCS)** optimized via the **Enhanced Meerkat Optimization Algorithm (EMOA)** for mRNA vaccine sentiment analysis.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31012/)

---

## 📜 License
This project is open-source and licensed under the [MIT License](LICENSE).

---

## 🛠️ Environment Setup & Installation

To ensure full experiment reproducibility, run this project in a **Python 3.10.12** environment.

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/passionatelearner123/vaccine.git
   cd vaccine
   ```

2. **Create and Activate Virtual Environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Pinned Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLP Models & Data Packages:**
   ```bash
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
   ```

---

## 📊 Reproducible Dataset Paths & Structure

Experiments are evaluated across three benchmark datasets:

| Dataset | Annotations | Description | Reference / Download Link |
| :--- | :--- | :--- | :--- |
| **SemEval-2017 (Task 4)** | Gold Standard (Human) | Primary benchmark dataset | [SemEval-2017 Task 4](https://alt.qcri.org/semeval2017/task4/) |
| **TD_1 (Indian Tweets)** | Silver Standard (VADER) | 110,000 domain-specific tweets | Processed via `data/download_td1.py` |
| **TD_2 (Global Tweets)** | Silver Standard (VADER) | 532,190 Kaggle global tweets | Processed via `data/download_td2.py` |

### Expected Directory Layout
Place or extract the datasets into the local `data/` directory:

```text
data/
├── semeval2017/
│   ├── train.csv
│   └── test.csv
├── td1_india/
│   └── indian_vaccine_tweets.csv
└── td2_global/
    └── global_vaccine_tweets.csv
```

---

## 🔁 Reproducibility & Fixed Random Seeds

All dataset splitting (70% train / 15% validation / 15% test) and parameter optimizations use a fixed random seed (`SEED = 42`). 

To run the model training pipeline deterministically:

```bash
python train.py --dataset semeval2017 --seed 42
```