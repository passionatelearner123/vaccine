import os
import random
import numpy as np
import tensorflow as tf
import torch

def set_reproducible_seeds(seed: int = 42) -> None:
    """
    Fixes global random seeds across Python, NumPy, TensorFlow, and PyTorch 
    to guarantee exact experiment reproducibility across runs.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[REPRODUCIBILITY] Global random seeds fixed to: {seed}")

if __name__ == "__main__":
    set_reproducible_seeds(seed=42)