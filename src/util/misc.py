import os
import sys
import logging
import coolname
from datetime import datetime
from transformers import set_seed
import torch, random
import numpy as np
def set_random_seed(seed=42):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    set_seed(seed) 


def setup_output_and_logging(log_dir):

    os.makedirs(log_dir, exist_ok=True)
    # Generate a human-readable base name
    base_name = "-".join(coolname.generate())

    # Generate precise timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Final file name stem
    filename_stem = f"{timestamp}_{base_name}"

    # Paths for .log and .txt
    log_file_path = os.path.join(log_dir, f"{filename_stem}.log")
    output_file_path = os.path.join(log_dir, f"{filename_stem}.txt")

    # Redirect stdout and stderr to .txt file
    output_file = open(output_file_path, "w", encoding="utf-8")
    sys.stdout = output_file
    sys.stderr = output_file

    # Set up verbose logging to the .log file
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    logger.debug(f"Logging to {log_file_path}")
    print(f"Standard output redirected to {output_file_path}")



