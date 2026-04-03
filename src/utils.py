from pathlib import Path
import random
import numpy as np
import yaml
import logging

# -----------------------------
# Reproducibilidad
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


# -----------------------------
# Gestión de carpetas
# -----------------------------
def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


# -----------------------------
# Cargar configuración YAML
# -----------------------------
def load_config(path: str) -> dict:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


# -----------------------------
# Logging básico
# -----------------------------
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )