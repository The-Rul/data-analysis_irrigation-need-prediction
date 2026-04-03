from src.utils import set_seed, ensure_dir, load_config

print("Probando utils.py...")

set_seed(42)
print("Seed fijada correctamente")

ensure_dir("reports/figures")
print("Carpeta reports/figures verificada")

config = load_config("configs/config.yaml")
print("Config leída correctamente")
print("Modelo configurado:", config["model"]["name"])