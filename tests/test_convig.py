from src.utils import load_config


''' Test de verificacion de config yaml.'''
config = load_config("configs/config.yaml")

print("Config cargada correctamente")
print("Modelo:", config["model"]["name"])
print("Target:", config["target"]["name"])
print("Ruta train:", config["paths"]["raw_train"])