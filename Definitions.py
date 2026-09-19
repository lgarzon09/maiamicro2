import os

# Ruta raíz del proyecto (se usa para construir rutas relativas robustas).
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, 'configuration.conf')
