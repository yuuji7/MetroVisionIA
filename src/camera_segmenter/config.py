# -*- coding: utf-8 -*-
"""
Configurações, caminhos e carregamento do modelo SAM 3 para o Camera Segmenter.
"""
import os
import sys
import numpy as np

# Determinar diretório raiz do projeto (onde está o .env e videoreal.py)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Carregamento de variáveis de ambiente (.env) na raiz do projeto
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT_DIR, ".env"))
except ImportError:
    pass  # python-dotenv não instalado

# Adicionar caminhos para o SAM 3
_sam3_path = os.environ.get("SAM3_PATH", "")
if _sam3_path:
    sys.path.append(_sam3_path)

# Adiciona o diretório 'src' ao sys.path para garantir compatibilidade
_src_path = os.path.join(ROOT_DIR, "src")
if _src_path not in sys.path:
    sys.path.append(_src_path)

try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
    SAM3_IMPORT_ERROR = None
except ImportError as e:
    SAM3_AVAILABLE = False
    SAM3_IMPORT_ERROR = str(e)

# Cores de alto contraste para exibição das máscaras
COLORS = np.array([
    [0.9, 0.1, 0.1],  # Vermelho
    [0.1, 0.8, 0.1],  # Verde
    [0.1, 0.5, 0.95], # Azul
    [0.95, 0.85, 0.1],# Amarelo
    [0.9, 0.1, 0.9],  # Magenta
    [0.1, 0.9, 0.9],  # Ciano
    [0.95, 0.5, 0.1], # Laranja
    [0.55, 0.1, 0.95],# Roxo
    [0.1, 0.95, 0.5]  # Lima
])
