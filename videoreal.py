# -*- coding: utf-8 -*-
"""
SAM 3.1 - Mapeador e Contabilizador de Câmera em Tempo Real (Zero-Shot)
====================================================================
Este utilitário permite capturar o feed da câmera do PC em tempo real, 
extrair frames em um ciclo temporizado configurável, aplicar a segmentação
com o modelo SAM 3.1 baseado em um prompt de texto e exibir a contagem
de objetos de forma contínua, com liberação rígida de memória a cada ciclo.

Este script é o ponto de entrada principal que inicializa o patch de
otimização SDPA e inicia a aplicação GUI modularizada.
"""

import os
import sys

# 1. Configuração de caminhos: garante que a pasta 'src' está no sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 2. Importa e aplica o patch de compatibilidade SDPA para GPUs RTX
from camera_segmenter import apply_sdpa_patch
apply_sdpa_patch()

# 3. Inicializa e executa a aplicação Tkinter
if __name__ == "__main__":
    from camera_segmenter import SAM3CameraSegmenterApp
    app = SAM3CameraSegmenterApp()
    app.mainloop()
