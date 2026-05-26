import sys
import os

# Configuração estrita de VRAM e desativação global de backends SDPA incompatíveis
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

try:
    import torch
    # Força o PyTorch a desativar kernels que geram erros em determinadas GPUs Laptop RTX 3060
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except ImportError:
    pass

# Adiciona o diretório src/ ao sys.path para importação dos módulos
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    from video_segmenter_gui import SAM3VideoTrackerApp
    app = SAM3VideoTrackerApp()
    app.mainloop()

