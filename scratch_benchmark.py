# -*- coding: utf-8 -*-
import os
import sys
import time
import torch
from PIL import Image

# Carregamento de variáveis de ambiente (.env)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

_sam3_path = os.environ.get("SAM3_PATH", "")
if _sam3_path:
    sys.path.append(_sam3_path)

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def run_benchmarks():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        
    print("\n--- LOADING MODEL ---")
    t_start = time.time()
    model = build_sam3_image_model(compile=False, load_from_HF=True, device=device)
    print(f"Model loaded in {time.time() - t_start:.2f} seconds.")
    
    # Create a dummy image (480p)
    img = Image.new("RGB", (854, 480), color=(128, 128, 128))
    prompt = "person"
    
    # We will test different configurations
    configs = [
        {"resolution": 1008, "sdpa": "default", "autocast": "bfloat16", "cache_text": False},
        {"resolution": 1008, "sdpa": "default", "autocast": "float16", "cache_text": False},
        {"resolution": 1008, "sdpa": "default", "autocast": "float16", "cache_text": True},
        {"resolution": 768, "sdpa": "default", "autocast": "float16", "cache_text": True},
        {"resolution": 512, "sdpa": "default", "autocast": "float16", "cache_text": True},
        {"resolution": 512, "sdpa": "math_only", "autocast": "float16", "cache_text": True},
    ]
    
    for idx, cfg in enumerate(configs):
        print(f"\n--- CONFIG {idx+1}: {cfg} ---")
        
        # Apply SDPA settings
        if cfg["sdpa"] == "math_only":
            try:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
            except Exception:
                pass
        else:
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
            except Exception:
                pass
                
        # Initialize processor with target resolution
        processor = Sam3Processor(model, resolution=cfg["resolution"], device=device)
        
        # Warm-up
        state = {}
        with torch.inference_mode():
            dtype = torch.bfloat16 if cfg["autocast"] == "bfloat16" else torch.float16
            with torch.autocast("cuda", dtype=dtype):
                state = processor.set_image(img, state)
                state = processor.set_text_prompt(prompt, state)
                
        # Benchmark loop (3 iterations)
        times_total = []
        times_image = []
        times_text = []
        times_forward = []
        
        # Pre-encode text if cache_text is True
        cached_text = None
        if cfg["cache_text"]:
            with torch.inference_mode():
                dtype = torch.bfloat16 if cfg["autocast"] == "bfloat16" else torch.float16
                with torch.autocast("cuda", dtype=dtype):
                    cached_text = model.backbone.forward_text([prompt], device=device)
                    
        for i in range(3):
            t0 = time.time()
            state = {}
            dtype = torch.bfloat16 if cfg["autocast"] == "bfloat16" else torch.float16
            
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=dtype):
                    t_img_start = time.time()
                    state = processor.set_image(img, state)
                    t_img_end = time.time()
                    times_image.append(t_img_end - t_img_start)
                    
                    if cfg["cache_text"]:
                        t_text_start = time.time()
                        state["backbone_out"].update(cached_text)
                        if "geometric_prompt" not in state:
                            state["geometric_prompt"] = model._get_dummy_prompt()
                        state = processor._forward_grounding(state)
                        t_text_end = time.time()
                        times_text.append(0.0) # cached
                        times_forward.append(t_text_end - t_text_start)
                    else:
                        t_text_start = time.time()
                        state = processor.set_text_prompt(prompt, state)
                        t_text_end = time.time()
                        times_text.append(t_text_end - t_text_start)
                        times_forward.append(0.0) # integrated
                        
            times_total.append(time.time() - t0)
            
        avg_total = sum(times_total) / len(times_total)
        avg_image = sum(times_image) / len(times_image)
        avg_text = sum(times_text) / len(times_text)
        avg_forward = sum(times_forward) / len(times_forward)
        
        print(f"  Average Total Frame Time: {avg_total:.4f}s")
        print(f"  Average Set Image Time:   {avg_image:.4f}s")
        print(f"  Average Text/Forward Time:{avg_text + avg_forward:.4f}s")

if __name__ == "__main__":
    run_benchmarks()
