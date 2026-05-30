# -*- coding: utf-8 -*-
"""
Utilitários de visualização e desenho de máscaras/bounding-boxes com Alpha Blending.

TAGS DE APRESENTAÇÃO:
- [CONVERSÃO_DE_ESPAÇO_DE_CORES] - cv2.cvtColor (BGR → RGB)
- [ALPHA_BLENDING_COMPOSITING] - draw_results() com α = 0.4
"""
import cv2
import numpy as np
from PIL import Image
from .config import COLORS

def draw_results(original_img, masks, boxes, scores):
    """
    Desenha máscaras e bounding boxes sobre o frame original usando Alpha Blending.
    Fórmula: C_out = alpha * C_mask + (1 - alpha) * C_orig
    """
    # [CONVERSÃO_DE_ESPAÇO_DE_CORES] Converte a imagem PIL para um array numpy (BGR) para processamento rápido
    frame = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    h, w, _ = frame.shape
    
    overlay_mask = np.zeros_like(frame, dtype=np.uint8)
    overlay_mask_bool = np.zeros((h, w), dtype=bool)
    
    if hasattr(masks, "cpu"):
        masks = masks.cpu().numpy()
    if hasattr(boxes, "cpu"):
        boxes = boxes.cpu().float().numpy()
        
    num_detected = len(masks)
    for idx in range(num_detected):
        mask = masks[idx]
        if mask.ndim == 3:
            mask = mask[0]
            
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            
        rgb = COLORS[idx % len(COLORS)]
        color_bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        
        overlay_mask[mask] = color_bgr
        overlay_mask_bool[mask] = True
        
    # [ALPHA_BLENDING_COMPOSITING] Aplica efeito Alpha blending (60% original, 40% cor da máscara, α = 0.4)
    if num_detected > 0:
        frame[overlay_mask_bool] = (frame[overlay_mask_bool] * 0.6 + overlay_mask[overlay_mask_bool] * 0.4).astype(np.uint8)
        
    # [CONVERSÃO_DE_ESPAÇO_DE_CORES] Retorna convertendo de BGR de volta para RGB para o formato PIL da interface
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

