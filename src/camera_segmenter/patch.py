# -*- coding: utf-8 -*-
"""
Patch de compatibilidade e otimização de SDPA para GPUs RTX.

TAG DE APRESENTAÇÃO:
[ATENÇÃO_MULTI_HEAD] - SDPA / FlashAttention no backbone do Transformer
"""
import os
import functools
import torch
import torch.nn.functional as _F_torch

def apply_sdpa_patch():
    """Ativa os backends otimizados de SDPA com fallback seguro."""
    try:
        # [ATENÇÃO_MULTI_HEAD] Ativação de FlashAttention e Memory-Efficient SDP
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    except Exception:
        pass


    _original_sdpa = _F_torch.scaled_dot_product_attention

    @functools.wraps(_original_sdpa)
    def _safe_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        """SDPA com fallback dinâmico e automático para backend MATH se falhar."""
        try:
            return _original_sdpa(query, key, value, attn_mask=attn_mask,
                                   dropout_p=dropout_p, is_causal=is_causal, **kwargs)
        except RuntimeError as e:
            err_str = str(e)
            if "No available kernel" in err_str or "not supported" in err_str.lower() or "abort" in err_str.lower():
                try:
                    # Fallback temporário para MATH
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    torch.backends.cuda.enable_math_sdp(True)
                    res = _original_sdpa(query, key, value, attn_mask=attn_mask,
                                         dropout_p=dropout_p, is_causal=is_causal, **kwargs)
                    # Restaura os kernels rápidos para as próximas chamadas
                    torch.backends.cuda.enable_flash_sdp(True)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                    return res
                except Exception:
                    pass
            raise

    # Patch em ambas as referências possíveis
    _F_torch.scaled_dot_product_attention = _safe_sdpa
    if hasattr(torch, "scaled_dot_product_attention"):
        torch.scaled_dot_product_attention = _safe_sdpa
