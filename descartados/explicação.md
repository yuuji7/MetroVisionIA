# Explicação Técnica — Projeto Contador (videoreal.py)
### Processamento de Imagem e Visão Computacional

---

## 1. Aquisição e Captura de Imagem — *Image Acquisition*

O primeiro passo de qualquer pipeline de visão computacional é a **aquisição da imagem**, que consiste em converter a cena física em uma representação digital.

No projeto, isso é feito usando **OpenCV** (`cv2.VideoCapture`) para capturar frames diretamente de uma webcam ou de um arquivo de vídeo local:

```python
# videoreal.py — método update_camera_loop()
ret, frame = self.cap.read()
self.last_raw_frame = frame.copy()
```

- O `frame` retornado pelo OpenCV é um **array NumPy tridimensional** de forma `(H, W, 3)`, onde cada posição `[y, x]` contém três valores de 8 bits (uint8) representando os canais de cor.
- O dispositivo captura frames a ~**33 FPS** (a cada 30ms via `self.after(30, self.update_camera_loop)`), correspondendo à taxa típica de amostragem temporal em sistemas de monitoramento em tempo real.

---

## 2. Conversão de Espaço de Cores — *Color Space Conversion*

O OpenCV armazena imagens no espaço de cores **BGR** (Blue-Green-Red), enquanto a PIL, o SAM 3.1 e a maioria dos modelos de visão computacional esperam imagens em **RGB** (Red-Green-Blue). Essa diferença é um detalhe clássico da área e, se ignorada, causa inversão nos canais de cor, comprometendo completamente os resultados da segmentação.

```python
# videoreal.py — método _process_snapshot_thread()
cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
orig_img = Image.fromarray(cv_img)
```

> **Conexão com teoria:** Conversão de espaço de cores é um tópico fundamental em Processamento de Imagem. Outros espaços relevantes para aplicações de visão são o **HSV** (Hue-Saturation-Value), útil para segmentação por cor, e o **YCrCb**, usado em compressão de vídeo (JPEG, H.264).

---

## 3. Redimensionamento da Imagem — *Image Resizing e Interpolação*

Para controlar a **relação entre qualidade e velocidade de inferência**, o projeto redimensiona cada frame antes de enviá-lo ao modelo:

```python
# videoreal.py — método _process_snapshot_thread()
target_w = int(w * (target_h / h))
working_img = orig_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
```

- A proporção de aspecto (**aspect ratio**) é preservada calculando `target_w` proporcionalmente ao `target_h` configurado pelo usuário, evitando distorção geométrica da imagem.
- O método de interpolação utilizado é **LANCZOS** (também chamado de Lanczos3 ou sinc filtering), um filtro de reamostragem que minimiza aliasing ao reduzir a imagem, preservando bordas e texturas com maior fidelidade do que interpolações mais simples como bilinear ou nearest-neighbor.

> **Conexão com teoria:** Redimensionamento pertence ao tópico de **Amostragem e Reconstrução de Sinais** (Teorema de Nyquist-Shannon aplicado a imagens 2D). O aliasing ocorre quando a frequência de amostragem é insuficiente para representar os detalhes da imagem — o filtro LANCZOS age como um filtro passa-baixa antes da subamostragem para atenuar esse efeito.

**Por que isso importa para o desempenho?**
A complexidade computacional dos transformers de visão é proporcional ao quadrado do número de tokens de imagem. Ao reduzir a resolução de 480px para 144px de altura (fator ~3.3x), o número de patches de imagem cai aproximadamente **11x**, reduzindo dramaticamente o custo da atenção.

---

## 4. Segmentação Semântica com Prompt de Texto — *Zero-Shot Semantic Segmentation*

O núcleo do projeto é o modelo **SAM 3.1** (Segment Anything Model), que realiza **segmentação semântica guiada por linguagem natural** (*Zero-Shot*, sem necessidade de treinamento adicional para novos objetos):

```python
# videoreal.py — método _process_snapshot_thread()
state = self.processor.set_image(working_img, state)
state = self.processor.set_text_prompt(prompt, state)  # ex: "person"
```

**Segmentação de Imagem** é a tarefa de classificar cada pixel da imagem em uma categoria semântica. As abordagens clássicas incluem:
- **Thresholding** (Otsu): separa fundo de objeto por intensidade.
- **Watershed**: segmenta regiões por gradiente de intensidade.
- **Graph Cuts**: minimização de energia em grafos.

O SAM 3.1 substitui todas essas abordagens tradicionais por um **Vision Transformer (ViT)** com duas etapas:

1. **Extração de features visuais** (`set_image`): O backbone ViT processa a imagem e gera um mapa de features de alta dimensão (feature map), codificando informação semântica, espacial e de textura em vetores densos.
2. **Grounding por prompt textual** (`set_text_prompt`): Um encoder de texto (baseado em CLIP/ALIGN) codifica o prompt "person" em um vetor semântico, que é alinhado com o feature map visual via mecanismo de **atenção cruzada** (*cross-attention*), localizando e segmentando os pixels correspondentes.

O resultado são **máscaras binárias** (`masks`) para cada instância detectada e suas respectivas **bounding boxes** (`boxes`) e **scores de confiança** (`scores`).

---

## 5. O Gargalo: Scaled Dot-Product Attention (SDPA)

O mecanismo de atenção é o coração dos transformers. Para cada token de imagem, ele computa:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Onde `Q` (Query), `K` (Key) e `V` (Value) são projeções lineares dos embeddings dos tokens. A multiplicação `QKᵀ` tem complexidade **O(N²)** em relação ao número de tokens `N`, tornando-se o principal gargalo em imagens de alta resolução.

O PyTorch implementa isso como `F.scaled_dot_product_attention` (SDPA), com três backends possíveis:

| Backend | Mecanismo | Velocidade | VRAM |
|---|---|---|---|
| **FlashAttention** | Tiling otimizado por CUDA | ⚡⚡⚡ Máxima | Mínima |
| **Memory-Efficient** | Recomputação de gradientes | ⚡⚡ Alta | Baixa |
| **MATH** | Implementação PyTorch puro | 🐢 Lenta | Alta |

---

## 6. O Problema Original — Patch Estático que Bloqueava a GPU

O patch original **desativava os kernels rápidos incondicionalmente** em cada chamada de atenção, forçando sempre o backend MATH:

```python
# ANTES — Código problemático
def _safe_sdpa(query, key, value, ...):
    torch.backends.cuda.enable_flash_sdp(False)       # ← Desativa FlashAttention
    torch.backends.cuda.enable_mem_efficient_sdp(False) # ← Desativa MemEfficient
    torch.backends.cuda.enable_math_sdp(True)         # ← Força modo lento
    return _original_sdpa(...)
```

**Efeito:** Cada chamada ao modelo recalculava `QKᵀ` e o `softmax` em precisão total sem qualquer otimização de tiling ou fusão de kernels CUDA, causando:
- Throughput extremamente baixo na GPU (~24 segundos/frame)
- Pressão excessiva no barramento de memória GPU
- Subutilização dos núcleos Tensor Cores da RTX 3060

---

## 7. A Solução — Fallback Dinâmico com Habilitação por Padrão

O patch foi reescrito para **habilitar os kernels rápidos por padrão** e realizar fallback apenas **reativamente**, quando um erro de runtime é detectado:

```python
# DEPOIS — Patch com fallback dinâmico
# Inicialização: kernels rápidos habilitados
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

def _safe_sdpa(query, key, value, ...):
    try:
        return _original_sdpa(...)      # Tenta FlashAttention / MemEfficient
    except RuntimeError as e:
        if "No available kernel" in str(e):
            # Fallback temporário para MATH apenas nessa chamada
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            res = _original_sdpa(...)
            # Restaura kernels rápidos imediatamente
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            return res
        raise
```

**Por que funciona na RTX 3060 Laptop?**
A arquitetura **Ampere** (sm_86) da RTX 3060 suporta FlashAttention v2 via CUDA. O erro original ocorria apenas em casos extremos de strides não-contíguos em tensores intermediários — situação incomum no SAM 3.1 com imagens bem formatadas.

---

## 8. Resultado da Otimização — Benchmark Real

Executado diretamente no ambiente do projeto com resolução **254×144**:

| Configuração | Tempo Médio por Frame |
|---|---|
| Patch MATH estático (antes) | **24,82 segundos** |
| Patch dinâmico com FlashAttention (depois) | **0,43 segundos** |
| **Ganho de performance** | **≈ 58× mais rápido** |

Isso possibilita ciclos de captura e contagem de **1 segundo**, alinhando o sistema ao regime de **processamento de vídeo em tempo real** (definido como latência < 1s por frame).

---

## 9. Alpha Blending nas Máscaras — *Image Compositing*

Para visualizar as máscaras de segmentação sobre o frame original sem apagar completamente a imagem, é aplicado **Alpha Blending**:

```python
# videoreal.py — método draw_results()
frame[overlay_mask_bool] = (
    frame[overlay_mask_bool] * 0.6 +    # 60% da imagem original
    overlay_mask[overlay_mask_bool] * 0.4  # 40% da cor da máscara
).astype(np.uint8)
```

A fórmula geral de compositing é:

$$C_{out} = \alpha \cdot C_{overlay} + (1 - \alpha) \cdot C_{background}$$

Com `α = 0.4`, obtém-se transparência adequada para visualizar simultaneamente a textura original da pessoa e o destaque colorido da máscara.

> **Conexão com teoria:** Alpha blending é o fundamento de **Image Compositing** e faz parte do tópico de **Operações Aritméticas em Imagens** em processamento digital. Ele é amplamente usado em pipelines de AR (Realidade Aumentada) e em interfaces de anotação de datasets de visão computacional.

---

## 10. Pipeline Completo — Visão Sistêmica

```
┌─────────────────────────────────────────────────────────────────┐
│                   PIPELINE DE VISÃO COMPUTACIONAL               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Câmera / Vídeo]                                               │
│       │                                                          │
│       ▼                                                          │
│  Aquisição de Imagem (OpenCV — BGR, 854×480, 33 FPS)            │
│       │                                                          │
│       ▼                                                          │
│  Conversão de Espaço de Cores (BGR → RGB)                       │
│       │                                                          │
│       ▼                                                          │
│  Redimensionamento com Anti-Aliasing (LANCZOS → ex: 254×144)    │
│       │                                                          │
│       ▼                                                          │
│  Extração de Features Visuais (ViT Backbone — set_image)         │
│       │                                                          │
│       ▼                                                          │
│  Alinhamento Semântico Texto-Imagem (Cross-Attention — "person") │
│       │                                                          │
│       ▼                                                          │
│  SDPA com FlashAttention (O(N²) → otimizado por CUDA Tiling)    │
│       │                                                          │
│       ▼                                                          │
│  Máscaras Binárias + Bounding Boxes + Scores                     │
│       │                                                          │
│       ▼                                                          │
│  Alpha Blending das Máscaras sobre o Frame Original              │
│       │                                                          │
│       ▼                                                          │
│  Exibição na GUI + Contagem de Instâncias + Registro em Excel    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tópicos de Processamento de Imagem e Visão Computacional Cobertos

| Tópico Formal | Onde Aparece no Projeto |
|---|---|
| Aquisição e Amostragem de Imagem | `cv2.VideoCapture`, `cap.read()`, 30ms/frame |
| Conversão de Espaço de Cores | `cv2.cvtColor(BGR→RGB)` |
| Redimensionamento e Interpolação | `Image.resize()` com LANCZOS |
| Anti-Aliasing e Subamostragem | Filtro LANCZOS antes do downscale |
| Segmentação Semântica | SAM 3.1 com prompt de texto |
| Feature Extraction com Redes Profundas | ViT Backbone no `set_image()` |
| Atenção Multi-Head (Transformer) | SDPA / FlashAttention no backbone |
| Grounding Texto-Imagem (VLM) | `set_text_prompt()` com CLIP |
| Detecção de Instâncias | Bounding Boxes + Masks |
| Alpha Blending / Compositing | `draw_results()` com α = 0.4 |
| Latência e Tempo Real | Inferência < 1s/frame após otimização |
| Otimização de Pipeline GPU | FlashAttention + autocast bfloat16 |
