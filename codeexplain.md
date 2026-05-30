# MetroVision IA — Explicação Técnica do Código
### Processamento de Imagem, Visão Computacional & Dashboard de Lotação

---

## Sumário

1. [Arquitetura e Estrutura do Projeto](#1-arquitetura-e-estrutura-do-projeto)
2. [Aquisição e Amostragem de Imagem](#2-aquisição-e-amostragem-de-imagem)
3. [Conversão de Espaço de Cores](#3-conversão-de-espaço-de-cores)
4. [Redimensionamento, Interpolação e Anti-Aliasing](#4-redimensionamento-interpolação-e-anti-aliasing)
5. [Segmentação Semântica Zero-Shot](#5-segmentação-semântica-zero-shot)
6. [Feature Extraction com Vision Transformer](#6-feature-extraction-com-vision-transformer)
7. [Atenção Multi-Head e Otimização SDPA](#7-atenção-multi-head-e-otimização-sdpa)
8. [Grounding Texto-Imagem (VLM)](#8-grounding-texto-imagem-vlm)
9. [Detecção de Instâncias](#9-detecção-de-instâncias)
10. [Alpha Blending e Compositing](#10-alpha-blending-e-compositing)
11. [Otimização de Pipeline GPU](#11-otimização-de-pipeline-gpu)
12. [Latência e Processamento em Tempo Real](#12-latência-e-processamento-em-tempo-real)
13. [Persistência em JSON e Dashboard em Tempo Real](#13-persistência-em-json-e-dashboard-em-tempo-real)
14. [Mapeamento Consolidado](#14-mapeamento-consolidado)

---

## 1. Arquitetura e Estrutura do Projeto

O projeto utiliza uma arquitetura modular distribuída em pacotes Python dentro do diretório `src/camera_segmenter/`:

```
ProjetoContador/
├── videoreal.py                     # Ponto de entrada principal (launcher)
├── dashboard.html                   # Painel HTML5 de lotação em tempo real
├── relatorio_contagem.json          # Banco de dados de medições (JSON)
│
├── src/
│   └── camera_segmenter/
│       ├── __init__.py              # Expõe a API pública do pacote
│       ├── patch.py                 # Otimização SDPA / FlashAttention
│       ├── config.py                # Variáveis de ambiente e constantes
│       ├── visualization.py         # Renderização de máscaras (Alpha Blending)
│       ├── json_db.py               # Persistência JSON + exportação JS
│       └── gui.py                   # Interface Tkinter + pipeline de inferência
│
└── media/
    ├── capturas/                    # Imagens capturadas com ID único
    └── dados_dashboard.js           # Dados sincronizados para o HTML
```

O fluxo de dados segue o pipeline clássico de visão computacional:

```
Câmera/Vídeo → Aquisição → Conversão de Cores → Redimensionamento
    → Extração de Features (ViT) → Grounding Texto-Imagem (CLIP)
    → Segmentação (Máscaras + Boxes) → Alpha Blending → Exibição + JSON
```

---

## 2. Aquisição e Amostragem de Imagem

**Conceito:** A aquisição de imagem é o primeiro passo de qualquer sistema de visão computacional. Consiste em converter a cena física em uma representação digital matricial.

**Implementação:** O OpenCV (`cv2.VideoCapture`) conecta-se ao dispositivo de captura (webcam ou arquivo de vídeo) e amostra quadros continuamente.

```python
# gui.py — _init_camera_thread()
# Conexão e inicialização do dispositivo de captura
cap = cv2.VideoCapture(cam_idx)

# Configuração da resolução de entrada para 480p
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

O loop principal de aquisição opera a cada **30 milissegundos** (~33 FPS), garantindo uma taxa de amostragem temporal adequada para monitoramento em tempo real:

```python
# gui.py — update_camera_loop()
def update_camera_loop(self):
    # Aquisição contínua do quadro (frame) da câmera
    ret, frame = self.cap.read()
    if ret:
        self.last_raw_frame = frame.copy()
        if self.show_live_feed:
            self.display_opencv_frame(frame)

    # Mantém atualizando com intervalo regulado de 30ms
    self.after(30, self.update_camera_loop)
```

**Detalhes técnicos:**
- O `frame` retornado é um **array NumPy tridimensional** de forma `(H, W, 3)`, onde cada pixel contém 3 canais de 8 bits (uint8).
- O método `self.after(30, ...)` do Tkinter agenda a próxima chamada em 30ms, criando um ciclo de amostragem temporal consistente.
- O `frame.copy()` garante que o snapshot para processamento é independente do buffer da câmera.

---

## 3. Conversão de Espaço de Cores

**Conceito:** O OpenCV armazena imagens no espaço de cores **BGR** (Blue-Green-Red), enquanto a PIL, o SAM 3.1 e a maioria dos modelos de visão computacional esperam **RGB** (Red-Green-Blue). Ignorar esta diferença causa inversão nos canais de cor.

**Implementação na exibição ao vivo:**

```python
# gui.py — display_opencv_frame()
# Converte de BGR (Padrão OpenCV) para RGB (Padrão PIL/GUI/SAM 3.1)
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(rgb_frame)
```

**Implementação no pipeline de segmentação:**

```python
# gui.py — _process_snapshot_thread()
# Conversão de cores BGR -> RGB do frame estático capturado
cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
orig_img = Image.fromarray(cv_img)
```

**Implementação na visualização de máscaras:**

```python
# visualization.py — draw_results()
# Converte a imagem PIL para array numpy (BGR) para processamento rápido
frame = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)

# ... (processamento das máscaras) ...

# Retorna convertendo de BGR de volta para RGB
return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
```

**Conexão teórica:** Conversão de espaço de cores é um tópico fundamental em Processamento de Imagem. Outros espaços relevantes incluem **HSV** (para segmentação por cor) e **YCrCb** (para compressão de vídeo como JPEG e H.264).

---

## 4. Redimensionamento, Interpolação e Anti-Aliasing

**Conceito:** Para controlar a relação entre qualidade e velocidade de inferência, o projeto redimensiona cada frame antes de enviá-lo ao modelo. O filtro **LANCZOS** minimiza aliasing ao atuar como filtro passa-baixa antes da subamostragem.

**Implementação:**

```python
# gui.py — _process_snapshot_thread()
# Cálculo proporcional mantendo aspect ratio
target_w = int(w * (target_h / h))
target_w = (target_w // 2) * 2   # Garante dimensões pares para o encoder
target_h = (target_h // 2) * 2

# Reamostragem LANCZOS atuando como filtro passa-baixa anti-aliasing
working_img = orig_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
```

**Por que isso importa para o desempenho?**

A complexidade computacional dos Vision Transformers é proporcional ao **quadrado** do número de tokens de imagem. Ao reduzir a resolução de 480px para 240px de altura (fator ~2x), o número de patches de imagem cai aproximadamente **4x**, reduzindo dramaticamente o custo da atenção.

**Conexão teórica:** Redimensionamento pertence ao tópico de **Amostragem e Reconstrução de Sinais** (Teorema de Nyquist-Shannon aplicado a imagens 2D). O filtro LANCZOS (sinc truncado) é um dos melhores filtros de reamostragem disponíveis, preservando bordas e texturas com fidelidade superior a interpolações bilineares ou nearest-neighbor.

---

## 5. Segmentação Semântica Zero-Shot

**Conceito:** O núcleo do projeto é o modelo **SAM 3.1** (Segment Anything Model), que realiza segmentação semântica guiada por linguagem natural sem necessidade de treinamento específico para novos objetos (zero-shot).

**Implementação:**

```python
# gui.py — _process_snapshot_thread()
# Inicialização da inferência zero-shot com SAM 3.1
state = {}
self.processor.set_confidence_threshold(user_thresh, state)

with torch.inference_mode():
    if device == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Vision Transformer Backbone extraindo mapa visual
            state = self.processor.set_image(working_img, state)
            # Alinhamento multimodal texto-imagem via CLIP
            state = self.processor.set_text_prompt(prompt, state)
```

O processo envolve duas etapas principais:

1. **`set_image()`** — O backbone ViT processa a imagem e gera um mapa de features de alta dimensão, codificando informação semântica, espacial e de textura.
2. **`set_text_prompt()`** — Um encoder de texto baseado em CLIP codifica o prompt (ex: "person") em um vetor semântico, que é alinhado com o feature map visual via atenção cruzada.

O resultado são **máscaras binárias** para cada instância detectada, com suas respectivas **bounding boxes** e **scores de confiança**.

---

## 6. Feature Extraction com Vision Transformer

**Conceito:** A extração de features é o processo de transformar pixels brutos em representações vetoriais de alta dimensão que codificam significado semântico. O SAM 3.1 utiliza um **Vision Transformer (ViT)** como backbone.

**Implementação:**

```python
# gui.py — _process_snapshot_thread()
# Vision Transformer (ViT) Backbone extraindo o mapa visual de alta dimensão
state = self.processor.set_image(working_img, state)
```

**Como funciona internamente:**

1. A imagem é dividida em **patches** (blocos de 16×16 pixels tipicamente).
2. Cada patch é projetado linearmente em um embedding vetorial.
3. Positional embeddings são adicionados para preservar informação espacial.
4. A sequência de embeddings passa por múltiplas camadas de **self-attention** e **feed-forward networks**.
5. O resultado é um **feature map denso** onde cada posição codifica contexto global da imagem.

---

## 7. Atenção Multi-Head e Otimização SDPA

**Conceito:** O mecanismo de atenção é o coração dos Transformers. Para cada token, computa:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

A multiplicação `QK^T` tem complexidade **O(N²)**, tornando-se o principal gargalo em imagens de alta resolução.

**Implementação do patch de otimização:**

```python
# patch.py — apply_sdpa_patch()
# Ativação de FlashAttention e Memory-Efficient SDP
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
```

**Fallback dinâmico com restauração automática:**

```python
# patch.py — _safe_sdpa()
def _safe_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    try:
        return _original_sdpa(query, key, value, ...)  # Tenta FlashAttention
    except RuntimeError as e:
        if "No available kernel" in str(e):
            # Fallback temporário para MATH apenas nesta chamada
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            res = _original_sdpa(query, key, value, ...)
            # Restaura kernels rápidos imediatamente
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            return res
        raise
```

**Comparação de backends SDPA:**

| Backend           | Mecanismo                    | Velocidade    | VRAM   |
|-------------------|------------------------------|---------------|--------|
| FlashAttention    | Tiling otimizado por CUDA    | ⚡⚡⚡ Máxima  | Mínima |
| Memory-Efficient  | Recomputação de gradientes   | ⚡⚡ Alta      | Baixa  |
| MATH              | Implementação PyTorch puro   | 🐢 Lenta      | Alta   |

**Resultado real:** A otimização reduziu o tempo de inferência de **24,82 segundos** para **0,43 segundos** por frame — um ganho de **≈ 58×**.

---

## 8. Grounding Texto-Imagem (VLM)

**Conceito:** O grounding texto-imagem permite localizar regiões em uma imagem que correspondem a uma descrição textual. Combina um **encoder de linguagem** (CLIP) com o **feature map visual** do ViT através de atenção cruzada (*cross-attention*).

**Implementação:**

```python
# gui.py — _process_snapshot_thread()
# Alinhamento multimodal texto-imagem do prompt usando o encoder CLIP
state = self.processor.set_text_prompt(prompt, state)
```

**Fluxo interno:**

1. O texto "person" é tokenizado e processado pelo encoder CLIP.
2. O resultado é um vetor semântico de 512/768 dimensões.
3. Este vetor é usado como *query* em um mecanismo de atenção cruzada contra os tokens visuais.
4. O modelo localiza e segmenta apenas os pixels correspondentes ao conceito textual.

---

## 9. Detecção de Instâncias

**Conceito:** A detecção de instâncias identifica e separa individualmente cada objeto detectado, produzindo máscaras binárias, caixas delimitadoras e scores de confiança.

**Implementação da extração de resultados:**

```python
# gui.py — _process_snapshot_thread()
# Extração das bounding boxes, máscaras binárias e scores
raw_scores = state.get("scores")
scores_np = raw_scores.cpu().float().numpy()

num_detected = len(scores_np)
keep_indices = np.arange(num_detected)
```

**Filtragem e separação por instância:**

```python
# gui.py — _process_snapshot_thread()
# Separação de caixas e máscaras filtradas por score
raw_masks = state.get("masks")
filtered_masks = raw_masks[keep_indices]

raw_boxes = state.get("boxes")
filtered_boxes = raw_boxes[keep_indices]
filtered_scores = raw_scores[keep_indices]

masked_pil = self.draw_results(orig_img, filtered_masks, filtered_boxes, filtered_scores)
```

Cada instância detectada recebe:
- **Máscara binária** `(H, W)` — indica quais pixels pertencem ao objeto.
- **Bounding box** `[x1, y1, x2, y2]` — retângulo envolvente.
- **Score de confiança** `[0.0, 1.0]` — grau de certeza do modelo.

---

## 10. Alpha Blending e Compositing

**Conceito:** Para visualizar as máscaras de segmentação sobre o frame original sem ocultá-lo, é aplicado **Alpha Blending** — uma operação aritmética de composição de imagens.

**Fórmula geral:**

```
C_out = α × C_overlay + (1 − α) × C_background
```

**Implementação:**

```python
# visualization.py — draw_results()
# Criação do overlay colorido por instância
overlay_mask = np.zeros_like(frame, dtype=np.uint8)

for idx in range(num_detected):
    mask = masks[idx]
    rgb = COLORS[idx % len(COLORS)]
    color_bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
    overlay_mask[mask] = color_bgr
    overlay_mask_bool[mask] = True

# Alpha Blending: 60% original, 40% cor da máscara (α = 0.4)
frame[overlay_mask_bool] = (
    frame[overlay_mask_bool] * 0.6 +
    overlay_mask[overlay_mask_bool] * 0.4
).astype(np.uint8)
```

Com `α = 0.4`, obtém-se transparência adequada para visualizar simultaneamente a textura original da pessoa e o destaque colorido da máscara. Cada instância recebe uma cor distinta de uma paleta de 9 cores de alto contraste.

---

## 11. Otimização de Pipeline GPU

**Conceito:** A precisão mista (mixed precision) permite que operações do modelo executem em **bfloat16** (16 bits) em vez de float32 (32 bits), dobrando o throughput nos Tensor Cores das GPUs NVIDIA RTX sem perda significativa de precisão.

**Implementação:**

```python
# gui.py — _process_snapshot_thread()
# Execução com precisão mista bfloat16 acelerada por hardware (RTX)
with torch.autocast("cuda", dtype=torch.bfloat16):
    state = self.processor.set_image(working_img, state)
    state = self.processor.set_text_prompt(prompt, state)
```

**Medidas adicionais de gerenciamento de memória:**

```python
# gui.py — _process_snapshot_thread()
# LIMPEZA ESTRITA DE MEMÓRIA
state.clear()
del state
gc.collect()                    # Coleta de lixo imediata do Python
torch.cuda.empty_cache()        # Libera cache de VRAM da GPU
```

A combinação de `torch.inference_mode()` (desativa autograd), `torch.autocast` (bfloat16) e limpeza agressiva de memória garante que cada ciclo de processamento é eficiente e não acumula pressão sobre a VRAM limitada de GPUs laptop.

---

## 12. Latência e Processamento em Tempo Real

**Conceito:** Um sistema de visão computacional é considerado "tempo real" quando sua latência de processamento por frame é inferior a 1 segundo, permitindo feedback contínuo ao operador.

**Implementação da medição de latência:**

```python
# gui.py — _process_snapshot_thread()
start_time = time.time()

# ... (pipeline completo de inferência) ...

# Cálculo e log de latência do processamento por frame
elapsed = time.time() - start_time

self.log(f"[MAPEAMENTO] Concluído em {elapsed:.2f}s!")
```

**Sistema de ciclo temporizado:**

```python
# gui.py — tick_countdown()
self.time_left -= 0.1
if self.time_left <= 0:
    self.time_left = self.interval
    self.is_processing = True
    snapshot_frame = self.last_raw_frame.copy()

    # Processamento em thread separada para não bloquear o feed
    threading.Thread(
        target=self._process_snapshot_thread,
        args=(snapshot_frame, prompt, video_time),
        daemon=True
    ).start()
```

O ciclo de captura é configurável pelo usuário (1 a 60 segundos) e executa o processamento em uma **thread separada**, garantindo que o feed da câmera nunca congela durante a inferência do modelo.

---

## 13. Persistência em JSON e Dashboard em Tempo Real

**Conceito:** O sistema utiliza um banco de dados JSON local com dupla exportação — arquivo `.json` para persistência e arquivo `.js` para sincronização em tempo real com o dashboard HTML.

**Implementação do banco de dados:**

```python
# json_db.py — save_db()
def save_db(settings, history):
    db_data = {"settings": settings, "history": history}

    # 1. Salvar arquivo JSON local
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db_data, f, indent=4, ensure_ascii=False)

    # 2. Exportar arquivo JS para contornar CORS no protocolo file://
    js_content = f"window.METRO_DATA = {json.dumps(db_data, ensure_ascii=False, indent=2)};"
    with open(JS_EXPORT_PATH, "w", encoding="utf-8") as f:
        f.write(js_content)
```

**Implementação da classificação de lotação:**

```python
# gui.py — save_to_json()
# Determinar classificação de status com base nos parâmetros de metrô
if count <= capacity * (start_pct / 100.0):
    status = "Livre"
elif count <= capacity:
    status = "Normal"
elif count <= overcapacity:
    status = "Lotado"
else:
    status = "Superlotado"
```

**Sincronização em tempo real no Dashboard (HTML):**

```javascript
// dashboard.html — Polling com cache-busting a cada 1 segundo
function reloadPythonData() {
    const script = document.createElement('script');
    script.src = 'media/dados_dashboard.js?t=' + Date.now(); // cache-busting
    script.onload = () => {
        if (window.METRO_DATA) {
            processIncomingData(window.METRO_DATA);
        }
    };
    document.body.appendChild(script);
}

setInterval(reloadPythonData, 1000); // Sincroniza a cada segundo
```

O dashboard utiliza **injeção dinâmica de tags `<script>`** com parâmetros de cache-busting (`?t=timestamp`) para garantir que o navegador sempre carrega a versão mais recente dos dados, mesmo operando via protocolo `file://` sem servidor web.

---

## 14. Mapeamento Consolidado

| Tópico Formal                       | TAG no Código                         | Onde Aparece no Projeto                        |
|--------------------------------------|---------------------------------------|------------------------------------------------|
| Aquisição e Amostragem de Imagem     | `[AQUISIÇÃO_E_AMOSTRAGEM]`           | `gui.py` — `VideoCapture`, `cap.read()`, 30ms  |
| Conversão de Espaço de Cores         | `[CONVERSÃO_DE_ESPAÇO_DE_CORES]`     | `gui.py` + `visualization.py` — `cvtColor`     |
| Redimensionamento e Interpolação     | `[REDIMENSIONAMENTO_E_INTERPOLAÇÃO]` | `gui.py` — `Image.resize()` com LANCZOS        |
| Anti-Aliasing e Subamostragem        | `[ANTI_ALIASING_E_SUBAMOSTRAGEM]`    | `gui.py` — Filtro LANCZOS antes do downscale    |
| Segmentação Semântica                | `[SEGMENTAÇÃO_SEMÂNTICA]`            | `gui.py` — SAM 3.1 com prompt de texto          |
| Feature Extraction (Deep Learning)   | `[FEATURE_EXTRACTION]`               | `gui.py` — ViT Backbone no `set_image()`        |
| Atenção Multi-Head (Transformer)     | `[ATENÇÃO_MULTI_HEAD]`               | `patch.py` — SDPA / FlashAttention               |
| Grounding Texto-Imagem (VLM)        | `[GROUNDING_TEXTO_IMAGEM]`           | `gui.py` — `set_text_prompt()` com CLIP          |
| Detecção de Instâncias               | `[DETECÇÃO_DE_INSTÂNCIAS]`           | `gui.py` — Bounding Boxes + Masks               |
| Alpha Blending / Compositing         | `[ALPHA_BLENDING_COMPOSITING]`       | `visualization.py` — `draw_results()` com α=0.4 |
| Otimização de Pipeline GPU           | `[OTIMIZAÇÃO_DE_PIPELINE_GPU]`       | `gui.py` — FlashAttention + autocast bfloat16   |
| Latência e Tempo Real                | `[LATÊNCIA_E_TEMPO_REAL]`            | `gui.py` — Inferência < 1s/frame                |

---

*Documento gerado para o Projeto Contador — MetroVision IA*
*Pipeline de Visão Computacional com SAM 3.1 e Dashboard de Lotação em Tempo Real*
