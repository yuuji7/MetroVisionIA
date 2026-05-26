<div align="center">

# 🚇 MetroVision AI — Projeto Contador

**Câmeras Inteligentes para Distribuição de Passageiros em Vagões de Metrô**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

## 📋 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Problema e Motivação](#-problema-e-motivação)
- [Solução Proposta](#-solução-proposta)
- [Arquitetura e Pipeline](#-arquitetura-e-pipeline)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação e Configuração](#-instalação-e-configuração)
- [Como Usar](#-como-usar)
- [Resultados e Benchmark](#-resultados-e-benchmark)
- [Apresentação](#-apresentação)
- [Licença](#-licença)

---

## 🎯 Sobre o Projeto

O **MetroVision AI** é um sistema de visão computacional em tempo real que utiliza o modelo **SAM 3.1** (Segment Anything Model) da Meta AI para **contar e segmentar pessoas** em feeds de câmeras de segurança, com foco na aplicação em **vagões de metrô**.

O sistema captura frames de câmeras (ou vídeos locais), aplica **segmentação semântica zero-shot** guiada por prompt de texto (`"person"`), e gera informações de **lotação por vagão** que podem ser exibidas em monitores espalhados pelas estações — permitindo que passageiros escolham o vagão menos lotado antes do embarque.

---

## ⚠️ Problema e Motivação

| Problema | Impacto |
|---|---|
| Passageiros se aglomeram nos vagões próximos às escadas/acessos | Superlotação localizada + desconforto |
| Não existem informações em tempo real sobre lotação por vagão | Escolha de vagão é aleatória |
| Vagões centrais a 120-150% da capacidade, extremidades a 60-70% | Distribuição desigual e ineficiente |
| Operador sem dados granulares por vagão | Tomada de decisão limitada |

> **Contexto:** O Metrô de São Paulo transporta ~4.7 milhões de passageiros/dia, com 75% da demanda concentrada em 4h de pico.

---

## 💡 Solução Proposta

```
📷 Câmera no Vagão  →  🔄 Pré-Processamento  →  🧠 SAM 3.1  →  📊 Contagem  →  🖥️ Monitor na Estação
```

1. **Reutilização da infraestrutura existente** — câmeras de segurança já instaladas nos vagões
2. **Processamento de imagem com IA** — SAM 3.1 com segmentação zero-shot (sem re-treinamento)
3. **Otimização para tempo real** — FlashAttention + autocast bfloat16 na GPU
4. **Exibição da lotação** — dados exibidos em monitores nas estações para redistribuir passageiros

---

## 🏗️ Arquitetura e Pipeline

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

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Uso no Projeto |
|---|---|
| **Python 3.10+** | Linguagem principal |
| **PyTorch 2.0+** | Framework de deep learning (inferência) |
| **SAM 3.1 (Meta AI)** | Modelo de segmentação zero-shot com prompt de texto |
| **OpenCV (cv2)** | Captura de vídeo, conversão de cores, resize |
| **CUDA / FlashAttention** | Aceleração de GPU (RTX 3060 Laptop) |
| **Tkinter** | Interface gráfica desktop (Dark Mode Premium) |
| **PIL / Pillow** | Manipulação e exibição de imagens |
| **NumPy** | Operações matriciais e alpha blending |
| **openpyxl** | Geração de relatórios Excel com imagens embutidas |
| **PptxGenJS (Node.js)** | Geração automatizada da apresentação PowerPoint |

---

## 📂 Estrutura do Repositório

```
MetroVisionIA/
├── .env.example              # Template de variáveis de ambiente
├── .gitignore                # Arquivos ignorados pelo Git
├── README.md                 # Este arquivo
├── explicação.md             # Documentação técnica detalhada do projeto
│
├── videoreal.py              # 🎥 App principal — Câmera em tempo real + contagem
├── Videoteste.py             # 🧪 Extração + segmentação sequencial de vídeo
├── conversorimage.py         # 🔄 Conversor de vídeo em imagens estáticas
├── main.py                   # 📦 Launcher do segmentador de imagens estáticas
├── video_segmenter.py        # 📦 Launcher do segmentador de vídeo por blocos
├── scratch_benchmark.py      # 📊 Script de benchmark de performance
│
├── src/                      # Módulos internos da aplicação
│   ├── image_segmenter.py    #   Segmentador de imagens estáticas (GUI)
│   └── video_segmenter_gui.py#   Rastreador de vídeo por blocos (GUI)
│
├── media/                    # Diretório de mídia (vídeos, imagens, capturas)
│   ├── videos/               #   Vídeos de entrada
│   ├── images/               #   Imagens extraídas de vídeos
│   ├── imagens/              #   Imagens para segmentação estática
│   └── capturas/             #   Capturas com máscaras salvas automaticamente
│
└── apresentacao/             # Geração da apresentação PowerPoint
    ├── apresentacao.js       #   Script PptxGenJS (Node.js)
    └── package.json          #   Dependências Node.js
```

---

## 📦 Pré-requisitos

- **Python** 3.10 ou superior
- **NVIDIA GPU** com CUDA 11.8+ (recomendado: RTX 3060 ou superior)
- **PyTorch** 2.0+ com suporte CUDA
- **SAM 3.1** instalado localmente ([repositório oficial](https://github.com/Meta-AI/sam3))
- **Node.js** 18+ (apenas para gerar a apresentação)

---

## ⚙️ Instalação e Configuração

### 1. Clone o repositório

```bash
git clone https://github.com/yuuji7/MetroVisionIA.git
cd MetroVisionIA
```

### 2. Configure as variáveis de ambiente

```bash
cp .env.example .env
```

Edite o arquivo `.env` com suas credenciais:

```env
# Token do HuggingFace (obtenha em https://huggingface.co/settings/tokens)
HF_TOKEN=hf_SEU_TOKEN_AQUI

# Caminho absoluto para a instalação local do SAM 3.1
SAM3_PATH=C:\Users\seu_usuario\sam3
```

### 3. Instale as dependências Python

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python pillow numpy openpyxl python-dotenv
```

### 4. (Opcional) Gerar a apresentação PowerPoint

```bash
cd apresentacao
npm install
node apresentacao.js
```

---

## 🚀 Como Usar

### Câmera em Tempo Real (principal)

```bash
python videoreal.py
```

Abre a interface de câmera em tempo real com:
- Captura de feed ao vivo da webcam ou vídeo local
- Ciclo de captura temporizado (1s a 60s)
- Segmentação zero-shot com prompt configurável
- Contagem automática e registro em Excel

### Segmentação de Imagens Estáticas

```bash
python main.py
```

### Rastreamento de Vídeo por Blocos

```bash
python video_segmenter.py
```

### Extração de Frames de Vídeo

```bash
python conversorimage.py
```

### Benchmark de Performance

```bash
python scratch_benchmark.py
```

---

## 📈 Resultados e Benchmark

Executado na **RTX 3060 Laptop (6GB VRAM)** com resolução **254×144**:

| Configuração | Tempo / Frame | Status |
|---|---|---|
| Patch MATH estático (antes) | **24,82 s** | ❌ Inviável |
| Patch dinâmico + FlashAttention (depois) | **0,43 s** | ✅ Tempo Real |
| **Ganho de performance** | **≈ 58× mais rápido** | 🚀 |

### Tópicos de Processamento de Imagem Cobertos

| Tópico | Implementação |
|---|---|
| Aquisição e Amostragem de Imagem | `cv2.VideoCapture`, 30ms/frame |
| Conversão de Espaço de Cores | `cv2.cvtColor` (BGR → RGB) |
| Redimensionamento e Interpolação | `Image.resize()` com LANCZOS |
| Segmentação Semântica Zero-Shot | SAM 3.1 com prompt de texto |
| Feature Extraction (Deep Learning) | ViT Backbone |
| Atenção Multi-Head (Transformer) | SDPA / FlashAttention |
| Grounding Texto-Imagem (VLM) | CLIP / Cross-Attention |
| Alpha Blending / Compositing | `draw_results()` com α = 0.4 |
| Otimização de Pipeline GPU | FlashAttention + autocast bfloat16 |

---

## 📊 Apresentação

A apresentação em PowerPoint é gerada programaticamente via **PptxGenJS** (JavaScript/Node.js):

```bash
cd apresentacao
node apresentacao.js
# → Gera: Projeto_Contador_MetroVision_AI.pptx (14 slides)
```

A apresentação segue o **Princípio da Pirâmide de Barbara Minto** — apresentando a conclusão primeiro, seguida pelos argumentos de suporte e evidências técnicas.

---

## 📄 Licença

Este projeto é de uso acadêmico. Desenvolvido como trabalho de grupo para a disciplina de Processamento de Imagem e Visão Computacional.

---

<div align="center">

**MetroVision AI** — *Inteligência que move a cidade.* 🚇🤖

</div>
