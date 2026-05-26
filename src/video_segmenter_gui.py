# -*- coding: utf-8 -*-
"""
SAM 3.1 - Rastreador e Contador de Vídeo Segregado por Blocos (Memory-Safe)
==========================================================================
Este módulo implementa a lógica dedicada para processamento de arquivos de vídeo,
focando estritamente em evitar estouros de VRAM por meio de uma estratégia de
execução em blocos curtos combinada com feedback geométrico por caixas delimitadoras
e mapeamento de identidades por correspondência de IoU (Intersection over Union).
"""

import os
import gc
import sys
import threading
import time
import random
import glob
import re
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2

# Configuração preventiva do alocador de cache CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
from PIL import Image, ImageTk, ImageDraw

# ============================================================
# PATCH DE COMPATIBILIDADE DE SDPA PARA RTX 3060 LAPTOP
# ============================================================
import functools
import torch.nn.functional as _F_torch

# Desativação global e forçada dos backends incompatíveis
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

_original_sdpa = _F_torch.scaled_dot_product_attention

@functools.wraps(_original_sdpa)
def _safe_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    """SDPA com fallback automático para backend MATH em caso de falha de kernel."""
    try:
        # Garante configuração estrita do backend MATH
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass
    try:
        return _original_sdpa(query, key, value, attn_mask=attn_mask,
                               dropout_p=dropout_p, is_causal=is_causal)
    except RuntimeError as e:
        err_str = str(e)
        if "No available kernel" in err_str or "not supported" in err_str.lower() or "abort" in err_str.lower():
            return _original_sdpa(query, key, value, attn_mask=attn_mask,
                                     dropout_p=dropout_p, is_causal=is_causal)
        raise

# Patch em ambas as referências possíveis (nn.functional e módulo root torch)
_F_torch.scaled_dot_product_attention = _safe_sdpa
if hasattr(torch, "scaled_dot_product_attention"):
    torch.scaled_dot_product_attention = _safe_sdpa


# Carregamento de variáveis de ambiente (.env)
try:
    from dotenv import load_dotenv
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_project_root, ".env"))
except ImportError:
    pass

# Adiciona o diretório do SAM 3 e a raiz local ao sys.path
_sam3_path = os.environ.get("SAM3_PATH", "")
if _sam3_path:
    sys.path.append(_sam3_path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sam3 import build_sam3_predictor, build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
except ImportError as e:
    SAM3_AVAILABLE = False
    SAM3_IMPORT_ERROR = str(e)

# Paleta premium de alto contraste para desenhar as máscaras
COLORS = np.array([
    [0.9, 0.1, 0.1],   # Vermelho Vibrante
    [0.1, 0.8, 0.1],   # Verde Esmeralda
    [0.1, 0.5, 0.95],  # Azul Safira
    [0.95, 0.85, 0.1], # Amarelo Ouro
    [0.9, 0.1, 0.9],   # Magenta Real
    [0.1, 0.9, 0.9],   # Ciano Elétrico
    [0.95, 0.5, 0.1],  # Laranja Cítrico
    [0.55, 0.1, 0.95], # Roxo Ametista
    [0.1, 0.95, 0.5]   # Verde Lima
])

def compute_iou(box1, box2):
    """Calcula a Intersection over Union (IoU) entre caixas normalized [xmin, ymin, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h
    
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

class SAM3VideoTrackerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SAM 3.1 - Rastreamento e Contabilizador de Vídeo Seguro (Blocos Segregados)")
        self.geometry("1280x920")
        self.configure(bg="#1e1e1f")
        
        self.predictor = None
        self.image_model = None
        self.image_processor = None
        self.video_path = None
        
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.temp_dir = os.path.join(self.project_root, "temp_frames")
        self.temp_block_dir = os.path.join(self.project_root, "temp_block_frames")
        
        self.tracked_frames = []         # Estrutura global carregada na CPU/RAM: (caminho_imagem, outputs_numpy)
        self.unique_objects_seen = set() # Histórico persistente de IDs de objetos estáveis
        
        # Variáveis exclusivas de sincronização Live/Tempo Real e Fila de Capturas
        self.frame_lock = threading.Lock()
        self.current_frame_data = None
        self.active_masks = None
        self.live_mask_overlay = None
        self.live_mask_bool = None
        self.live_opaque_overlay = None
        self.live_opaque_bool = None
        self.live_capture_interval = 0.0
        self.cap = None
        self.orig_w = 0
        self.orig_h = 0
        self.orig_fps = 30.0
        self.total_frames = 0
        
        # Estruturas do benchmark do vídeo por intervalos
        self.video_benchmark_results = []
        self.next_capture_time = 1.0  # Primeira captura ocorre aos 1.0 segundos
        self.capture_ready_flag = False
        self.capture_in_progress = False
        self.current_capture_time = 0.0
        self.current_capture_idx = 0


        self.is_processing_active = False
        
        # Player de Vídeo
        self.playback_active = False
        self.current_frame_idx = 0
        self.playback_fps = 10
        self.playback_timer = None
        
        # Diretório de vídeos padrão
        self.default_video_dir = os.path.join(self.project_root, "media", "videos")
        os.makedirs(self.default_video_dir, exist_ok=True)
        
        # Configurar Estilo Dark Mode Premium
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background="#1e1e1f", foreground="#e1e1e1", fieldbackground="#2d2d2d")
        self.style.configure("TButton", background="#3c3c3d", foreground="#ffffff", borderwidth=0, padding=6, font=("Helvetica", 9, "bold"))
        self.style.map("TButton", background=[("active", "#4d4d4f")])
        self.style.configure("TLabel", background="#1e1e1f", foreground="#e1e1e1", font=("Helvetica", 9))
        self.style.configure("TEntry", fieldbackground="#2d2d2d", foreground="#ffffff", insertcolor="#ffffff")
        self.style.configure("TCombobox", fieldbackground="#2d2d2d", background="#3c3c3d", foreground="#ffffff", selectbackground="#2d2d2d")
        
        self.create_widgets()
        
        # Carregamento do modelo
        if SAM3_AVAILABLE:
            self.log("[STATUS] Carregando motor de rastreamento SAM 3.1...")
            threading.Thread(target=self.load_sam3_video_model, daemon=True).start()
        else:
            self.log(f"[ERRO CRÍTICO] Falha ao importar SAM 3.1: {SAM3_IMPORT_ERROR}")
            messagebox.showerror("Erro de Importação", f"Erro: {SAM3_IMPORT_ERROR}")

    def create_widgets(self):
        # Painel Superior: Configurações e Controles
        ctrl_frame = tk.Frame(self, bg="#252526", pady=10, padx=15, highlightbackground="#3c3c3c", highlightthickness=1)
        ctrl_frame.pack(fill="x", side="top", padx=10, pady=10)
        
        # Linha 1: Seleção de Vídeo
        tk.Label(ctrl_frame, text="Pasta de Vídeos:", bg="#252526", fg="#ffffff", font=("Helvetica", 10, "bold")).grid(row=0, column=0, sticky="w", pady=5)
        self.entry_folder_path = ttk.Entry(ctrl_frame, width=40)
        self.entry_folder_path.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.entry_folder_path.insert(0, self.default_video_dir)
        
        self.btn_select_folder = ttk.Button(ctrl_frame, text="Selecionar Pasta...", command=self.select_folder)
        self.btn_select_folder.grid(row=0, column=2, padx=5, pady=5)
        
        tk.Label(ctrl_frame, text="Vídeo na Pasta:", bg="#252526", fg="#ffffff", font=("Helvetica", 10, "bold")).grid(row=0, column=3, sticky="w", pady=5, padx=(15, 0))
        self.combo_videos = ttk.Combobox(ctrl_frame, width=28, state="readonly")
        self.combo_videos.grid(row=0, column=4, padx=5, pady=5)
        self.combo_videos.bind("<<ComboboxSelected>>", self.on_video_selected)
        
        self.btn_browse_file = ttk.Button(ctrl_frame, text="Procurar Arquivo...", command=self.browse_video_file)
        self.btn_browse_file.grid(row=0, column=5, padx=5, pady=5)
        
        # Linha 2: Prompt e Confiança
        tk.Label(ctrl_frame, text="Prompt (Texto):", bg="#252526", fg="#ffffff", font=("Helvetica", 10, "bold")).grid(row=1, column=0, sticky="w", pady=5)
        self.entry_prompt = ttk.Entry(ctrl_frame, width=40)
        self.entry_prompt.insert(0, "person")
        self.entry_prompt.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        tk.Label(ctrl_frame, text="Limiar Confiança:", bg="#252526", fg="#ffffff", font=("Helvetica", 10, "bold")).grid(row=1, column=2, sticky="w", pady=5, padx=5)
        self.val_confidence = tk.DoubleVar(value=0.45)
        self.slider_conf = ttk.Scale(ctrl_frame, from_=0.1, to=0.9, variable=self.val_confidence, orient="horizontal", command=self.update_conf_label)
        self.slider_conf.grid(row=1, column=3, sticky="ew", padx=5, pady=5)
        self.lbl_conf = tk.Label(ctrl_frame, text="0.45", bg="#252526", fg="#a0a0a0", font=("Helvetica", 9, "bold"))
        self.lbl_conf.grid(row=1, column=4, sticky="w", padx=5, pady=5)
        
        # Linha 3: Otimizações de Memória e Modo de Operação
        tk.Label(ctrl_frame, text="Resolução Alvo:", bg="#252526", fg="#00aaff", font=("Helvetica", 10, "bold")).grid(row=2, column=0, sticky="w", pady=5)
        self.combo_resolution = ttk.Combobox(ctrl_frame, values=["240p (Muito Baixa)", "360p (Baixa)", "480p (Média)", "720p (HD)", "Original"], width=20, state="readonly")
        self.combo_resolution.set("360p (Baixa)")
        self.combo_resolution.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        tk.Label(ctrl_frame, text="Modo Operação:", bg="#252526", fg="#ffaa00", font=("Helvetica", 10, "bold")).grid(row=2, column=2, sticky="w", pady=5, padx=(15, 0))
        self.combo_mode = ttk.Combobox(ctrl_frame, values=["Live (Tempo Real - Asíncrono)", "Offline (Blocos - ID Persistente)"], width=22, state="readonly")
        self.combo_mode.set("Live (Tempo Real - Asíncrono)")
        self.combo_mode.grid(row=2, column=3, padx=5, pady=5, sticky="w")
        self.combo_mode.bind("<<ComboboxSelected>>", self.on_mode_changed)
        
        tk.Label(ctrl_frame, text="Frequência Live:", bg="#252526", fg="#ffaa00", font=("Helvetica", 10, "bold")).grid(row=2, column=4, sticky="w", pady=5, padx=(15, 0))
        self.combo_live_interval = ttk.Combobox(ctrl_frame, values=[
            "A cada 10.0 segundos",
            "A cada 15.0 segundos",
            "A cada 30.0 segundos",
            "A cada 5.0 segundos",
            "A cada 2.0 segundos",
            "A cada 1.0 segundo",
            "A cada 0.5 segundos",
            "Máxima (Adaptativo GPU)"
        ], width=20, state="readonly")
        self.combo_live_interval.set("A cada 10.0 segundos")
        self.combo_live_interval.grid(row=2, column=5, padx=5, pady=5, sticky="w")
        
        # Linha 4: Limites de Exibição de Quadros (Modo Offline)
        tk.Label(ctrl_frame, text="Qtd. Máx. Quadros:", bg="#252526", fg="#e1e1e1", font=("Helvetica", 10, "bold")).grid(row=3, column=0, sticky="w", pady=5)
        self.combo_duration = ttk.Combobox(ctrl_frame, values=["100 frames", "150 frames", "200 frames", "500 frames", "Completo"], width=20, state="readonly")
        self.combo_duration.set("100 frames")
        self.combo_duration.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        self.combo_duration.config(state="disabled")
        
        tk.Label(ctrl_frame, text="Tamanho Bloco:", bg="#252526", fg="#00ff00", font=("Helvetica", 10, "bold")).grid(row=3, column=2, sticky="w", pady=5, padx=(15, 0))
        self.combo_block_size = ttk.Combobox(ctrl_frame, values=["5 (Super Otimizado VRAM)", "10", "15 (Padrão)", "20", "30"], width=22, state="readonly")
        self.combo_block_size.set("15 (Padrão)")
        self.combo_block_size.grid(row=3, column=3, padx=5, pady=5, sticky="w")
        self.combo_block_size.config(state="disabled")
        
        tk.Label(ctrl_frame, text="Temporizador (FPS):", bg="#252526", fg="#00aaff", font=("Helvetica", 10, "bold")).grid(row=3, column=4, sticky="w", pady=5, padx=(15, 0))
        self.combo_fps = ttk.Combobox(ctrl_frame, values=["5 FPS (Muito Seguro)", "10 FPS (Baixo)", "15 FPS (Médio)", "24 FPS (Cinema)", "Original FPS"], width=20, state="readonly")
        self.combo_fps.set("10 FPS (Baixo)")
        self.combo_fps.grid(row=3, column=5, padx=5, pady=5, sticky="w")
        self.combo_fps.config(state="disabled")
        
        # Ações
        action_frame = tk.Frame(ctrl_frame, bg="#252526")
        action_frame.grid(row=3, column=2, columnspan=4, sticky="ew", pady=(10, 0))
        
        self.btn_run = tk.Button(action_frame, text="⚡ Iniciar Rastreamento Live (Tempo Real)", bg="#007acc", fg="#ffffff", font=("Helvetica", 11, "bold"), activebackground="#005999", activeforeground="#ffffff", relief="flat", state="disabled", command=self.start_video_processing)
        self.btn_run.pack(side="left", fill="x", expand=True, padx=(10, 10))
        self.make_btn_hoverable(self.btn_run, "#007acc", "#005999")
        
        self.btn_stop = tk.Button(action_frame, text="🛑 Cancelar / Esvaziar", bg="#dc3545", fg="#ffffff", font=("Helvetica", 11, "bold"), activebackground="#bd2130", activeforeground="#ffffff", relief="flat", command=self.stop_current_action)
        self.btn_stop.pack(side="right", padx=5)
        self.make_btn_hoverable(self.btn_stop, "#dc3545", "#bd2130")
        
        # Painel Central
        main_container = tk.Frame(self, bg="#1e1e1f")
        main_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Player (Esquerda)
        viewer_frame = ttk.LabelFrame(main_container, text="Visualizador do Rastreamento Premium", padding=5)
        viewer_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        self.canvas_video = tk.Canvas(viewer_frame, bg="#151515", highlightthickness=0)
        self.canvas_video.pack(fill="both", expand=True)
        
        # Métricas (Direita)
        side_panel = ttk.LabelFrame(main_container, text="Estado de Recursos & Benchmark", padding=10, width=320)
        side_panel.pack(side="right", fill="y", padx=(5, 0))
        side_panel.pack_propagate(False)
        
        tk.Label(side_panel, text="Métricas da Execução:", bg="#1e1e1f", fg="#00aaff", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 10))
        
        self.lbl_orig_info = tk.Label(side_panel, text="Vídeo Original: --", bg="#1e1e1f", fg="#a0a0a0", anchor="w")
        self.lbl_orig_info.pack(fill="x", pady=2)
        
        self.lbl_comp_info = tk.Label(side_panel, text="Frames Compactados: --", bg="#1e1e1f", fg="#a0a0a0", anchor="w")
        self.lbl_comp_info.pack(fill="x", pady=2)
        
        self.lbl_time_info = tk.Label(side_panel, text="Tempo Total da IA: --", bg="#1e1e1f", fg="#a0a0a0", anchor="w")
        self.lbl_time_info.pack(fill="x", pady=2)
        
        self.lbl_vram_info = tk.Label(side_panel, text="VRAM Alocada: --", bg="#1e1e1f", fg="#a0a0a0", anchor="w")
        self.lbl_vram_info.pack(fill="x", pady=2)
        
        self.lbl_unique_counter = tk.Label(side_panel, text="Objetos Únicos: 0", bg="#1e1e1f", fg="#00ff00", font=("Helvetica", 12, "bold"), anchor="w")
        self.lbl_unique_counter.pack(fill="x", pady=(15, 10))
        
        ttk.Separator(side_panel, orient="horizontal").pack(fill="x", pady=10)
        
        # Player Controles
        tk.Label(side_panel, text="Controles de Reprodução:", bg="#1e1e1f", fg="#00aaff", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 5))
        
        play_btn_frame = tk.Frame(side_panel, bg="#1e1e1f")
        play_btn_frame.pack(fill="x", pady=5)
        
        self.btn_play_pause = tk.Button(play_btn_frame, text="▶ Reproduzir", bg="#28a745", fg="#ffffff", font=("Helvetica", 10, "bold"), activebackground="#218838", relief="flat", state="disabled", command=self.toggle_playback)
        self.btn_play_pause.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.make_btn_hoverable(self.btn_play_pause, "#28a745", "#218838")
        
        # Slider
        self.val_timeline = tk.IntVar(value=0)
        self.slider_timeline = ttk.Scale(side_panel, from_=0, to=100, variable=self.val_timeline, orient="horizontal", command=self.on_scrub)
        self.slider_timeline.pack(fill="x", pady=10)
        self.slider_timeline.config(state="disabled")
        
        self.lbl_timeline_frame = tk.Label(side_panel, text="Frame: 0 / 0", bg="#1e1e1f", fg="#a0a0a0", font=("Courier", 10))
        self.lbl_timeline_frame.pack(pady=2)
        
        # Console Inferior
        bottom_frame = tk.Frame(self, bg="#1e1e1f")
        bottom_frame.pack(fill="x", side="bottom", padx=10, pady=(0, 10))
        
        self.progress_bar = ttk.Progressbar(bottom_frame, orient="horizontal", mode="determinate")
        self.progress_bar.pack(fill="x", pady=(0, 5))
        
        self.log_frame = ttk.LabelFrame(bottom_frame, text="Log de Execução do Processamento em Blocos Temporais", padding=5)
        self.log_frame.pack(fill="x")
        
        self.text_log = tk.Text(self.log_frame, wrap="word", height=6, bg="#101010", fg="#00ff66", insertbackground="white", state="disabled", font=("Courier", 9))
        self.text_log.pack(fill="both", expand=True)
        
        self.scan_videos_in_folder()

    def on_mode_changed(self, event):
        mode = self.combo_mode.get()
        if "Live" in mode:
            self.combo_live_interval.config(state="readonly")
            self.combo_block_size.config(state="disabled")
            self.combo_fps.config(state="disabled")
            self.combo_duration.config(state="disabled")
            self.btn_run.config(text="⚡ Iniciar Rastreamento Live (Tempo Real)")
            self.log("[SISTEMA] Modo Live selecionado. Rastreamento em tempo real direto no player.")
        else:
            self.combo_live_interval.config(state="disabled")
            self.combo_block_size.config(state="readonly")
            self.combo_fps.config(state="readonly")
            self.combo_duration.config(state="readonly")
            self.btn_run.config(text="⚡ Iniciar Rastreamento Offline (Memory-Safe)")
            self.log("[SISTEMA] Modo Offline selecionado. Processamento incremental em blocos temporais.")

    def make_btn_hoverable(self, btn, normal_bg, hover_bg):
        btn.bind("<Enter>", lambda e: btn.config(bg=hover_bg) if btn["state"] != "disabled" else None)
        btn.bind("<Leave>", lambda e: btn.config(bg=normal_bg) if btn["state"] != "disabled" else None)

    def log(self, msg):
        def _log():
            self.text_log.config(state="normal")
            self.text_log.insert("end", f"{msg}\n")
            self.text_log.config(state="disabled")
            self.text_log.see("end")
        self.after(0, _log)

    def update_conf_label(self, val):
        self.lbl_conf.config(text=f"{float(val):.2f}")
        if self.tracked_frames and not self.playback_active:
            self.show_frame_idx(self.current_frame_idx)

    def select_folder(self):
        folder = filedialog.askdirectory(title="Selecionar Pasta de Vídeos")
        if folder:
            self.entry_folder_path.delete(0, "end")
            self.entry_folder_path.insert(0, folder)
            self.scan_videos_in_folder()

    def scan_videos_in_folder(self):
        folder = self.entry_folder_path.get().strip()
        if not os.path.isdir(folder):
            self.log(f"[AVISO] Pasta não encontrada: {folder}")
            return
            
        extensions = ["*.mp4", "*.avi", "*.mkv", "*.mov", "*.MP4", "*.AVI", "*.MKV", "*.MOV"]
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(folder, ext)))
            
        names = [os.path.basename(f) for f in files]
        self.combo_videos.config(values=names)
        
        if names:
            self.combo_videos.current(0)
            self.on_video_selected(None)
            self.log(f"[OK] {len(names)} vídeos localizados na pasta.")
        else:
            self.combo_videos.set("")
            self.log("[AVISO] Nenhum vídeo encontrado nesta pasta.")

    def on_video_selected(self, event):
        folder = self.entry_folder_path.get().strip()
        name = self.combo_videos.get()
        if name:
            self.video_path = os.path.join(folder, name)
            self.log(f"[OK] Vídeo selecionado: {name}")
            self.load_video_metadata()

    def browse_video_file(self):
        file = filedialog.askopenfilename(
            title="Selecionar Arquivo de Vídeo",
            filetypes=[("Vídeos", "*.mp4 *.avi *.mkv *.mov")]
        )
        if file:
            self.video_path = file
            self.combo_videos.set(os.path.basename(file))
            self.log(f"[OK] Vídeo selecionado manualmente: {file}")
            self.load_video_metadata()

    def load_video_metadata(self):
        if not self.video_path or not os.path.exists(self.video_path):
            return
        try:
            cap = cv2.VideoCapture(self.video_path)
            self.orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.orig_fps = cap.get(cv2.CAP_PROP_FPS)
            if self.orig_fps <= 0: self.orig_fps = 24.0
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            self.playback_fps = self.orig_fps
            
            self.lbl_orig_info.config(text=f"Vídeo Original: {self.orig_w}x{self.orig_h} | {self.orig_fps:.2f} FPS | {self.total_frames} frames")
            
            # Primeiro frame preview
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.display_image(Image.fromarray(rgb))
            cap.release()
        except Exception as e:
            self.log(f"[ERRO] Falha ao ler metadados: {e}")

    def display_image(self, pil_img):
        def _display():
            self.canvas_video.delete("all")
            c_w = self.canvas_video.winfo_width()
            c_h = self.canvas_video.winfo_height()
            if c_w <= 1: c_w = 640
            if c_h <= 1: c_h = 480
            
            w, h = pil_img.size
            ratio = min(c_w / w, c_h / h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            
            resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(resized)
            self._tk_img_show = tk_img
            self.canvas_video.create_image(c_w // 2, c_h // 2, anchor="center", image=tk_img)
        self.after(0, _display)

    def load_sam3_video_model(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.log(f"[STATUS] Inicializando Predictor SAM 3.1 no dispositivo '{device.upper()}'...")
            
            if device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.log(f"[SISTEMA] VRAM total detectada: {vram_total:.1f} GB.")
                
            start = time.time()
            self.predictor = build_sam3_predictor(
                version="sam3.1",
                compile=False,
                use_fa3=False,
                async_loading_frames=False
            )
            
            self.log("[STATUS] Inicializando Detector de Texto (Grounding) SAM 3.1...")
            self.image_model = build_sam3_image_model(compile=False, load_from_HF=True, device=device)
            self.image_processor = Sam3Processor(self.image_model, device=device)
            
            if hasattr(self.predictor, "model"):
                self.predictor.model.eval()
                self.predictor.model.postprocess_batch_size = 1
                self.predictor.model.use_batched_grounding = False
                self.log("[SISTEMA] Modo eval() e batch_size=1 ativados para inferência limpa.")
                
            if hasattr(self.predictor, "model") and hasattr(self.predictor.model, "init_state"):
                original_init = self.predictor.model.init_state
                def patched_init(*args, **kwargs):
                    kwargs.pop("offload_state_to_cpu", None)
                    return original_init(*args, **kwargs)
                self.predictor.model.init_state = patched_init
                self.log("[SISTEMA] Monkeypatch 'offload_state_to_cpu' operacional.")
                
            elapsed = time.time() - start
            self.log(f"[OK] SAM 3.1 e Detector de Texto carregados com sucesso em {elapsed:.2f}s!")
            self.after(0, lambda: self.btn_run.config(state="normal"))
        except Exception as e:
            self.log(f"[ERRO] Falha crítica ao inicializar o motor: {e}")

    def parse_resolution_height(self, raw_val):
        raw_val = str(raw_val).strip().lower()
        if "original" in raw_val:
            return None
        nums = re.findall(r"\d+", raw_val)
        if not nums:
            return None
        h = int(nums[0])
        return max(240, h)

    def parse_fps(self, raw_val):
        raw_val = str(raw_val).strip().lower()
        if "original" in raw_val:
            return None
        nums = re.findall(r"\d+", raw_val)
        if not nums:
            return None
        fps = int(nums[0])
        return max(5, fps)

    def parse_block_size(self, raw_val):
        nums = re.findall(r"\d+", str(raw_val))
        if not nums:
            return 15
        val = int(nums[0])
        return max(3, min(val, 60))

    def parse_max_frames(self, raw_val):
        raw_val = str(raw_val).strip().lower()
        if "completo" in raw_val:
            return None
        nums = re.findall(r"\d+", raw_val)
        if not nums:
            return None
        return int(nums[0])

    def start_video_processing(self):
        if not self.video_path or not os.path.exists(self.video_path):
            return
        prompt = self.entry_prompt.get().strip()
        if not prompt:
            return
            
        self.stop_playback()
        self.is_processing_active = True
        
        mode = self.combo_mode.get()
        if "Live" in mode:
            self.btn_run.config(state="disabled", text="⚡ Rastreando Live...")
            
            # Configura intervalo live
            interval_str = self.combo_live_interval.get().lower()
            if "0.5" in interval_str:
                self.live_capture_interval = 0.5
            elif "1.0" in interval_str:
                self.live_capture_interval = 1.0
            elif "2.0" in interval_str:
                self.live_capture_interval = 2.0
            elif "5.0" in interval_str:
                self.live_capture_interval = 5.0
            elif "10.0" in interval_str or "10" in interval_str:
                self.live_capture_interval = 10.0
            elif "15.0" in interval_str or "15" in interval_str:
                self.live_capture_interval = 15.0
            elif "30.0" in interval_str or "30" in interval_str:
                self.live_capture_interval = 30.0
            else:
                self.live_capture_interval = 0.0  # Máxima GPU
                
            # Inicializa variáveis do benchmark de vídeo e limpa estados anteriores
            self.video_benchmark_results = []
            self.next_capture_time = 1.0  # Primeira captura ocorre aos 1.0 segundos
            self.capture_ready_flag = False
            self.capture_in_progress = False
            
            with self.frame_lock:
                self.current_frame_data = None
                self.captured_frame_data = None
                self.active_masks = None
                self.live_mask_overlay = None
                self.live_mask_bool = None
                self.live_opaque_overlay = None
                self.live_opaque_bool = None
                
            # Abre o vídeo para o player
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            self.current_frame_idx = 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Habilita controles do player
            self.btn_play_pause.config(state="normal")
            self.slider_timeline.config(state="normal", from_=0, to=self.total_frames - 1)
            self.slider_timeline.set(0)
            self.lbl_timeline_frame.config(text=f"Frame: 1 / {self.total_frames}")
            
            # Inicia playback automático
            self.playback_active = True
            self.btn_play_pause.config(text="⏸ Pausar", bg="#ffc107", activebackground="#e0a800")
            self._live_play_loop()
            
            # Dispara thread de segmentação asíncrona
            threading.Thread(target=self._live_segmentation_worker, args=(prompt,), daemon=True).start()
            self.log(f"[STATUS] Iniciado Modo Live. Frequência: {self.combo_live_interval.get()}")
        else:
            # Modo Offline
            self.btn_run.config(state="disabled", text="Processando Blocos...")
            self.btn_play_pause.config(state="disabled")
            self.slider_timeline.config(state="disabled")
            
            threading.Thread(target=self._process_video_thread, args=(prompt,), daemon=True).start()

    def _process_video_thread(self, prompt):
        try:
            self.log("\n=======================================================")
            self.log("[STATUS] INICIALIZANDO PIPELINE DE RASTREAMENTO...")
            
            target_height = self.parse_resolution_height(self.combo_resolution.get())
            target_fps = self.parse_fps(self.combo_fps.get())
            block_size = self.parse_block_size(self.combo_block_size.get())
            max_frames = self.parse_max_frames(self.combo_duration.get())
            
            cap = cv2.VideoCapture(self.video_path)
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            orig_fps = cap.get(cv2.CAP_PROP_FPS)
            orig_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            max_read = min(max_frames, orig_total) if max_frames is not None else orig_total
            
            if target_height is not None:
                scale = target_height / orig_h
                dest_w = int(orig_w * scale)
                dest_w = (dest_w // 2) * 2
                dest_h = (target_height // 2) * 2
            else:
                dest_w = (orig_w // 2) * 2
                dest_h = (orig_h // 2) * 2
                
            frame_skip = max(1, round(orig_fps / target_fps)) if target_fps is not None else 1
            actual_fps = orig_fps / frame_skip
            
            # Calcula os índices reais dos frames do vídeo original de forma instantânea
            frame_indices = []
            read_idx = 0
            while read_idx < max_read:
                if read_idx % frame_skip == 0:
                    frame_indices.append(read_idx)
                read_idx += 1
            total_extracted = len(frame_indices)
            
            self.log(f"[STATUS] Resolução Alvo: {dest_w}x{dest_h} | FPS Efetivo: {actual_fps:.2f}")
            self.log(f"[STATUS] Partição de Memória: Bloco de {block_size} frames")
            self.log(f"[STATUS] Frames Totais Calculados: {total_extracted}")
            
            self.after(0, lambda: self.lbl_comp_info.config(text=f"Frames: {dest_w}x{dest_h} | {actual_fps:.2f} FPS | {total_extracted} frames"))
            
            if total_extracted == 0:
                raise RuntimeError("Nenhum frame foi selecionado para amostragem.")
                
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            os.makedirs(self.temp_dir, exist_ok=True)
            
            # Loop de Rastreamento por Blocos
            self.log("[STATUS] Iniciando processamento do pipeline de vídeo...")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            start_time = time.time()
            vram_init = torch.cuda.mem_get_info()[0] / 1024**3 if device == "cuda" else 0.0
            
            self.tracked_frames = [None] * total_extracted
            self.unique_objects_seen.clear()
            
            # Blocos temporais de frames com sobreposição de 1 frame
            blocks = []
            start = 0
            while start < total_extracted - 1:
                end = min(start + block_size, total_extracted)
                blocks.append((start, end))
                if end == total_extracted:
                    break
                start = end - 1
                
            if not blocks:
                blocks = [(0, total_extracted)]
                
            last_detections = []
            
            for b_idx, (b_start, b_end) in enumerate(blocks):
                if not self.is_processing_active:
                    break
                    
                self.log(f"\n[BLOCO {b_idx+1}/{len(blocks)}] Processando frames {b_start} até {b_end-1}...")
                
                # Criar subdiretório do bloco
                if os.path.exists(self.temp_block_dir):
                    shutil.rmtree(self.temp_block_dir)
                os.makedirs(self.temp_block_dir, exist_ok=True)
                
                # Extração sob demanda (On-The-Fly) dos frames do bloco
                block_len = b_end - b_start
                cap = cv2.VideoCapture(self.video_path)
                for local_i in range(block_len):
                    global_i = b_start + local_i
                    orig_idx = frame_indices[global_i]
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, orig_idx)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    resized = cv2.resize(frame, (dest_w, dest_h), interpolation=cv2.INTER_AREA) if target_height is not None else frame
                    
                    cv2.imwrite(os.path.join(self.temp_block_dir, f"{local_i:05d}.jpg"), resized)
                    cv2.imwrite(os.path.join(self.temp_dir, f"{global_i:05d}.jpg"), resized)
                cap.release()
                
                if device == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    vram_cur = torch.cuda.mem_get_info()[0] / 1024**3
                    self.log(f"[MEMÓRIA] VRAM Livre (Início do Bloco): {vram_cur:.2f} GB")
                    
                autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.inference_mode()
                
                with torch.inference_mode(), autocast:
                    # Cria sessão
                    res = self.predictor.handle_request({
                        "type": "start_session",
                        "resource_path": self.temp_block_dir,
                        "offload_video_to_cpu": True,
                        "offload_state_to_cpu": True
                    })
                    sid = res["session_id"]
                    
                    # Prompting frame 0 do Bloco
                    if b_idx > 0 and last_detections:
                        # Propagação geométrica das caixas delimitadoras
                        boxes = [d["box"] for d in last_detections]
                        labels = [1] * len(boxes)
                        self.predictor.handle_request({
                            "type": "add_prompt",
                            "session_id": sid,
                            "frame_index": 0,
                            "bounding_boxes": boxes,
                            "bounding_box_labels": labels
                        })
                    else:
                        # Bloco inicial ou fallback: Usamos o image_processor para detectar pelo prompt de texto
                        f0_path = os.path.join(self.temp_block_dir, "00000.jpg")
                        norm_boxes = []
                        if os.path.exists(f0_path):
                            f0_img = Image.open(f0_path).convert("RGB")
                            state0 = {}
                            self.image_processor.set_confidence_threshold(self.val_confidence.get(), state0)
                            
                            with torch.inference_mode(), autocast:
                                state0 = self.image_processor.set_image(f0_img, state0)
                                state0 = self.image_processor.set_text_prompt(prompt, state0)
                                
                            raw_scores = state0.get("scores")
                            if raw_scores is not None:
                                scores_np = raw_scores.cpu().float().numpy() if hasattr(raw_scores, "cpu") else np.array(raw_scores)
                            else:
                                scores_np = np.array([])
                                
                            thresh = self.val_confidence.get()
                            keep = np.where(scores_np >= thresh)[0] if len(scores_np) > 0 else []
                            
                            if len(keep) > 0:
                                raw_boxes = state0.get("boxes")
                                if isinstance(raw_boxes, torch.Tensor):
                                    boxes_np = raw_boxes[keep].cpu().numpy()
                                else:
                                    boxes_np = np.array([raw_boxes[i] for i in keep])
                                    
                                # Converte de absoluto [x0, y0, x1, y1] para normalizado [x0_rel, y0_rel, bw_rel, bh_rel]
                                f_w, f_h = f0_img.size
                                for box in boxes_np:
                                    x0, y0, x1, y1 = box
                                    x0_rel = x0 / f_w
                                    y0_rel = y0 / f_h
                                    bw_rel = (x1 - x0) / f_w
                                    bh_rel = (y1 - y0) / f_h
                                    norm_boxes.append([x0_rel, y0_rel, bw_rel, bh_rel])
                                    
                        if norm_boxes:
                            labels = [1] * len(norm_boxes)
                            self.predictor.handle_request({
                                "type": "add_prompt",
                                "session_id": sid,
                                "frame_index": 0,
                                "bounding_boxes": norm_boxes,
                                "bounding_box_labels": labels
                            })
                            self.log(f"[INFO] Grounding detectou {len(norm_boxes)} objetos para o prompt '{prompt}' no frame inicial.")
                        else:
                            self.log(f"[AVISO] Nenhum objeto correspondente ao prompt '{prompt}' detectado no frame inicial.")
                        
                    # Stream propagation no bloco
                    local_outputs = {}
                    for step in self.predictor.handle_stream_request({
                        "type": "propagate_in_video",
                        "session_id": sid
                    }):
                        local_frame = step["frame_index"]
                        out = step["outputs"]
                        
                        # Move tensores para CPU imediatamente para liberar VRAM
                        cpu_out = {}
                        for k, v in out.items():
                            if isinstance(v, torch.Tensor):
                                cpu_out[k] = v.detach().cpu().numpy()
                            elif isinstance(v, np.ndarray):
                                cpu_out[k] = v.copy()
                            else:
                                cpu_out[k] = v
                        local_outputs[local_frame] = cpu_out
                        
                    # Fechar sessão para desalocar tensores pesados no SAM 3.1
                    self.predictor.handle_request({
                        "type": "close_session",
                        "session_id": sid
                    })
                    
                # Limpeza forçada de VRAM pós-sessão de bloco
                if device == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                # Mapeamento de identidades por correspondência de IoU
                local_to_global = {}
                
                # Resolve frame local 0
                if 0 in local_outputs:
                    out0 = local_outputs[0]
                    loc_ids = out0.get("out_obj_ids", [])
                    loc_boxes = out0.get("out_boxes_xywh", [])
                    
                    for idx_l, loc_id in enumerate(loc_ids):
                        loc_box = loc_boxes[idx_l]
                        best_iou = 0.0
                        best_global = None
                        
                        if b_idx > 0 and last_detections:
                            for det in last_detections:
                                iou = compute_iou(loc_box, det["box"])
                                if iou > best_iou:
                                    best_iou = iou
                                    best_global = det["global_id"]
                                    
                        if best_iou > 0.35 and best_global is not None:
                            local_to_global[loc_id] = best_global
                        else:
                            new_global = max(self.unique_objects_seen) + 1 if self.unique_objects_seen else 0
                            self.unique_objects_seen.add(new_global)
                            local_to_global[loc_id] = new_global
                            
                # Traduzir e preencher resultados globais na CPU
                for local_i in range(block_len):
                    global_i = b_start + local_i
                    
                    if local_i not in local_outputs:
                        # Cria frames vazios se nenhuma detecção foi feita
                        self.tracked_frames[global_i] = (
                            os.path.join(self.temp_dir, f"{global_i:05d}.jpg"),
                            {
                                "out_obj_ids": np.zeros(0, dtype=np.int64),
                                "out_probs": np.zeros(0, dtype=np.float32),
                                "out_boxes_xywh": np.zeros((0, 4), dtype=np.float32),
                                "out_binary_masks": np.zeros((0, dest_h, dest_w), dtype=bool)
                            }
                        )
                        continue
                        
                    out_loc = local_outputs[local_i]
                    loc_ids = out_loc.get("out_obj_ids", [])
                    loc_probs = out_loc.get("out_probs", np.ones(len(loc_ids)))
                    loc_boxes = out_loc.get("out_boxes_xywh", [])
                    loc_masks = out_loc.get("out_binary_masks", [])
                    
                    mapped_ids = []
                    for loc_id in loc_ids:
                        if loc_id not in local_to_global:
                            # Nova detecção surge no meio do bloco
                            new_global = max(self.unique_objects_seen) + 1 if self.unique_objects_seen else 0
                            self.unique_objects_seen.add(new_global)
                            local_to_global[loc_id] = new_global
                        mapped_ids.append(local_to_global[loc_id])
                        
                    mapped_outputs = {
                        "out_obj_ids": np.array(mapped_ids, dtype=np.int64),
                        "out_probs": loc_probs,
                        "out_boxes_xywh": loc_boxes,
                        "out_binary_masks": loc_masks
                    }
                    
                    if self.tracked_frames[global_i] is None:
                        self.tracked_frames[global_i] = (
                            os.path.join(self.temp_dir, f"{global_i:05d}.jpg"),
                            mapped_outputs
                        )
                        
                    # Salva caixas delimitadoras do último frame para feedback geométrico no bloco seguinte
                    if local_i == block_len - 1:
                        last_detections = []
                        thresh = self.val_confidence.get()
                        for idx_l, loc_id in enumerate(loc_ids):
                            if loc_probs[idx_l] >= thresh:
                                last_detections.append({
                                    "box": loc_boxes[idx_l].tolist(),
                                    "global_id": local_to_global[loc_id]
                                })
                                
                    # Exibe no player em tempo real
                    f_path = os.path.join(self.temp_dir, f"{global_i:05d}.jpg")
                    if os.path.exists(f_path):
                        pil_f = Image.open(f_path)
                        drawn = self.draw_tracking_on_image(pil_f, mapped_outputs)
                        self.display_image(drawn)
                        
                # Progresso
                prog = int(((b_idx + 1) / len(blocks)) * 100)
                self.after(0, lambda p=prog: self.progress_bar.configure(value=p))
                
                if os.path.exists(self.temp_block_dir):
                    shutil.rmtree(self.temp_block_dir)
                    
            # Fim do rastreamento
            elapsed = time.time() - start_time
            avg_fps = total_extracted / elapsed if elapsed > 0 else 0
            
            vram_end = torch.cuda.mem_get_info()[0] / 1024**3 if device == "cuda" else 0.0
            if device == "cuda":
                self.after(0, lambda: self.lbl_vram_info.config(text=f"VRAM Livre: {vram_init:.2f} GB -> {vram_end:.2f} GB"))
                
            self.log("\n=======================================================")
            self.log("[STATUS] RASTREAMENTO MULTI-BLOCO FINALIZADO COM SUCESSO!")
            self.log(f"[STATUS] Tempo Total: {elapsed:.2f}s | Velocidade: {avg_fps:.2f} FPS")
            self.log("=======================================================")
            
            self.after(0, lambda: self.lbl_time_info.config(text=f"Tempo Total: {elapsed:.2f} segundos"))
            self.after(0, lambda: self.lbl_unique_counter.config(text=f"Objetos Únicos: {len(self.unique_objects_seen)}"))
            
            self.playback_fps = actual_fps
            self.current_frame_idx = 0
            self.after(0, self._enable_player_controls)
            self.show_frame_idx(0)
            
            self.after(0, lambda: messagebox.showinfo("Processamento Concluído", 
                                f"O rastreamento do vídeo foi finalizado com sucesso!\n\n"
                                f"• Blocos Executados: {len(blocks)}\n"
                                f"• Quadros Processados: {len(self.tracked_frames)}\n"
                                f"• Objetos Únicos Mapeados: {len(self.unique_objects_seen)}\n"
                                f"• Velocidade Média: {avg_fps:.2f} FPS\n\n"
                                f"Sessões de memória liberadas com sucesso!"))
            
        except Exception as e:
            self.log(f"[ERRO] Ocorreu um erro no processamento: {e}")
        finally:
            self.is_processing_active = False
            self.after(0, lambda: self.btn_run.config(state="normal", text="⚡ Iniciar Rastreamento Segregado em Blocos (Memory-Safe)"))
            self.after(0, lambda: self.progress_bar.configure(value=0))

    def _enable_player_controls(self):
        self.btn_play_pause.config(state="normal")
        self.slider_timeline.config(state="normal", from_=0, to=len(self.tracked_frames) - 1)
        self.val_timeline.set(0)
        self.lbl_timeline_frame.config(text=f"Frame: 1 / {len(self.tracked_frames)}")

    def draw_tracking_on_image(self, pil_img, outputs):
        rgba = pil_img.convert("RGBA")
        overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        masks = outputs.get("out_binary_masks", [])
        obj_ids = outputs.get("out_obj_ids", [])
        boxes = outputs.get("out_boxes_xywh", [])
        probs = outputs.get("out_probs", [])
        
        w, h = pil_img.size
        thresh = self.val_confidence.get()
        
        for i in range(len(masks)):
            prob = probs[i] if i < len(probs) else 1.0
            if prob < thresh:
                continue
                
            obj_id = obj_ids[i]
            mask = masks[i]
            if mask.ndim == 3:
                mask = mask[0]
                
            rgb = COLORS[obj_id % len(COLORS)]
            color = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), 100)
            
            mask_im = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
            if mask_im.size != pil_img.size:
                mask_im = mask_im.resize(pil_img.size, Image.Resampling.BILINEAR)
                
            color_im = Image.new("RGBA", pil_img.size, color)
            overlay = Image.composite(color_im, overlay, mask_im)
            
            # O usuário solicitou remover a marcação quadrada (boxes) e labels, mantendo apenas o sombreado de cores
            # if len(boxes) > i:
            #     box = boxes[i]
            #     # Detecção inteligente de formato: se qualquer coordenada for > 1.0, é absoluto [x0, y0, x1, y1]
            #     if any(v > 1.0 for v in box):
            #         x0, y0, x1, y1 = box
            #         x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            #     else:
            #         # Formato relativo [x0_rel, y0_rel, bw_rel, bh_rel]
            #         x0_rel, y0_rel, bw_rel, bh_rel = box
            #         x0 = int(x0_rel * w)
            #         y0 = int(y0_rel * h)
            #         x1 = int((x0_rel + bw_rel) * w)
            #         y1 = int((y0_rel + bh_rel) * h)
            #     
            #     x0, y0 = max(0, x0), max(0, y0)
            #     x1, y1 = min(w - 1, x1), min(h - 1, y1)
            #     
            #     draw.rectangle([x0, y0, x1, y1], outline=tuple(color[:3]), width=2)
            #     draw.text((x0 + 4, max(y0 - 13, 0)), f"ID:{obj_id} ({prob:.2f})", fill="#00ff00")
                
        final = Image.alpha_composite(rgba, overlay)
        return final.convert("RGB")

    def show_frame_idx(self, idx):
        if not self.tracked_frames or idx < 0 or idx >= len(self.tracked_frames):
            return
            
        data = self.tracked_frames[idx]
        if data is None:
            return
            
        f_path, outputs = data
        if os.path.exists(f_path):
            try:
                pil_f = Image.open(f_path)
                drawn = self.draw_tracking_on_image(pil_f, outputs)
                self.display_image(drawn)
                self.after(0, lambda: self.lbl_timeline_frame.config(text=f"Frame: {idx + 1} / {len(self.tracked_frames)}"))
            except Exception as e:
                self.log(f"[ERRO] Falha ao exibir frame {idx}: {e}")

    def toggle_playback(self):
        mode = self.combo_mode.get()
        if "Live" in mode:
            if self.playback_active:
                self.stop_playback()
            else:
                self.playback_active = True
                self.btn_play_pause.config(text="⏸ Pausar", bg="#ffc107", activebackground="#e0a800")
                if not self.cap or not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.video_path)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                self._live_play_loop()
        else:
            if not self.tracked_frames:
                return
            if self.playback_active:
                self.stop_playback()
            else:
                self.playback_active = True
                self.btn_play_pause.config(text="⏸ Pausar", bg="#ffc107", activebackground="#e0a800")
                self._playback_loop()

    def stop_playback(self):
        self.playback_active = False
        self.btn_play_pause.config(text="▶ Reproduzir", bg="#28a745", activebackground="#218838")
        if self.playback_timer:
            self.after_cancel(self.playback_timer)
            self.playback_timer = None

    def _playback_loop(self):
        if not self.playback_active:
            return
        self.show_frame_idx(self.current_frame_idx)
        self.val_timeline.set(self.current_frame_idx)
        self.current_frame_idx = (self.current_frame_idx + 1) % len(self.tracked_frames)
        delay = int(1000 / self.playback_fps)
        self.playback_timer = self.after(delay, self._playback_loop)

    def _live_play_loop(self):
        if not self.playback_active or not self.cap:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.stop_playback()
            self.log("[STATUS] Fim do vídeo.")
            self.finish_video_benchmark()
            return
            
        self.current_frame_idx += 1
        current_time_sec = self.current_frame_idx / self.orig_fps
        
        # Redimensiona para exibição rápida no canvas e inferência asíncrona segura (previne VRAM OOM)
        target_height = self.parse_resolution_height(self.combo_resolution.get())
        if target_height is not None:
            scale = target_height / self.orig_h
            dest_w = int(self.orig_w * scale)
            dest_w = (dest_w // 2) * 2
            dest_h = (target_height // 2) * 2
            resized = cv2.resize(frame, (dest_w, dest_h), interpolation=cv2.INTER_AREA)
        else:
            resized = frame.copy()
            
        # Copia o frame thread-safely para exibição instantânea
        with self.frame_lock:
            self.current_frame_data = resized.copy()
            
        # Determina se devemos capturar um novo quadro para segmentação e contagem
        should_capture = False
        if self.live_capture_interval > 0.0:
            if current_time_sec >= self.next_capture_time:
                should_capture = True
                self.next_capture_time = current_time_sec + self.live_capture_interval
        else:
            # Modo Máxima GPU: captura o próximo frame disponível assim que a thread estiver livre
            if not self.capture_ready_flag and not self.capture_in_progress:
                should_capture = True
                
        if should_capture:
            with self.frame_lock:
                # O frame capturado já está perfeitamente redimensionado para escala e prevenção de OOM
                self.captured_frame_data = resized.copy()
                self.current_capture_time = current_time_sec
                self.current_capture_idx = self.current_frame_idx
                self.capture_ready_flag = True
                self.capture_in_progress = True
            
            formatted_time = time.strftime('%H:%M:%S')
            self.log(f"[{formatted_time}] [CAPTURA #{len(self.video_benchmark_results) + 1}] Frame capturado aos {current_time_sec:.2f}s (Frame #{self.current_frame_idx}) para processamento.")
            
        # Aplicação ultra-rápida do overlay NumPy (Sub-milissegundo)
        with self.frame_lock:
            mask_overlay = self.live_mask_overlay
            mask_bool = self.live_mask_bool
            opaque_overlay = self.live_opaque_overlay
            opaque_bool = self.live_opaque_bool
            
        if mask_overlay is not None and mask_bool is not None and mask_overlay.shape == resized.shape:
            # Alpha blending das máscaras de segmentação (60% vídeo original, 40% cor da máscara)
            resized[mask_bool] = (resized[mask_bool] * 0.6 + mask_overlay[mask_bool] * 0.4).astype(np.uint8)
            
        if opaque_overlay is not None and opaque_bool is not None and opaque_overlay.shape == resized.shape:
            # Desenha as caixas delimitadoras e IDs sobrepostos de forma opaca
            resized[opaque_bool] = opaque_overlay[opaque_bool]
            
        pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        self.display_image(pil_img)
        
        # Sincroniza slider e label da timeline
        self.val_timeline.set(self.current_frame_idx)
        self.lbl_timeline_frame.config(text=f"Frame: {self.current_frame_idx} / {self.total_frames}")
        
        # Executa no tempo de FPS original do vídeo
        delay = int(1000 / self.playback_fps)
        self.playback_timer = self.after(delay, self._live_play_loop)

    def _live_segmentation_worker(self, prompt):
        self.log("[STATUS] Thread de segmentação asíncrona ativa.")
        
        while self.is_processing_active:
            # Espera até que um novo frame esteja disponível para processamento
            with self.frame_lock:
                ready = self.capture_ready_flag
                frame = None
                if ready:
                    frame = self.captured_frame_data.copy() if self.captured_frame_data is not None else None
                    capture_time = self.current_capture_time
                    capture_idx = self.current_capture_idx
            
            if not ready or frame is None:
                time.sleep(0.01)
                continue
                
            t0 = time.time()
            
            try:
                # O frame já vem redimensionado corretamente de _live_play_loop
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Configura o image_processor para zero-shot text-prompting com filtragem precoce
                state = {}
                user_thresh = self.val_confidence.get()
                self.image_processor.set_confidence_threshold(user_thresh, state)
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.inference_mode()
                
                with torch.inference_mode(), autocast:
                    state = self.image_processor.set_image(pil_img, state)
                    state = self.image_processor.set_text_prompt(prompt, state)
                    
                # Extrai resultados brutos
                raw_masks = state.get("masks")
                raw_boxes = state.get("boxes")
                raw_scores = state.get("scores")
                
                # ---> CORREÇÃO: Extração segura imune a falhas de tensores CUDA <---
                if raw_scores is not None:
                    if hasattr(raw_scores, "cpu"):
                        scores_np = raw_scores.cpu().float().numpy()
                    else:
                        scores_np = np.array([s.cpu().float().numpy() if hasattr(s, "cpu") else s for s in raw_scores])
                else:
                    scores_np = np.array([])
                    
                user_thresh = self.val_confidence.get()
                keep_indices = np.where(scores_np >= user_thresh)[0] if len(scores_np) > 0 else []
                num_detected = len(keep_indices)
                
                # Pre-renderiza as matrizes para overlay ultra-rápido no player
                h, w, c_dim = frame.shape
                overlay_mask = np.zeros_like(frame, dtype=np.uint8)
                overlay_mask_bool = np.zeros((h, w), dtype=bool)
                overlay_opaque = np.zeros_like(frame, dtype=np.uint8)
                
                if num_detected > 0:
                    for out_idx, orig_idx in enumerate(keep_indices):
                        score = scores_np[orig_idx]
                        
                        # Máscara segura
                        m = None
                        if raw_masks is not None:
                            m_raw = raw_masks[orig_idx]
                            if hasattr(m_raw, "cpu"):
                                m_raw = m_raw.cpu().numpy()
                            if m_raw.ndim == 3:
                                m_raw = m_raw[0]
                                
                            if m_raw.shape != (h, w):
                                m = cv2.resize(m_raw.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                            else:
                                m = m_raw.astype(bool)
                                
                        if m is not None:
                            rgb = COLORS[out_idx % len(COLORS)]
                            color_bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
                            overlay_mask[m] = color_bgr
                            overlay_mask_bool[m] = True
                            
                        # Caixa segura
                        b = None
                        if raw_boxes is not None:
                            b_raw = raw_boxes[orig_idx]
                            if hasattr(b_raw, "cpu"):
                                b = b_raw.cpu().numpy()
                            else:
                                b = b_raw
                                
                        if b is not None and len(b) >= 4:
                            if any(v > 1.0 for v in b):
                                x0, y0, x1, y1 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                            else:
                                x0 = int(b[0] * w)
                                y0 = int(b[1] * h)
                                x1 = int((b[0] + b[2]) * w)
                                y1 = int((b[1] + b[3]) * h)
                                
                            x0, y0 = max(0, x0), max(0, y0)
                            x1, y1 = min(w - 1, x1), min(h - 1, y1)
                            
                            # Evita falha se a máscara falhar mas a box existir
                            rgb = COLORS[out_idx % len(COLORS)]
                            color_bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
                                
                            # O usuário solicitou remover a marcação quadrada (boxes) e labels, mantendo apenas o sombreado de cores
                            # cv2.rectangle(overlay_opaque, (x0, y0), (x1, y1), color_bgr, 2)
                            # label_text = f"ID:{out_idx} ({score:.2f})"
                            # cv2.putText(overlay_opaque, label_text, (x0 + 4, max(y0 - 6, 15)),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                                        
                overlay_opaque_bool = np.any(overlay_opaque != 0, axis=-1)
                
                # Prepara cpu_out para compatibilidade retroativa e sincronização completa
                if num_detected > 0:
                    if isinstance(raw_masks, torch.Tensor):
                        filtered_masks = raw_masks[keep_indices].cpu().numpy()
                        filtered_boxes = raw_boxes[keep_indices].cpu().numpy() if raw_boxes is not None else np.zeros((num_detected, 4), dtype=np.float32)
                        filtered_scores = raw_scores[keep_indices].cpu().numpy()
                    else:
                        filtered_masks = np.array([raw_masks[i] for i in keep_indices])
                        filtered_boxes = np.array([raw_boxes[i] for i in keep_indices]) if raw_boxes is not None else np.zeros((num_detected, 4), dtype=np.float32)
                        filtered_scores = np.array([raw_scores[i] for i in keep_indices])
                else:
                    filtered_masks = np.zeros((0, h, w), dtype=bool)
                    filtered_boxes = np.zeros((0, 4), dtype=np.float32)
                    filtered_scores = np.zeros(0, dtype=np.float32)

                cpu_out = {
                    "out_obj_ids": np.arange(num_detected, dtype=np.int64),
                    "out_probs": filtered_scores,
                    "out_boxes_xywh": filtered_boxes,
                    "out_binary_masks": filtered_masks
                }
                
                # Sincroniza a máscara ativa e os overlays thread-safely
                with self.frame_lock:
                    self.active_masks = cpu_out
                    self.live_mask_overlay = overlay_mask
                    self.live_mask_bool = overlay_mask_bool
                    self.live_opaque_overlay = overlay_opaque
                    self.live_opaque_bool = overlay_opaque_bool
                    
                elapsed = time.time() - t0
                
                # Atualiza estatísticas da interface gráfica
                self.after(0, lambda c=num_detected: self.lbl_unique_counter.config(text=f"Objetos na Tela: {c}"))
                self.after(0, lambda e=elapsed: self.lbl_time_info.config(text=f"IA Frame: {e:.2f} segundos"))
                
                # Adiciona entrada ao benchmark de vídeo
                self.video_benchmark_results.append({
                    "capture_num": len(self.video_benchmark_results) + 1,
                    "time_sec": capture_time,
                    "frame_idx": capture_idx,
                    "count": num_detected,
                    "elapsed": elapsed
                })
                
                self.log(f"[RESULTADO] Captura #{len(self.video_benchmark_results)} aos {capture_time:.2f}s: Mapeados {num_detected} '{prompt}'(s) em {elapsed:.2f}s")
                
            except Exception as e:
                self.log(f"[ERRO IA] Falha no frame aos {capture_time:.2f}s: {e}")
                
            # Limpa as flags de controle para aceitar nova captura periódica
            with self.frame_lock:
                self.capture_ready_flag = False
                self.capture_in_progress = False
                
            # Limpa VRAM periodicamente pós-captura
            if device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
                
        self.log("[STATUS] Thread de segmentação asíncrona finalizada.")



    def on_scrub(self, val):
        idx = int(float(val))
        if idx != self.current_frame_idx:
            if self.playback_active:
                self.stop_playback()
            self.current_frame_idx = idx
            
            mode = self.combo_mode.get()
            if "Live" in mode:
                if not self.cap or not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.video_path)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = self.cap.read()
                if ret:
                    with self.frame_lock:
                        self.current_frame_data = frame.copy()
                        
                    target_height = self.parse_resolution_height(self.combo_resolution.get())
                    if target_height is not None:
                        scale = target_height / self.orig_h
                        dest_w = int(self.orig_w * scale)
                        dest_w = (dest_w // 2) * 2
                        dest_h = (target_height // 2) * 2
                        resized = cv2.resize(frame, (dest_w, dest_h), interpolation=cv2.INTER_AREA)
                    else:
                        resized = frame
                        
                    # Aplicação ultra-rápida do overlay NumPy (Sub-milissegundo)
                    with self.frame_lock:
                        mask_overlay = self.live_mask_overlay
                        mask_bool = self.live_mask_bool
                        opaque_overlay = self.live_opaque_overlay
                        opaque_bool = self.live_opaque_bool
                        
                    if mask_overlay is not None and mask_bool is not None and mask_overlay.shape == resized.shape:
                        # Alpha blending das máscaras de segmentação (60% vídeo original, 40% cor da máscara)
                        resized[mask_bool] = (resized[mask_bool] * 0.6 + mask_overlay[mask_bool] * 0.4).astype(np.uint8)
                        
                    if opaque_overlay is not None and opaque_bool is not None and opaque_overlay.shape == resized.shape:
                        # Desenha as caixas delimitadoras e IDs sobrepostos de forma opaca
                        resized[opaque_bool] = opaque_overlay[opaque_bool]
                        
                    pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                    self.display_image(pil_img)
                    self.lbl_timeline_frame.config(text=f"Frame: {idx} / {self.total_frames}")
            else:
                self.show_frame_idx(idx)

    def stop_current_action(self):
        self.is_processing_active = False
        self.stop_playback()
        self.finish_video_benchmark()
        self.tracked_frames = []
        self.unique_objects_seen.clear()
        self.current_frame_idx = 0
        
        # Libera VideoCapture
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
            
        with self.frame_lock:
            self.current_frame_data = None
            self.captured_frame_data = None
            self.active_masks = None
            
        self.lbl_unique_counter.config(text="Objetos Únicos: 0")
        self.lbl_timeline_frame.config(text="Frame: 0 / 0")
        self.btn_play_pause.config(state="disabled")
        self.slider_timeline.config(state="disabled")
        self.val_timeline.set(0)
        self.progress_bar.configure(value=0)
        
        for d in [self.temp_dir, self.temp_block_dir]:
            if os.path.exists(d):
                try:
                    shutil.rmtree(d)
                    os.makedirs(d, exist_ok=True)
                except Exception:
                    pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.log("[STATUS] Processamento interrompido e memórias liberadas.")
        self.after(0, lambda: self.btn_run.config(state="normal", text="⚡ Iniciar Rastreamento Live (Tempo Real)"))
        self.load_video_metadata()

    def finish_video_benchmark(self):
        results = self.video_benchmark_results
        if not results:
            return
            
        num_caps = len(results)
        total_time = sum(r["elapsed"] for r in results)
        avg_time = total_time / num_caps if num_caps > 0 else 0.0
        avg_objs = sum(r["count"] for r in results) / num_caps if num_caps > 0 else 0.0
        
        prompt = self.entry_prompt.get().strip()
        video_dir = os.path.dirname(self.video_path) if self.video_path else self.default_video_dir
        
        # O usuário solicitou não gerar relatórios em TXT/CSV no modo Live
        # para economizar processamento e manter o tempo de execução 100% otimizado.
        # Mas queremos mostrar a janela modal do benchmark como processo de exibição interativo.
        self.after(0, lambda: self.show_video_benchmark_results_window(prompt, video_dir, num_caps, total_time, avg_time, avg_objs, results))

    def show_video_benchmark_results_window(self, prompt, folder, num_caps, total_time, avg_time, avg_objs, results):
        # Janela Toplevel estilizada e dark
        report_win = tk.Toplevel(self)
        report_win.title("📊 Relatório de Benchmark em Vídeo - SAM 3.1")
        report_win.geometry("820x640")
        report_win.configure(bg="#1e1e1f")
        report_win.transient(self)
        report_win.grab_set()
        
        # Centralizar na tela
        report_win.update_idletasks()
        w = report_win.winfo_width()
        h = report_win.winfo_height()
        x = (report_win.winfo_screenwidth() // 2) - (w // 2)
        y = (report_win.winfo_screenheight() // 2) - (h // 2)
        report_win.geometry(f"+{x}+{y}")
        
        # Título superior
        title_frame = tk.Frame(report_win, bg="#252526", pady=10, padx=15, highlightbackground="#3c3c3c", highlightthickness=1)
        title_frame.pack(fill="x", side="top", padx=10, pady=10)
        
        tk.Label(title_frame, text="RESULTADO DO BENCHMARK EM VÍDEO (AO VIVO)", bg="#252526", fg="#ffaa00", font=("Helvetica", 12, "bold")).pack(anchor="w")
        tk.Label(title_frame, text=f"Prompt/Objeto: '{prompt}' | Limiar de Confiança: {self.val_confidence.get():.2f} | Intervalo: {self.combo_live_interval.get()}", bg="#252526", fg="#a0a0a0", font=("Helvetica", 9, "italic")).pack(anchor="w")
        
        # Painel de Cards de Métricas
        metrics_frame = tk.Frame(report_win, bg="#1e1e1f")
        metrics_frame.pack(fill="x", padx=10, pady=5)
        
        def create_metric_card(parent, title, val, col, color="#00ff66"):
            card = tk.Frame(parent, bg="#252526", padx=10, pady=8, highlightbackground="#3c3c3c", highlightthickness=1)
            card.grid(row=0, column=col, sticky="nsew", padx=4, pady=2)
            tk.Label(card, text=title, bg="#252526", fg="#8e8e93", font=("Helvetica", 8, "bold")).pack()
            tk.Label(card, text=val, bg="#252526", fg=color, font=("Helvetica", 11, "bold")).pack(pady=(2, 0))
            parent.columnconfigure(col, weight=1)
            
        create_metric_card(metrics_frame, "Capturas Totais", f"{num_caps}", 0, "#ffffff")
        create_metric_card(metrics_frame, "Tempo Total IA", f"{total_time:.2f}s", 1, "#ffcc00")
        create_metric_card(metrics_frame, "Tempo Médio IA", f"{avg_time:.3f}s", 2, "#00aaff")
        create_metric_card(metrics_frame, "Média de Objetos", f"{avg_objs:.2f}", 3, "#00ff66")
        
        # Treeview / Tabela
        table_frame = tk.Frame(report_win, bg="#1e1e1f")
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Customizar Treeview para Dark Mode
        style = ttk.Style()
        style.configure("Treeview", background="#252526", foreground="#ffffff", fieldbackground="#252526", rowheight=24, font=("Helvetica", 9))
        style.map("Treeview", background=[("selected", "#007acc")], foreground=[("selected", "#ffffff")])
        style.configure("Treeview.Heading", background="#3c3c3d", foreground="#ffffff", font=("Helvetica", 9, "bold"))
        style.map("Treeview.Heading", background=[("active", "#4d4d4f")])
        
        columns = ("cap_num", "time_sec", "frame_idx", "count", "elapsed")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", style="Treeview")
        tree.heading("cap_num", text="Captura #")
        tree.heading("time_sec", text="Tempo Vídeo")
        tree.heading("frame_idx", text="Frame ID")
        tree.heading("count", text="Mapeados")
        tree.heading("elapsed", text="Tempo IA")
        
        tree.column("cap_num", anchor="center", width=80)
        tree.column("time_sec", anchor="center", width=120)
        tree.column("frame_idx", anchor="center", width=120)
        tree.column("count", anchor="center", width=150)
        tree.column("elapsed", anchor="center", width=150)
        
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        for r in results:
            tree.insert("", "end", values=(
                f"#{r['capture_num']}",
                f"{r['time_sec']:.2f}s",
                f"F_{r['frame_idx']}",
                f"{r['count']} '{prompt}'(s)",
                f"{r['elapsed']:.3f}s"
            ))
            
        # Botões inferiores
        btn_frame = tk.Frame(report_win, bg="#1e1e1f", pady=10)
        btn_frame.pack(fill="x", side="bottom", padx=10)
        
        def open_folder():
            os.startfile(folder)
                
        btn_open_folder = ttk.Button(btn_frame, text="📂 Abrir Pasta do Vídeo", command=open_folder)
        btn_open_folder.pack(side="left", padx=5)
        
        btn_close = tk.Button(btn_frame, text="Fechar Relatório", bg="#dc3545", fg="#ffffff", relief="flat", font=("Helvetica", 9, "bold"), activebackground="#bd2130", activeforeground="#ffffff", padx=15, command=report_win.destroy)
        btn_close.pack(side="right", padx=5)

if __name__ == "__main__":
    app = SAM3VideoTrackerApp()
    app.mainloop()
