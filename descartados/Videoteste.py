# -*- coding: utf-8 -*-
"""
SAM 3.1 - Teste de Extração e Segmentação Sequencial (Videoteste)
================================================================
Este script realiza a extração uniforme de quadros de um vídeo local e 
executa a segmentação sequencial multi-imagens (Grounding Zero-Shot) na tela, 
simulando a segmentação do vídeo frame por frame com máxima economia de memória.
"""

import os
import gc
import sys
import time
import glob
import shutil
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import torch
from PIL import Image, ImageTk, ImageDraw

# ============================================================
# PATCH DE COMPATIBILIDADE DE SDPA PARA RTX 3060 LAPTOP
# ============================================================
import functools
import torch.nn.functional as _F_torch

try:
    # Tenta ativar as otimizações por padrão (se a máquina suportar, rodará super rápido)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

_original_sdpa = _F_torch.scaled_dot_product_attention

@functools.wraps(_original_sdpa)
def _safe_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    try:
        # Tenta primeiramente com os kernels rápidos ativados oportunisticamente
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass
    try:
        return _original_sdpa(query, key, value, attn_mask=attn_mask,
                               dropout_p=dropout_p, is_causal=is_causal)
    except RuntimeError as e:
        err_str = str(e)
        if "No available kernel" in err_str or "not supported" in err_str.lower() or "abort" in err_str.lower() or "compatibility" in err_str.lower():
            # Fallback seguro para o modo matemático puro na GPU do notebook RTX 3060 Laptop
            try:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
            except Exception:
                pass
            return _original_sdpa(query, key, value, attn_mask=attn_mask,
                                     dropout_p=dropout_p, is_causal=is_causal)
        raise

_F_torch.scaled_dot_product_attention = _safe_sdpa
if hasattr(torch, "scaled_dot_product_attention"):
    torch.scaled_dot_product_attention = _safe_sdpa


# Carregamento de variáveis de ambiente (.env)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

# Adiciona caminhos do SAM 3 e do src do projeto
_sam3_path = os.environ.get("SAM3_PATH", "")
if _sam3_path:
    sys.path.append(_sam3_path)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

try:
    from sam3 import build_sam3_image_model
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

class VideoTesteSegmenterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SAM 3.1 - Simulador de Segmentação Sequencial de Vídeo (Videoteste)")
        self.geometry("1200x900")
        self.configure(bg="#1e1e1f")
        
        self.video_path = None
        self.fps = 24.0
        self.total_frames = 0
        self.duration_seconds = 0.0
        
        self.model = None
        self.processor = None
        self.current_frame_image = None
        self.is_running = False
        
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.default_video_dir = os.path.join(self.project_root, "media", "videos")
        self.default_image_dir = os.path.join(self.project_root, "media", "images")
        
        os.makedirs(self.default_video_dir, exist_ok=True)
        os.makedirs(self.default_image_dir, exist_ok=True)
        
        # Estilos Dark Mode Premium
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background="#1e1e1f", foreground="#ffffff", fieldbackground="#2d2d2d")
        self.style.configure("TButton", background="#3c3c3d", foreground="#ffffff", borderwidth=0, padding=6, font=("Helvetica", 9, "bold"))
        self.style.map("TButton", background=[("active", "#4d4d4f")])
        self.style.configure("TLabel", background="#1e1e1f", foreground="#ffffff")
        self.style.configure("TEntry", fieldbackground="#2d2d2d", foreground="#ffffff", insertcolor="#ffffff")
        self.style.configure("TCombobox", fieldbackground="#2d2d2d", background="#3c3c3d", foreground="#ffffff")
        self.style.map("TCombobox", fieldbackground=[("readonly", "#2d2d2d")], foreground=[("readonly", "#ffffff")])
        
        self.create_widgets()
        self.scan_videos_in_folder()
        
        # Carga em background do modelo
        if SAM3_AVAILABLE:
            self.log("[STATUS] Carregando pesos do SAM 3.1 Image Model em background...")
            threading.Thread(target=self.load_sam3_image_model, daemon=True).start()
        else:
            self.log(f"[ERRO CRÍTICO] Falha ao importar SAM 3.1: {SAM3_IMPORT_ERROR}")
            messagebox.showerror("Erro de Importação", f"Erro: {SAM3_IMPORT_ERROR}")

    def create_widgets(self):
        ctrl_frame = tk.Frame(self, bg="#252526", pady=12, padx=15, highlightbackground="#3c3c3c", highlightthickness=1)
        ctrl_frame.pack(fill="x", side="top", padx=10, pady=10)
        
        # Título
        tk.Label(ctrl_frame, text="PAINEL DE CONFIGURAÇÃO - EXTRAÇÃO E SEGMENTAÇÃO DE VÍDEO", bg="#252526", fg="#ffaa00", font=("Helvetica", 11, "bold")).grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 10))
        
        # Linha 1: Seleção de Origem
        tk.Label(ctrl_frame, text="Pasta de Vídeos:", bg="#252526", fg="#ffffff", font=("Helvetica", 10, "bold")).grid(row=1, column=0, sticky="w", pady=5)
        self.entry_folder_path = ttk.Entry(ctrl_frame, width=40)
        self.entry_folder_path.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.entry_folder_path.insert(0, self.default_video_dir)
        
        self.btn_select_folder = ttk.Button(ctrl_frame, text="Selecionar Pasta...", command=self.select_folder)
        self.btn_select_folder.grid(row=1, column=2, padx=5, pady=5)
        
        tk.Label(ctrl_frame, text="Vídeo Local:", bg="#252526", fg="#ffffff", font=("Helvetica", 10, "bold")).grid(row=1, column=3, sticky="w", pady=5, padx=(15, 0))
        self.combo_videos = ttk.Combobox(ctrl_frame, width=25, state="readonly")
        self.combo_videos.grid(row=1, column=4, padx=5, pady=5)
        self.combo_videos.bind("<<ComboboxSelected>>", self.on_video_selected)
        
        # Linha 2: Vídeo Avulso e Metadados
        self.btn_browse_video = ttk.Button(ctrl_frame, text="Procurar Vídeo...", command=self.browse_video)
        self.btn_browse_video.grid(row=2, column=1, sticky="w", padx=5, pady=2)
        
        self.lbl_metadata = tk.Label(ctrl_frame, text="Duração: -- | FPS: -- | Total Frames: --", bg="#252526", fg="#00ff66", font=("Helvetica", 9, "bold"))
        self.lbl_metadata.grid(row=2, column=2, columnspan=3, sticky="w", padx=5, pady=2)
        
        ttk.Separator(ctrl_frame, orient="horizontal").grid(row=3, column=0, columnspan=5, sticky="ew", pady=8)
        
        # Parâmetros
        tk.Label(ctrl_frame, text="Tempo Inicial (s):", bg="#252526", fg="#ffffff", font=("Helvetica", 10, "bold")).grid(row=4, column=0, sticky="w", pady=5)
        self.entry_start_time = ttk.Entry(ctrl_frame, width=15)
        self.entry_start_time.insert(0, "0.0")
        self.entry_start_time.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        
        tk.Label(ctrl_frame, text="Tempo Final (s):", bg="#252526", fg="#ffffff", font=("Helvetica", 10, "bold")).grid(row=4, column=2, sticky="w", pady=5, padx=(15, 0))
        self.entry_end_time = ttk.Entry(ctrl_frame, width=15)
        self.entry_end_time.grid(row=4, column=3, padx=5, pady=5, sticky="w")
        
        # Quantidade de Imagens
        tk.Label(ctrl_frame, text="Qtd. Imagens:", bg="#252526", fg="#ffffff", font=("Helvetica", 10, "bold")).grid(row=5, column=0, sticky="w", pady=5)
        self.entry_quantity = ttk.Entry(ctrl_frame, width=15)
        self.entry_quantity.insert(0, "10")
        self.entry_quantity.grid(row=5, column=1, padx=5, pady=5, sticky="w")
        
        # Prompt e Confiança
        tk.Label(ctrl_frame, text="Prompt (Texto):", bg="#252526", fg="#ffffff", font=("Helvetica", 10, "bold")).grid(row=5, column=2, sticky="w", pady=5, padx=(15, 0))
        self.entry_prompt = ttk.Entry(ctrl_frame, width=20)
        self.entry_prompt.insert(0, "person")
        self.entry_prompt.grid(row=5, column=3, padx=5, pady=5, sticky="w")
        
        # Limiar Confiança
        tk.Label(ctrl_frame, text="Confiança:", bg="#252526", fg="#ffffff", font=("Helvetica", 10, "bold")).grid(row=5, column=4, sticky="w", pady=5, padx=5)
        self.val_confidence = tk.DoubleVar(value=0.45)
        self.slider_conf = ttk.Scale(ctrl_frame, from_=0.1, to=0.9, variable=self.val_confidence, orient="horizontal", command=self.update_conf_label)
        self.slider_conf.grid(row=6, column=4, sticky="ew", padx=5, pady=2)
        self.lbl_conf = tk.Label(ctrl_frame, text="0.45", bg="#252526", fg="#a0a0a0", font=("Helvetica", 9, "bold"))
        self.lbl_conf.grid(row=6, column=3, sticky="e", padx=5, pady=2)
        
        # Resolução e Pasta de Destino
        tk.Label(ctrl_frame, text="Resolução Alvo:", bg="#252526", fg="#00aaff", font=("Helvetica", 10, "bold")).grid(row=6, column=0, sticky="w", pady=5)
        self.combo_resolution = ttk.Combobox(ctrl_frame, values=["Original", "720p (HD)", "480p (SD)", "360p (Baixa)"], width=15, state="readonly")
        self.combo_resolution.set("480p (SD)")
        self.combo_resolution.grid(row=6, column=1, padx=5, pady=5, sticky="w")
        
        tk.Label(ctrl_frame, text="Pasta Destino:", bg="#252526", fg="#ffffff", font=("Helvetica", 10, "bold")).grid(row=7, column=0, sticky="w", pady=5)
        self.entry_dest_folder = ttk.Entry(ctrl_frame, width=40)
        self.entry_dest_folder.grid(row=7, column=1, columnspan=3, padx=5, pady=5, sticky="ew")
        
        self.btn_select_dest = ttk.Button(ctrl_frame, text="Destino...", command=self.select_dest_folder)
        self.btn_select_dest.grid(row=7, column=4, padx=5, pady=5)
        
        # Botão Principal Executar
        self.btn_run = tk.Button(ctrl_frame, text="⚡ Iniciar Extração e Segmentação Sequencial", bg="#28a745", fg="#ffffff", font=("Helvetica", 11, "bold"), activebackground="#218838", activeforeground="#ffffff", relief="flat", state="disabled", command=self.start_process)
        self.btn_run.grid(row=8, column=0, columnspan=5, sticky="ew", pady=(15, 0))
        
        # Painel Central: Canvas
        viewer_container = tk.Frame(self, bg="#1e1e1f")
        viewer_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        viewer_frame = ttk.LabelFrame(viewer_container, text="Visualizador de Sequência Mapeada (Simulador de Vídeo)", padding=5)
        viewer_frame.pack(fill="both", expand=True)
        
        self.canvas_image = tk.Canvas(viewer_frame, bg="#151515", highlightthickness=0)
        self.canvas_image.pack(fill="both", expand=True)
        
        # Overlay do Contador
        self.lbl_counter = tk.Label(self, text="Objetos Mapeados no Frame Atual: 0", bg="#1e1e1f", fg="#00ff00", font=("Helvetica", 14, "bold"), anchor="w")
        self.lbl_counter.pack(fill="x", padx=15, pady=5)
        
        # Barra de progresso
        self.progress_bar = ttk.Progressbar(self, orient="horizontal", mode="determinate")
        self.progress_bar.pack(fill="x", padx=15, pady=5)
        
        # Console inferior
        self.log_frame = ttk.LabelFrame(self, text="Console de Execução & IA Pipeline", padding=5)
        self.log_frame.pack(fill="x", side="bottom", padx=10, pady=(0, 10))
        
        self.text_log = tk.Text(self.log_frame, wrap="word", height=6, bg="#101010", fg="#00ff66", insertbackground="white", state="disabled", font=("Courier", 9))
        self.text_log.pack(fill="both", expand=True)

    def log(self, msg):
        def _log():
            self.text_log.config(state="normal")
            self.text_log.insert("end", f"{msg}\n")
            self.text_log.config(state="disabled")
            self.text_log.see("end")
        self.after(0, _log)

    def update_conf_label(self, val):
        self.lbl_conf.config(text=f"{float(val):.2f}")

    def select_folder(self):
        folder = filedialog.askdirectory(title="Selecionar Pasta de Vídeos")
        if folder:
            self.entry_folder_path.delete(0, "end")
            self.entry_folder_path.insert(0, folder)
            self.scan_videos_in_folder()

    def select_dest_folder(self):
        folder = filedialog.askdirectory(title="Selecionar Pasta de Destino")
        if folder:
            self.entry_dest_folder.delete(0, "end")
            self.entry_dest_folder.insert(0, folder)

    def scan_videos_in_folder(self):
        folder = self.entry_folder_path.get().strip()
        if not os.path.isdir(folder):
            self.log(f"[AVISO] Pasta não encontrada: {folder}")
            return
            
        extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm", "*.MP4", "*.AVI", "*.MOV", "*.MKV", "*.WEBM"]
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(folder, ext)))
            
        names = [os.path.basename(f) for f in files]
        self.combo_videos.config(values=names)
        
        if names:
            self.combo_videos.current(0)
            self.on_video_selected(None)
            self.log(f"[OK] {len(names)} vídeos identificados na pasta.")
        else:
            self.combo_videos.set("")
            self.log("[AVISO] Nenhum vídeo encontrado nesta pasta.")

    def on_video_selected(self, event):
        folder = self.entry_folder_path.get().strip()
        name = self.combo_videos.get()
        if name:
            self.video_path = os.path.join(folder, name)
            self.load_video_metadata()

    def browse_video(self):
        file = filedialog.askopenfilename(
            title="Selecionar Vídeo Avulso",
            filetypes=[("Vídeos", "*.mp4 *.avi *.mov *.mkv *.webm")]
        )
        if file:
            self.video_path = file
            self.combo_videos.set(os.path.basename(file))
            self.log(f"[OK] Vídeo selecionado: {os.path.basename(file)}")
            self.load_video_metadata()

    def load_video_metadata(self):
        if not self.video_path or not os.path.exists(self.video_path):
            return
            
        cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0: self.fps = 24.0
        
        self.duration_seconds = self.total_frames / self.fps
        cap.release()
        
        self.lbl_metadata.config(text=f"Duração: {self.duration_seconds:.2f}s | FPS: {self.fps:.2f} | Total Frames: {self.total_frames}")
        
        self.entry_end_time.delete(0, "end")
        self.entry_end_time.insert(0, f"{self.duration_seconds:.2f}")
        self.entry_start_time.delete(0, "end")
        self.entry_start_time.insert(0, "0.0")
        
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        suggested = os.path.join(self.default_image_dir, f"{video_name}_extracted_frames")
        self.entry_dest_folder.delete(0, "end")
        self.entry_dest_folder.insert(0, suggested)
        
        # Preview do frame 0
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.display_image(Image.fromarray(rgb))
        cap.release()
        
        self.log(f"[STATUS] Metadados carregados para {os.path.basename(self.video_path)}")

    def display_image(self, pil_img):
        def _display():
            self.canvas_image.delete("all")
            c_w = self.canvas_image.winfo_width()
            c_h = self.canvas_image.winfo_height()
            if c_w <= 1: c_w = 640
            if c_h <= 1: c_h = 480
            
            w, h = pil_img.size
            ratio = min(c_w / w, c_h / h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            
            resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(resized)
            self._tk_img_show = img_tk
            self.canvas_image.create_image(c_w // 2, c_h // 2, anchor="center", image=img_tk)
        self.after(0, _display)

    def load_sam3_image_model(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.log(f"[STATUS] Inicializando motor SAM 3.1 no dispositivo '{device.upper()}'...")
            
            self.model = build_sam3_image_model(compile=False, load_from_HF=True, device=device)
            self.processor = Sam3Processor(self.model, device=device)
            
            self.log("[OK] SAM 3.1 Image Predictor inicializado com sucesso!")
            self.after(0, lambda: self.btn_run.config(state="normal"))
        except Exception as e:
            self.log(f"[ERRO] Falha ao carregar o motor SAM 3.1: {e}")

    def start_process(self):
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showwarning("Aviso", "Selecione um vídeo válido primeiro.")
            return
            
        dest = self.entry_dest_folder.get().strip()
        if not dest:
            return
            
        prompt = self.entry_prompt.get().strip()
        if not prompt:
            messagebox.showwarning("Aviso", "Digite um prompt de texto (ex: person).")
            return
            
        try:
            start = float(self.entry_start_time.get().strip())
            end = float(self.entry_end_time.get().strip())
            qty = int(self.entry_quantity.get().strip())
        except ValueError:
            messagebox.showwarning("Aviso", "Parâmetros de tempo ou quantidade inválidos.")
            return
            
        if start < 0 or start >= self.duration_seconds:
            return
        if end <= start or end > self.duration_seconds:
            return
        if qty <= 0:
            return
            
        self.btn_run.config(state="disabled", text="Processando Pipeline...")
        self.is_running = True
        
        threading.Thread(target=self._process_pipeline_thread, args=(start, end, qty, dest, prompt), daemon=True).start()

    def _process_pipeline_thread(self, start, end, qty, dest, prompt):
        try:
            # ----------------------------------------------------
            # ETAPA 1: Extração Uniforme de Frames
            # ----------------------------------------------------
            self.log("\n=======================================================")
            self.log("[ETAPA 1/2] EXTRAÇÃO DOS QUADROS DO VÍDEO...")
            self.log(f"[STATUS] Origem: {os.path.basename(self.video_path)}")
            self.log(f"[STATUS] Intervalo: {start:.2f}s - {end:.2f}s | Quadros a extrair: {qty}")
            self.log(f"[STATUS] Pasta Destino: {dest}")
            
            # Limpa pasta anterior se existir para evitar mistura de frames
            if os.path.exists(dest):
                shutil.rmtree(dest)
            os.makedirs(dest, exist_ok=True)
            
            start_frame = int(start * self.fps)
            end_frame = int(end * self.fps)
            frame_range = end_frame - start_frame
            
            if qty == 1:
                indices = [start_frame]
            else:
                indices = [int(start_frame + i * (frame_range - 1) / (qty - 1)) for i in range(qty)]
                
            cap = cv2.VideoCapture(self.video_path)
            extracted_paths = []
            
            self.after(0, lambda: self.progress_bar.configure(value=0, maximum=qty))
            
            success_count = 0
            for idx, f_idx in enumerate(indices):
                if not self.is_running:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if ret:
                    f_name = f"frame_{idx+1:04d}_f{f_idx}.jpg"
                    full_path = os.path.join(dest, f_name)
                    cv2.imwrite(full_path, frame)
                    extracted_paths.append(full_path)
                    success_count += 1
                    self.log(f"[EXTRAÇÃO] [{success_count}/{qty}] Salvo: {f_name}")
                else:
                    self.log(f"[EXTRAÇÃO] [ERRO] Falha no quadro {f_idx}")
                    
                self.after(0, lambda v=idx+1: self.progress_bar.configure(value=v))
                
            cap.release()
            self.log(f"[SUCESSO] Extraídos {success_count} frames na pasta.")
            
            if not extracted_paths:
                raise RuntimeError("Nenhum frame pôde ser extraído do vídeo.")
                
            # ----------------------------------------------------
            # ETAPA 2: Segmentação Sequencial Multi-Imagens
            # ----------------------------------------------------
            self.log("\n=======================================================")
            self.log("[ETAPA 2/2] INICIANDO SEGMENTAÇÃO DE FRAMES COM IA...")
            self.log(f"[STATUS] Prompt: '{prompt}' | Limiar: {self.val_confidence.get():.2f}")
            self.log("=======================================================")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            target_res = self.combo_resolution.get()
            user_thresh = self.val_confidence.get()
            
            # OTIMIZAÇÃO 1: Inicialização dinâmica da resolução do processador SAM 3.1
            res_map = {
                "Original": 1024,
                "720p (HD)": 768,
                "480p (SD)": 512,
                "360p (Baixa)": 384
            }
            internal_res = res_map.get(target_res, 512)
            self.log(f"[IA] Configurando processador SAM 3.1 com resolução interna de {internal_res}px...")
            self.processor = Sam3Processor(self.model, resolution=internal_res, device=device)
            
            # OTIMIZAÇÃO 2: Pré-codificação do prompt de texto (Cache de Embeddings)
            self.log(f"[IA] Pré-codificando prompt de texto '{prompt}' para caching...")
            cached_text = None
            with torch.inference_mode():
                if device == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        cached_text = self.model.backbone.forward_text([prompt], device=device)
                else:
                    cached_text = self.model.backbone.forward_text([prompt], device=device)
            
            self.after(0, lambda: self.progress_bar.configure(value=0, maximum=len(extracted_paths)))
            
            for idx, path in enumerate(extracted_paths):
                if not self.is_running:
                    break
                    
                t0 = time.time()
                
                # 1. Carrega imagem
                orig_img = Image.open(path).convert("RGB")
                w, h = orig_img.size
                working_img = orig_img
                
                # Redimensionamento dinâmico imune a OOM
                if target_res != "Original":
                    target_h = 480
                    if "720p" in target_res:
                        target_h = 720
                    elif "360p" in target_res:
                        target_h = 360
                    target_w = int(w * (target_h / h))
                    target_w = (target_w // 2) * 2
                    target_h = (target_h // 2) * 2
                    working_img = orig_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                
                # 2. Roda inferência de Grounding com a IA (com filtragem precoce no threshold do usuário)
                state = {}
                self.processor.set_confidence_threshold(user_thresh, state)
                
                # OTIMIZAÇÃO: Injeta os embeddings de texto cacheados em vez de re-codificar o prompt a cada frame
                with torch.inference_mode():
                    if device == "cuda":
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            state = self.processor.set_image(working_img, state)
                            state["backbone_out"].update(cached_text)
                            if "geometric_prompt" not in state:
                                state["geometric_prompt"] = self.model._get_dummy_prompt()
                            state = self.processor._forward_grounding(state)
                    else:
                        state = self.processor.set_image(working_img, state)
                        state["backbone_out"].update(cached_text)
                        if "geometric_prompt" not in state:
                            state["geometric_prompt"] = self.model._get_dummy_prompt()
                        state = self.processor._forward_grounding(state)
                    
                # 3. Processamento de resultados já pré-filtrados
                raw_scores = state.get("scores")
                if raw_scores is not None:
                    scores_np = raw_scores.cpu().float().numpy() if hasattr(raw_scores, "cpu") else np.array(raw_scores)
                else:
                    scores_np = np.array([])
                    
                num_detected = len(scores_np)
                keep_indices = np.arange(num_detected)
                
                if num_detected == 0:
                    processed_img = working_img
                else:
                    raw_masks = state.get("masks")
                    raw_boxes = state.get("boxes")
                    raw_scores = state.get("scores")
                    
                    if isinstance(raw_masks, torch.Tensor):
                        filtered_masks = raw_masks[keep_indices]
                        filtered_boxes = raw_boxes[keep_indices]
                        filtered_scores = raw_scores[keep_indices]
                    else:
                        filtered_masks = [raw_masks[i] for i in keep_indices]
                        filtered_boxes = [raw_boxes[i] for i in keep_indices]
                        filtered_scores = [raw_scores[i] for i in keep_indices]
                        
                    processed_img = self.draw_results(working_img, filtered_masks, filtered_boxes, filtered_scores)
                
                # Exibição dinâmica na tela simulando o vídeo reproduzindo
                self.display_image(processed_img)
                self.after(0, lambda c=num_detected: self.lbl_counter.config(text=f"Objetos Mapeados no Frame Atual: {c}"))
                
                elapsed = time.time() - t0
                self.log(f"[IA SEGMENTER] [{idx+1}/{len(extracted_paths)}] Quadro {os.path.basename(path)}: Mapeados {num_detected} '{prompt}'(s) em {elapsed:.2f}s")
                
                self.after(0, lambda v=idx+1: self.progress_bar.configure(value=v))
                
                # OTIMIZAÇÃO: Limpeza periódica de cache CUDA a cada 10 frames para evitar overhead excessivo
                if (idx + 1) % 10 == 0:
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()
                
                # OTIMIZAÇÃO: Curtíssima pausa para dar respiro à GUI sem atrasar o loop
                time.sleep(0.002)
                
            self.log("\n=======================================================")
            self.log("[OK] PIPELINE COMPACTO FINALIZADO COM SUCESSO!")
            self.log("=======================================================")
            
            self.after(0, lambda: messagebox.showinfo("Simulação Concluída", 
                                "O pipeline de extração e segmentação sequencial de vídeo foi finalizado com sucesso!\n\n"
                                f"• Frames Segmentados: {len(extracted_paths)}\n"
                                f"• Pasta de Destino: {dest}\n\n"
                                "O simulador completou todas as etapas."))
                                
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.log(f"[ERRO PIPELINE] Falha no processamento: {e}\n{tb}")
        finally:
            self.is_running = False
            # OTIMIZAÇÃO: Limpeza final robusta para liberar toda a VRAM de volta para o sistema
            gc.collect()
            if "device" in locals() and device == "cuda":
                torch.cuda.empty_cache()
            self.after(0, lambda: self.btn_run.config(state="normal", text="⚡ Iniciar Extração e Segmentação Sequencial"))
            self.after(0, lambda: self.progress_bar.configure(value=0))

    def draw_results(self, original_img, masks, boxes, scores):
        # Converte a imagem PIL para um array numpy (BGR) para processamento ultra-rápido com OpenCV/NumPy
        frame = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
        h, w, _ = frame.shape
        
        overlay_mask = np.zeros_like(frame, dtype=np.uint8)
        overlay_mask_bool = np.zeros((h, w), dtype=bool)
        overlay_opaque = np.zeros_like(frame, dtype=np.uint8)
        
        if hasattr(masks, "cpu"):
            masks = masks.cpu().numpy()
        if hasattr(boxes, "cpu"):
            boxes = boxes.cpu().float().numpy()
        if hasattr(scores, "cpu"):
            scores = scores.cpu().float().numpy()
            
        num_detected = len(masks)
        for idx in range(num_detected):
            mask = masks[idx]
            if mask.ndim == 3:
                mask = mask[0]
                
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                
            rgb = COLORS[idx % len(COLORS)]
            # COLORS está em formato RGB, OpenCV utiliza BGR
            color_bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            
            overlay_mask[mask] = color_bgr
            overlay_mask_bool[mask] = True
            
            box = boxes[idx]
            x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w - 1, x1), min(h - 1, y1)
            
            # O usuário solicitou remover a marcação quadrada (boxes) e labels, mantendo apenas o sombreado de cores
            # cv2.rectangle(overlay_opaque, (x0, y0), (x1, y1), color_bgr, 2)
            
            # Escreve ID e confiança do objeto
            # label_text = f"ID:{idx+1} ({scores[idx]:.2f})"
            # cv2.putText(overlay_opaque, label_text, (x0 + 4, max(y0 - 6, 15)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                        
        # Mesclagem ultra-rápida (Alpha blending 60% original, 40% máscara)
        if num_detected > 0:
            frame[overlay_mask_bool] = (frame[overlay_mask_bool] * 0.6 + overlay_mask[overlay_mask_bool] * 0.4).astype(np.uint8)
            opaque_bool = np.any(overlay_opaque != 0, axis=-1)
            frame[opaque_bool] = overlay_opaque[opaque_bool]
            
        # Converte de volta para imagem PIL RGB
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    app = VideoTesteSegmenterApp()
    app.mainloop()
