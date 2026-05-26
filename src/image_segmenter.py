# -*- coding: utf-8 -*-
"""
SAM 3.1 - Mapeador e Contabilizador de Imagens Estáticas (Zero-Shot)
===================================================================
Este módulo implementa a interface gráfica para segmentação de imagens estáticas.
Roda em processo independente, livre de resíduos de memória do pipeline de vídeo.
"""

import os
import gc
import sys
import threading
import time
import random
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
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

# Adiciona caminhos do SAM 3
_sam3_path = os.environ.get("SAM3_PATH", "")
if _sam3_path:
    sys.path.append(_sam3_path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
except ImportError as e:
    SAM3_AVAILABLE = False
    SAM3_IMPORT_ERROR = str(e)

# Cores de alto contraste para exibição das máscaras
COLORS = np.array([
    [0.9, 0.1, 0.1],  # Vermelho
    [0.1, 0.8, 0.1],  # Verde
    [0.1, 0.5, 0.95], # Azul
    [0.95, 0.85, 0.1],# Amarelo
    [0.9, 0.1, 0.9],  # Magenta
    [0.1, 0.9, 0.9],  # Ciano
    [0.95, 0.5, 0.1], # Laranja
    [0.55, 0.1, 0.95],# Roxo
    [0.1, 0.95, 0.5]  # Lima
])

class SAM3ImageSegmenterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SAM 3.1 - Segmentação de Imagem & Contabilizador Premium (Zero-Shot)")
        self.geometry("1200x860")
        self.configure(bg="#1e1e1f")
        
        self.model = None
        self.processor = None
        self.image_path = None
        self.current_image = None
        self.processed_image = None
        
        # Diretório padrão
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.default_image_dir = os.path.join(self.project_root, "media", "images")
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
        
        # Carga em background do modelo
        if SAM3_AVAILABLE:
            self.log("[STATUS] Carregando pesos do SAM 3.1 Image Model...")
            threading.Thread(target=self.load_sam3_image_model, daemon=True).start()
        else:
            self.log(f"[ERRO CRÍTICO] Falha ao importar SAM 3.1: {SAM3_IMPORT_ERROR}")
            messagebox.showerror("Erro de Importação", f"Erro: {SAM3_IMPORT_ERROR}")

    def create_widgets(self):
        ctrl_frame = tk.Frame(self, bg="#252526", pady=10, padx=15, highlightbackground="#3c3c3c", highlightthickness=1)
        ctrl_frame.pack(fill="x", side="top", padx=10, pady=10)
        
        # Linha 1: Seleção de Imagem
        tk.Label(ctrl_frame, text="Pasta de Imagens:", bg="#252526", fg="#ffffff", font=("Helvetica", 10, "bold")).grid(row=0, column=0, sticky="w", pady=5)
        self.entry_folder_path = ttk.Entry(ctrl_frame, width=45)
        self.entry_folder_path.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.entry_folder_path.insert(0, self.default_image_dir)
        
        self.btn_select_folder = ttk.Button(ctrl_frame, text="Selecionar Pasta...", command=self.select_folder)
        self.btn_select_folder.grid(row=0, column=2, padx=5, pady=5)
        
        tk.Label(ctrl_frame, text="Imagem na Pasta:", bg="#252526", fg="#ffffff", font=("Helvetica", 10, "bold")).grid(row=0, column=3, sticky="w", pady=5, padx=(15, 0))
        self.combo_images = ttk.Combobox(ctrl_frame, width=25, state="readonly")
        self.combo_images.grid(row=0, column=4, padx=5, pady=5)
        self.combo_images.bind("<<ComboboxSelected>>", self.on_image_selected)
        
        self.btn_browse = ttk.Button(ctrl_frame, text="Procurar...", command=self.browse_single_file)
        self.btn_browse.grid(row=0, column=5, padx=5, pady=5)
        
        # Linha 2: Prompt e Confiança
        tk.Label(ctrl_frame, text="Prompt (Texto):", bg="#252526", fg="#ffffff", font=("Helvetica", 10, "bold")).grid(row=1, column=0, sticky="w", pady=5)
        self.entry_prompt = ttk.Entry(ctrl_frame, width=45)
        self.entry_prompt.insert(0, "person")
        self.entry_prompt.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        tk.Label(ctrl_frame, text="Limiar Confiança:", bg="#252526", fg="#ffffff", font=("Helvetica", 10, "bold")).grid(row=1, column=2, sticky="w", pady=5, padx=5)
        self.val_confidence = tk.DoubleVar(value=0.45)
        self.slider_conf = ttk.Scale(ctrl_frame, from_=0.1, to=0.9, variable=self.val_confidence, orient="horizontal", command=self.update_conf_label)
        self.slider_conf.grid(row=1, column=3, sticky="ew", padx=5, pady=5)
        self.lbl_conf = tk.Label(ctrl_frame, text="0.45", bg="#252526", fg="#a0a0a0", font=("Helvetica", 9, "bold"))
        self.lbl_conf.grid(row=1, column=4, sticky="w", padx=5, pady=5)
        
        # Linha 3: Resolução (Otimização RAM)
        tk.Label(ctrl_frame, text="Resolução Alvo:", bg="#252526", fg="#00aaff", font=("Helvetica", 10, "bold")).grid(row=2, column=0, sticky="w", pady=5)
        self.combo_resolution = ttk.Combobox(ctrl_frame, values=["Original", "720p (HD)", "480p (SD)", "360p (Baixa)"], width=20, state="readonly")
        self.combo_resolution.set("480p (SD)")
        self.combo_resolution.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        tk.Label(ctrl_frame, text="(Previne estouros OOM em imagens de altíssima resolução)", bg="#252526", fg="#8e8e93", font=("Helvetica", 9, "italic")).grid(row=2, column=2, columnspan=4, sticky="w", padx=5, pady=5)
        
        # Linha 4: Ações
        self.btn_run = tk.Button(ctrl_frame, text="⚡ Mapear Objeto na Imagem", bg="#007acc", fg="#ffffff", font=("Helvetica", 11, "bold"), activebackground="#005999", activeforeground="#ffffff", relief="flat", state="disabled", command=self.start_image_processing)
        self.btn_run.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        
        self.btn_benchmark = tk.Button(ctrl_frame, text="📊 Executar Benchmark de Pasta (Multi-Image)", bg="#28a745", fg="#ffffff", font=("Helvetica", 11, "bold"), activebackground="#218838", activeforeground="#ffffff", relief="flat", state="disabled", command=self.start_folder_benchmark)
        self.btn_benchmark.grid(row=3, column=3, columnspan=3, sticky="ew", pady=(10, 0), padx=(10, 0))
        
        # Painel Central: Canvas
        viewer_container = tk.Frame(self, bg="#1e1e1f")
        viewer_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        viewer_frame = ttk.LabelFrame(viewer_container, text="Painel Visualizador de Imagem", padding=5)
        viewer_frame.pack(fill="both", expand=True)
        
        self.canvas_image = tk.Canvas(viewer_frame, bg="#151515", highlightthickness=0)
        self.canvas_image.pack(fill="both", expand=True)
        
        # Overlay do Contador
        self.lbl_counter = tk.Label(self, text="Objetos Encontrados: 0", bg="#1e1e1f", fg="#00ff00", font=("Helvetica", 14, "bold"), anchor="w")
        self.lbl_counter.pack(fill="x", padx=15, pady=5)
        
        # Barra de progresso
        self.progress_bar = ttk.Progressbar(self, orient="horizontal", mode="indeterminate")
        self.progress_bar.pack(fill="x", padx=15, pady=5)
        
        # Console inferior
        self.log_frame = ttk.LabelFrame(self, text="Display de Etapas e Benchmark", padding=5)
        self.log_frame.pack(fill="x", side="bottom", padx=10, pady=(0, 10))
        
        self.text_log = tk.Text(self.log_frame, wrap="word", height=6, bg="#101010", fg="#00ff66", insertbackground="white", state="disabled", font=("Courier", 9))
        self.text_log.pack(fill="both", expand=True)
        
        self.scan_images_in_folder()

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
        folder = filedialog.askdirectory(title="Selecionar Pasta de Imagens")
        if folder:
            self.entry_folder_path.delete(0, "end")
            self.entry_folder_path.insert(0, folder)
            self.scan_images_in_folder()

    def scan_images_in_folder(self):
        folder = self.entry_folder_path.get().strip()
        if not os.path.isdir(folder):
            self.log(f"[AVISO] Pasta não encontrada: {folder}")
            return
            
        exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.PNG", "*.JPG", "*.JPEG", "*.BMP", "*.WEBP"]
        img_files = []
        for ext in exts:
            img_files.extend(glob.glob(os.path.join(folder, ext)))
            
        img_names = [os.path.basename(f) for f in img_files]
        self.combo_images.config(values=img_names)
        
        if img_names:
            self.combo_images.current(0)
            self.on_image_selected(None)
            self.log(f"[OK] {len(img_names)} imagens localizadas na pasta.")
        else:
            self.combo_images.set("")
            self.log("[AVISO] Nenhuma imagem encontrada nesta pasta.")

    def on_image_selected(self, event):
        folder = self.entry_folder_path.get().strip()
        name = self.combo_images.get()
        if name:
            self.image_path = os.path.join(folder, name)
            self.log(f"[OK] Selecionada imagem: {name}")
            self.load_image_metadata()

    def browse_single_file(self):
        file = filedialog.askopenfilename(
            title="Selecionar Imagem Estática",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if file:
            self.image_path = file
            self.combo_images.set(os.path.basename(file))
            self.log(f"[OK] Imagem selecionada manualmente: {file}")
            self.load_image_metadata()

    def load_image_metadata(self):
        if not self.image_path or not os.path.exists(self.image_path):
            return
        try:
            self.current_image = Image.open(self.image_path).convert("RGB")
            self.processed_image = None
            self.lbl_counter.config(text="Objetos Encontrados: 0")
            self.log(f"[STATUS] Resolução original: {self.current_image.width}x{self.current_image.height}")
            self.display_image(self.current_image)
        except Exception as e:
            self.log(f"[ERRO] Falha ao carregar imagem: {e}")

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
            self.log(f"[STATUS] Inicializando SAM 3.1 no dispositivo '{device.upper()}'...")
            
            self.model = build_sam3_image_model(compile=False, load_from_HF=True, device=device)
            self.processor = Sam3Processor(self.model, device=device)
            
            self.log("[OK] SAM 3.1 Image Predictor carregado com sucesso!")
            self.after(0, lambda: self.btn_run.config(state="normal"))
            self.after(0, lambda: self.btn_benchmark.config(state="normal"))
        except Exception as e:
            self.log(f"[ERRO] Falha ao carregar o SAM 3.1: {e}")

    def start_image_processing(self):
        if not self.image_path or not os.path.exists(self.image_path):
            return
        prompt = self.entry_prompt.get().strip()
        if not prompt:
            return
            
        self.btn_run.config(state="disabled", text="Processando...")
        self.progress_bar.start(10)
        
        threading.Thread(target=self._process_image_thread, args=(prompt,), daemon=True).start()

    def _process_image_thread(self, prompt):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            start_time = time.time()
            
            # Carrega imagem original
            orig_img = Image.open(self.image_path).convert("RGB")
            
            # Redimensionamento
            target_res = self.combo_resolution.get()
            w, h = orig_img.size
            working_img = orig_img
            
            if target_res != "Original":
                target_h = 480
                if "720p" in target_res:
                    target_h = 720
                elif "360p" in target_res:
                    target_h = 360
                
                target_w = int(w * (target_h / h))
                target_w = (target_w // 2) * 2
                target_h = (target_h // 2) * 2
                self.log(f"[STATUS] Ajustando resolução de inferência para {target_w}x{target_h}...")
                working_img = orig_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                
            user_thresh = self.val_confidence.get()
            state = {}
            self.processor.set_confidence_threshold(user_thresh, state)
            
            with torch.inference_mode():
                if device == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        state = self.processor.set_image(working_img, state)
                        state = self.processor.set_text_prompt(prompt, state)
                else:
                    state = self.processor.set_image(working_img, state)
                    state = self.processor.set_text_prompt(prompt, state)
                
            elapsed = time.time() - start_time
            self.log(f"[OK] Segmentação concluída em {elapsed:.2f}s!")
            
            # Detecções
            raw_scores = state.get("scores")
            if raw_scores is not None:
                scores_np = raw_scores.cpu().float().numpy() if hasattr(raw_scores, "cpu") else np.array(raw_scores)
            else:
                scores_np = np.array([])
                
            num_detected = len(scores_np)
            keep_indices = np.arange(num_detected)
            self.log(f"[RESULTADO] Encontrados {num_detected} objetos '{prompt}' (conf > {user_thresh:.2f}).")
            self.lbl_counter.config(text=f"Objetos Encontrados: {num_detected}")
            
            if num_detected == 0:
                self.processed_image = working_img
                self.display_image(self.processed_image)
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
                    
                self.processed_image = self.draw_results(working_img, filtered_masks, filtered_boxes, filtered_scores)
                self.display_image(self.processed_image)
                
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            self.log(f"[ERRO] Ocorreu uma falha no processamento: {e}")
        finally:
            self.progress_bar.stop()
            self.after(0, lambda: self.btn_run.config(state="normal", text="⚡ Mapear Objeto na Imagem"))

    def start_folder_benchmark(self):
        folder = self.entry_folder_path.get().strip()
        if not os.path.isdir(folder):
            messagebox.showwarning("Aviso", "Por favor, selecione uma pasta de imagens válida primeiro.")
            return
            
        prompt = self.entry_prompt.get().strip()
        if not prompt:
            return
            
        self.btn_run.config(state="disabled")
        self.btn_benchmark.config(state="disabled", text="Rodando Benchmark...")
        self.progress_bar.start(10)
        
        threading.Thread(target=self._process_folder_benchmark, args=(prompt, folder), daemon=True).start()

    def _process_folder_benchmark(self, prompt, folder):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.PNG", "*.JPG", "*.JPEG", "*.BMP", "*.WEBP"]
            img_files = []
            for ext in exts:
                img_files.extend(glob.glob(os.path.join(folder, ext)))
                
            if not img_files:
                self.log("[AVISO] Nenhuma imagem encontrada na pasta para o benchmark.")
                return
                
            img_files = sorted(img_files)
            num_imgs = len(img_files)
            
            self.log("\n=======================================================")
            self.log(f"[BENCHMARK] INICIANDO BENCHMARK SEQUENCIAL DA PASTA...")
            self.log(f"[BENCHMARK] Imagens Encontradas: {num_imgs}")
            self.log(f"[BENCHMARK] Prompt: '{prompt}' | Dispositivo: {device.upper()}")
            self.log("=======================================================")
            
            times = []
            total_objs = 0
            benchmark_results = []
            
            target_res = self.combo_resolution.get()
            user_thresh = self.val_confidence.get()
            
            context = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.inference_mode()
            
            for idx, path in enumerate(img_files):
                img_name = os.path.basename(path)
                
                # 1. Carrega imagem
                orig_img = Image.open(path).convert("RGB")
                w, h = orig_img.size
                working_img = orig_img
                
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
                # 2. Executa inferência e cronometra (com filtragem precoce no threshold do usuário)
                state = {}
                self.processor.set_confidence_threshold(user_thresh, state)
                
                t0 = time.time()
                with torch.inference_mode():
                    if device == "cuda":
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            state = self.processor.set_image(working_img, state)
                            state = self.processor.set_text_prompt(prompt, state)
                    else:
                        state = self.processor.set_image(working_img, state)
                        state = self.processor.set_text_prompt(prompt, state)
                t1 = time.time()
                
                taken = t1 - t0
                times.append(taken)
                
                # 3. Processamento de resultados já pré-filtrados
                raw_scores = state.get("scores")
                if raw_scores is not None:
                    scores_np = raw_scores.cpu().float().numpy() if hasattr(raw_scores, "cpu") else np.array(raw_scores)
                else:
                    scores_np = np.array([])
                    
                num_detected = len(scores_np)
                keep_indices = np.arange(num_detected)
                total_objs += num_detected
                benchmark_results.append((img_name, num_detected, taken))
                
                # Atualiza display
                self.lbl_counter.config(text=f"Objetos Encontrados: {num_detected}")
                if num_detected == 0:
                    self.processed_image = working_img
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
                        
                    self.processed_image = self.draw_results(working_img, filtered_masks, filtered_boxes, filtered_scores)
                
                self.display_image(self.processed_image)
                self.log(f"[OK] [{idx+1}/{num_imgs}] {img_name} processada em {taken:.3f}s. Encontrados: {num_detected}.")
                
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                    
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            
            # O usuário solicitou não gerar relatórios físicos para máxima velocidade de execução
            self.log("\n--- DETALHAMENTO DE MAPEAMENTO POR IMAGEM (BENCHMARK) ---")
            for name, count, _ in benchmark_results:
                self.log(f"• {name}: {count} '{prompt}'(s) mapeados")
            self.log("---------------------------------------------------------")
            
            self.log("\n=======================================================")
            self.log("[OK] BENCHMARK SEQUENCIAL COMPLETO!")
            self.log(f"[BENCHMARK] Tempo Total: {total_time:.3f}s")
            self.log(f"[BENCHMARK] Tempo Médio: {avg_time:.3f}s por imagem")
            self.log(f"[BENCHMARK] Total de Objetos Mapeados: {total_objs}")
            self.log("=======================================================")
            
            self.after(0, lambda: self.show_benchmark_results_window(prompt, folder, num_imgs, total_time, avg_time, total_objs, benchmark_results))
                                
        except Exception as e:
            self.log(f"[ERRO] Ocorreu uma falha no benchmark: {e}")
        finally:
            self.progress_bar.stop()
            self.after(0, lambda: self.btn_run.config(state="normal"))
            self.after(0, lambda: self.btn_benchmark.config(state="normal", text="📊 Executar Benchmark de Pasta (Multi-Image)"))

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

    def show_benchmark_results_window(self, prompt, folder, num_imgs, total_time, avg_time, total_objs, benchmark_results):
        # Cria uma janela Toplevel estilizada
        report_win = tk.Toplevel(self)
        report_win.title("📊 Relatório Detalhado de Benchmark - SAM 3.1")
        report_win.geometry("780x620")
        report_win.configure(bg="#1e1e1f")
        report_win.transient(self) # Janela filha
        report_win.grab_set() # Foco exclusivo
        
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
        
        tk.Label(title_frame, text="RESULTADO DO BENCHMARK DE PASTA (DETALHADO)", bg="#252526", fg="#ffaa00", font=("Helvetica", 12, "bold")).pack(anchor="w")
        tk.Label(title_frame, text=f"Filtro/Prompt: '{prompt}' | Limiar de Confiança: {self.val_confidence.get():.2f}", bg="#252526", fg="#a0a0a0", font=("Helvetica", 9, "italic")).pack(anchor="w")
        
        # Painel de Métricas Rápidas
        metrics_frame = tk.Frame(report_win, bg="#1e1e1f")
        metrics_frame.pack(fill="x", padx=10, pady=5)
        
        # Grid para 4 métricas com visual moderno
        def create_metric_card(parent, title, val, col, color="#00ff66"):
            card = tk.Frame(parent, bg="#252526", padx=10, pady=8, highlightbackground="#3c3c3c", highlightthickness=1)
            card.grid(row=0, column=col, sticky="nsew", padx=4, pady=2)
            tk.Label(card, text=title, bg="#252526", fg="#8e8e93", font=("Helvetica", 8, "bold")).pack()
            tk.Label(card, text=val, bg="#252526", fg=color, font=("Helvetica", 11, "bold")).pack(pady=(2, 0))
            parent.columnconfigure(col, weight=1)
            
        create_metric_card(metrics_frame, "Imagens Processadas", f"{num_imgs}", 0, "#ffffff")
        create_metric_card(metrics_frame, "Tempo Total", f"{total_time:.2f}s", 1, "#ffcc00")
        create_metric_card(metrics_frame, "Média por Imagem", f"{avg_time:.3f}s", 2, "#00aaff")
        create_metric_card(metrics_frame, "Total Encontrado", f"{total_objs} '{prompt}'", 3, "#00ff66")
        
        # Treeview / Tabela centralizada
        table_frame = tk.Frame(report_win, bg="#1e1e1f")
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Customizar Treeview para Dark Mode
        style = ttk.Style()
        style.configure("Treeview", background="#252526", foreground="#ffffff", fieldbackground="#252526", rowheight=24, font=("Helvetica", 9))
        style.map("Treeview", background=[("selected", "#007acc")], foreground=[("selected", "#ffffff")])
        style.configure("Treeview.Heading", background="#3c3c3d", foreground="#ffffff", font=("Helvetica", 9, "bold"))
        style.map("Treeview.Heading", background=[("active", "#4d4d4f")])
        
        # Configurar colunas
        columns = ("frame", "mapped", "time")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", style="Treeview")
        tree.heading("frame", text="Nome da Imagem (Frame)")
        tree.heading("mapped", text=f"Qtd de '{prompt}' Mapeados")
        tree.heading("time", text="Tempo de Inferência (s)")
        
        tree.column("frame", anchor="w", width=350)
        tree.column("mapped", anchor="center", width=180)
        tree.column("time", anchor="center", width=150)
        
        # Scrollbars
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Inserir dados na Treeview
        for name, count, t_taken in benchmark_results:
            tree.insert("", "end", values=(name, f"{count} mapeados", f"{t_taken:.3f}s"))
            
        # Botões de Ação Inferiores
        btn_frame = tk.Frame(report_win, bg="#1e1e1f", pady=10)
        btn_frame.pack(fill="x", side="bottom", padx=10)
        
        def open_folder():
            os.startfile(folder)
                
        btn_open_folder = ttk.Button(btn_frame, text="📂 Abrir Pasta de Destino", command=open_folder)
        btn_open_folder.pack(side="left", padx=5)
        
        btn_close = tk.Button(btn_frame, text="Fechar Relatório", bg="#dc3545", fg="#ffffff", relief="flat", font=("Helvetica", 9, "bold"), activebackground="#bd2130", activeforeground="#ffffff", padx=15, command=report_win.destroy)
        btn_close.pack(side="right", padx=5)

if __name__ == "__main__":
    app = SAM3ImageSegmenterApp()
    app.mainloop()
