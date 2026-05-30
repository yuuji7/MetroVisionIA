# -*- coding: utf-8 -*-
"""
SAM 3.1 - Conversor de Vídeo em Imagens Estáticas (Extrator Uniforme)
====================================================================
Este utilitário permite extrair quadros uniformemente espaçados de um vídeo local,
salvando-os diretamente na pasta padrão do projeto para testes no Image Segmenter.
"""

import os
import sys
import glob
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2

class VideoToImageConverterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SAM 3.1 - Conversor de Vídeo para Imagens (ProjetoContador)")
        self.geometry("960x720")
        self.configure(bg="#1e1e1f")
        
        self.video_path = None
        self.fps = 24.0
        self.total_frames = 0
        self.duration_seconds = 0.0
        
        # Estrutura do projeto
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.default_video_dir = os.path.join(self.project_root, "media", "videos")
        self.default_image_dir = os.path.join(self.project_root, "media", "images")
        
        os.makedirs(self.default_video_dir, exist_ok=True)
        os.makedirs(self.default_image_dir, exist_ok=True)
        
        # Configurar Estilo Dark Mode Premium
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

    def create_widgets(self):
        ctrl_frame = tk.Frame(self, bg="#252526", pady=15, padx=15, highlightbackground="#3c3c3c", highlightthickness=1)
        ctrl_frame.pack(fill="x", side="top", padx=10, pady=10)
        
        # Título
        tk.Label(ctrl_frame, text="EXTRATOR DE FRAMES UNIFORMES - PROJETOCONTADOR", bg="#252526", fg="#ffaa00", font=("Helvetica", 11, "bold")).grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 10))
        
        # Linha 1: Pasta de Origem
        tk.Label(ctrl_frame, text="Pasta de Vídeos:", bg="#252526", fg="#ffffff", font=("Helvetica", 10, "bold")).grid(row=1, column=0, sticky="w", pady=5)
        self.entry_folder_path = ttk.Entry(ctrl_frame, width=48)
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
        
        ttk.Separator(ctrl_frame, orient="horizontal").grid(row=3, column=0, columnspan=5, sticky="ew", pady=10)
        
        # Parâmetros
        tk.Label(ctrl_frame, text="Tempo Inicial (s):", bg="#252526", fg="#ffffff", font=("Helvetica", 10, "bold")).grid(row=4, column=0, sticky="w", pady=5)
        self.entry_start_time = ttk.Entry(ctrl_frame, width=15)
        self.entry_start_time.insert(0, "0.0")
        self.entry_start_time.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        
        tk.Label(ctrl_frame, text="Tempo Final (s):", bg="#252526", fg="#ffffff", font=("Helvetica", 10, "bold")).grid(row=4, column=2, sticky="w", pady=5, padx=(15, 0))
        self.entry_end_time = ttk.Entry(ctrl_frame, width=15)
        self.entry_end_time.grid(row=4, column=3, padx=5, pady=5, sticky="w")
        
        # Quantidade de Imagens
        tk.Label(ctrl_frame, text="Qtd. de Imagens:", bg="#252526", fg="#ffffff", font=("Helvetica", 10, "bold")).grid(row=5, column=0, sticky="w", pady=5)
        self.entry_quantity = ttk.Entry(ctrl_frame, width=15)
        self.entry_quantity.insert(0, "10")
        self.entry_quantity.grid(row=5, column=1, padx=5, pady=5, sticky="w")
        
        # Pasta de Destino
        tk.Label(ctrl_frame, text="Pasta Destino:", bg="#252526", fg="#ffffff", font=("Helvetica", 10, "bold")).grid(row=6, column=0, sticky="w", pady=5)
        self.entry_dest_folder = ttk.Entry(ctrl_frame, width=48)
        self.entry_dest_folder.grid(row=6, column=1, columnspan=3, padx=5, pady=5, sticky="ew")
        
        self.btn_select_dest = ttk.Button(ctrl_frame, text="Destino...", command=self.select_dest_folder)
        self.btn_select_dest.grid(row=6, column=4, padx=5, pady=5)
        
        # Botão Principal
        self.btn_run = tk.Button(ctrl_frame, text="Iniciar Extração Uniforme de Imagens", bg="#007acc", fg="#ffffff", font=("Helvetica", 11, "bold"), activebackground="#005999", activeforeground="#ffffff", relief="flat", command=self.start_extraction)
        self.btn_run.grid(row=7, column=0, columnspan=5, sticky="ew", pady=(15, 0))
        
        # Console de status
        steps_container = tk.Frame(self, bg="#1e1e1f")
        steps_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        steps_frame = ttk.LabelFrame(steps_container, text="Etapas de Execução do Processo", padding=5)
        steps_frame.pack(fill="both", expand=True)
        
        self.text_log = tk.Text(steps_frame, wrap="word", bg="#101010", fg="#00ff66", insertbackground="white", font=("Courier", 10))
        self.text_log.pack(fill="both", expand=True)
        
        self.progress_bar = ttk.Progressbar(self, orient="horizontal", mode="determinate")
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 10))

    def log(self, msg):
        self.text_log.insert("end", f"{msg}\n")
        self.text_log.see("end")
        self.update_idletasks()

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
            
        extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(folder, ext)))
            
        names = [os.path.basename(f) for f in files]
        self.combo_videos.config(values=names)
        
        if names:
            self.combo_videos.current(0)
            self.on_video_selected(None)
            self.log(f"[OK] {len(names)} vídeos identificados na pasta de origem.")
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
        suggested = os.path.join(self.default_image_dir, f"{video_name}_frames")
        self.entry_dest_folder.delete(0, "end")
        self.entry_dest_folder.insert(0, suggested)
        
        self.log(f"[STATUS] Metadados carregados para {os.path.basename(self.video_path)}")

    def start_extraction(self):
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showwarning("Aviso", "Por favor, selecione um vídeo válido primeiro.")
            return
            
        dest = self.entry_dest_folder.get().strip()
        if not dest:
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
            
        self.btn_run.config(state="disabled", text="Extraindo Quadros...")
        self.progress_bar["value"] = 0
        self.progress_bar["maximum"] = qty
        
        threading.Thread(target=self._extraction_thread, args=(start, end, qty, dest), daemon=True).start()

    def _extraction_thread(self, start, end, qty, dest):
        try:
            self.log("\n=======================================================")
            self.log("[STATUS] INICIANDO PREPARAÇÃO DA EXTRAÇÃO UNIFORME...")
            self.log(f"[STATUS] Origem: {self.video_path}")
            self.log(f"[STATUS] Intervalo: {start:.2f}s até {end:.2f}s | Qtd: {qty}")
            
            os.makedirs(dest, exist_ok=True)
            
            start_frame = int(start * self.fps)
            end_frame = int(end * self.fps)
            frame_range = end_frame - start_frame
            
            if qty == 1:
                indices = [start_frame]
            else:
                indices = [int(start_frame + i * (frame_range - 1) / (qty - 1)) for i in range(qty)]
                
            self.log(f"[STATUS] Frame range: {start_frame} até {end_frame} (Intervalos calculados: {indices})")
            
            cap = cv2.VideoCapture(self.video_path)
            success = 0
            
            for idx, f_idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                
                if ret:
                    f_name = f"frame_{idx+1:04d}_f{f_idx}.png"
                    cv2.imwrite(os.path.join(dest, f_name), frame)
                    success += 1
                    self.log(f"[SUCESSO] [{success}/{qty}] Quadro {f_idx} extraído e salvo como: {f_name}")
                else:
                    self.log(f"[ERRO] Falha ao capturar quadro {f_idx}")
                    
                self.progress_bar["value"] = idx + 1
                self.update_idletasks()
                
            cap.release()
            self.log(f"\n[OK] EXTRAÇÃO CONCLUÍDA! {success} quadros salvos em:")
            self.log(f" => {dest}")
            self.log("=======================================================")
            
            messagebox.showinfo("Extração Concluída", f"{success} imagens salvas com sucesso em:\n{dest}")
        except Exception as e:
            self.log(f"[ERRO CRÍTICO] Falha no pipeline de extração: {e}")
        finally:
            self.btn_run.config(state="normal", text="Iniciar Extração Uniforme de Imagens")

if __name__ == "__main__":
    app = VideoToImageConverterApp()
    app.mainloop()
