# -*- coding: utf-8 -*-
"""
Gerenciador de banco de dados em JSON e exportação de dados para o Dashboard HTML.
"""
import os
import json

DB_PATH = "relatorio_contagem.json"
JS_EXPORT_DIR = os.path.join("media")
JS_EXPORT_PATH = os.path.join(JS_EXPORT_DIR, "dados_dashboard.js")

DEFAULT_DB = {
    "settings": {
        "capacity_seats": 5,
        "overcapacity_threshold": 5,
        "absolute_max": 40,
        "occupancy_start_pct": 30
    },
    "history": []
}

def load_db():
    """Carrega as configurações e o histórico do arquivo JSON."""
    if not os.path.exists(DB_PATH):
        return DEFAULT_DB.copy()
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Garantir chaves padrão se estiverem faltando
            if "settings" not in data:
                data["settings"] = DEFAULT_DB["settings"].copy()
            else:
                # Preencher chaves individuais se faltantes
                for k, v in DEFAULT_DB["settings"].items():
                    if k not in data["settings"]:
                        data["settings"][k] = v
            if "history" not in data:
                data["history"] = []
            return data
    except Exception:
        return DEFAULT_DB.copy()

def save_db(settings, history):
    """
    Grava as configurações e histórico em JSON e
    exporta em JS com cache-busting para leitura em tempo real no HTML local.
    """
    db_data = {
        "settings": settings,
        "history": history
    }
    
    # 1. Salvar arquivo JSON local
    try:
        with open(DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db_data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[ERRO JSON] Falha ao gravar JSON: {e}")
        
    # 2. Exportar arquivo JS para contornar restrições de CORS no protocolo file://
    try:
        os.makedirs(JS_EXPORT_DIR, exist_ok=True)
        js_content = f"window.METRO_DATA = {json.dumps(db_data, ensure_ascii=False, indent=2)};"
        with open(JS_EXPORT_PATH, "w", encoding="utf-8") as f:
            f.write(js_content)
    except Exception as e:
        print(f"[ERRO JS] Falha ao exportar JS: {e}")
