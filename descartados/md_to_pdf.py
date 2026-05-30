# -*- coding: utf-8 -*-
"""
Script utilitário para converter codeexplain.md em PDF via HTML intermediário.
Usa a biblioteca 'markdown' para renderizar o MD e gera um HTML estilizado
que pode ser impresso diretamente como PDF pelo navegador.
"""
import os
import sys
import markdown
import webbrowser

INPUT_MD = "codeexplain.md"
OUTPUT_HTML = "codeexplain_print.html"

# CSS premium para impressão em PDF
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #ffffff;
    --text: #1a1a2e;
    --text-secondary: #4a4a6a;
    --accent: #0066cc;
    --accent-light: #e8f0fe;
    --border: #e0e0e8;
    --code-bg: #f5f5fa;
    --table-header: #1a1a2e;
    --table-stripe: #f8f8fc;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 11pt;
    line-height: 1.7;
    color: var(--text);
    background: var(--bg);
    max-width: 210mm;
    margin: 0 auto;
    padding: 25mm 20mm;
}

h1 {
    font-size: 22pt;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 4px;
    padding-bottom: 12px;
    border-bottom: 3px solid var(--accent);
    letter-spacing: -0.5px;
    page-break-after: avoid;
}

h2 {
    font-size: 15pt;
    font-weight: 700;
    color: var(--accent);
    margin-top: 28px;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
    page-break-after: avoid;
}

h3 {
    font-size: 12pt;
    font-weight: 600;
    color: var(--text-secondary);
    margin-top: 20px;
    margin-bottom: 8px;
    page-break-after: avoid;
}

p {
    margin-bottom: 10px;
    text-align: justify;
}

strong {
    font-weight: 600;
    color: var(--text);
}

em {
    font-style: italic;
    color: var(--text-secondary);
}

hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 24px 0;
}

/* Code blocks */
pre {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 6px;
    padding: 14px 16px;
    margin: 14px 0;
    overflow-x: auto;
    font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
    font-size: 9pt;
    line-height: 1.55;
    page-break-inside: avoid;
}

code {
    font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
    font-size: 9pt;
    background: var(--code-bg);
    padding: 2px 5px;
    border-radius: 3px;
    border: 1px solid var(--border);
    color: #c7254e;
}

pre code {
    background: none;
    padding: 0;
    border: none;
    color: var(--text);
    font-size: 9pt;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}

thead th {
    background: var(--table-header);
    color: #ffffff;
    font-weight: 600;
    padding: 10px 12px;
    text-align: left;
    font-size: 9pt;
    letter-spacing: 0.3px;
}

tbody td {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
}

tbody tr:nth-child(even) {
    background: var(--table-stripe);
}

tbody tr:hover {
    background: var(--accent-light);
}

/* Lists */
ul, ol {
    margin: 8px 0 12px 24px;
}

li {
    margin-bottom: 4px;
}

/* Blockquotes */
blockquote {
    border-left: 4px solid var(--accent);
    background: var(--accent-light);
    padding: 12px 16px;
    margin: 14px 0;
    border-radius: 0 6px 6px 0;
    font-style: italic;
    color: var(--text-secondary);
}

/* Print-specific */
@media print {
    body {
        padding: 15mm;
        max-width: 100%;
    }
    
    h1, h2, h3 {
        page-break-after: avoid;
    }
    
    pre, table, blockquote {
        page-break-inside: avoid;
    }
    
    .no-print {
        display: none;
    }
}

/* Print instruction banner */
.print-banner {
    background: linear-gradient(135deg, #0066cc 0%, #004499 100%);
    color: white;
    padding: 16px 24px;
    border-radius: 10px;
    margin-bottom: 30px;
    text-align: center;
    font-size: 13pt;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(0,102,204,0.3);
}

.print-banner small {
    display: block;
    font-size: 9pt;
    font-weight: 400;
    opacity: 0.85;
    margin-top: 4px;
}
"""

def convert():
    if not os.path.exists(INPUT_MD):
        print(f"[ERRO] Arquivo '{INPUT_MD}' não encontrado.")
        sys.exit(1)

    with open(INPUT_MD, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Converter MD para HTML com extensões de tabela e code highlighting
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "codehilite", "toc"],
        extension_configs={
            "codehilite": {"linenums": False, "css_class": "highlight"}
        }
    )

    full_html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MetroVision IA — Explicação Técnica do Código</title>
    <style>{CSS}</style>
</head>
<body>
    <div class="print-banner no-print">
        📄 Pressione <strong>Ctrl+P</strong> para salvar como PDF
        <small>Selecione "Salvar como PDF" na impressora e clique em Salvar</small>
    </div>
    {html_body}
</body>
</html>
"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"[OK] HTML gerado em: {os.path.abspath(OUTPUT_HTML)}")
    print("[INFO] Abrindo no navegador para impressão como PDF...")

    # Abrir no navegador para que o usuário salve como PDF via Ctrl+P
    webbrowser.open(os.path.abspath(OUTPUT_HTML))

if __name__ == "__main__":
    convert()
