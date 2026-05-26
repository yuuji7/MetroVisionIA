// ============================================================
// APRESENTAÇÃO — Projeto Contador: Visão Computacional em Tempo Real
// Geração automatizada de PowerPoint via PptxGenJS (Node.js)
// Estrutura: Princípio da Pirâmide de Barbara Minto
// ============================================================
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();

// ────────────────── CONFIGURAÇÕES GLOBAIS ──────────────────
pres.author = "Equipe Projeto Contador";
pres.company = "MetroVision AI";
pres.subject = "Visão Computacional Aplicada a Transporte Urbano";
pres.title = "Projeto Contador — Câmeras Inteligentes para Distribuição de Passageiros no Metrô";
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 polegadas

// ────────────────── PALETA DE CORES ──────────────────
const COLORS = {
  darkBg:       "0F1724",  // fundo escuro azulado
  cardBg:       "1A2744",  // cards sobre fundo
  accent:       "00B4D8",  // azul ciano vibrante
  accentDark:   "0077B6",  // azul profundo
  accentGreen:  "00E676",  // verde neon para resultados
  accentOrange: "FF9100",  // laranja para destaques
  accentRed:    "FF5252",  // vermelho para problemas
  white:        "FFFFFF",
  lightGray:    "B0BEC5",
  midGray:      "607D8B",
  gold:         "FFD600",
};

// ────────────────── FUNÇÕES AUXILIARES ──────────────────

/** Adiciona barra inferior de marca e número de página */
function addFooter(slide, pageNum, totalPages) {
  // Barra inferior
  slide.addShape(pres.ShapeType.rect, {
    x: 0, y: 6.9, w: 13.33, h: 0.6,
    fill: { color: COLORS.cardBg },
  });
  slide.addText("MetroVision AI  •  Projeto Contador  •  Visão Computacional & IA", {
    x: 0.4, y: 6.95, w: 9, h: 0.4,
    fontSize: 9, color: COLORS.midGray, fontFace: "Segoe UI",
  });
  slide.addText(`${pageNum} / ${totalPages}`, {
    x: 11.5, y: 6.95, w: 1.5, h: 0.4,
    fontSize: 9, color: COLORS.midGray, fontFace: "Segoe UI", align: "right",
  });
}

/** Adiciona fundo escuro gradiente padrão */
function darkBg(slide) {
  slide.background = { fill: COLORS.darkBg };
}

/** Linha decorativa de acento */
function accentLine(slide, x, y, w, color) {
  slide.addShape(pres.ShapeType.rect, {
    x, y, w, h: 0.04,
    fill: { color: color || COLORS.accent },
  });
}

/** Card escuro com borda lateral colorida */
function addCard(slide, opts) {
  const { x, y, w, h, borderColor } = opts;
  // Fundo do card
  slide.addShape(pres.ShapeType.rect, {
    x, y, w, h,
    fill: { color: COLORS.cardBg },
    rectRadius: 0.1,
  });
  // Borda lateral esquerda colorida
  if (borderColor) {
    slide.addShape(pres.ShapeType.rect, {
      x, y: y + 0.05, w: 0.06, h: h - 0.1,
      fill: { color: borderColor },
      rectRadius: 0.03,
    });
  }
}

/** Título de slide padronizado */
function addSlideTitle(slide, title, subtitle) {
  slide.addText(title, {
    x: 0.6, y: 0.3, w: 12, h: 0.7,
    fontSize: 28, color: COLORS.white, fontFace: "Segoe UI Light", bold: true,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.6, y: 0.95, w: 12, h: 0.4,
      fontSize: 14, color: COLORS.accent, fontFace: "Segoe UI", italic: true,
    });
  }
  accentLine(slide, 0.6, 1.35, 2.5);
}

const TOTAL_PAGES = 14;

// ================================================================
//  SLIDE 1 — CAPA
// ================================================================
(function slideCapa() {
  const slide = pres.addSlide();
  darkBg(slide);

  // Faixa superior decorativa
  slide.addShape(pres.ShapeType.rect, {
    x: 0, y: 0, w: 13.33, h: 0.12,
    fill: { color: COLORS.accent },
  });

  // Faixa lateral esquerda
  slide.addShape(pres.ShapeType.rect, {
    x: 0, y: 0, w: 0.12, h: 7.5,
    fill: { color: COLORS.accent },
  });

  // Título principal
  slide.addText("PROJETO CONTADOR", {
    x: 1.0, y: 1.5, w: 11, h: 1.2,
    fontSize: 48, color: COLORS.white, fontFace: "Segoe UI Light", bold: true,
  });

  // Subtítulo
  slide.addText("Câmeras Inteligentes para Distribuição\nde Passageiros em Vagões de Metrô", {
    x: 1.0, y: 2.7, w: 11, h: 1.0,
    fontSize: 22, color: COLORS.accent, fontFace: "Segoe UI",
  });

  accentLine(slide, 1.0, 3.8, 5.0);

  // Informações do grupo
  slide.addText("Processamento de Imagem e Visão Computacional", {
    x: 1.0, y: 4.2, w: 11, h: 0.5,
    fontSize: 16, color: COLORS.lightGray, fontFace: "Segoe UI",
  });

  slide.addText("Tecnologias: Python • OpenCV • SAM 3.1 • PyTorch • CUDA • Tkinter", {
    x: 1.0, y: 4.7, w: 11, h: 0.4,
    fontSize: 13, color: COLORS.midGray, fontFace: "Segoe UI",
  });

  // Caixa com emoji de metrô
  slide.addText("🚇", {
    x: 10.5, y: 1.5, w: 2, h: 2,
    fontSize: 72, align: "center", valign: "middle",
  });

  // Ícones de IA
  slide.addText("🤖  📷  🧠", {
    x: 10.0, y: 3.8, w: 3, h: 0.8,
    fontSize: 28, align: "center", color: COLORS.white,
  });

  addFooter(slide, 1, TOTAL_PAGES);
})();

// ================================================================
//  SLIDE 2 — MENSAGEM-CHAVE (Pirâmide de Minto: Resposta Primeiro)
// ================================================================
(function slideKeyMessage() {
  const slide = pres.addSlide();
  darkBg(slide);
  addSlideTitle(slide, "Mensagem Central", "Princípio da Pirâmide — A resposta em primeiro lugar");

  // Card principal com a tese
  addCard(slide, { x: 0.6, y: 1.7, w: 12.1, h: 2.4, borderColor: COLORS.gold });

  slide.addText([
    { text: "Câmeras de segurança acopladas a um processador de imagens com IA ", options: { fontSize: 18, color: COLORS.white, fontFace: "Segoe UI" } },
    { text: "geram dados de lotação em tempo real ", options: { fontSize: 18, color: COLORS.accentGreen, fontFace: "Segoe UI", bold: true } },
    { text: "dentro de vagões de metrô, possibilitando a exibição dessas informações em monitores pela estação e redistribuindo o fluxo de passageiros de forma inteligente.", options: { fontSize: 18, color: COLORS.white, fontFace: "Segoe UI" } },
  ], {
    x: 1.0, y: 1.9, w: 11.3, h: 2.0,
    valign: "middle",
    lineSpacingMultiple: 1.3,
  });

  // Três pilares de suporte
  const pillars = [
    { icon: "🏢", title: "Segmento", desc: "Transporte Urbano\n& Mobilidade Inteligente", color: COLORS.accent },
    { icon: "⚙️", title: "Tecnologia", desc: "SAM 3.1 + FlashAttention\n+ Visão Computacional", color: COLORS.accentOrange },
    { icon: "📊", title: "Resultado", desc: "58× mais rápido\n< 1 segundo por frame", color: COLORS.accentGreen },
  ];

  pillars.forEach((p, i) => {
    const px = 0.6 + i * 4.1;
    addCard(slide, { x: px, y: 4.5, w: 3.8, h: 2.0, borderColor: p.color });
    slide.addText(p.icon, { x: px + 0.2, y: 4.6, w: 1, h: 0.8, fontSize: 36 });
    slide.addText(p.title, { x: px + 1.2, y: 4.65, w: 2.4, h: 0.4, fontSize: 16, color: p.color, bold: true, fontFace: "Segoe UI" });
    slide.addText(p.desc, { x: px + 1.2, y: 5.05, w: 2.4, h: 1.2, fontSize: 12, color: COLORS.lightGray, fontFace: "Segoe UI", lineSpacingMultiple: 1.2 });
  });

  addFooter(slide, 2, TOTAL_PAGES);
})();

// ================================================================
//  SLIDE 3 — SEGMENTO DA EMPRESA (6.1)
// ================================================================
(function slideSegmento() {
  const slide = pres.addSlide();
  darkBg(slide);
  addSlideTitle(slide, "Segmento da Empresa", "6.1 — Identificação do setor de atuação");

  // Card principal
  addCard(slide, { x: 0.6, y: 1.7, w: 12.1, h: 1.8, borderColor: COLORS.accent });
  slide.addText([
    { text: "Setor: ", options: { fontSize: 16, color: COLORS.accent, bold: true, fontFace: "Segoe UI" } },
    { text: "Tecnologia aplicada a Transporte Público e Mobilidade Urbana Inteligente (Smart Mobility)", options: { fontSize: 16, color: COLORS.white, fontFace: "Segoe UI" } },
  ], { x: 1.0, y: 1.85, w: 11.3, h: 0.6 });

  slide.addText([
    { text: "Subsetor: ", options: { fontSize: 14, color: COLORS.accentOrange, bold: true, fontFace: "Segoe UI" } },
    { text: "Visão Computacional e Inteligência Artificial para monitoramento de fluxo de passageiros em sistemas metroviários e ferroviários.", options: { fontSize: 14, color: COLORS.lightGray, fontFace: "Segoe UI" } },
  ], { x: 1.0, y: 2.5, w: 11.3, h: 0.8, lineSpacingMultiple: 1.2 });

  // Justificativa
  addCard(slide, { x: 0.6, y: 3.8, w: 12.1, h: 2.6, borderColor: COLORS.accentOrange });
  slide.addText("Justificativa do Segmento", {
    x: 1.0, y: 3.95, w: 5, h: 0.4,
    fontSize: 16, color: COLORS.accentOrange, bold: true, fontFace: "Segoe UI",
  });

  const justItems = [
    "O projeto utiliza câmeras de segurança já existentes na infraestrutura do metrô, aplicando processamento de imagem com o modelo SAM 3.1 para contagem zero-shot de pessoas.",
    "A solução se encaixa no contexto de Smart Cities e IoT urbano, onde dados visuais são convertidos em informações operacionais para otimização de recursos.",
    "O mercado de mobilidade inteligente no Brasil é impulsionado pela expansão do Metrô de São Paulo (Linhas 6 e 17) e pela crescente demanda por soluções data-driven em transporte público.",
    "Empresas como Thales, Alstom e Huawei já investem em soluções similares globalmente, validando o segmento.",
  ];

  justItems.forEach((item, i) => {
    slide.addText(`▸  ${item}`, {
      x: 1.0, y: 4.4 + i * 0.5, w: 11.3, h: 0.5,
      fontSize: 11, color: COLORS.lightGray, fontFace: "Segoe UI", lineSpacingMultiple: 1.1,
      valign: "top",
    });
  });

  addFooter(slide, 3, TOTAL_PAGES);
})();

// ================================================================
//  SLIDE 4 — NOME DA EMPRESA + VISÃO, MISSÃO E VALORES (6.2)
// ================================================================
(function slideEmpresa() {
  const slide = pres.addSlide();
  darkBg(slide);
  addSlideTitle(slide, "MetroVision AI", "6.2 — Empresa Fictícia • Visão, Missão e Valores");

  // Logo area
  slide.addShape(pres.ShapeType.rect, {
    x: 0.6, y: 1.7, w: 4.5, h: 3.5,
    fill: { color: COLORS.cardBg },
    rectRadius: 0.15,
  });

  slide.addText("🚇🤖", { x: 0.8, y: 1.9, w: 4.1, h: 1.2, fontSize: 56, align: "center" });
  slide.addText("MetroVision AI", {
    x: 0.8, y: 3.1, w: 4.1, h: 0.6,
    fontSize: 24, color: COLORS.accent, fontFace: "Segoe UI Light", bold: true, align: "center",
  });
  slide.addText("Inteligência que move a cidade.", {
    x: 0.8, y: 3.7, w: 4.1, h: 0.5,
    fontSize: 12, color: COLORS.lightGray, fontFace: "Segoe UI", italic: true, align: "center",
  });
  slide.addText("Fundada em 2026 • São Paulo, Brasil", {
    x: 0.8, y: 4.2, w: 4.1, h: 0.4,
    fontSize: 10, color: COLORS.midGray, fontFace: "Segoe UI", align: "center",
  });

  // Visão, Missão, Valores
  const vmv = [
    {
      title: "🔭  Visão",
      text: "Ser referência nacional em soluções de visão computacional para transporte público, transformando dados visuais em eficiência operacional e conforto para milhões de passageiros.",
      color: COLORS.accent,
    },
    {
      title: "🎯  Missão",
      text: "Desenvolver tecnologia acessível de contagem e monitoramento inteligente de passageiros, integrando câmeras existentes a modelos de IA de ponta para gerar informações em tempo real que redistribuam o fluxo de pessoas nas estações.",
      color: COLORS.accentOrange,
    },
    {
      title: "💎  Valores",
      text: "Inovação Responsável  •  Segurança e Privacidade  •  Eficiência Operacional  •  Acessibilidade Tecnológica  •  Sustentabilidade Urbana  •  Excelência Técnica",
      color: COLORS.accentGreen,
    },
  ];

  vmv.forEach((item, i) => {
    const cy = 1.7 + i * 1.2;
    addCard(slide, { x: 5.5, y: cy, w: 7.2, h: 1.05, borderColor: item.color });
    slide.addText(item.title, {
      x: 5.8, y: cy + 0.05, w: 6.6, h: 0.35,
      fontSize: 14, color: item.color, fontFace: "Segoe UI", bold: true,
    });
    slide.addText(item.text, {
      x: 5.8, y: cy + 0.38, w: 6.6, h: 0.6,
      fontSize: 10.5, color: COLORS.lightGray, fontFace: "Segoe UI", lineSpacingMultiple: 1.15,
      valign: "top",
    });
  });

  // Diferencial competitivo
  addCard(slide, { x: 0.6, y: 5.5, w: 12.1, h: 1.1, borderColor: COLORS.gold });
  slide.addText("⭐ Diferencial Competitivo", {
    x: 1.0, y: 5.55, w: 4, h: 0.35,
    fontSize: 13, color: COLORS.gold, fontFace: "Segoe UI", bold: true,
  });
  slide.addText("Utilização do SAM 3.1 (Segment Anything Model), o modelo de segmentação mais avançado da Meta AI, com capacidade zero-shot — sem necessidade de re-treinamento para novos cenários. Combinado com otimização FlashAttention para GPU, permite inferência em tempo real (<1s) em hardware acessível (RTX 3060 Laptop).", {
    x: 1.0, y: 5.9, w: 11.3, h: 0.6,
    fontSize: 10.5, color: COLORS.lightGray, fontFace: "Segoe UI", lineSpacingMultiple: 1.15,
  });

  addFooter(slide, 4, TOTAL_PAGES);
})();

// ================================================================
//  SLIDE 5 — PROBLEMA A SER RESOLVIDO (6.3)
// ================================================================
(function slideProblema() {
  const slide = pres.addSlide();
  darkBg(slide);
  addSlideTitle(slide, "O Problema", "6.3 — Identificação da dor e oportunidade de inovação");

  // Card do problema
  addCard(slide, { x: 0.6, y: 1.7, w: 12.1, h: 2.0, borderColor: COLORS.accentRed });
  slide.addText("⚠️  Distribuição Desigual de Passageiros nos Vagões do Metrô", {
    x: 1.0, y: 1.8, w: 11, h: 0.5,
    fontSize: 18, color: COLORS.accentRed, fontFace: "Segoe UI", bold: true,
  });

  const problems = [
    "Passageiros se aglomeram nos vagões próximos às escadas e acessos da estação, enquanto vagões das extremidades permanecem com baixa ocupação.",
    "Não existem informações em tempo real acessíveis aos passageiros sobre o nível de lotação de cada vagão antes do embarque.",
    "Superlotação localizada causa desconforto, atrasos no embarque/desembarque e riscos à segurança dos passageiros.",
    "O operador metroviário não possui dados granulares por vagão para tomada de decisão operacional em tempo real.",
  ];

  problems.forEach((p, i) => {
    slide.addText(`✖  ${p}`, {
      x: 1.0, y: 2.4 + i * 0.33, w: 11.3, h: 0.33,
      fontSize: 11, color: COLORS.lightGray, fontFace: "Segoe UI",
    });
  });

  // Dados de contexto
  addCard(slide, { x: 0.6, y: 4.0, w: 5.8, h: 2.5, borderColor: COLORS.accentOrange });
  slide.addText("📊  Contexto Numérico", {
    x: 1.0, y: 4.1, w: 5, h: 0.4,
    fontSize: 14, color: COLORS.accentOrange, fontFace: "Segoe UI", bold: true,
  });

  const stats = [
    "O Metrô de São Paulo transporta ~4.7 milhões de passageiros/dia.",
    "Horários de pico concentram 75% da demanda em 4h do dia.",
    "Vagões centrais operam a 120-150% da capacidade enquanto extremos ficam a 60-70%.",
    "Intervalo entre trens: 90s no pico — cada segundo de atraso no embarque é amplificado.",
  ];

  stats.forEach((s, i) => {
    slide.addText(`▸  ${s}`, {
      x: 1.0, y: 4.6 + i * 0.45, w: 5.0, h: 0.4,
      fontSize: 10.5, color: COLORS.lightGray, fontFace: "Segoe UI", lineSpacingMultiple: 1.1,
    });
  });

  // Oportunidade
  addCard(slide, { x: 6.8, y: 4.0, w: 5.9, h: 2.5, borderColor: COLORS.accentGreen });
  slide.addText("💡  A Oportunidade", {
    x: 7.2, y: 4.1, w: 5, h: 0.4,
    fontSize: 14, color: COLORS.accentGreen, fontFace: "Segoe UI", bold: true,
  });

  slide.addText(
    "Reutilizar a infraestrutura de câmeras de segurança já instaladas nos vagões, acoplando um processador de imagem com IA capaz de contar pessoas em tempo real e exibir a informação de lotação em monitores espalhados pela estação, permitindo que o passageiro escolha o vagão menos lotado antes do trem chegar.", {
    x: 7.2, y: 4.6, w: 5.2, h: 1.8,
    fontSize: 11, color: COLORS.lightGray, fontFace: "Segoe UI", lineSpacingMultiple: 1.25,
    valign: "top",
  });

  addFooter(slide, 5, TOTAL_PAGES);
})();

// ================================================================
//  SLIDE 6 — SOLUÇÃO PROPOSTA
// ================================================================
(function slideSolucao() {
  const slide = pres.addSlide();
  darkBg(slide);
  addSlideTitle(slide, "A Solução — MetroVision", "Visão geral da arquitetura e fluxo de dados");

  // Fluxo principal em 5 etapas
  const steps = [
    { icon: "📷", label: "Câmera\nno Vagão", desc: "Captura contínua\n854×480 @ 33FPS", color: COLORS.accent },
    { icon: "🔄", label: "Pré-\nProcessamento", desc: "BGR→RGB\nResize LANCZOS", color: COLORS.accentOrange },
    { icon: "🧠", label: "SAM 3.1\nInferência", desc: "Segmentação\nZero-Shot", color: COLORS.gold },
    { icon: "📊", label: "Contagem\ne Registro", desc: "Nº de pessoas\n+ Excel log", color: COLORS.accentGreen },
    { icon: "🖥️", label: "Monitor\nna Estação", desc: "Exibição\nda lotação", color: COLORS.accent },
  ];

  steps.forEach((s, i) => {
    const sx = 0.4 + i * 2.6;
    addCard(slide, { x: sx, y: 1.7, w: 2.3, h: 2.8, borderColor: s.color });
    slide.addText(s.icon, { x: sx, y: 1.8, w: 2.3, h: 0.8, fontSize: 40, align: "center" });
    slide.addText(s.label, { x: sx + 0.15, y: 2.6, w: 2.0, h: 0.6, fontSize: 13, color: s.color, fontFace: "Segoe UI", bold: true, align: "center", lineSpacingMultiple: 1.1 });
    slide.addText(s.desc, { x: sx + 0.15, y: 3.2, w: 2.0, h: 0.8, fontSize: 10, color: COLORS.lightGray, fontFace: "Segoe UI", align: "center", lineSpacingMultiple: 1.15 });

    // Seta entre etapas
    if (i < steps.length - 1) {
      slide.addText("→", {
        x: sx + 2.2, y: 2.5, w: 0.5, h: 0.6,
        fontSize: 28, color: COLORS.midGray, align: "center", valign: "middle",
      });
    }
  });

  // Descrição embaixo
  addCard(slide, { x: 0.6, y: 4.8, w: 12.1, h: 1.7, borderColor: COLORS.accent });
  slide.addText("Como funciona na prática?", {
    x: 1.0, y: 4.9, w: 5, h: 0.35,
    fontSize: 14, color: COLORS.accent, fontFace: "Segoe UI", bold: true,
  });

  slide.addText([
    { text: "1. ", options: { bold: true, color: COLORS.accent } },
    { text: "A câmera de segurança do vagão captura frames em tempo real via OpenCV.  ", options: { color: COLORS.lightGray } },
    { text: "2. ", options: { bold: true, color: COLORS.accentOrange } },
    { text: "O frame é convertido de BGR→RGB e redimensionado com filtro LANCZOS para otimizar velocidade.  ", options: { color: COLORS.lightGray } },
    { text: "3. ", options: { bold: true, color: COLORS.gold } },
    { text: "O SAM 3.1 recebe o prompt \"person\" e segmenta cada indivíduo com máscaras binárias.  ", options: { color: COLORS.lightGray } },
    { text: "4. ", options: { bold: true, color: COLORS.accentGreen } },
    { text: "A contagem é registrada em planilha Excel com timestamp e imagem embutida.  ", options: { color: COLORS.lightGray } },
    { text: "5. ", options: { bold: true, color: COLORS.accent } },
    { text: "Os dados alimentam os monitores da estação, mostrando a lotação por vagão em tempo real.", options: { color: COLORS.lightGray } },
  ], {
    x: 1.0, y: 5.3, w: 11.3, h: 1.1,
    fontSize: 10.5, fontFace: "Segoe UI", lineSpacingMultiple: 1.3,
    valign: "top",
  });

  addFooter(slide, 6, TOTAL_PAGES);
})();

// ================================================================
//  SLIDE 7 — CÓDIGO: AQUISIÇÃO E CONVERSÃO (5.2)
// ================================================================
(function slideCode1() {
  const slide = pres.addSlide();
  darkBg(slide);
  addSlideTitle(slide, "Código — Aquisição e Pré-Processamento", "5.2 — Captura de imagem, conversão de cores e redimensionamento");

  // Bloco de código 1
  addCard(slide, { x: 0.4, y: 1.7, w: 6.2, h: 2.5, borderColor: COLORS.accent });
  slide.addText("📸  Captura do Frame (videoreal.py)", {
    x: 0.7, y: 1.75, w: 5.5, h: 0.35,
    fontSize: 12, color: COLORS.accent, fontFace: "Segoe UI", bold: true,
  });

  slide.addText(
`# Loop de câmera a cada 30ms (~33 FPS)
ret, frame = self.cap.read()
self.last_raw_frame = frame.copy()

# Conversão BGR → RGB
cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
orig_img = Image.fromarray(cv_img)`, {
    x: 0.7, y: 2.15, w: 5.6, h: 1.9,
    fontSize: 10, color: COLORS.accentGreen, fontFace: "Consolas",
    fill: { color: "0D1117" },
    lineSpacingMultiple: 1.35,
    valign: "top",
    paraSpaceAfter: 4,
  });

  // Bloco de código 2
  addCard(slide, { x: 6.9, y: 1.7, w: 6.0, h: 2.5, borderColor: COLORS.accentOrange });
  slide.addText("🔧  Redimensionamento Inteligente", {
    x: 7.2, y: 1.75, w: 5.5, h: 0.35,
    fontSize: 12, color: COLORS.accentOrange, fontFace: "Segoe UI", bold: true,
  });

  slide.addText(
`# Preserva aspect ratio
target_w = int(w * (target_h / h))

# Aplica filtro anti-aliasing LANCZOS
working_img = orig_img.resize(
    (target_w, target_h),
    Image.Resampling.LANCZOS
)`, {
    x: 7.2, y: 2.15, w: 5.4, h: 1.9,
    fontSize: 10, color: COLORS.accentGreen, fontFace: "Consolas",
    fill: { color: "0D1117" },
    lineSpacingMultiple: 1.35,
    valign: "top",
  });

  // Explicação
  addCard(slide, { x: 0.4, y: 4.5, w: 12.5, h: 2.0, borderColor: COLORS.gold });
  slide.addText("💡  O que está acontecendo?", {
    x: 0.7, y: 4.55, w: 5, h: 0.35,
    fontSize: 13, color: COLORS.gold, fontFace: "Segoe UI", bold: true,
  });

  const explanations = [
    { label: "cv2.VideoCapture:", desc: "Converte a cena física (luz) em um array NumPy (H, W, 3) — amostragem espacial da imagem." },
    { label: "BGR → RGB:", desc: "OpenCV usa BGR por padrão; modelos de IA e PIL esperam RGB. Sem a conversão, as cores ficam invertidas e a segmentação falha." },
    { label: "LANCZOS:", desc: "Filtro de reamostragem que atua como passa-baixa antes do downscale, minimizando aliasing (Nyquist-Shannon aplicado a imagens 2D)." },
    { label: "Resultado:", desc: "Reduzir de 480p para 144p reduz os tokens de imagem em ~11×, acelerando drasticamente a atenção do transformer (complexidade O(N²))." },
  ];

  explanations.forEach((e, i) => {
    slide.addText([
      { text: `${e.label}  `, options: { bold: true, color: COLORS.accent, fontSize: 10.5 } },
      { text: e.desc, options: { color: COLORS.lightGray, fontSize: 10.5 } },
    ], {
      x: 0.7, y: 5.0 + i * 0.35, w: 11.8, h: 0.35,
      fontFace: "Segoe UI",
    });
  });

  addFooter(slide, 7, TOTAL_PAGES);
})();

// ================================================================
//  SLIDE 8 — CÓDIGO: SEGMENTAÇÃO SAM 3.1 (5.2)
// ================================================================
(function slideCode2() {
  const slide = pres.addSlide();
  darkBg(slide);
  addSlideTitle(slide, "Código — Segmentação com SAM 3.1", "5.2 — Inferência zero-shot com prompt de texto");

  // Bloco de código
  addCard(slide, { x: 0.4, y: 1.7, w: 7.5, h: 3.3, borderColor: COLORS.gold });
  slide.addText("🧠  Inferência do Modelo (videoreal.py)", {
    x: 0.7, y: 1.75, w: 6, h: 0.35,
    fontSize: 12, color: COLORS.gold, fontFace: "Segoe UI", bold: true,
  });

  slide.addText(
`# Configura limiar de confiança definido pelo usuário
self.processor.set_confidence_threshold(
    user_thresh, state
)

# Executa inferência com autocast bfloat16 na GPU
with torch.inference_mode():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = self.processor.set_image(
            working_img, state
        )
        state = self.processor.set_text_prompt(
            prompt, state    # prompt = "person"
        )

# Resultado: masks, boxes, scores
num_detected = len(scores_np)`, {
    x: 0.7, y: 2.15, w: 7.0, h: 2.7,
    fontSize: 10, color: COLORS.accentGreen, fontFace: "Consolas",
    fill: { color: "0D1117" },
    lineSpacingMultiple: 1.3,
    valign: "top",
  });

  // Explicação lateral
  addCard(slide, { x: 8.2, y: 1.7, w: 4.7, h: 3.3, borderColor: COLORS.accent });
  slide.addText("O que o SAM 3.1 faz?", {
    x: 8.5, y: 1.75, w: 4, h: 0.35,
    fontSize: 13, color: COLORS.accent, fontFace: "Segoe UI", bold: true,
  });

  const samSteps = [
    { step: "1", text: "set_image(): O backbone Vision Transformer (ViT) processa a imagem e gera um mapa de features semânticas de alta dimensão.", color: COLORS.accent },
    { step: "2", text: "set_text_prompt(\"person\"): Um encoder CLIP converte o texto em vetor semântico e alinha com as features visuais via cross-attention.", color: COLORS.accentOrange },
    { step: "3", text: "Saída: Máscaras binárias individuais para cada pessoa detectada + bounding boxes + scores de confiança.", color: COLORS.accentGreen },
  ];

  samSteps.forEach((s, i) => {
    slide.addText([
      { text: `${s.step}.  `, options: { bold: true, color: s.color, fontSize: 11 } },
      { text: s.text, options: { color: COLORS.lightGray, fontSize: 10.5 } },
    ], {
      x: 8.5, y: 2.2 + i * 0.85, w: 4.1, h: 0.8,
      fontFace: "Segoe UI", lineSpacingMultiple: 1.15, valign: "top",
    });
  });

  // Conceitos chave
  addCard(slide, { x: 0.4, y: 5.3, w: 12.5, h: 1.3, borderColor: COLORS.accentOrange });
  slide.addText("📚  Conceitos-Chave de Processamento de Imagem", {
    x: 0.7, y: 5.35, w: 8, h: 0.35,
    fontSize: 12, color: COLORS.accentOrange, fontFace: "Segoe UI", bold: true,
  });

  slide.addText([
    { text: "Zero-Shot: ", options: { bold: true, color: COLORS.accent } },
    { text: "Segmenta qualquer objeto sem re-treinamento — basta mudar o prompt textual.   ", options: { color: COLORS.lightGray } },
    { text: "bfloat16: ", options: { bold: true, color: COLORS.accentOrange } },
    { text: "Precisão reduzida que acelera cálculos na GPU sem perda significativa de qualidade.   ", options: { color: COLORS.lightGray } },
    { text: "inference_mode(): ", options: { bold: true, color: COLORS.accentGreen } },
    { text: "Desativa o grafo de gradientes, economizando memória (não há backpropagation na inferência).", options: { color: COLORS.lightGray } },
  ], {
    x: 0.7, y: 5.75, w: 11.8, h: 0.7,
    fontSize: 10, fontFace: "Segoe UI", lineSpacingMultiple: 1.3,
  });

  addFooter(slide, 8, TOTAL_PAGES);
})();

// ================================================================
//  SLIDE 9 — CÓDIGO: OTIMIZAÇÃO SDPA / FLASHATTENTION (5.2)
// ================================================================
(function slideCode3() {
  const slide = pres.addSlide();
  darkBg(slide);
  addSlideTitle(slide, "Código — Otimização FlashAttention", "5.2 — O gargalo resolvido: de 24.8s para 0.43s por frame");

  // ANTES
  addCard(slide, { x: 0.4, y: 1.7, w: 6.2, h: 2.2, borderColor: COLORS.accentRed });
  slide.addText("❌  ANTES — Patch Estático (Lento)", {
    x: 0.7, y: 1.75, w: 5.5, h: 0.35,
    fontSize: 12, color: COLORS.accentRed, fontFace: "Segoe UI", bold: true,
  });

  slide.addText(
`def _safe_sdpa(query, key, value, ...):
    # Desativava TODOS os kernels rápidos
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return _original_sdpa(...)
    # → Forçava backend MATH (lento)`, {
    x: 0.7, y: 2.15, w: 5.6, h: 1.6,
    fontSize: 9.5, color: COLORS.accentRed, fontFace: "Consolas",
    fill: { color: "1A0000" },
    lineSpacingMultiple: 1.3,
    valign: "top",
  });

  // DEPOIS
  addCard(slide, { x: 6.9, y: 1.7, w: 6.0, h: 2.2, borderColor: COLORS.accentGreen });
  slide.addText("✅  DEPOIS — Fallback Dinâmico (Rápido)", {
    x: 7.2, y: 1.75, w: 5.5, h: 0.35,
    fontSize: 12, color: COLORS.accentGreen, fontFace: "Segoe UI", bold: true,
  });

  slide.addText(
`# Kernels rápidos habilitados por padrão
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

def _safe_sdpa(query, key, value, ...):
    try:
        return _original_sdpa(...)  # Flash!
    except RuntimeError:
        # Fallback MATH apenas nessa chamada
        ...`, {
    x: 7.2, y: 2.15, w: 5.4, h: 1.6,
    fontSize: 9.5, color: COLORS.accentGreen, fontFace: "Consolas",
    fill: { color: "001A00" },
    lineSpacingMultiple: 1.3,
    valign: "top",
  });

  // Tabela de resultados
  addCard(slide, { x: 0.4, y: 4.2, w: 12.5, h: 2.3, borderColor: COLORS.gold });
  slide.addText("📈  Resultado do Benchmark (resolução 254×144)", {
    x: 0.7, y: 4.25, w: 8, h: 0.35,
    fontSize: 13, color: COLORS.gold, fontFace: "Segoe UI", bold: true,
  });

  // Tabela
  const tableRows = [
    [
      { text: "Configuração", options: { fill: { color: COLORS.accentDark }, color: COLORS.white, bold: true, fontSize: 11, fontFace: "Segoe UI", align: "center" } },
      { text: "Tempo / Frame", options: { fill: { color: COLORS.accentDark }, color: COLORS.white, bold: true, fontSize: 11, fontFace: "Segoe UI", align: "center" } },
      { text: "Status", options: { fill: { color: COLORS.accentDark }, color: COLORS.white, bold: true, fontSize: 11, fontFace: "Segoe UI", align: "center" } },
    ],
    [
      { text: "Patch MATH Estático (antes)", options: { fill: { color: "1A1A2E" }, color: COLORS.accentRed, fontSize: 11, fontFace: "Segoe UI" } },
      { text: "24,82 segundos", options: { fill: { color: "1A1A2E" }, color: COLORS.accentRed, bold: true, fontSize: 14, fontFace: "Segoe UI", align: "center" } },
      { text: "❌ Inviável", options: { fill: { color: "1A1A2E" }, color: COLORS.accentRed, fontSize: 11, fontFace: "Segoe UI", align: "center" } },
    ],
    [
      { text: "Patch Dinâmico + FlashAttention (depois)", options: { fill: { color: "0A1A0A" }, color: COLORS.accentGreen, fontSize: 11, fontFace: "Segoe UI" } },
      { text: "0,43 segundos", options: { fill: { color: "0A1A0A" }, color: COLORS.accentGreen, bold: true, fontSize: 14, fontFace: "Segoe UI", align: "center" } },
      { text: "✅ Tempo Real", options: { fill: { color: "0A1A0A" }, color: COLORS.accentGreen, fontSize: 11, fontFace: "Segoe UI", align: "center" } },
    ],
    [
      { text: "Ganho de Performance", options: { fill: { color: COLORS.cardBg }, color: COLORS.gold, bold: true, fontSize: 11, fontFace: "Segoe UI" } },
      { text: "≈ 58× mais rápido", options: { fill: { color: COLORS.cardBg }, color: COLORS.gold, bold: true, fontSize: 16, fontFace: "Segoe UI", align: "center" } },
      { text: "🚀", options: { fill: { color: COLORS.cardBg }, color: COLORS.gold, fontSize: 20, fontFace: "Segoe UI", align: "center" } },
    ],
  ];

  slide.addTable(tableRows, {
    x: 1.0, y: 4.7, w: 11.3,
    border: { type: "solid", pt: 0.5, color: COLORS.midGray },
    rowH: [0.35, 0.35, 0.35, 0.4],
    colW: [5.5, 3.5, 2.3],
  });

  addFooter(slide, 9, TOTAL_PAGES);
})();

// ================================================================
//  SLIDE 10 — CÓDIGO: ALPHA BLENDING E VISUALIZAÇÃO (5.2)
// ================================================================
(function slideCode4() {
  const slide = pres.addSlide();
  darkBg(slide);
  addSlideTitle(slide, "Código — Visualização e Alpha Blending", "5.2 — Composição das máscaras sobre a imagem original");

  // Código
  addCard(slide, { x: 0.4, y: 1.7, w: 7.0, h: 2.8, borderColor: COLORS.accent });
  slide.addText("🎨  Alpha Blending (videoreal.py → draw_results)", {
    x: 0.7, y: 1.75, w: 6, h: 0.35,
    fontSize: 12, color: COLORS.accent, fontFace: "Segoe UI", bold: true,
  });

  slide.addText(
`# Para cada instância detectada:
for idx in range(num_detected):
    mask = masks[idx]
    rgb = COLORS[idx % len(COLORS)]
    overlay_mask[mask] = color_bgr
    overlay_mask_bool[mask] = True

# Composição com alpha blending:
frame[overlay_mask_bool] = (
    frame[overlay_mask_bool] * 0.6 +   # 60% original
    overlay_mask[overlay_mask_bool]*0.4 # 40% máscara
).astype(np.uint8)`, {
    x: 0.7, y: 2.15, w: 6.4, h: 2.2,
    fontSize: 10, color: COLORS.accentGreen, fontFace: "Consolas",
    fill: { color: "0D1117" },
    lineSpacingMultiple: 1.3,
    valign: "top",
  });

  // Fórmula e explicação
  addCard(slide, { x: 7.7, y: 1.7, w: 5.2, h: 2.8, borderColor: COLORS.accentOrange });
  slide.addText("📐  Fórmula de Compositing", {
    x: 8.0, y: 1.75, w: 4.5, h: 0.35,
    fontSize: 12, color: COLORS.accentOrange, fontFace: "Segoe UI", bold: true,
  });

  slide.addText("C_out = α · C_overlay + (1 − α) · C_background", {
    x: 8.0, y: 2.3, w: 4.6, h: 0.5,
    fontSize: 14, color: COLORS.gold, fontFace: "Consolas", bold: true,
    align: "center",
  });

  slide.addText("onde α = 0.4", {
    x: 8.0, y: 2.8, w: 4.6, h: 0.3,
    fontSize: 12, color: COLORS.accent, fontFace: "Consolas", align: "center",
  });

  slide.addText(
    "O resultado é uma imagem onde cada pessoa detectada aparece com uma sobreposição colorida semi-transparente, permitindo visualizar simultaneamente a textura original e a máscara de segmentação.\n\n" +
    "Cada instância recebe uma cor distinta do array COLORS, facilitando a diferenciação visual entre indivíduos próximos.\n\n" +
    "Essa técnica é fundamental em pipelines de Realidade Aumentada e anotação de datasets de CV.", {
    x: 8.0, y: 3.2, w: 4.6, h: 1.2,
    fontSize: 9.5, color: COLORS.lightGray, fontFace: "Segoe UI", lineSpacingMultiple: 1.2,
    valign: "top",
  });

  // Registro em Excel
  addCard(slide, { x: 0.4, y: 4.8, w: 12.5, h: 1.7, borderColor: COLORS.accentGreen });
  slide.addText("📊  Registro Automatizado em Planilha Excel", {
    x: 0.7, y: 4.85, w: 8, h: 0.35,
    fontSize: 13, color: COLORS.accentGreen, fontFace: "Segoe UI", bold: true,
  });

  slide.addText(
`# Salva no Excel com openpyxl — timestamp, vídeo, contagem, prompt e imagem embutida
row_data = [timestamp, video_time, count, prompt, img_path]
ws.append(row_data)
# Embutir thumbnail 150×100 na coluna F
img = OpenpyxlImage(img_path)
ws.add_image(img, f"F{row_idx}")`, {
    x: 0.7, y: 5.25, w: 11.8, h: 1.1,
    fontSize: 9.5, color: COLORS.accentGreen, fontFace: "Consolas",
    fill: { color: "0D1117" },
    lineSpacingMultiple: 1.3,
    valign: "top",
  });

  addFooter(slide, 10, TOTAL_PAGES);
})();

// ================================================================
//  SLIDE 11 — PIPELINE COMPLETO (Visão Sistêmica)
// ================================================================
(function slidePipeline() {
  const slide = pres.addSlide();
  darkBg(slide);
  addSlideTitle(slide, "Pipeline Completo de Visão Computacional", "Visão sistêmica — Do frame à informação para o passageiro");

  // Pipeline em formato vertical com setas
  const pipeline = [
    { label: "Câmera / Vídeo", detail: "Aquisição (OpenCV, BGR, 854×480, 33 FPS)", color: COLORS.accent, icon: "📷" },
    { label: "Conversão de Cores", detail: "BGR → RGB (cv2.cvtColor)", color: COLORS.accentOrange, icon: "🎨" },
    { label: "Redimensionamento", detail: "LANCZOS Anti-Aliasing → 254×144", color: COLORS.gold, icon: "📐" },
    { label: "Feature Extraction", detail: "ViT Backbone (set_image)", color: COLORS.accent, icon: "🔍" },
    { label: "Grounding Texto-Imagem", detail: "Cross-Attention + CLIP (\"person\")", color: COLORS.accentOrange, icon: "🧠" },
    { label: "SDPA + FlashAttention", detail: "O(N²) otimizado por CUDA Tiling", color: COLORS.accentRed, icon: "⚡" },
    { label: "Máscaras + Boxes + Scores", detail: "Segmentação por instância", color: COLORS.accentGreen, icon: "🎯" },
    { label: "Alpha Blending", detail: "Composição visual α=0.4", color: COLORS.gold, icon: "✨" },
    { label: "GUI + Contagem + Excel", detail: "Exibição + Registro auditável", color: COLORS.accent, icon: "📊" },
  ];

  // Organizar em 3 colunas de 3
  pipeline.forEach((p, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const px = 0.5 + col * 4.3;
    const py = 1.7 + row * 1.7;

    addCard(slide, { x: px, y: py, w: 3.9, h: 1.2, borderColor: p.color });
    slide.addText(p.icon, { x: px + 0.1, y: py + 0.1, w: 0.6, h: 0.5, fontSize: 24 });
    slide.addText(p.label, {
      x: px + 0.7, y: py + 0.1, w: 3.0, h: 0.4,
      fontSize: 12, color: p.color, fontFace: "Segoe UI", bold: true,
    });
    slide.addText(p.detail, {
      x: px + 0.7, y: py + 0.5, w: 3.0, h: 0.55,
      fontSize: 9.5, color: COLORS.lightGray, fontFace: "Segoe UI", lineSpacingMultiple: 1.15,
    });

    // Seta horizontal dentro da mesma linha
    if (col < 2 && i < pipeline.length - 1) {
      slide.addText("→", {
        x: px + 3.8, y: py + 0.15, w: 0.5, h: 0.5,
        fontSize: 22, color: COLORS.midGray, align: "center", valign: "middle",
      });
    }
  });

  // Setas verticais entre linhas
  for (let row = 0; row < 2; row++) {
    slide.addText("↓", {
      x: 6.4, y: 2.85 + row * 1.7, w: 0.5, h: 0.5,
      fontSize: 22, color: COLORS.midGray, align: "center", valign: "middle",
    });
  }

  addFooter(slide, 11, TOTAL_PAGES);
})();

// ================================================================
//  SLIDE 12 — TÓPICOS ACADÊMICOS COBERTOS
// ================================================================
(function slideTopicos() {
  const slide = pres.addSlide();
  darkBg(slide);
  addSlideTitle(slide, "Tópicos de Processamento de Imagem Cobertos", "Mapeamento entre teoria acadêmica e implementação prática");

  const topics = [
    { topic: "Aquisição e Amostragem de Imagem", where: "cv2.VideoCapture, cap.read(), 30ms/frame", color: COLORS.accent },
    { topic: "Conversão de Espaço de Cores", where: "cv2.cvtColor (BGR → RGB)", color: COLORS.accentOrange },
    { topic: "Redimensionamento e Interpolação", where: "Image.resize() com LANCZOS", color: COLORS.gold },
    { topic: "Anti-Aliasing e Subamostragem", where: "Filtro LANCZOS antes do downscale", color: COLORS.accent },
    { topic: "Segmentação Semântica", where: "SAM 3.1 com prompt de texto", color: COLORS.accentGreen },
    { topic: "Feature Extraction (Deep Learning)", where: "ViT Backbone no set_image()", color: COLORS.accentOrange },
    { topic: "Atenção Multi-Head (Transformer)", where: "SDPA / FlashAttention no backbone", color: COLORS.accentRed },
    { topic: "Grounding Texto-Imagem (VLM)", where: "set_text_prompt() com CLIP", color: COLORS.gold },
    { topic: "Detecção de Instâncias", where: "Bounding Boxes + Masks", color: COLORS.accent },
    { topic: "Alpha Blending / Compositing", where: "draw_results() com α = 0.4", color: COLORS.accentGreen },
    { topic: "Latência e Tempo Real", where: "Inferência < 1s/frame após otimização", color: COLORS.accentOrange },
    { topic: "Otimização de Pipeline GPU", where: "FlashAttention + autocast bfloat16", color: COLORS.accentRed },
  ];

  const tableRows = [
    [
      { text: "Tópico Formal", options: { fill: { color: COLORS.accentDark }, color: COLORS.white, bold: true, fontSize: 11, fontFace: "Segoe UI", align: "center" } },
      { text: "Onde Aparece no Projeto", options: { fill: { color: COLORS.accentDark }, color: COLORS.white, bold: true, fontSize: 11, fontFace: "Segoe UI", align: "center" } },
    ],
  ];

  topics.forEach((t) => {
    tableRows.push([
      { text: t.topic, options: { fill: { color: COLORS.cardBg }, color: t.color, fontSize: 10, fontFace: "Segoe UI", bold: true } },
      { text: t.where, options: { fill: { color: COLORS.cardBg }, color: COLORS.lightGray, fontSize: 10, fontFace: "Consolas" } },
    ]);
  });

  slide.addTable(tableRows, {
    x: 0.6, y: 1.7, w: 12.1,
    border: { type: "solid", pt: 0.5, color: COLORS.midGray },
    rowH: 0.37,
    colW: [5.5, 6.6],
  });

  addFooter(slide, 12, TOTAL_PAGES);
})();

// ================================================================
//  SLIDE 13 — RESULTADOS E IMPACTO
// ================================================================
(function slideResultados() {
  const slide = pres.addSlide();
  darkBg(slide);
  addSlideTitle(slide, "Resultados e Impacto Esperado", "O que o MetroVision entrega para a operação metroviária");

  // KPIs
  const kpis = [
    { value: "58×", label: "Mais rápido\n(otimização SDPA)", color: COLORS.accentGreen },
    { value: "<1s", label: "Latência por\nframe (tempo real)", color: COLORS.accent },
    { value: "0.43s", label: "Tempo médio\nde inferência", color: COLORS.gold },
    { value: "∞", label: "Objetos sem\nre-treinamento", color: COLORS.accentOrange },
  ];

  kpis.forEach((k, i) => {
    const kx = 0.5 + i * 3.2;
    addCard(slide, { x: kx, y: 1.7, w: 2.9, h: 1.8, borderColor: k.color });
    slide.addText(k.value, {
      x: kx, y: 1.8, w: 2.9, h: 0.8,
      fontSize: 36, color: k.color, fontFace: "Segoe UI Light", bold: true, align: "center",
    });
    slide.addText(k.label, {
      x: kx, y: 2.6, w: 2.9, h: 0.7,
      fontSize: 11, color: COLORS.lightGray, fontFace: "Segoe UI", align: "center", lineSpacingMultiple: 1.2,
    });
  });

  // Benefícios operacionais
  addCard(slide, { x: 0.5, y: 3.8, w: 6.0, h: 2.7, borderColor: COLORS.accentGreen });
  slide.addText("✅  Benefícios Operacionais", {
    x: 0.8, y: 3.85, w: 5, h: 0.35,
    fontSize: 14, color: COLORS.accentGreen, fontFace: "Segoe UI", bold: true,
  });

  const benefits = [
    "Redistribuição inteligente dos passageiros entre vagões",
    "Redução do tempo de embarque/desembarque no pico",
    "Melhora na experiência do passageiro (conforto)",
    "Dados operacionais granulares por vagão em tempo real",
    "Aproveitamento da infraestrutura de câmeras existentes",
    "Registros auditáveis em Excel com imagens embutidas",
    "Modelo zero-shot adaptável a outros cenários (ônibus, aeroportos)",
  ];

  benefits.forEach((b, i) => {
    slide.addText(`▸  ${b}`, {
      x: 0.8, y: 4.3 + i * 0.31, w: 5.4, h: 0.31,
      fontSize: 10.5, color: COLORS.lightGray, fontFace: "Segoe UI",
    });
  });

  // Escalabilidade futura
  addCard(slide, { x: 6.8, y: 3.8, w: 6.0, h: 2.7, borderColor: COLORS.accent });
  slide.addText("🔮  Escalabilidade Futura", {
    x: 7.1, y: 3.85, w: 5, h: 0.35,
    fontSize: 14, color: COLORS.accent, fontFace: "Segoe UI", bold: true,
  });

  const future = [
    "Integração com sistemas SCADA/CBTC do metrô",
    "Dashboard web em tempo real para centro de controle",
    "Alertas automáticos de superlotação via API",
    "Expansão para análise de fluxo em plataformas",
    "Modelo treinado fine-tuned para maior precisão",
    "Deploy em edge computing (NVIDIA Jetson Orin)",
    "Aplicativo mobile para passageiros consultarem lotação",
  ];

  future.forEach((f, i) => {
    slide.addText(`▸  ${f}`, {
      x: 7.1, y: 4.3 + i * 0.31, w: 5.4, h: 0.31,
      fontSize: 10.5, color: COLORS.lightGray, fontFace: "Segoe UI",
    });
  });

  addFooter(slide, 13, TOTAL_PAGES);
})();

// ================================================================
//  SLIDE 14 — ENCERRAMENTO
// ================================================================
(function slideEncerramento() {
  const slide = pres.addSlide();
  darkBg(slide);

  // Faixa decorativa superior
  slide.addShape(pres.ShapeType.rect, {
    x: 0, y: 0, w: 13.33, h: 0.12,
    fill: { color: COLORS.accent },
  });

  slide.addShape(pres.ShapeType.rect, {
    x: 0, y: 0, w: 0.12, h: 7.5,
    fill: { color: COLORS.accent },
  });

  // Título
  slide.addText("Obrigado!", {
    x: 1.0, y: 1.5, w: 11, h: 1.2,
    fontSize: 52, color: COLORS.white, fontFace: "Segoe UI Light", bold: true,
  });

  slide.addText("Projeto Contador — MetroVision AI", {
    x: 1.0, y: 2.7, w: 11, h: 0.6,
    fontSize: 22, color: COLORS.accent, fontFace: "Segoe UI",
  });

  accentLine(slide, 1.0, 3.4, 4.0);

  // Resumo final (Pirâmide de Minto — reafirmação da mensagem central)
  addCard(slide, { x: 0.8, y: 3.8, w: 11.7, h: 1.5, borderColor: COLORS.gold });
  slide.addText([
    { text: "Demonstramos que é possível ", options: { color: COLORS.lightGray, fontSize: 14 } },
    { text: "reutilizar câmeras de segurança ", options: { color: COLORS.accent, fontSize: 14, bold: true } },
    { text: "já instaladas nos vagões de metrô, acoplando um ", options: { color: COLORS.lightGray, fontSize: 14 } },
    { text: "processador de imagem com IA (SAM 3.1) ", options: { color: COLORS.gold, fontSize: 14, bold: true } },
    { text: "otimizado com FlashAttention, para gerar ", options: { color: COLORS.lightGray, fontSize: 14 } },
    { text: "dados de lotação em tempo real (<1s/frame) ", options: { color: COLORS.accentGreen, fontSize: 14, bold: true } },
    { text: "exibidos em monitores nas estações, ", options: { color: COLORS.lightGray, fontSize: 14 } },
    { text: "redistribuindo passageiros de forma inteligente.", options: { color: COLORS.accent, fontSize: 14, bold: true } },
  ], {
    x: 1.2, y: 3.95, w: 10.9, h: 1.2,
    fontFace: "Segoe UI", lineSpacingMultiple: 1.4, valign: "middle",
  });

  // Tecnologias
  slide.addText("Python  •  OpenCV  •  SAM 3.1  •  PyTorch  •  FlashAttention  •  CUDA  •  Tkinter  •  openpyxl", {
    x: 1.0, y: 5.6, w: 11, h: 0.5,
    fontSize: 12, color: COLORS.midGray, fontFace: "Segoe UI", align: "center",
  });

  // Emojis finais
  slide.addText("🚇  📷  🧠  ⚡  📊", {
    x: 3.5, y: 6.1, w: 6, h: 0.6,
    fontSize: 32, align: "center",
  });

  addFooter(slide, 14, TOTAL_PAGES);
})();

// ================================================================
//  GERAR O ARQUIVO .PPTX
// ================================================================
pres.writeFile({ fileName: "Projeto_Contador_MetroVision_AI.pptx" })
  .then(() => {
    console.log("✅ Apresentação gerada com sucesso!");
    console.log("📂 Arquivo: Projeto_Contador_MetroVision_AI.pptx");
    console.log("📄 Total de slides: " + TOTAL_PAGES);
  })
  .catch((err) => {
    console.error("❌ Erro ao gerar a apresentação:", err);
  });
