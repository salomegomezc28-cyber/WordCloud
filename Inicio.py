"""☁️ WordCloud Studio — Nube de Palabras Profesional

Instalación:
pip install streamlit wordcloud matplotlib pandas Pillow numpy

Ejecución:
streamlit run wordcloud_app.py
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import io
from collections import Counter
from wordcloud import WordCloud, STOPWORDS


# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="WordCloud Studio",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# ESTILOS — diseño rosado profesional
# ─────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, #ffe4f1 0, transparent 28%),
            linear-gradient(135deg, #fff7fb 0%, #fff0f6 45%, #ffffff 100%);
        color: #2f1022;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #831843 0%, #be185d 48%, #f472b6 100%);
        border-right: 1px solid rgba(255,255,255,0.25);
    }

    section[data-testid="stSidebar"] * {
        color: #fff7fb !important;
    }

    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] select {
        background: rgba(255,255,255,0.95) !important;
        color: #4a102a !important;
        border-radius: 14px !important;
        border: 1px solid #fbcfe8 !important;
    }

    .main-title {
        padding: 26px 30px;
        border-radius: 28px;
        background: linear-gradient(135deg, #be185d 0%, #ec4899 52%, #f9a8d4 100%);
        box-shadow: 0 20px 45px rgba(190, 24, 93, 0.22);
        margin-bottom: 24px;
        color: white;
    }

    .main-title h1 {
        margin: 0;
        font-size: 2.5rem;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.04em;
    }

    .main-title p {
        margin: 10px 0 0;
        color: #fff1f7;
        font-size: 1rem;
        max-width: 760px;
    }

    .section-card {
        background: rgba(255,255,255,0.88);
        border: 1px solid #fbcfe8;
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 14px 35px rgba(190, 24, 93, 0.10);
        backdrop-filter: blur(10px);
    }

    .section-card h3 {
        color: #831843;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    .info-item {
        display: flex;
        gap: 12px;
        align-items: flex-start;
        background: #fff5fa;
        border: 1px solid #fce7f3;
        padding: 13px 14px;
        border-radius: 18px;
        margin: 10px 0;
    }

    .uso-tag {
        display: inline-block;
        padding: 9px 12px;
        margin: 5px 5px 5px 0;
        border-radius: 999px;
        background: #fce7f3;
        color: #831843 !important;
        border: 1px solid #f9a8d4;
        font-size: 0.88rem;
        font-weight: 600;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.92);
        border: 1px solid #fbcfe8;
        border-radius: 22px;
        padding: 18px;
        box-shadow: 0 10px 28px rgba(190, 24, 93, 0.10);
    }

    div[data-testid="stMetricLabel"] {
        color: #9d174d !important;
        font-weight: 700;
    }

    div[data-testid="stMetricValue"] {
        color: #4a102a !important;
        font-weight: 800;
    }

    .freq-row {
        display: flex;
        align-items: center;
        gap: 10px;
        background: #fff7fb;
        border: 1px solid #fce7f3;
        border-radius: 16px;
        padding: 9px 11px;
        margin: 7px 0;
    }

    .rank-tag {
        background: #be185d;
        color: white;
        border-radius: 999px;
        padding: 5px 9px;
        font-size: 0.78rem;
        font-weight: 800;
    }

    .freq-bar {
        height: 12px;
        border-radius: 999px;
        background: linear-gradient(90deg, #f9a8d4, #ec4899, #be185d);
    }

    .cloud-card {
        background: white;
        border: 1px solid #fbcfe8;
        border-radius: 26px;
        padding: 18px;
        box-shadow: 0 16px 40px rgba(190, 24, 93, 0.12);
    }

    .cloud-caption {
        color: #831843;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(135deg, #be185d, #ec4899) !important;
        color: white !important;
        border: 0 !important;
        border-radius: 16px !important;
        font-weight: 800 !important;
        padding: 0.7rem 1rem !important;
        box-shadow: 0 10px 24px rgba(190, 24, 93, 0.22);
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #9d174d, #db2777) !important;
        color: white !important;
        border: 0 !important;
    }

    hr {
        border-color: rgba(251, 207, 232, 0.75) !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# STOPWORDS
# ─────────────────────────────────────────────

STOPWORDS_ES = {
    "de","la","el","en","y","a","los","del","se","las","un","por","con","no",
    "una","su","para","es","al","lo","como","mas","pero","sus","le","ya","o",
    "este","si","porque","esta","entre","cuando","muy","sin","sobre","tambien",
    "me","hasta","hay","donde","quien","desde","nos","durante","ni","contra",
    "ese","eso","ante","bajo","tras","que","fue","son","han","ha","ser","era",
    "estan","siendo","sido","he","has","hemos","habian","tiene","tienen",
    "hacer","puede","pueden","asi","tan","parte","todo","todos","todas","cada",
    "otro","otra","otros","otras","mismo","misma","nuestro","nuestra","ellos",
    "ellas","nosotros","les","esa","esos","esas","aquel","aquella","aquellos",
}


def obtener_stopwords(idioma):
    sw = set(STOPWORDS)
    if idioma in ("Español", "Ambos"):
        sw |= STOPWORDS_ES
    return sw


# ─────────────────────────────────────────────
# PALETAS PROFESIONALES
# ─────────────────────────────────────────────

PALETAS = {
    "Rosado editorial": ["#831843", "#9d174d", "#be185d", "#db2777", "#ec4899", "#f472b6", "#f9a8d4"],
    "Rosado suave": ["#500724", "#831843", "#be185d", "#ec4899", "#f9a8d4", "#fbcfe8", "#fce7f3"],
    "Escala de grises": ["#111827", "#1f2937", "#374151", "#4b5563", "#6b7280", "#9ca3af", "#d1d5db"],
    "Azul corporativo": ["#1e3a5f", "#1d4ed8", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#0f2942"],
    "Verde institucional": ["#064e3b", "#065f46", "#047857", "#059669", "#10b981", "#34d399", "#6ee7b7"],
    "Gris azulado": ["#0f172a", "#1e293b", "#334155", "#475569", "#64748b", "#94a3b8", "#cbd5e1"],
    "Terracota": ["#7c2d12", "#9a3412", "#c2410c", "#ea580c", "#f97316", "#fb923c", "#fdba74"],
    "Índigo profundo": ["#1e1b4b", "#312e81", "#3730a3", "#4338ca", "#4f46e5", "#6366f1", "#818cf8"],
    "Monocromático negro": ["#000000", "#111111", "#222222", "#444444", "#666666", "#888888", "#aaaaaa"],
}

FORMAS = {
    "Rectángulo": None,
    "Círculo": "circle",
}


def crear_mascara(forma, size=500):
    if forma == "circle":
        y, x = np.ogrid[:size, :size]
        cx, cy = size // 2, size // 2
        mascara = np.ones((size, size), dtype=np.uint8) * 255
        mascara[(x - cx) ** 2 + (y - cy) ** 2 <= (size // 2 - 12) ** 2] = 0
        return mascara
    return None


# ─────────────────────────────────────────────
# FUNCIONES CORE
# ─────────────────────────────────────────────

def limpiar_texto(texto, stopwords, min_longitud):
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"[^a-záéíóúüñàâèêîôùûäëïöü\s]", " ", texto, flags=re.UNICODE)
    palabras = [p for p in texto.split() if p not in stopwords and len(p) >= min_longitud]
    return " ".join(palabras)


def contar_palabras(texto_limpio):
    return pd.DataFrame(
        Counter(texto_limpio.split()).most_common(50),
        columns=["Palabra", "Frecuencia"],
    )


def generar_wordcloud(texto_limpio, paleta_nombre, max_words, fondo, forma, ancho=1000, alto=520):
    import random

    colores = PALETAS[paleta_nombre]

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        rng = random_state or random.Random()
        return colores[rng.randint(0, len(colores) - 1)]

    mascara = crear_mascara(forma, size=min(ancho, alto))

    wc = WordCloud(
        width=ancho,
        height=alto,
        max_words=max_words,
        background_color=fondo,
        color_func=color_func,
        mask=mascara,
        collocations=False,
        min_font_size=11,
        max_font_size=120,
        prefer_horizontal=0.75,
        relative_scaling=0.5,
        margin=5,
    ).generate(texto_limpio)

    fig, ax = plt.subplots(figsize=(ancho / 100, alto / 100))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor(fondo)
    plt.tight_layout(pad=0)
    return fig


def fig_a_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ☁️ WordCloud Studio")
    st.markdown("Crea nubes de palabras con una estética limpia y personalizable.")
    st.divider()

    st.markdown("### 📥 Fuente de texto")
    fuente = st.radio(
        "fuente",
        ["✍️ Escribir / Pegar", "📂 Subir archivo"],
        label_visibility="collapsed",
    )

    texto_input = ""

    if fuente == "✍️ Escribir / Pegar":
        texto_input = st.text_area(
            "Texto:",
            height=190,
            placeholder="Pega aquí un artículo, reseña, discurso, encuesta...",
        )

        with st.expander("Cargar texto de ejemplo"):
            ejemplos = {
                "Inteligencia Artificial": """
                La inteligencia artificial es una disciplina de la informática orientada a desarrollar
                sistemas capaces de ejecutar tareas que requieren capacidades cognitivas humanas.
                El aprendizaje automático, las redes neuronales profundas y el procesamiento del
                lenguaje natural constituyen los pilares técnicos de los sistemas modernos de
                inteligencia artificial. Los modelos de lenguaje de gran escala, la visión
                computacional y la robótica autónoma representan aplicaciones de vanguardia.
                La inteligencia artificial transforma sectores como la salud, la educación,
                la manufactura, las finanzas y el transporte, generando eficiencias significativas.
                """,
                "Colombia": """
                Colombia es una nación situada en el extremo noroccidental de América del Sur,
                reconocida por su excepcional biodiversidad, riqueza cultural y diversidad de paisajes.
                Bogotá es la capital y principal centro económico, seguida de Medellín, Cali y
                Barranquilla como ciudades de relevancia nacional. El café colombiano goza de
                reconocimiento internacional por su calidad y perfil aromático. La floricultura
                colombiana abastece mercados globales con alta competitividad. El país alberga
                ecosistemas del Amazonas, los Andes, el Caribe y el Pacífico, constituyéndose
                como uno de los territorios con mayor biodiversidad del planeta.
                """,
                "Tecnología 4.0": """
                La cuarta revolución industrial redefine los modelos productivos mediante la
                convergencia de tecnologías digitales avanzadas. El Internet de las cosas,
                la inteligencia artificial, el análisis de grandes datos, la robótica colaborativa
                y la automatización inteligente son pilares estratégicos de la industria moderna.
                Las fábricas inteligentes integran sensores, conectividad y analítica para
                optimizar procesos en tiempo real. La manufactura aditiva, los gemelos digitales
                y la realidad aumentada transforman la ingeniería de producción. La computación
                en la nube y la ciberseguridad son habilitadores fundamentales de la
                transformación digital empresarial.
                """,
            }

            ejemplo_sel = st.selectbox(
                "Ejemplo:",
                list(ejemplos.keys()),
                label_visibility="collapsed",
            )

            if st.button("Cargar texto seleccionado"):
                st.session_state["texto_ejemplo"] = ejemplos[ejemplo_sel]
                st.rerun()

        if "texto_ejemplo" in st.session_state and not texto_input:
            texto_input = st.session_state["texto_ejemplo"]

    else:
        archivo = st.file_uploader(
            "Archivo:",
            type=["txt", "csv"],
            label_visibility="collapsed",
        )

        if archivo:
            if archivo.name.endswith(".txt"):
                texto_input = archivo.read().decode("utf-8", errors="ignore")
            elif archivo.name.endswith(".csv"):
                df_csv = pd.read_csv(archivo)
                col_txt = st.selectbox("Columna de texto:", df_csv.columns.tolist())
                texto_input = " ".join(df_csv[col_txt].dropna().astype(str).tolist())

            st.success(f"Archivo cargado — {len(texto_input):,} caracteres")

    st.divider()

    st.markdown("### ⚙️ Procesamiento")
    idioma = st.selectbox("Stopwords:", ["Español", "Inglés", "Ambos", "Ninguno"])
    min_longitud = st.slider("Longitud mínima de palabra", 2, 8, 3)
    palabras_extra = st.text_input(
        "Excluir palabras adicionales:",
        placeholder="ej: también, así, aquí",
    )

    st.divider()

    st.markdown("### 🎨 Apariencia")
    paleta_sel = st.selectbox("Paleta:", list(PALETAS.keys()))
    fondo_sel = st.radio("Fondo:", ["Blanco", "Negro"], horizontal=True)
    fondo_color = "white" if fondo_sel == "Blanco" else "black"
    forma_sel = st.selectbox("Forma:", list(FORMAS.keys()))
    max_words = st.slider("Máximo de palabras:", 20, 200, 80)

    st.divider()
    generar = st.button("GENERAR NUBE  ↗", use_container_width=True)


# ─────────────────────────────────────────────
# CONTENIDO PRINCIPAL
# ─────────────────────────────────────────────

st.markdown(
    """
    <div class="main-title">
        <h1>☁️ WordCloud Studio</h1>
        <p>Analiza textos, detecta términos dominantes y genera nubes de palabras con una presentación visual más clara, moderna y editable.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# PANTALLA DE BIENVENIDA
# ─────────────────────────────────────────────

if not generar or not texto_input.strip():
    col_izq, col_der = st.columns([3, 2], gap="large")

    with col_izq:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Acerca de esta herramienta")
        st.markdown(
            """
            Una **nube de palabras** representa visualmente la frecuencia de términos en un texto:
            las palabras más frecuentes aparecen con mayor tamaño, lo que permite identificar
            los temas centrales de un corpus de manera rápida.
            """
        )

        for icono, titulo, desc in [
            ("📊", "Análisis de frecuencia", "Identifica los términos dominantes de cualquier corpus textual."),
            ("🔍", "Filtrado inteligente", "Elimina palabras vacías (*stopwords*) en español e inglés."),
            ("🎨", "Personalización visual", "Selecciona paleta, forma y densidad de la nube."),
            ("⬇️", "Exportación", "Descarga la imagen en alta resolución y la tabla de frecuencias en CSV."),
        ]:
            st.markdown(
                f'<div class="info-item">'
                f'<span style="font-size:1.3rem; flex-shrink:0;">{icono}</span>'
                f'<div><strong style="color:#831843;">{titulo}</strong>'
                f'<p style="margin:2px 0 0 0; color:#6b2146 !important; font-size:0.88rem;">{desc}</p></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### Instrucciones")
        for i, paso in enumerate(
            [
                "Ingresa o sube un texto en el panel lateral.",
                "Configura el idioma de *stopwords*, paleta y número de palabras.",
                "Haz clic en **GENERAR NUBE ↗**.",
                "Descarga la imagen PNG o la tabla CSV.",
            ],
            1,
        ):
            st.markdown(f"**{i}.** {paso}")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_der:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Aplicaciones frecuentes")

        for caso in [
            "📰 Análisis de prensa y noticias",
            "📋 Resultados de encuestas abiertas",
            "💬 Reseñas y comentarios de clientes",
            "🎓 Análisis de textos académicos",
            "🗳️ Discursos y documentos políticos",
            "📚 Estudios literarios y de corpus",
            "📊 Informes de inteligencia de negocio",
        ]:
            st.markdown(
                f'<span class="uso-tag">{caso}</span>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="section-card" style="margin-top:16px;">',
            unsafe_allow_html=True,
        )
        st.markdown("### Paletas disponibles")

        for nombre in PALETAS.keys():
            st.markdown(
                f'<div style="padding:7px 0; border-bottom:1px solid #fce7f3;">'
                f'<span style="color:#831843; font-size:0.9rem; font-weight:700;">{nombre}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    if not texto_input.strip() and generar:
        st.warning("Ingresa un texto en el panel lateral antes de generar la nube.")

    st.stop()


# ─────────────────────────────────────────────
# PROCESAMIENTO
# ─────────────────────────────────────────────

stopwords_set = obtener_stopwords(idioma) if idioma != "Ninguno" else set()

if palabras_extra.strip():
    stopwords_set |= {
        p.strip().lower()
        for p in palabras_extra.split(",")
        if p.strip()
    }

texto_limpio = limpiar_texto(texto_input, stopwords_set, min_longitud)

if not texto_limpio.strip():
    st.error("El texto resultante está vacío. Reduce la longitud mínima o cambia la configuración de stopwords.")
    st.stop()

df_freq = contar_palabras(texto_limpio)
total_palabras = len(texto_limpio.split())
vocabulario = len(df_freq)


# ─────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────

m1, m2, m3, m4 = st.columns(4)

m1.metric("Palabras procesadas", f"{total_palabras:,}")
m2.metric("Vocabulario único", f"{vocabulario:,}")
m3.metric("Término más frecuente", df_freq.iloc[0]["Palabra"] if not df_freq.empty else "—")
m4.metric("Frecuencia máxima", int(df_freq.iloc[0]["Frecuencia"]) if not df_freq.empty else 0)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# NUBE
# ─────────────────────────────────────────────

with st.spinner("Generando nube de palabras..."):
    fig_wc = generar_wordcloud(
        texto_limpio,
        paleta_sel,
        max_words,
        fondo_color,
        FORMAS[forma_sel],
        ancho=1000,
        alto=520,
    )

st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
st.markdown(
    f'<div class="cloud-caption">Nube de palabras · Paleta: {paleta_sel} · Fondo: {fondo_sel} · {max_words} palabras máx.</div>',
    unsafe_allow_html=True,
)
st.pyplot(fig_wc, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

img_bytes = fig_a_bytes(fig_wc)

st.download_button(
    "⬇️ Descargar imagen PNG",
    data=img_bytes,
    file_name="wordcloud.png",
    mime="image/png",
    use_container_width=True,
)

st.divider()


# ─────────────────────────────────────────────
# ANÁLISIS
# ─────────────────────────────────────────────

col_freq, col_tabla = st.columns([3, 2], gap="large")

with col_freq:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Frecuencia léxica — Top 20")

    top20 = df_freq.head(20)
    max_freq = top20["Frecuencia"].max()

    for rank, (_, row) in enumerate(top20.iterrows(), 1):
        p = row["Palabra"]
        f = int(row["Frecuencia"])
        barra_w = max(12, int((f / max_freq) * 210))

        st.markdown(
            f'<div class="freq-row">'
            f'<span class="rank-tag">#{rank:02d}</span>'
            f'<span style="font-weight:700; color:#831843; min-width:130px; font-size:0.93rem;">{p}</span>'
            f'<div class="freq-bar" style="width:{barra_w}px; opacity:{0.5 + 0.5 * (f / max_freq):.2f};"></div>'
            f'<span style="font-family:monospace; font-size:0.88rem; color:#4a102a; min-width:28px; text-align:right; font-weight:700;">{f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

with col_tabla:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Tabla de frecuencias")

    st.dataframe(
        df_freq.head(30)
        .style.background_gradient(subset=["Frecuencia"], cmap="RdPu")
        .format({"Frecuencia": "{:,}"}),
        use_container_width=True,
        height=500,
    )

    csv_bytes = df_freq.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Exportar tabla (.csv)",
        data=csv_bytes,
        file_name="frecuencias.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

with st.expander("Ver texto procesado tras eliminación de stopwords"):
    preview = texto_limpio[:2500] + ("..." if len(texto_limpio) > 2500 else "")
    st.markdown(
        f'<div style="background:#fff7fb; border:1px solid #fbcfe8; border-radius:18px; padding:16px; color:#4a102a;">{preview}</div>',
        unsafe_allow_html=True,
    )

plt.close("all")
