"""
☁️ PinkCloud Studio — Nube de Palabras Moderna
Diseño aesthetic rosado + UI premium para Streamlit
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
    page_title="PinkCloud Studio",
    page_icon="💖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# ESTILOS PERSONALIZADOS
# ─────────────────────────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Fondo principal */
.stApp {
    background: linear-gradient(135deg, #0f0f14, #1b1022);
    color: white;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161621 !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Títulos sidebar */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ff85c2 !important;
    font-weight: 700 !important;
}

/* Texto sidebar */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #f5f5f7 !important;
}

/* Inputs */
textarea, input[type="text"] {
    background-color: #1f1f2e !important;
    border: 2px solid #ff4fa3 !important;
    border-radius: 12px !important;
    color: white !important;
    padding: 12px !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: #ff85c2 !important;
    box-shadow: 0 0 12px rgba(255,79,163,0.3) !important;
}

/* Selectbox */
[data-baseweb="select"] > div {
    background: #1f1f2e !important;
    border: 2px solid #ff4fa3 !important;
    border-radius: 12px !important;
    color: white !important;
}

/* Radio y sliders */
.stSlider label, .stRadio label {
    color: #f5f5f7 !important;
}

/* Botones */
.stButton > button {
    background: linear-gradient(135deg, #ff4fa3, #ff85c2) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    padding: 0.7rem 1.4rem !important;
    transition: 0.3s ease !important;
    box-shadow: 0 4px 18px rgba(255,79,163,0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(255,79,163,0.4);
}

/* Download button */
[data-testid="stDownloadButton"] button {
    background: #2a2235 !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #ff4fa3 !important;
}

/* Header principal */
.header-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 35px;
    margin-bottom: 25px;
    box-shadow: 0 8px 28px rgba(0,0,0,0.25);
}

/* Cards */
.section-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(10px);
    border-radius: 22px;
    padding: 24px;
    margin-bottom: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 6px 20px rgba(0,0,0,0.2);
}

/* Títulos */
h1 {
    color: #ff4fa3 !important;
    font-size: 3rem !important;
    font-weight: 700 !important;
}

h2, h3 {
    color: #ff85c2 !important;
}

/* Texto */
p, li, span {
    color: #f5f5f7 !important;
}

/* Métricas */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.2);
}

[data-testid="metric-container"] label {
    color: #ffb3d9 !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: white !important;
    font-size: 1.8rem !important;
}

/* Nube container */
.wc-container {
    background: rgba(255,255,255,0.04);
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}

/* Expander */
div[data-testid="stExpander"] {
    border-radius: 18px !important;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08) !important;
    background: rgba(255,255,255,0.04) !important;
}

/* Frecuencia rows */
.freq-row {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px;
    margin-bottom: 10px;
    background: rgba(255,255,255,0.04);
    border-radius: 14px;
    transition: 0.2s ease;
}

.freq-row:hover {
    transform: translateX(4px);
    background: rgba(255,255,255,0.07);
}

/* Barras */
.freq-bar {
    height: 10px;
    border-radius: 8px;
    background: linear-gradient(90deg, #ff4fa3, #ff85c2);
}

/* Rank */
.rank-tag {
    background: rgba(255,79,163,0.15);
    color: #ff85c2;
    padding: 4px 10px;
    border-radius: 10px;
    font-size: 0.8rem;
    font-weight: 700;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #ff4fa3;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <h1>☁️ PinkCloud Studio</h1>
    <p style="font-size:18px;">
        Genera nubes de palabras modernas con una experiencia visual premium.
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RESTO DEL CÓDIGO
# ─────────────────────────────────────────────

st.info("✨ Mantén el resto de tu lógica original exactamente igual debajo de este bloque de estilos.")
