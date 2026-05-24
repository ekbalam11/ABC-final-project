import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from pathlib import Path
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# DATOS DEL MODELO
# ─────────────────────────────────────────────────────────────────────────────

MODELO = {
    "nombre":     "XGBoost",
    "n_arboles":  600,
    "color":      "#3498DB",

    # Métricas globales
    "accuracy":   0.68,
    "f1_macro":   0.68,
    "auc_roc":    0.7459,

    # Métricas por clase
    "clases": {
        "no_riesgo": {"precision": 0.68, "recall": 0.63, "f1": 0.65},
        "riesgo":    {"precision": 0.68, "recall": 0.73, "f1": 0.71},
    },

    # Matriz de confusión [TN FP / FN TP]
    "confusion_matrix": [
        [4135, 2478],   # [TN, FP]
        [1946, 5378],   # [FN, TP]
    ],

    # Hiperparámetros óptimos
    "hiperparametros": {
        "n_estimators":    600,
        "learning_rate":   0.02,
        "max_depth":       7,
        "subsample":       0.9,
        "colsample_bytree":0.8,
        "reg_alpha":       1,
        "reg_lambda":      1,
        "gamma":           0.5,
    },

    # Feature importances — actualizar con los valores reales del modelo
    "feature_importances": {
        "elevation":        0.166750,
        "season_summer":    0.153434,
        "season_winter":    0.121504,
        "ndvi":             0.089679,
        "season_spring":    0.068757,
        "nbr":              0.061618,
        "ndwi":             0.058517,
        "mndwi":            0.049134,
        "swir2":            0.048130,
        "nir":              0.047852,
        "swir1":            0.046644,
        "ndmi":             0.044848,
        "ndbi":             0.043131
    },

    # Dataset
    "n_total":  69683,
    "n_train":  55746,
    "n_test":   13937,
}

# ─────────────────────────────────────────────────────────────────────────────
# ESTILO GLOBAL MATPLOTLIB
# ─────────────────────────────────────────────────────────────────────────────

BG_MAIN  = "#0E1117"
BG_CARD  = "#1A1D27"
BG_CARD2 = "#1F2235"
BORDER   = "#2D3045"
TEXT_DIM = "#8890A4"
TEXT_MED = "#C8CDD8"
TEXT_BRT = "#E0E4EF"
ACCENT   = MODELO["color"]
RED      = "#E74C3C"

plt.rcParams.update({
    "figure.facecolor": BG_MAIN,
    "axes.facecolor":   BG_CARD,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT_MED,
    "xtick.color":      TEXT_DIM,
    "ytick.color":      TEXT_DIM,
    "text.color":       TEXT_BRT,
    "grid.color":       BORDER,
    "grid.linewidth":   0.5,
    "font.family":      "monospace",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ─────────────────────────────────────────────────────────────────────────────
# COLORES DEL MAPA
# ─────────────────────────────────────────────────────────────────────────────

COLORES_MAPA = {
    "TP": "#2ECC71",
    "TN": "#3498DB",
    "FP": "#F39C12",
    "FN": "#E74C3C",
}

COLORES_PRED = {
    "riesgo":    "#E74C3C",
    "no_riesgo": "#3498DB",
}

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DEL CSV DEL MAPA
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def cargar_mapa_data():
    path = Path(__file__).parent.parent / "data" / "processed" / "mapa_predicciones.csv"
    try:
        return pd.read_csv(path), None
    except FileNotFoundError:
        return None, str(path)
    
# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="UHI · XGBoost",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0E1117; }

.metric-card {
    background: linear-gradient(135deg, #1A1D27 0%, #1F2235 100%);
    border: 1px solid #2D3045;
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
}
.metric-label { font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: #6B7280; margin-bottom: 8px; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 30px; font-weight: 700; line-height: 1; }
.metric-sub   { font-size: 11px; margin-top: 6px; color: #6B7280; }

.class-card {
    background: #1A1D27;
    border: 1px solid #2D3045;
    border-radius: 10px;
    padding: 16px 18px;
}
.class-title { font-family: 'Space Mono', monospace; font-size: 13px; font-weight: 700; margin-bottom: 12px; }

.insight-box {
    background: #13151F;
    border-left: 3px solid #3498DB;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 13px;
    line-height: 1.6;
    color: #C8CDD8;
}
.insight-box.warn { border-left-color: #F39C12; }
.insight-box.info { border-left-color: #2ECC71; }

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 11px; letter-spacing: 3px; text-transform: uppercase;
    color: #4A5080; margin: 28px 0 14px; padding-bottom: 8px;
    border-bottom: 1px solid #2D3045;
}

.tag {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; font-family: 'Space Mono', monospace;
    margin-right: 6px;
}
.tag-blue   { background: #12243D; color: #3498DB; border: 1px solid #3498DB33; }

div[data-testid="stSidebar"] { background-color: #13151F; border-right: 1px solid #2D3045; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🌡️ UHI Risk Model")
    st.markdown(f"""
    <div style='font-size:12px;color:#8890A4;line-height:1.8;'>
    <b style='color:#C8CDD8;'>Modelo:</b> XGBoost<br>
    <b style='color:#C8CDD8;'>Estimadores:</b> 600<br>
    <b style='color:#C8CDD8;'>Features:</b> 9 índices + elevation + season<br>
    <b style='color:#C8CDD8;'>Fuente:</b> Sentinel-2 / MODIS<br>
    <b style='color:#C8CDD8;'>Target:</b> Riesgo UHI (binario)<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:12px;color:#8890A4;line-height:1.9;'>
    <b style='color:#C8CDD8;'>Total obs.:</b> {MODELO['n_total']:,}<br>
    <b style='color:#C8CDD8;'>Train:</b> {MODELO['n_train']:,}<br>
    <b style='color:#C8CDD8;'>Test:</b> {MODELO['n_test']:,}<br>
    <b style='color:#C8CDD8;'>Split:</b> 80 / 20<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    vista = st.radio(
        "Sección",
        [
            "📊 Resumen ejecutivo",
            "🔬 Análisis detallado",
            "🌿 Importancia de variables",
            "🗺️ Mapa de predicciones",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#4A5080;line-height:1.6;'>
    <b style='color:#6B7280;'>Features</b><br>
    MNDWI · NBR · NDBI · NDMI<br>
    NDVI · NDWI · NIR · SWIR1<br>
    SWIR2 · Elevation · Season
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("""
    <div style='padding:20px 0 8px;'>
        <h1 style='font-family:Space Mono,monospace;font-size:22px;margin:0;letter-spacing:1px;'>
            XGBoost · Riesgo UHI
        </h1>
        <p style='color:#6B7280;margin:6px 0 0;font-size:13px;'>
            Clasificación binaria · Índices espectrales satelitales → anomalía térmica nocturna
        </p>
    </div>
    """, unsafe_allow_html=True)
with col_h2:
    st.markdown("""
    <div style='padding-top:24px;text-align:right;'>
        <span class='tag tag-blue'>✅ XGB — 600 estimadores</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# VISTA 1 — RESUMEN EJECUTIVO
# ══════════════════════════════════════════════════════════════════════════════

if vista == "📊 Resumen ejecutivo":

    st.markdown('<div class="section-title">Métricas globales</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)

    kpis = [
        (k1, "ACCURACY",  f"{MODELO['accuracy']:.0%}",  "vs. 50% baseline aleatorio"),
        (k2, "F1 MACRO",  f"{MODELO['f1_macro']:.2f}",  "media entre clases"),
        (k3, "AUC-ROC",   f"{MODELO['auc_roc']:.4f}",   "área bajo la curva ROC"),
        (k4, "N TEST",    f"{MODELO['n_test']:,}",       "observaciones de evaluación"),
    ]

    for col, label, val, sub in kpis:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value" style="color:{ACCENT};">{val}</div>'
                f'<div class="metric-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown(
        '<div class="insight-box">'
        '🌡️ &nbsp;El modelo predice el riesgo UHI exclusivamente a partir de índices espectrales '
        'derivados de imágenes satelitales y la elevación del terreno, '
        '<b>sin ninguna variable de temperatura como input</b>. '
        'Con un AUC-ROC de 0.7459, el modelo tiene capacidad real de separar zonas de riesgo '
        'desde la firma espectral del suelo urbano.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Rendimiento por clase</div>', unsafe_allow_html=True)

    c_nr, c_r = st.columns(2)

    for col_ui, clase, datos_clase, color_clase, icono, comentario in [
        (c_nr, "no_riesgo", MODELO["clases"]["no_riesgo"], "#3498DB", "🟦",
         "Precision 0.68 · el modelo identifica correctamente las zonas sin riesgo en el 68% de los casos."),
        (c_r,  "riesgo",    MODELO["clases"]["riesgo"],    ACCENT,    "🟥",
         "Recall 0.73 · el modelo detecta 3 de cada 4 zonas con riesgo UHI real. Indicador clave en planificación urbana."),
    ]:
        with col_ui:
            p, r, f = datos_clase["precision"], datos_clase["recall"], datos_clase["f1"]
            st.markdown(
                f'<div class="class-card">'
                f'<div class="class-title" style="color:{color_clase};">{icono} &nbsp;<code>{clase}</code></div>',
                unsafe_allow_html=True,
            )
            fig_b, ax_b = plt.subplots(figsize=(4, 1.8))
            metricas = ["Precision", "Recall", "F1-score"]
            valores  = [p, r, f]
            bars = ax_b.barh(metricas, valores, color=color_clase, alpha=0.8, height=0.55, zorder=3)
            for bar, val in zip(bars, valores):
                ax_b.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                          f"{val:.2f}", va="center", fontsize=10, color=color_clase, fontweight="bold")
            ax_b.set_xlim(0, 1.15)
            ax_b.axvline(0.5, color=BORDER, linewidth=0.8, linestyle="--", zorder=0)
            ax_b.grid(axis="x", alpha=0.2, zorder=0)
            fig_b.tight_layout(pad=0.4)
            st.pyplot(fig_b, use_container_width=True)
            plt.close(fig_b)
            st.markdown(f'<div class="metric-sub" style="margin-top:6px;font-size:12px;color:{TEXT_DIM};">{comentario}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Hiperparámetros
    st.markdown('<div class="section-title">Hiperparámetros óptimos</div>', unsafe_allow_html=True)
    hp = MODELO["hiperparametros"]
    cols_hp = st.columns(4)
    for i, (param, val) in enumerate(hp.items()):
        with cols_hp[i % 4]:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{param}</div>'
                f'<div class="metric-value" style="color:{ACCENT};font-size:20px;">{val}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# VISTA 2 — ANÁLISIS DETALLADO
# ══════════════════════════════════════════════════════════════════════════════

elif vista == "🔬 Análisis detallado":

    st.markdown('<div class="section-title">Matriz de Confusión</div>', unsafe_allow_html=True)

    cm = np.array(MODELO["confusion_matrix"])
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    total = tn + fp + fn + tp

    col_cm, col_stats = st.columns([1.1, 1])

    with col_cm:
        fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.5))
        cmap = sns.light_palette(ACCENT, as_cmap=True)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap=cmap, ax=ax_cm,
            linewidths=2, linecolor=BG_MAIN,
            annot_kws={"size": 22, "weight": "bold", "color": TEXT_BRT},
            cbar_kws={"shrink": 0.8}
        )
        ax_cm.set_xlabel("Predicho", fontsize=11, labelpad=10)
        ax_cm.set_ylabel("Real", fontsize=11, labelpad=10)
        ax_cm.set_xticklabels(["no_riesgo", "riesgo"], fontsize=10)
        ax_cm.set_yticklabels(["no_riesgo", "riesgo"], fontsize=10, rotation=0)
        etiquetas = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax_cm.text(j+0.5, i+0.78, etiquetas[i][j],
                           ha='center', fontsize=9, color=TEXT_DIM)
        ax_cm.set_title(f"Predicciones en test (N = {total:,})", fontsize=11, pad=14, color=TEXT_MED)
        fig_cm.tight_layout()
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)

    with col_stats:
        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
        for label, val, color_val, desc in [
            ("Verdaderos Negativos (TN)", tn, ACCENT, "no_riesgo → no_riesgo ✓"),
            ("Verdaderos Positivos (TP)", tp, ACCENT, "riesgo → riesgo ✓"),
            ("Falsos Positivos (FP)",     fp, RED,    "no_riesgo → riesgo ✗"),
            ("Falsos Negativos (FN)",     fn, RED,    "riesgo → no_riesgo ✗"),
        ]:
            st.markdown(
                f'<div style="background:#1A1D27;border:1px solid #2D3045;border-radius:8px;'
                f'padding:10px 14px;margin-bottom:8px;display:flex;align-items:center;gap:12px;">'
                f'<div style="font-family:Space Mono,monospace;font-size:22px;font-weight:700;'
                f'color:{color_val};min-width:60px;">{val:,}</div>'
                f'<div><div style="font-size:12px;color:#C8CDD8;">{label}</div>'
                f'<div style="font-size:11px;color:#6B7280;">{desc}</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div style="font-size:11px;color:#4A5080;text-align:right;margin-top:4px;">'
            f'Total: {total:,} obs. · Correctos: {(tn+tp):,} ({(tn+tp)/total:.1%})</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Reporte de clasificación</div>', unsafe_allow_html=True)

    df_rep = pd.DataFrame({
        "Clase":     ["no_riesgo", "riesgo", "—", "macro avg"],
        "Precision": [0.68, 0.68, "—", 0.68],
        "Recall":    [0.63, 0.73, "—", 0.68],
        "F1-score":  [0.65, 0.71, "—", 0.68],
        "Support":   [f"{tn+fp:,}", f"{fn+tp:,}", "—", f"{total:,}"],
    })

    def style_report(row):
        styles = [""] * len(row)
        if row["Clase"] == "riesgo":
            styles = [f"color: {ACCENT}; font-weight: 600"] * len(row)
        elif row["Clase"] == "macro avg":
            styles = ["color: #8890A4; font-style: italic"] * len(row)
        return styles

    st.dataframe(
        df_rep.style.apply(style_report, axis=1),
        use_container_width=True,
        hide_index=True,
        height=175,
    )

    st.markdown(
        '<div class="insight-box warn">'
        '⚖️ &nbsp;<b>FP vs FN en planificación urbana:</b> Los falsos positivos '
        '(zonas predichas como riesgo que no lo son) implican intervenciones preventivas '
        'innecesarias — un coste asumible. Los falsos negativos (zonas de riesgo '
        'no detectadas) representan intervenciones omitidas, potencialmente más dañinas. '
        'El modelo prioriza correctamente el recall en la clase <code>riesgo</code> (0.73).'
        '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# VISTA 3 — IMPORTANCIA DE VARIABLES
# ══════════════════════════════════════════════════════════════════════════════

elif vista == "🌿 Importancia de variables":

    st.markdown('<div class="section-title">Importancia por índice espectral</div>', unsafe_allow_html=True)

    fi = dict(sorted(MODELO["feature_importances"].items(), key=lambda x: x[1], reverse=True))
    nombres = list(fi.keys())
    valores  = list(fi.values())
    colores_fi = [ACCENT if v >= 0.20 else ("#2ECC71" if v >= 0.10 else "#8890A4") for v in valores]

    fig_fi, ax_fi = plt.subplots(figsize=(8, 5))
    bars_fi = ax_fi.barh(
        nombres[::-1], valores[::-1],
        color=colores_fi[::-1], alpha=0.85, zorder=3, height=0.6
    )
    for bar, val, c in zip(bars_fi, valores[::-1], colores_fi[::-1]):
        ax_fi.text(
            bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            f"{val:.0%}", va="center", fontsize=10, color=c, fontweight="bold"
        )
    ax_fi.set_xlabel("Importancia relativa (Gini)")
    ax_fi.set_xlim(0, max(valores) * 1.25)
    ax_fi.grid(axis="x", alpha=0.25, zorder=0)
    ax_fi.set_title("Feature importances · XGBoost", fontsize=11, pad=12, color=TEXT_MED)
    fig_fi.tight_layout()
    st.pyplot(fig_fi, use_container_width=True)
    plt.close(fig_fi)

    st.markdown('<div class="section-title">Interpretación física</div>', unsafe_allow_html=True)

    interpretaciones = {
        "ndvi":     ("NDVI",      ACCENT,    "Normalized Difference Vegetation Index",
                     "Variable más relevante (~29%). Actúa como proxy de refrigeración urbana: "
                     "la vegetación densa mitiga el calor mediante evapotranspiración y sombreado. "
                     "Zonas con bajo NDVI presentan mayor riesgo UHI."),
        "nbr":      ("NBR",       "#2ECC71", "Normalized Burn Ratio",
                     "Alta importancia (~12%). Sensible a la cobertura vegetal y la humedad del suelo. "
                     "Complementa al NDVI distinguiendo vegetación sana de suelo degradado."),
        "ndwi":     ("NDWI",      "#2ECC71", "Normalized Difference Water Index",
                     "~12% de importancia. Detecta presencia de agua superficial. "
                     "Las masas de agua tienen efecto regulador térmico sobre las zonas circundantes."),
        "swir2":    ("SWIR2",     "#8890A4", "Short-Wave Infrared Band 2",
                     "~11%. Sensible a la humedad del suelo y materiales de construcción. "
                     "Superficies impermeables y secas retienen más calor que suelos húmedos."),
        "ndmi":     ("NDMI",      "#8890A4", "Normalized Difference Moisture Index",
                     "~8%. Mide el contenido de humedad en la vegetación y el suelo. "
                     "Zonas con baja humedad presentan mayor vulnerabilidad al UHI."),
        "swir1":    ("SWIR1",     "#8890A4", "Short-Wave Infrared Band 1",
                     "~8%. Relacionado con la impermeabilización del suelo urbano. "
                     "Complementa a SWIR2 en la caracterización de superficies construidas."),
        "ndbi":     ("NDBI",      "#8890A4", "Normalized Difference Built-up Index",
                     "~8%. Índice de urbanización: distingue superficies construidas de vegetación. "
                     "Alta correlación con islas de calor en entornos densamente edificados."),
        "mndwi":    ("MNDWI",     "#8890A4", "Modified Normalized Difference Water Index",
                     "~7%. Versión mejorada del NDWI usando SWIR en lugar de NIR. "
                     "Más preciso para detectar agua urbana como fuentes reguladoras de temperatura."),
        "nir":      ("NIR",       "#8890A4", "Near-Infrared Reflectance",
                     "~6%. Complementa la distinción entre vegetación activa y suelos desnudos "
                     "o superficies construidas. Base de cálculo de NDVI y otros índices."),
        "elevation":("Elevation", "#8890A4", "Elevación del terreno (metros)",
                     "Contexto topográfico del punto. La altitud influye en la temperatura "
                     "base del entorno y modula el efecto UHI en zonas con relieve significativo."),
    }

    cols_fi = st.columns(3)
    for i, (key, (sigla, col_fi, nombre_completo, texto)) in enumerate(interpretaciones.items()):
        with cols_fi[i % 3]:
            imp_val = MODELO["feature_importances"].get(key, 0)
            st.markdown(
                f'<div style="background:#1A1D27;border:1px solid #2D3045;border-radius:10px;'
                f'padding:14px 16px;margin-bottom:12px;">'
                f'<div style="font-family:Space Mono,monospace;font-size:14px;font-weight:700;'
                f'color:{col_fi};margin-bottom:4px;">{sigla}</div>'
                f'<div style="font-size:10px;color:#6B7280;margin-bottom:10px;">{nombre_completo}</div>'
                f'<div style="font-size:12px;color:#C8CDD8;line-height:1.55;">{texto}</div>'
                f'<div style="margin-top:10px;font-family:Space Mono,monospace;font-size:14px;'
                f'font-weight:700;color:{col_fi};">{imp_val:.0%}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="insight-box info">'
        '🌿 &nbsp;<b>NDVI domina con ~29%</b> de la importancia total, confirmando que la cobertura '
        'vegetal es el principal predictor del riesgo UHI desde la firma espectral. '
        'El resto de índices contribuyen de forma más equilibrada (6-12%), indicando que el modelo '
        'integra múltiples dimensiones del entorno físico urbano.'
        '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# VISTA 4 — MAPA DE PREDICCIONES
# ══════════════════════════════════════════════════════════════════════════════

elif vista == "🗺️ Mapa de predicciones":

    df_mapa, error_path = cargar_mapa_data()

    if df_mapa is None:
        st.error(
            f"No se encontró `{error_path}`.\n\n"
            "**Genera el archivo primero ejecutando:**\n"
            "```bash\npython src/exportar_mapa.py\n```"
        )
        st.stop()

    st.markdown('<div class="section-title">Configuración del mapa</div>', unsafe_allow_html=True)

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

    with col_ctrl1:
        capa = st.selectbox(
            "Capa a visualizar",
            [
                "🔴🟢 Errores del modelo (TP/TN/FP/FN)",
                "🎯 Predicción (riesgo / no_riesgo)",
                "🌡️ Mapa de calor (probabilidad de riesgo)",
            ],
        )

    with col_ctrl2:
        muestra = st.slider(
            "Puntos a mostrar",
            min_value=1000,
            max_value=min(len(df_mapa), 20000),
            value=15000,
            step=1000,
        )

    with col_ctrl3:
        if "TP/TN/FP/FN" in capa:
            filtro_resultado = st.multiselect(
                "Filtrar por resultado",
                ["TP", "TN", "FP", "FN"],
                default=["TP", "TN", "FP", "FN"],
            )
        else:
            filtro_resultado = None
            st.markdown('<div style="height:58px;"></div>', unsafe_allow_html=True)

    # Estadísticas rápidas
    conteos = df_mapa["resultado"].value_counts()
    total_v = len(df_mapa)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    for col_s, key, label in [
        (col_s1, "TP", "Verdaderos +"),
        (col_s2, "TN", "Verdaderos −"),
        (col_s3, "FP", "Falsas alarmas"),
        (col_s4, "FN", "Riesgo omitido"),
    ]:
        n = conteos.get(key, 0)
        with col_s:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value" style="color:{COLORES_MAPA[key]};font-size:22px;">{n:,}</div>'
                f'<div class="metric-sub">{n/total_v:.1%} del total</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    # Preparar datos
    df_vis = df_mapa.dropna(subset=["latitude", "longitude"]).copy()

    if filtro_resultado and "TP/TN/FP/FN" in capa:
        df_vis = df_vis[df_vis["resultado"].isin(filtro_resultado)]

    if len(df_vis) > muestra:
        df_vis = df_vis.sample(n=muestra, random_state=42)

    # Construir mapa
    centro_lat = df_vis["latitude"].mean()
    centro_lon = df_vis["longitude"].mean()

    m = folium.Map(
        location=[centro_lat, centro_lon],
        zoom_start=12,
        tiles="CartoDB dark_matter",
    )

    if "calor" in capa:
        heat_data = df_vis[["latitude", "longitude", "pred_proba"]].values.tolist()
        HeatMap(
        heat_data,
        min_opacity=0.4,
        max_zoom=18,
        radius=40,        # era 25
        blur=35,          # era 20
        gradient={0.0: "#3498DB", 0.5: "#E74C3C"},
    ).add_to(m)
        leyenda_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:#1A1D27;border:1px solid #2D3045;border-radius:8px;
                    padding:12px 16px;font-family:monospace;font-size:12px;color:#C8CDD8;">
            <b style="color:#E0E4EF;">Probabilidad de riesgo UHI</b><br><br>
            <span style="color:#3498DB;">●</span> No riesgo — prob. &lt; 0.5<br>
            <span style="color:#E74C3C;">●</span> Riesgo — prob. ≥ 0.5
        </div>"""

    elif "Predicción" in capa:
        cluster = MarkerCluster(
            options={"maxClusterRadius": 40, "disableClusteringAtZoom": 14}
        ).add_to(m)
        for _, row in df_vis.iterrows():
            c = COLORES_PRED.get(row["pred_label"], "#AAAAAA")
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=4, color=c, fill=True, fill_color=c, fill_opacity=0.75,
                popup=folium.Popup(
                    f"<b>Pred:</b> {row['pred_label']}<br>"
                    f"<b>Prob:</b> {row['pred_proba']:.2f}<br>"
                    f"<b>Real:</b> {row['real_label']}<br>"
                    f"<b>NDVI:</b> {row.get('ndvi', 0):.3f} · "
                    f"<b>NDBI:</b> {row.get('ndbi', 0):.3f}",
                    max_width=200,
                ),
            ).add_to(cluster)
        leyenda_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:#1A1D27;border:1px solid #2D3045;border-radius:8px;
                    padding:12px 16px;font-family:monospace;font-size:12px;color:#C8CDD8;">
            <b style="color:#E0E4EF;">Predicción del modelo</b><br><br>
            <span style="color:#E74C3C;">●</span> riesgo<br>
            <span style="color:#3498DB;">●</span> no_riesgo
        </div>"""

    else:
        cluster = MarkerCluster(
            options={"maxClusterRadius": 40, "disableClusteringAtZoom": 14}
        ).add_to(m)
        for _, row in df_vis.iterrows():
            res = row["resultado"]
            c   = COLORES_MAPA.get(res, "#AAAAAA")
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=4, color=c, fill=True, fill_color=c, fill_opacity=0.75,
                popup=folium.Popup(
                    f"<b>Resultado:</b> {res}<br>"
                    f"<b>Real:</b> {row['real_label']}<br>"
                    f"<b>Pred:</b> {row['pred_label']} ({row['pred_proba']:.2f})<br>"
                    f"<b>NDVI:</b> {row.get('ndvi', 0):.3f} · "
                    f"<b>NDBI:</b> {row.get('ndbi', 0):.3f}",
                    max_width=200,
                ),
            ).add_to(cluster)
        leyenda_html = f"""
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:#1A1D27;border:1px solid #2D3045;border-radius:8px;
                    padding:12px 16px;font-family:monospace;font-size:12px;color:#C8CDD8;">
            <b style="color:#E0E4EF;">Resultado del modelo</b><br><br>
            <span style="color:{COLORES_MAPA['TP']};">●</span> TP — riesgo detectado ✓<br>
            <span style="color:{COLORES_MAPA['TN']};">●</span> TN — no riesgo detectado ✓<br>
            <span style="color:{COLORES_MAPA['FP']};">●</span> FP — falsa alarma ✗<br>
            <span style="color:{COLORES_MAPA['FN']};">●</span> FN — riesgo omitido ✗
        </div>"""

    m.get_root().html.add_child(folium.Element(leyenda_html))

    st.markdown('<div class="section-title">Mapa interactivo · Barcelona</div>', unsafe_allow_html=True)
    st_folium(m, width="100%", height=560, returned_objects=[])

    st.markdown(
        f'<div class="insight-box info" style="margin-top:12px;">'
        f'🗺️ Mostrando <b>{len(df_vis):,}</b> de <b>{len(df_mapa):,}</b> puntos. '
        f'Zoom para desagrupar · Clic en un punto para ver sus índices espectrales · '
        f'Los FN (<span style="color:{COLORES_MAPA["FN"]};">rojo</span>) son las zonas más '
        f'críticas: riesgo real no detectado por el modelo.'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    f'<p style="text-align:center;color:#4A5080;font-size:11px;">'
    f'XGBoost · 600 estimadores · 9 índices espectrales + elevation + season · '
    f'{MODELO["n_total"]:,} obs. · Accuracy 68% · AUC-ROC 0.7459'
    f'</p>',
    unsafe_allow_html=True,
)