import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import joblib
from pathlib import Path
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

BG_MAIN  = "#0E1117"
BG_CARD  = "#1A1D27"
BG_CARD2 = "#1F2235"
BORDER   = "#2D3045"
TEXT_DIM = "#8890A4"
TEXT_MED = "#C8CDD8"
TEXT_BRT = "#E0E4EF"
ACCENT   = "#3498DB"
RED      = "#E74C3C"

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
# CARGA DE DATOS Y MODELO
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def cargar_modelo():
    path = Path(__file__).parent.parent / "models" / "xgboost_uhi_no_geo_optimized.sav"
    return joblib.load(path)

@st.cache_data
def cargar_mapa_data():
    path = Path(__file__).parent.parent / "data" / "processed" / "mapa_predicciones.csv"
    try:
        return pd.read_csv(path), None
    except FileNotFoundError:
        return None, str(path)

@st.cache_data
def cargar_metricas():
    modelo = cargar_modelo()
    df     = pd.read_csv(
        Path(__file__).parent.parent / "data" / "processed" / "dataset_modeling.csv"
    )

    features = [
        'elevation', 'mndwi', 'nbr', 'ndbi', 'ndmi',
        'ndvi', 'ndwi', 'nir', 'swir1', 'swir2', 'season'
    ]
    X = pd.get_dummies(df[features], columns=['season'], drop_first=True)
    y = df['uhi_risk']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred  = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    auc    = roc_auc_score(y_test, y_proba)
    fi     = pd.Series(
        modelo.feature_importances_,
        index=modelo.feature_names_in_
    ).sort_values(ascending=False)

    return {
        "accuracy":  report["accuracy"],
        "f1_macro":  report["macro avg"]["f1-score"],
        "auc_roc":   auc,
        "clases": {
            "no_riesgo": {
                "precision": report["0"]["precision"],
                "recall":    report["0"]["recall"],
                "f1":        report["0"]["f1-score"],
            },
            "riesgo": {
                "precision": report["1"]["precision"],
                "recall":    report["1"]["recall"],
                "f1":        report["1"]["f1-score"],
            },
        },
        "confusion_matrix": cm.tolist(),
        "feature_importances": fi.to_dict(),
        "n_total":  len(df),
        "n_train":  len(df) - len(X_test),
        "n_test":   len(X_test),
        "n_arboles": modelo.n_estimators,
        "hiperparametros": {
            "n_estimators":     modelo.n_estimators,
            "learning_rate":    modelo.learning_rate,
            "max_depth":        modelo.max_depth,
            "subsample":        modelo.subsample,
            "colsample_bytree": modelo.colsample_bytree,
            "reg_alpha":        modelo.reg_alpha,
            "reg_lambda":       modelo.reg_lambda,
            "gamma":            modelo.gamma,
        }
    }

# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="UHI  · XGBoost",
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
.tag-blue { background: #12243D; color: #3498DB; border: 1px solid #3498DB33; }

div[data-testid="stSidebar"] { background-color: #13151F; border-right: 1px solid #2D3045; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CARGAR MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

metricas = cargar_metricas()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🌡️ UHI Risk Model")
    st.markdown(f"""
    <div style='font-size:12px;color:#8890A4;line-height:1.8;'>
    <b style='color:#C8CDD8;'>Modelo:</b> XGBoost<br>
    <b style='color:#C8CDD8;'>Estimadores:</b> {metricas['n_arboles']}<br>
    <b style='color:#C8CDD8;'>Features:</b> 9 índices + elevation + season<br>
    <b style='color:#C8CDD8;'>Fuente:</b> Sentinel-2 / MODIS<br>
    <b style='color:#C8CDD8;'>Target:</b> Riesgo UHI (binario)<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:12px;color:#8890A4;line-height:1.9;'>
    <b style='color:#C8CDD8;'>Total obs.:</b> {metricas['n_total']:,}<br>
    <b style='color:#C8CDD8;'>Train:</b> {metricas['n_train']:,}<br>
    <b style='color:#C8CDD8;'>Test:</b> {metricas['n_test']:,}<br>
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
        "🔮 Predicción",
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
    </div>
    """, unsafe_allow_html=True)
with col_h2:
    st.markdown(f"""
    <div style='padding-top:24px;text-align:right;'>
        <span class='tag tag-blue'>✅ XGB — {metricas['n_arboles']} estimadores</span>
    </div>
    """, unsafe_allow_html=True)

# Párrafo introductorio
st.markdown("""
<div style='background:#13151F;border-left:3px solid #3498DB;border-radius:0 8px 8px 0;
padding:18px 22px;margin:16px 0 8px;font-size:13px;line-height:1.8;color:#C8CDD8;'>

Las <b style='color:#E0E4EF;'>islas de calor urbanas</b> (UHI, por sus siglas en inglés) son zonas 
de las ciudades que acumulan y retienen más calor que su entorno, especialmente durante la noche. 
Este fenómeno está directamente relacionado con la densidad de edificios, la falta de vegetación 
y la impermeabilización del suelo urbano.
<br><br>
Este modelo predice el riesgo UHI en Barcelona a partir de imágenes satelitales, 
<b style='color:#E0E4EF;'>sin utilizar datos de temperatura directos</b>. Analiza la firma 
espectral del suelo — cómo refleja y absorbe la luz en distintas longitudes de onda — para 
identificar zonas con características físicas asociadas a mayor retención de calor nocturno.
<br><br>
Comprender y anticipar estas zonas permite a urbanistas y planificadores priorizar intervenciones 
como la creación de zonas verdes, el uso de materiales reflectantes o la mejora del drenaje urbano, 
con el objetivo de <b style='color:#E0E4EF;'>reducir el impacto del calor en la salud y el bienestar 
de la ciudadanía</b>.

</div>
""", unsafe_allow_html=True)

# Desplegable de variables
with st.expander("🔭 Variables de análisis"):
    st.markdown("""
    <table style='width:100%;font-size:13px;color:#C8CDD8;border-collapse:collapse;'>
        <thead>
            <tr style='border-bottom:1px solid #2D3045;'>
                <th style='text-align:left;padding:8px 12px;color:#E0E4EF;font-family:Space Mono,monospace;'>Variable</th>
                <th style='text-align:left;padding:8px 12px;color:#E0E4EF;font-family:Space Mono,monospace;'>En palabras simples</th>
            </tr>
        </thead>
        <tbody>
            <tr style='border-bottom:1px solid #1F2235;'>
                <td style='padding:8px 12px;'><b style='color:#3498DB;'>NDVI</b></td>
                <td style='padding:8px 12px;'>¿Cuánta vegetación hay? Parques y jardines tienen valores altos. Asfalto y edificios, valores bajos.</td>
            </tr>
            <tr style='border-bottom:1px solid #1F2235;'>
                <td style='padding:8px 12px;'><b style='color:#3498DB;'>NDBI</b></td>
                <td style='padding:8px 12px;'>¿Qué tan urbanizada está la zona? Más edificios y superficies construidas = valor más alto.</td>
            </tr>
            <tr style='border-bottom:1px solid #1F2235;'>
                <td style='padding:8px 12px;'><b style='color:#3498DB;'>MNDWI</b></td>
                <td style='padding:8px 12px;'>¿Hay agua cerca? Ríos, fuentes o zonas húmedas tienen valores altos.</td>
            </tr>
            <tr style='border-bottom:1px solid #1F2235;'>
                <td style='padding:8px 12px;'><b style='color:#3498DB;'>NBR</b></td>
                <td style='padding:8px 12px;'>¿Está la vegetación sana o degradada? Complementa al NDVI detectando suelo quemado o seco.</td>
            </tr>
            <tr style='border-bottom:1px solid #1F2235;'>
                <td style='padding:8px 12px;'><b style='color:#3498DB;'>NDMI</b></td>
                <td style='padding:8px 12px;'>¿Qué tan húmedo está el suelo y la vegetación? Zonas secas tienen valores bajos.</td>
            </tr>
            <tr style='border-bottom:1px solid #1F2235;'>
                <td style='padding:8px 12px;'><b style='color:#3498DB;'>NDWI</b></td>
                <td style='padding:8px 12px;'>Similar a MNDWI pero menos preciso en entornos urbanos. Detecta humedad general.</td>
            </tr>
            <tr style='border-bottom:1px solid #1F2235;'>
                <td style='padding:8px 12px;'><b style='color:#3498DB;'>NIR</b></td>
                <td style='padding:8px 12px;'>Reflejo infrarrojo cercano. La vegetación sana refleja mucho; el asfalto, poco.</td>
            </tr>
            <tr style='border-bottom:1px solid #1F2235;'>
                <td style='padding:8px 12px;'><b style='color:#3498DB;'>SWIR1 / SWIR2</b></td>
                <td style='padding:8px 12px;'>Infrarrojo de onda corta. Sensible a materiales de construcción y humedad del suelo.</td>
            </tr>
            <tr style='border-bottom:1px solid #1F2235;'>
                <td style='padding:8px 12px;'><b style='color:#3498DB;'>Elevation</b></td>
                <td style='padding:8px 12px;'>¿A qué altura está el punto? Las zonas más altas tienden a ser más frescas.</td>
            </tr>
            <tr>
                <td style='padding:8px 12px;'><b style='color:#3498DB;'>Season</b></td>
                <td style='padding:8px 12px;'>¿En qué estación del año se tomó la imagen? El riesgo UHI varía entre verano e invierno.</td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# VISTA 1 — RESUMEN EJECUTIVO
# ══════════════════════════════════════════════════════════════════════════════

if vista == "📊 Resumen ejecutivo":

    st.markdown('<div class="section-title">Métricas globales</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)

    kpis = [
        (k1, "ACCURACY", f"{metricas['accuracy']:.0%}",  "vs. 50% baseline aleatorio"),
        (k2, "F1 MACRO",  f"{metricas['f1_macro']:.2f}", "media entre clases"),
        (k3, "AUC-ROC",   f"{metricas['auc_roc']:.4f}",  "área bajo la curva ROC"),
        (k4, "N TEST",    f"{metricas['n_test']:,}",      "observaciones de evaluación"),
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
        f'Con un AUC-ROC de {metricas["auc_roc"]:.4f}, el modelo tiene capacidad real de separar '
        'zonas de riesgo desde la firma espectral del suelo urbano.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Rendimiento por clase</div>', unsafe_allow_html=True)

    c_nr, c_r = st.columns(2)

    for col_ui, clase, datos_clase, color_clase, icono, comentario in [
        (c_nr, "no_riesgo", metricas["clases"]["no_riesgo"], "#3498DB", "🟦",
         f"Precision {metricas['clases']['no_riesgo']['precision']:.2f} · el modelo identifica correctamente las zonas sin riesgo."),
        (c_r,  "riesgo",    metricas["clases"]["riesgo"],    ACCENT,    "🟥",
         f"Recall {metricas['clases']['riesgo']['recall']:.2f} · indicador clave en planificación urbana."),
    ]:
        with col_ui:
            p = datos_clase["precision"]
            r = datos_clase["recall"]
            f = datos_clase["f1"]
            st.markdown(
                f'<div class="class-card">'
                f'<div class="class-title" style="color:{color_clase};">{icono} &nbsp;<code>{clase}</code></div>',
                unsafe_allow_html=True,
            )
            fig_b, ax_b = plt.subplots(figsize=(4, 1.8))
            bars = ax_b.barh(
                ["Precision", "Recall", "F1-score"], [p, r, f],
                color=color_clase, alpha=0.8, height=0.55, zorder=3
            )
            for bar, val in zip(bars, [p, r, f]):
                ax_b.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                          f"{val:.2f}", va="center", fontsize=10,
                          color=color_clase, fontweight="bold")
            ax_b.set_xlim(0, 1.15)
            ax_b.axvline(0.5, color=BORDER, linewidth=0.8, linestyle="--", zorder=0)
            ax_b.grid(axis="x", alpha=0.2, zorder=0)
            fig_b.tight_layout(pad=0.4)
            st.pyplot(fig_b, use_container_width=True)
            plt.close(fig_b)
            st.markdown(
                f'<div class="metric-sub" style="margin-top:6px;font-size:12px;color:{TEXT_DIM};">'
                f'{comentario}</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

    # Hiperparámetros
    st.markdown('<div class="section-title">Hiperparámetros óptimos</div>', unsafe_allow_html=True)
    cols_hp = st.columns(4)
    for i, (param, val) in enumerate(metricas["hiperparametros"].items()):
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

    cm = np.array(metricas["confusion_matrix"])
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
        ax_cm.set_title(f"Predicciones en test (N = {total:,})",
                        fontsize=11, pad=14, color=TEXT_MED)
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
        "Precision": [
            round(metricas["clases"]["no_riesgo"]["precision"], 2),
            round(metricas["clases"]["riesgo"]["precision"], 2),
            "—",
            round(metricas["f1_macro"], 2)
        ],
        "Recall": [
            round(metricas["clases"]["no_riesgo"]["recall"], 2),
            round(metricas["clases"]["riesgo"]["recall"], 2),
            "—",
            round(metricas["f1_macro"], 2)
        ],
        "F1-score": [
            round(metricas["clases"]["no_riesgo"]["f1"], 2),
            round(metricas["clases"]["riesgo"]["f1"], 2),
            "—",
            round(metricas["f1_macro"], 2)
        ],
        "Support": [f"{tn+fp:,}", f"{fn+tp:,}", "—", f"{total:,}"],
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
        f'El modelo prioriza correctamente el recall en la clase <code>riesgo</code> '
        f'({metricas["clases"]["riesgo"]["recall"]:.2f}).'
        '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# VISTA 3 — IMPORTANCIA DE VARIABLES
# ══════════════════════════════════════════════════════════════════════════════

elif vista == "🌿 Importancia de variables":

    st.markdown('<div class="section-title">Importancia por índice espectral</div>', unsafe_allow_html=True)

    fi = metricas["feature_importances"]
    nombres = list(fi.keys())
    valores  = list(fi.values())
    colores_fi = [ACCENT if v >= 0.15 else ("#2ECC71" if v >= 0.08 else "#8890A4") for v in valores]

    fig_fi, ax_fi = plt.subplots(figsize=(8, 5))
    bars_fi = ax_fi.barh(
        nombres[::-1], valores[::-1],
        color=colores_fi[::-1], alpha=0.85, zorder=3, height=0.6
    )
    for bar, val, c in zip(bars_fi, valores[::-1], colores_fi[::-1]):
        ax_fi.text(
            bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            f"{val:.1%}", va="center", fontsize=10, color=c, fontweight="bold"
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
        "elevation":     ("Elevation",    ACCENT,    "Elevación del terreno (metros)",
                          "Variable más relevante. La altitud influye en la temperatura base "
                          "del entorno y modula el efecto UHI en zonas con relieve significativo."),
        "season_summer": ("Season Summer","#2ECC71",  "Estación: verano",
                          "El verano es el período de mayor riesgo UHI — las temperaturas "
                          "nocturnas son más altas y la retención de calor urbano más pronunciada."),
        "season_winter": ("Season Winter","#2ECC71",  "Estación: invierno",
                          "El invierno representa el contraste térmico más claro respecto al verano, "
                          "ayudando al modelo a discriminar entre estaciones extremas."),
        "ndvi":          ("NDVI",         "#8890A4", "Normalized Difference Vegetation Index",
                          "La vegetación densa mitiga el calor mediante evapotranspiración y sombreado. "
                          "Zonas con bajo NDVI presentan mayor riesgo UHI."),
        "season_spring": ("Season Spring","#8890A4", "Estación: primavera",
                          "Estación de transición con señal moderada. Ayuda al modelo a "
                          "contextualizar las observaciones entre invierno y verano."),
        "nbr":           ("NBR",          "#8890A4", "Normalized Burn Ratio",
                          "Sensible a la cobertura vegetal y la humedad del suelo. "
                          "Complementa al NDVI distinguiendo vegetación sana de suelo degradado."),
        "ndwi":          ("NDWI",         "#8890A4", "Normalized Difference Water Index",
                          "Detecta presencia de agua superficial. Las masas de agua tienen "
                          "efecto regulador térmico sobre las zonas circundantes."),
        "mndwi":         ("MNDWI",        "#8890A4", "Modified Normalized Difference Water Index",
                          "Versión mejorada del NDWI usando SWIR en lugar de NIR. "
                          "Más preciso para detectar agua urbana como fuente reguladora de temperatura."),
        "swir2":         ("SWIR2",        "#8890A4", "Short-Wave Infrared Band 2",
                          "Sensible a la humedad del suelo y materiales de construcción. "
                          "Superficies impermeables y secas retienen más calor que suelos húmedos."),
        "nir":           ("NIR",          "#8890A4", "Near-Infrared Reflectance",
                          "Complementa la distinción entre vegetación activa y suelos desnudos "
                          "o superficies construidas. Base de cálculo de NDVI y otros índices."),
        "swir1":         ("SWIR1",        "#8890A4", "Short-Wave Infrared Band 1",
                          "Relacionado con la impermeabilización del suelo urbano. "
                          "Complementa a SWIR2 en la caracterización de superficies construidas."),
        "ndmi":          ("NDMI",         "#8890A4", "Normalized Difference Moisture Index",
                          "Mide el contenido de humedad en la vegetación y el suelo. "
                          "Zonas con baja humedad presentan mayor vulnerabilidad al UHI."),
        "ndbi":          ("NDBI",         "#8890A4", "Normalized Difference Built-up Index",
                          "Índice de urbanización: distingue superficies construidas de vegetación. "
                          "Alta correlación con islas de calor en entornos densamente edificados."),
    }

    cols_fi = st.columns(3)
    for i, (key, (sigla, col_fi, nombre_completo, texto)) in enumerate(interpretaciones.items()):
        with cols_fi[i % 3]:
            imp_val = fi.get(key, 0)
            st.markdown(
                f'<div style="background:#1A1D27;border:1px solid #2D3045;border-radius:10px;'
                f'padding:14px 16px;margin-bottom:12px;">'
                f'<div style="font-family:Space Mono,monospace;font-size:14px;font-weight:700;'
                f'color:{col_fi};margin-bottom:4px;">{sigla}</div>'
                f'<div style="font-size:10px;color:#6B7280;margin-bottom:10px;">{nombre_completo}</div>'
                f'<div style="font-size:12px;color:#C8CDD8;line-height:1.55;">{texto}</div>'
                f'<div style="margin-top:10px;font-family:Space Mono,monospace;font-size:14px;'
                f'font-weight:700;color:{col_fi};">{imp_val:.1%}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    top_feature = max(fi, key=fi.get)
    top_val     = fi[top_feature]
    st.markdown(
        f'<div class="insight-box info">'
        f'🌿 &nbsp;<b>{top_feature.upper()} lidera con {top_val:.1%}</b> de la importancia total. '
        f'El modelo integra múltiples dimensiones del entorno físico urbano — '
        f'topografía, estacionalidad y firma espectral — para detectar el riesgo UHI.'
        f'</div>',
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
            radius=40,
            blur=35,
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
    
# ══════════════════════════════════════════════════════════════════════════════
# VISTA 5 — PREDICCIÓN MANUAL
# ══════════════════════════════════════════════════════════════════════════════

elif vista == "🔮 Predicción":

    st.markdown('<div class="section-title">Introduce los valores del punto</div>', unsafe_allow_html=True)

    modelo = cargar_modelo()

    col_inputs1, col_inputs2 = st.columns(2)

    with col_inputs1:
        elevation = st.number_input("Elevation (m)",        min_value=0,    max_value=1000, value=50)
        ndvi      = st.slider("NDVI",   min_value=-1.0, max_value=1.0, value=0.3,  step=0.01)
        ndbi      = st.slider("NDBI",   min_value=-1.0, max_value=1.0, value=0.0,  step=0.01)
        ndmi      = st.slider("NDMI",   min_value=-1.0, max_value=1.0, value=0.0,  step=0.01)
        ndwi      = st.slider("NDWI",   min_value=-1.0, max_value=1.0, value=-0.3, step=0.01)
        nbr       = st.slider("NBR",    min_value=-1.0, max_value=1.0, value=0.2,  step=0.01)

    with col_inputs2:
        season    = st.selectbox("Season", ["autumn", "spring", "summer", "winter"])
        mndwi     = st.slider("MNDWI",  min_value=-1.0, max_value=1.0, value=-0.3, step=0.01)
        nir       = st.slider("NIR",    min_value=0.0,  max_value=1.0, value=0.25, step=0.01)
        swir1     = st.slider("SWIR1",  min_value=0.0,  max_value=1.0, value=0.15, step=0.01)
        swir2     = st.slider("SWIR2",  min_value=0.0,  max_value=1.0, value=0.10, step=0.01)

    st.markdown("")

    if st.button("🔍 Predecir", use_container_width=True):

        # Construir dataframe con los mismos features que el modelo
        input_data = pd.DataFrame([{
            "elevation": elevation,
            "mndwi":     mndwi,
            "nbr":       nbr,
            "ndbi":      ndbi,
            "ndmi":      ndmi,
            "ndvi":      ndvi,
            "ndwi":      ndwi,
            "nir":       nir,
            "swir1":     swir1,
            "swir2":     swir2,
            "season":    season,
        }])

        # One-hot encoding igual que en el entrenamiento
        input_enc = pd.get_dummies(input_data, columns=["season"], drop_first=True)

        # Asegurar que tiene todas las columnas que espera el modelo
        for col in modelo.feature_names_in_:
            if col not in input_enc.columns:
                input_enc[col] = 0

        input_enc = input_enc[modelo.feature_names_in_]

        # Predicción
        pred = modelo.predict(input_enc)[0]

        # Mostrar resultado
        if pred == 1:
            st.markdown(
                '<div style="background:#2D1A1A;border:2px solid #E74C3C;border-radius:12px;'
                'padding:24px;text-align:center;margin-top:16px;">'
                '<div style="font-family:Space Mono,monospace;font-size:32px;font-weight:700;'
                'color:#E74C3C;">⚠️ RIESGO UHI</div>'
                '<div style="color:#C8CDD8;margin-top:8px;font-size:14px;">'
                'El punto analizado presenta características espectrales asociadas a riesgo de isla de calor urbana.'
                '</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#1A2D1A;border:2px solid #2ECC71;border-radius:12px;'
                'padding:24px;text-align:center;margin-top:16px;">'
                '<div style="font-family:Space Mono,monospace;font-size:32px;font-weight:700;'
                'color:#2ECC71;">✅ SIN RIESGO UHI</div>'
                '<div style="color:#C8CDD8;margin-top:8px;font-size:14px;">'
                'El punto analizado no presenta características espectrales asociadas a riesgo de isla de calor urbana.'
                '</div></div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    f'<p style="text-align:center;color:#4A5080;font-size:11px;">'
    f'XGBoost · {metricas["n_arboles"]} estimadores · '
    f'9 índices espectrales + elevation + season · '
    f'{metricas["n_total"]:,} obs. · '
    f'Accuracy {metricas["accuracy"]:.0%} · '
    f'AUC-ROC {metricas["auc_roc"]:.4f}'
    f'</p>',
    unsafe_allow_html=True,
)