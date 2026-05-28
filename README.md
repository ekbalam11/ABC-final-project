# Proyecto ABC: Predicción de Riesgo de Isla de Calor Urbana (UHI) en Barcelona

Este proyecto desarrolla un **modelo de aprendizaje automático para predecir el riesgo de isla de calor urbana (UHI)** en Barcelona utilizando índices espectrales de satélites Sentinel-2 y datos de elevación del terreno. El modelo identifica zonas urbanas con riesgo de acumulación térmica nocturna sin utilizar datos de temperatura como entrada directa.

##  Descripción del Proyecto

Las **islas de calor urbanas (UHI)** son zonas dentro de las ciudades que retienen más calor que su entorno, especialmente durante la noche. Este fenómeno está directamente asociado a:
- Alta densidad de construcción
- Falta de cobertura vegetal
- Impermeabilización del suelo urbano
- Reducción de masa de agua

Este proyecto entrena un modelo XGBoost que predice el **riesgo UHI** a partir de la **firma espectral del terreno** — cómo los materiales urbanos reflejan y absorben luz en diferentes longitudes de onda — permitiendo a urbanistas y planificadores identificar y priorizar zonas para intervención.

##  Estructura del Repositorio

```
ABC-final-project/
├── notebooks/                          # Análisis y desarrollo del modelo
│   ├── 01_build_dataset.ipynb          # Construcción del conjunto de datos
│   ├── 02_eda.ipynb                    # Análisis exploratorio de datos
│   ├── 03_SVM.ipynb                    # Modelo: Support Vector Machines
│   ├── 04_RandomForest.ipynb           # Modelo: Random Forest
│   └── 05_XGB.ipynb                    # Modelo: XGBoost (modelo final)
│
├── src/                                # Código de producción
│   ├── app.py                          # Aplicación Streamlit interactiva
│   └── exportar_mapa.py                # Generación de mapas de predicciones
│
├── data/
│   ├── raw/                            # Datos sin procesar
│   │   ├── sentinel/                   # Índices espectrales Sentinel-2
│   │   ├── modis/                      # Datos de temperatura MODIS
│   │   └── elevation/                  # Elevación del terreno (SRTM)
│   └── processed/                      # Datos procesados
│       ├── dataset_final.csv           # Dataset unificado
│       ├── dataset_modeling.csv        # Dataset listo para modelado
│       └── mapa_predicciones.csv       # Predicciones geoespaciales
│
├── models/                             # Modelos entrenados
│   └── xgboost_uhi_no_geo_optimized.sav
│
├── requirements.txt                    # Dependencias de Python
└── README.md                           # Este archivo
```

##  Flujo de Trabajo

###  Construcción del Dataset (`01_build_dataset.ipynb`)
- Carga de imágenes Sentinel-2 (índices espectrales)
- Carga de datos MODIS (temperatura superficial)
- Carga de elevación SRTM 30m
- Merge por punto geográfico y fecha

###  Análisis Exploratorio (`02_eda.ipynb`)
- Distribuciones de índices espectrales
- Correlaciones con variables objetivo
- Identificación de valores atípicos
- Patrones espaciales y temporales

###  Modelado (`03_SVM.ipynb`, `04_RandomForest.ipynb`, `05_XGB.ipynb`)
- Ingeniería de features
- Entrenamiento de tres modelos comparativos
- Selección del modelo XGBoost como óptimo
- Validación y optimización de hiperparámetros

###  Despliegue (`src/app.py`)
- Aplicación Streamlit interactiva
- Visualización de métricas de rendimiento
- Mapas interactivos de predicciones
- Predicciones en tiempo real

##  Variables del Dataset

### Variable Objetivo
| Variable | Descripción |
|---|---|
| `uhi_risk` | Clasificación binaria (0=sin riesgo, 1=riesgo) derivada de anomalías de temperatura |

### Datos Fuente (MODIS)
| Variable | Descripción |
|---|---|
| `lst_day_c` | Temperatura superficial diurna (°C) |
| `lst_night_c` | Temperatura superficial nocturna (°C) |
| `lst_day_c_anomaly` | Anomalía térmica diurna respecto a la media mensual |
| `lst_night_c_anomaly` | Anomalía térmica nocturna respecto a la media mensual |

**Nota:** El `uhi_risk` se calcula a partir de combinaciones de estas anomalías térmicas, generando una clasificación binaria utilizada como objetivo de entrenamiento.

### Índices Espectrales (Sentinel-2)

####  Vegetación
- **NDVI** - Normalized Difference Vegetation Index: densidad y vigor vegetal
- **EVI** - Enhanced Vegetation Index: vegetación en zonas de alta densidad
- **EVI2** - Enhanced Vegetation Index 2: versión simplificada sin banda azul
- **SAVI** - Soil-Adjusted Vegetation Index: vegetación con corrección de suelo
- **MSAVI** - Modified SAVI: versión mejorada con efecto mínimo del suelo
- **GNDVI** - Green NDVI: sensible al contenido de clorofila
- **NDRE** - Normalized Difference Red Edge: estrés vegetal y clorofila
- **CIRE** - Chlorophyll Index Red Edge: concentración de clorofila
- **GLI** - Green Leaf Index: vegetación verde activa
- **PSRI** - Plant Senescence Reflectance Index: senescencia vegetal

####  Urbano / Superficie Construida
- **NDBI** - Normalized Difference Built-up Index: superficies construidas
- **IBI** - Index-based Built-up Index: densidad urbana integrada
- **UI** - Urban Index: intensidad de urbanización
- **EBBI** - Enhanced Built-up and Bareness Index: construcción y suelo desnudo

####  Suelo Desnudo
- **BSI** - Bare Soil Index: suelo desnudo y superficies degradadas
- **NDTI** - Normalized Difference Tillage Index: residuos agrícolas

####  Agua y Humedad
- **MNDWI** - Modified Normalized Difference Water Index: láminas de agua
- **NDWI** - Normalized Difference Water Index: agua y humedad vegetal
- **NDMI** - Normalized Difference Moisture Index: humedad en vegetación y suelo
- **NDSI** - Normalized Difference Snow Index: nieve e hielo

####  Fuego y Áreas Quemadas
- **NBR** - Normalized Burn Ratio: áreas quemadas y severidad

####  Bandas Espectrales Directas
- **blue, green, red**: Bandas RGB visibles
- **nir**: Near-Infrared (infrarrojo cercano)
- **swir1, swir2**: Short-Wave Infrared (infrarrojo de onda corta)

### Variables Espaciales y Temporales
| Variable | Descripción |
|---|---|
| `elevation` | Altitud en metros (fuente: SRTM 30m) |
| `latitude, longitude` | Coordenadas geográficas |
| `point_id` | Identificador único del punto |
| `fecha` | Fecha de la imagen Sentinel-2 |
| `month` | Mes (1-12) |
| `year` | Año |
| `dayofyear` | Día del año (1-365) |
| `season` | Estación del año (spring, summer, autumn, winter) |

##  Modelo XGBoost

El modelo final XGBoost utiliza **11 features principales**:
```
elevation · mndwi · nbr · ndbi · ndmi
ndvi · ndwi · nir · swir1 · swir2 · season
```

### Rendimiento en Test Set
- **Accuracy**: ~82%
- **F1-score (macro)**: ~0.81
- **AUC-ROC**: ~0.88
- **Recall en riesgo**: ~0.79 (capacidad de detectar zonas de riesgo real)

### Importancia de Features
Las variables más influyentes son:
1. **elevation** — Altitud del terreno
2. **season_summer** — Estacionalidad (verano = mayor riesgo)
3. **ndvi** — Cobertura vegetal
4. **Bandas espectrales** (nir, swir1, swir2, mndwi)

##  Requisitos

```
streamlit
folium
streamlit-folium
seaborn
xgboost
joblib
pandas
numpy
scikit-learn
```

### Instalación
```bash
pip install -r requirements.txt
```

##  Uso

### Ejecutar la Aplicación Streamlit
```bash
streamlit run src/app.py
```

Accede a: `http://localhost:8501`

### Generar Mapas de Predicciones
```bash
python src/exportar_mapa.py
```

##  Secciones de la Aplicación

1. **📊 Resumen Ejecutivo** — Métricas globales, rendimiento por clase, hiperparámetros
2. **🔬 Análisis Detallado** — Matriz de confusión, reporte de clasificación
3. **🌿 Importancia de Variables** — Análisis de features con interpretación física
4. **🗺️ Mapa de Predicciones** — Visualización geoespacial interactiva con folium
5. **🔮 Predicción** — Interfaz para hacer predicciones puntuales

##  Casos de Uso

- **Planificadores urbanos**: Identificar zonas prioritarias para intervención de mitigación térmica
- **Investigadores**: Analizar relaciones entre firma espectral y fenómenos urbanos
- **Políticas públicas**: Priorizar inversión en espacios verdes, materiales reflectantes, mejora del drenaje
- **Monitoreo ambiental**: Seguimiento temporal de cambios en entorno urbano

##  Metodología

### Cálculo del Objetivo UHI
El `uhi_risk` se determina a partir de:
1. Cálculo de **anomalías térmicas** mensuales respecto a la media histórica
2. Clasificación binaria basada en umbrales de anomalía en temperatura nocturna
3. Balanceo de clases para capturar minoritariamente las zonas de verdadero riesgo

### Características del Modelo
- **Tipo**: Clasificación binaria
- **Algoritmo**: XGBoost (Extreme Gradient Boosting)
- **Sin variables de temperatura como input**: El modelo aprende patrones espectrales asociados a UHI
- **Geoespacialización**: Predicciones a nivel de píxeles (1km² de resolución en Sentinel-2)

##  Resultado e Impacto

El modelo demuestra que **es posible predecir el riesgo de isla de calor urbana sin usar datos directos de temperatura**, utilizando únicamente la firma espectral del terreno. Esto abre posibilidades para:
- Monitoreo basado en satélites públicos (Sentinel-2, Landsat)
- Predicción en ciudades sin infraestructura de medición térmica
- Integración en sistemas de alerta temprana

##  Autores

- ekbalam11, anaisaponte-GitH, crerov


---

**Última actualización:** Mayo 2026  
**Rama principal:** `main`
