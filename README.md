# Índices espectrales — Sentinel-2

## Variables objetivo (MODIS)
| Variable | Descripción |
|---|---|
| `lst_day_c` | Temperatura superficial diurna (°C) |
| `lst_night_c` | Temperatura superficial nocturna (°C) |
| `lst_day_c_anomaly` | Anomalía térmica diurna respecto a la media mensual |
| `lst_night_c_anomaly` | Anomalía térmica nocturna respecto a la media mensual |

## Vegetación
| Índice | Nombre completo | Qué mide |
|---|---|---|
| `ndvi` | Normalized Difference Vegetation Index | Densidad y vigor de la vegetación |
| `evi` | Enhanced Vegetation Index | Vegetación en zonas de alta densidad, menos sensible a la atmósfera |
| `evi2` | Enhanced Vegetation Index 2 | Versión simplificada de EVI sin banda azul |
| `savi` | Soil-Adjusted Vegetation Index | Vegetación con corrección de suelo desnudo |
| `msavi` | Modified SAVI | Versión mejorada de SAVI, minimiza el efecto del suelo |
| `gndvi` | Green NDVI | Sensible al contenido de clorofila |
| `ndre` | Normalized Difference Red Edge | Estrés vegetal y contenido de clorofila |
| `cire` | Chlorophyll Index Red Edge | Concentración de clorofila en vegetación |
| `gli` | Green Leaf Index | Presencia de vegetación verde activa |
| `psri` | Plant Senescence Reflectance Index | Senescencia y madurez de la vegetación |

## Urbano / superficie sellada
| Índice | Nombre completo | Qué mide |
|---|---|---|
| `ndbi` | Normalized Difference Built-up Index | Superficie construida e impermeable |
| `ibi` | Index-based Built-up Index | Superficie urbana combinando vegetación, agua y suelo |
| `ui` | Urban Index | Intensidad de urbanización |
| `ebbi` | Enhanced Built-up and Bareness Index | Superficie construida y suelo desnudo combinados |

## Suelo desnudo
| Índice | Nombre completo | Qué mide |
|---|---|---|
| `bsi` | Bare Soil Index | Suelo desnudo, diferencia zonas agrícolas e industriales |
| `ndti` | Normalized Difference Tillage Index | Residuos agrícolas y suelo labrado |

## Agua y humedad
| Índice | Nombre completo | Qué mide |
|---|---|---|
| `mndwi` | Modified Normalized Difference Water Index | Láminas de agua superficial |
| `ndwi` | Normalized Difference Water Index | Humedad en vegetación y agua superficial |
| `ndmi` | Normalized Difference Moisture Index | Contenido de humedad en vegetación |
| `ndsi` | Normalized Difference Snow Index | Nieve y hielo (sin relevancia en BCN, incluido para EDA) |

## Fuego y vegetación quemada
| Índice | Nombre completo | Qué mide |
|---|---|---|
| `nbr` | Normalized Burn Ratio | Áreas quemadas y severidad del fuego |

## Variables topográficas y espaciales
| Variable | Descripción |
|---|---|
| `elevation` | Altitud en metros (fuente: SRTM 30m) |
| `grid_id` | Identificador de celda de 1km² en proyección UTM 32631 |

## Variables temporales
| Variable | Descripción |
|---|---|
| `fecha` | Fecha de la imagen Sentinel-2 |
| `month` | Mes (1-12) |
| `year` | Año |
| `dayofyear` | Día del año (1-365) |



## Rangos teóricos de los índices

| Índice | Rango teórico | Acotado | Valor mínimo | Valor máximo |
|---|---|---|---|---|
| `ndvi` | -1 a 1 | ✅ | Agua o superficies artificiales | Vegetación densa y sana |
| `ndbi` | -1 a 1 | ✅ | Vegetación o agua | Alta densidad de superficie construida |
| `gndvi` | -1 a 1 | ✅ | Agua o suelo desnudo | Alta concentración de clorofila |
| `ndre` | -1 a 1 | ✅ | Vegetación estresada o ausente | Vegetación sana con alto contenido de clorofila |
| `ndwi` | -1 a 1 | ✅ | Suelo seco o superficie construida | Lámina de agua superficial |
| `mndwi` | -1 a 1 | ✅ | Suelo seco o superficie construida | Lámina de agua superficial |
| `ndmi` | -1 a 1 | ✅ | Vegetación seca o suelo sin humedad | Vegetación con alta humedad |
| `ndsi` | -1 a 1 | ✅ | Suelo desnudo o superficie construida | Nieve o hielo |
| `nbr` | -1 a 1 | ✅ | Área quemada severa | Vegetación densa sin quemar |
| `ui` | -1 a 1 | ✅ | Vegetación o agua | Alta intensidad de urbanización |
| `bsi` | -1 a 1 | ✅ | Vegetación densa | Suelo desnudo o superficie degradada |
| `gli` | -1 a 1 | ✅ | Superficie no vegetada | Vegetación verde activa |
| `msavi` | -1 a 1 | ✅ | Suelo desnudo o superficie artificial | Vegetación densa con mínima influencia del suelo |
| `evi` | -1 a ~2 | ⚠️ | Agua o superficies artificiales | Vegetación muy densa en zonas tropicales |
| `evi2` | -1 a ~2 | ⚠️ | Agua o superficies artificiales | Vegetación muy densa |
| `savi` | -1.5 a 1.5 | ⚠️ | Superficies artificiales o agua | Vegetación densa sobre suelo de alta reflectancia |
| `ibi` | -1 a 1 | ⚠️ | Agua o vegetación densa | Alta densidad urbana integrada |
| `cire` | 0 a ∞ | ❌ | Vegetación con bajo contenido de clorofila | Alta concentración de clorofila |
| `ebbi` | sin límite | ❌ | Vegetación densa o agua | Superficie construida o suelo desnudo expuesto |
| `psri` | sin límite | ❌ | Vegetación joven y verde | Vegetación senescente o en proceso de maduración |

> ✅ Acotado por construcción matemática  
> ⚠️ Teóricamente acotado pero puede producir outliers en píxeles con valores de banda cercanos a cero  
> ❌ Sin límite teórico — requiere filtrado de outliers en el EDA

