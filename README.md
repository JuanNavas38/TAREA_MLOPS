# Taller 1 - Nivel 0: Palmer Penguins 🐧

## MLOps - Pontificia Universidad Javeriana

Clasificación de especies de pingüinos (Adelie, Chinstrap, Gentoo) usando el dataset [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/).

---

## Estructura del proyecto

```
TAREA/
├── taller_1.ipynb                  # Notebook con todo el desarrollo
├── requirements.txt                # Dependencias del proyecto
├── README.md
└── models/
    ├── random_forest_model.pkl     # Modelo RandomForest entrenado
    ├── label_encoder_island.pkl    # Encoder de la variable island
    └── label_encoder_sex.pkl       # Encoder de la variable sex
```

## Etapas completadas

### Etapa 1: Preparación de datos
1. Carga del dataset Palmer Penguins (344 filas, 8 columnas)
2. Exploración inicial (tipos de datos, valores nulos)
3. Limpieza: eliminación de 11 filas con valores nulos (333 filas finales)
4. Estadísticas descriptivas de variables numéricas y categóricas
5. Visualización exploratoria (boxplots, scatter plots, heatmap de correlación)

### Etapa 2: Creación de modelo
6. Transformación: eliminación de columna `year`, codificación de variables categóricas con LabelEncoder
7. División train/test (80/20, estratificada)
8. Entrenamiento de RandomForestClassifier (100 árboles)
9. Validación: **Accuracy = 100%** en datos de prueba (67 muestras)
10. Guardado del modelo y encoders con `joblib`

## Resultados del modelo

| Especie   | Precision | Recall | F1-Score | Muestras |
|-----------|-----------|--------|----------|----------|
| Adelie    | 1.00      | 1.00   | 1.00     | 29       |
| Chinstrap | 1.00      | 1.00   | 1.00     | 14       |
| Gentoo    | 1.00      | 1.00   | 1.00     | 24       |

## Cómo ejecutar

```bash
# Instalar dependencias
pip install -r requirements.txt

# Abrir el notebook
jupyter notebook taller_1.ipynb
```

## Tecnologías

- Python 3.10
- pandas, numpy, matplotlib, seaborn
- scikit-learn (RandomForestClassifier)
- joblib (serialización del modelo)
- palmerpenguins (dataset)

## Autor

Juan Navas — Maestría en Ingeniería de Sistemas y Computación, PUJ
