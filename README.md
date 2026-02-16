# Taller 1 - Nivel 0: Palmer Penguins

**MLOps - Pontificia Universidad Javeriana**

**Clasificación de especies de pingüinos (Adelie, Chinstrap, Gentoo) usando el dataset Palmer Penguins.**

## Autores
*   **Juan Navas**
*   **Camila Cuellar**
*   Maestría en Ingeniería de Sistemas y Computación, PUJ

---

## Descripción General
Este proyecto abarca desde la exploración y modelado de datos en un entorno de notebook hasta la operacionalización del modelo mediante un pipeline de MLOps. El sistema incluye procesamiento de datos, entrenamiento de modelos, y una API REST dockerizada.

## Tecnologías
*   **Python 3.10**
*   **pandas, numpy, matplotlib, seaborn**
*   **scikit-learn** (RandomForestClassifier, SVM)
*   **joblib** (serialización del modelo)
*   **palmerpenguins** (dataset)
*   **FastAPI** (API REST)
*   **Docker** (Contenedorización)

## Estructura del Proyecto

```text
TAREA_MLOPS/
├── taller_1.ipynb                  # Notebook con el análisis exploratorio y desarrollo inicial
├── train.py                        # Script de entrenamiento automatizado (MLOps)
├── api.py                          # API para servir predicciones (MLOps)
├── Dockerfile                      # Definición del contenedor (MLOps)
├── requirements.txt                # Dependencias del proyecto
├── README.md                       # Documentación
└── models/                         # Artefactos generados
    ├── random_forest_model.pkl
    ├── svm_model.pkl
    ├── label_encoder_island.pkl
    ├── label_encoder_sex.pkl
    └── label_encoder_species.pkl
```

---

## Parte 1: Exploración y Experimentación (Notebook)

Esta fase corresponde al desarrollo realizado en `taller_1.ipynb`.

### Etapas Completadas

#### Etapa 1: Preparación de datos
*   Carga del dataset Palmer Penguins (344 filas, 8 columnas).
*   Exploración inicial (tipos de datos, valores nulos).
*   Limpieza: eliminación de 11 filas con valores nulos (333 filas finales).
*   Estadísticas descriptivas de variables numéricas y categóricas.
*   Visualización exploratoria (boxplots, scatter plots, heatmap de correlación).

#### Etapa 2: Creación de modelo
*   Transformación: eliminación de columna `year`, codificación de variables categóricas con `LabelEncoder`.
*   División train/test (80/20, estratificada).
*   Entrenamiento de **RandomForestClassifier** (100 árboles).
*   Validación: Accuracy reportada en notebook (varía según semilla, aprox 98%-100%).
*   Guardado del modelo y encoders con joblib.

### Resultados del Modelo (Referencia Notebook)

| Especie | Precision | Recall | F1-Score | Muestras |
| :--- | :--- | :--- | :--- | :--- |
| **Adelie** | 1.00 | 1.00 | 1.00 | 29 |
| **Chinstrap** | 1.00 | 1.00 | 1.00 | 14 |
| **Gentoo** | 1.00 | 1.00 | 1.00 | 24 |

### Ejecución del Notebook
```bash
# Instalar dependencias
pip install -r requirements.txt

# Abrir el notebook
jupyter notebook taller_1.ipynb
```

---

## Parte 2: Pipeline MLOps y Despliegue

Esta fase automatiza el entrenamiento y expone el modelo vía API. Se ha añadido un modelo SVM adicional como bono.

### Características del Pipeline
1.  **Entrenamiento Automatizado (`train.py`)**: Script que descarga datos frescos, re-entrena Random Forest y SVM, y regenera los artefactos en `models/`.
2.  **API REST (`api.py`)**: Servicio FastAPI con selectores de modelo.
3.  **Contenedorización**: Despliegue portátil con Docker.

### Instrucciones de Ejecución MLOps

#### 1. Entrenamiento
Genera modelos frescos:
```bash
python train.py
```

#### 2. Ejecución con Docker
Construye y corre el servicio en el puerto 8989:
```bash
docker build -t penguins-api .
docker run -d -p 8989:8989 --name penguins-container penguins-api
```

#### 3. Uso de la API (Endpoints)
*   **Swagger UI**: `http://localhost:8989/docs`
*   **Predicción (RF)**: `POST /predict/rf`
*   **Predicción (SVM)**: `POST /predict/svm`

**Ejemplo de Petición:**
```bash
curl -X POST "http://localhost:8989/predict/rf" \
     -H "Content-Type: application/json" \
     -d '{
           "island": "Biscoe",
           "bill_length_mm": 45.2,
           "bill_depth_mm": 14.8,
           "flipper_length_mm": 212.0,
           "body_mass_g": 5200.0,
           "sex": "female"
         }'
```
