# Taller 1: Palmer Penguins

**MLOps - Pontificia Universidad Javeriana**

**Clasificación de especies de pingüinos (Adelie, Chinstrap, Gentoo) usando el dataset Palmer Penguins.**

## Autores
*   **Juan Navas**
*   **Camila Cuellar**
*   **Jhonathan Murcia**
*   Maestría en Ingeniería de Sistemas y Computación, PUJ

---

## Descripción General
Este proyecto abarca desde la exploración y modelado de datos en un entorno de notebook hasta la operacionalización del modelo mediante un pipeline de MLOps. El sistema incluye procesamiento de datos, entrenamiento de modelos, y una API REST dockerizada.

## Tecnologías
*   **Python 3.9+**
*   **pandas, scikit-learn** (RandomForestClassifier, SVM con Pipeline + StandardScaler)
*   **joblib** (serialización de modelos)
*   **palmerpenguins** (dataset)
*   **FastAPI** (API REST)
*   **pytest** (tests)
*   **Docker** (contenedorización)

## Estructura del Proyecto

```text
TAREA_MLOPS/
├── src/                                # Código fuente principal
│   ├── __init__.py
│   ├── config.py                       # Configuración centralizada (paths, params, enums)
│   ├── preprocessing.py                # Lógica compartida de limpieza y encoding
│   ├── train.py                        # Pipeline de entrenamiento automatizado
│   └── api.py                          # API REST para predicciones
├── tests/                              # Tests automatizados
│   ├── __init__.py
│   ├── test_preprocessing.py           # Tests de limpieza y encoding
│   ├── test_api.py                     # Tests de endpoints de la API
│   └── test_train.py                   # Tests del pipeline de entrenamiento
├── models/                             # Artefactos generados
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   ├── label_encoder_island.pkl
│   ├── label_encoder_sex.pkl
│   ├── label_encoder_species.pkl
│   └── metrics.json                    # Métricas de evaluación
├── taller_1.ipynb                      # Notebook con análisis exploratorio
├── Dockerfile                          # Definición del contenedor
├── .dockerignore                       # Exclusiones para Docker
├── requirements.txt                    # Dependencias de producción (pinneadas)
├── requirements-dev.txt                # Dependencias de desarrollo (pytest, httpx)
└── README.md
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
*   Validación: Accuracy reportada en notebook (aprox 98%-100%).
*   Guardado del modelo y encoders con joblib.

---

## Parte 2: Pipeline MLOps y Despliegue

Esta fase automatiza el entrenamiento y expone el modelo vía API. Incluye dos modelos: Random Forest y SVM (con escalado via `StandardScaler`).

### Arquitectura

```
palmerpenguins → src/preprocessing.py → src/train.py → models/*.pkl + metrics.json
                                                              ↓
                                                        src/api.py (FastAPI)
                                                              ↓
                                                        Dockerfile → Container
```

### Características del Pipeline
1.  **Configuración centralizada (`src/config.py`)**: Paths, hiperparámetros, enums y rangos de validación en un solo lugar.
2.  **Preprocesamiento compartido (`src/preprocessing.py`)**: Lógica de limpieza y encoding reutilizada entre entrenamiento y API.
3.  **Entrenamiento automatizado (`src/train.py`)**: Descarga datos frescos, entrena RF y SVM (con `Pipeline` + `StandardScaler`), guarda artefactos y métricas.
4.  **API REST (`src/api.py`)**: FastAPI con validación de inputs (rangos numéricos, categorías válidas), logging estructurado, y healthcheck.
5.  **Tests (`tests/`)**: 17 tests cubriendo preprocessing, endpoints y training.
6.  **Contenedorización**: Docker con usuario no-root y healthcheck.

### Instrucciones de Ejecución

#### 1. Instalación de dependencias
```bash
# Solo producción
pip install -r requirements.txt

# Con herramientas de desarrollo (pytest, httpx)
pip install -r requirements-dev.txt
```

#### 2. Entrenamiento
Genera modelos frescos y métricas:
```bash
python -m src.train
```

#### 3. Ejecución de la API
```bash
python -m src.api
```
La API estará disponible en `http://localhost:8989`.

#### 4. Tests
```bash
python -m pytest tests/ -v
```

#### 5. Ejecución con Docker
```bash
docker build -t penguins-api .
docker run -d -p 8989:8989 --name penguins-container penguins-api
```

### Endpoints de la API

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Mensaje de bienvenida |
| GET | `/health` | Estado de la API y modelos cargados |
| GET | `/docs` | Swagger UI (documentación interactiva) |
| POST | `/predict/rf` | Predicción con Random Forest |
| POST | `/predict/svm` | Predicción con SVM |

**Ejemplo de petición:**
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

**Respuesta:**
```json
{
    "model_used": "rf",
    "prediction": "Gentoo"
}
```

### Validaciones de la API
*   **Categorías**: `island` debe ser Biscoe, Dream o Torgersen. `sex` debe ser male o female.
*   **Rangos numéricos**: Los valores de bill_length, bill_depth, flipper_length y body_mass deben estar dentro de rangos biológicamente razonables.
*   **Modelo**: Solo acepta `rf` o `svm` como tipo de modelo.
