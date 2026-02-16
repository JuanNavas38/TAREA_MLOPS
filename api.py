from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import os

app = FastAPI(title="Penguin Species Prediction API", description="API for predicting penguin species using MLOps pipeline")

# Cargar modelos y encoders
MODELS_DIR = "models"
try:
    rf_model = joblib.load(os.path.join(MODELS_DIR, 'random_forest_model.pkl'))
    svm_model = joblib.load(os.path.join(MODELS_DIR, 'svm_model.pkl'))
    le_island = joblib.load(os.path.join(MODELS_DIR, 'label_encoder_island.pkl'))
    le_sex = joblib.load(os.path.join(MODELS_DIR, 'label_encoder_sex.pkl'))
    le_species = joblib.load(os.path.join(MODELS_DIR, 'label_encoder_species.pkl'))
except FileNotFoundError:
    print("Error: Models not found. Please run train.py first.")
    
# Definir modelo de datos de entrada
class Penguin(BaseModel):
    island: str
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Penguin Prediction API"}

@app.post("/predict/{model_type}")
def predict_species(model_type: str, penguin: Penguin):
    # Preprocesamiento
    try:
        # Codificar variables categóricas
        # Manejo de errores si el valor no existe en el encoder
        if penguin.island not in le_island.classes_:
            raise HTTPException(status_code=400, detail=f"Island '{penguin.island}' unknown.")
        if penguin.sex not in le_sex.classes_:
            raise HTTPException(status_code=400, detail=f"Sex '{penguin.sex}' unknown.")
            
        island_encoded = le_island.transform([penguin.island])[0]
        sex_encoded = le_sex.transform([penguin.sex])[0]
        
        # Crear DataFrame para la predicción
        data = pd.DataFrame([[
            island_encoded, 
            penguin.bill_length_mm, 
            penguin.bill_depth_mm, 
            penguin.flipper_length_mm, 
            penguin.body_mass_g, 
            sex_encoded
        ]], columns=['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex'])
        
        # Seleccionar modelo
        if model_type == "rf":
            prediction_idx = rf_model.predict(data)[0]
        elif model_type == "svm":
            prediction_idx = svm_model.predict(data)[0]
        else:
            raise HTTPException(status_code=400, detail="Model type must be 'rf' or 'svm'")
            
        prediction_name = le_species.inverse_transform([prediction_idx])[0]
        
        return {
            "model_used": model_type,
            "prediction": prediction_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict_species_default(penguin: Penguin):
    return predict_species("rf", penguin)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8989)
