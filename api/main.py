from fastapi import FastAPI, HTTPException
import uvicorn

import sys
sys.path.append('../')
import common
import numpy as np

from pydantic import BaseModel, PrivateAttr, Field, PositiveFloat, computed_field, PositiveInt

model = common.load_model("../models//trip_duration.model")

app = FastAPI()


class PredictionInput(BaseModel):
    weekday: PositiveInt = Field(..., ge=1, le=7, description="Day of the week (1=Monday, ..., 7=Sunday)")
    month: PositiveInt = Field(..., ge=1, le=12, description="Month of the year (1=January, ..., 12=December)")
    hour: PositiveInt = Field(..., ge=0, le=23, description="Hour of the day (0 to 23)")

@app.post("/predict")
def predict(input_data: PredictionInput):
    """
    Endpoint pour prédire la durée du trajet.
    """
    # Convertir les données d'entrée en format compatible avec le modèle
    features = np.array([[input_data.weekday, input_data.month, input_data.hour]])

    # Prédiction
    prediction = model.predict(features)[0]

    return {"predicted_trip_duration": prediction}


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0",
                port=8000, reload=True)
