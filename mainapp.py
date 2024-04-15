from fastapi import FastAPI
import mlflow.pyfunc
import numpy as np

# Load the MLflow model as a PyFunc model
model = mlflow.pyfunc.load_model('./337761508546523430/06c7529135bb4305bff712e6f9794078/artifacts/keras_model')

app = FastAPI()

@app.post('/predict')
def predict(data: dict):
    # Assuming data is a dictionary with model input
    prediction = model.predict(np.expand_dims(np.asarray(data['data']), axis=0))
    return {"prediction": str(np.argmax(prediction))}
