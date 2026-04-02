
from fastapi import FastAPI
from app.schema import PredictionInput
from src.predict import InferencePipeline
from src.utils import logger

app = FastAPI(title="Failure Prediction API")

# Load model once
pipeline = InferencePipeline()


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        input_data = data.dict()

        logger.info(f"Received input: {input_data}")

        result = pipeline.predict(input_data)

        logger.info(f"Prediction result: {result}")

        return result

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}
