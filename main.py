import pickle
import numpy as np
from fastapi import FastAPI
import uvicorn

app = FastAPI()

import pymongo


def create_database():
    with pymongo.MongoClient("mongodb://localhost:27017/") as myclient:
        db = myclient["crop_prediction"]
        db["feedback"]
        return "Database created successfully"


schema = {
    "type": "object",
    "properties": {
        "N": {"type": "number"},
        "P": {"type": "number"},
        "K": {"type": "number"},
        "temperature": {"type": "number"},
        "humidity": {"type": "number"},
        "ph": {"type": "number"},
        "rainfall": {"type": "number"},
        "feedback": {"type": "string"},
    },
    "required": [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
        "feedback",
    ],
}


def insert_feedback(data: dict):
    from jsonschema import validate

    try:
        validate(instance=data, schema=schema)
    except:
        return "Invalid data"
    finally:
        with pymongo.MongoClient("mongodb://localhost:27017/") as myclient:
            db = myclient["crop_prediction"]
            feedback = db["feedback"]
            feedback.insert_one(data)
            return "Feedback added successfully"


@app.get("/")
def main():
    return {"message": "Welcome to the Crop Prediction API"}


loaded_model = pickle.load(open("Model.pkl", "rb"))
print(create_database())


@app.get("/predict")
def predict(
    N: float,
    P: float,
    K: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float,
):
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = loaded_model.predict(data)
    return {"prediction": prediction[0]}


@app.get("/feedback")
def feedback(
    N: float,
    P: float,
    K: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float,
    prediction: str,
):
    return insert_feedback(
        {
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
            "feedback": prediction,
        }
    )


@app.get("/get_feedbacks")
def get_feedbacks():
    import pymongo

    with pymongo.MongoClient("mongodb://localhost:27017/") as myclient:
        db = myclient["crop_prediction"]
        feedback = db["feedback"]
        all_feedback = feedback.find()
        feedback_list = []
        for feedback in all_feedback:
            feedback_list.append(
                {
                    "N": feedback["N"],
                    "P": feedback["P"],
                    "K": feedback["K"],
                    "temperature": feedback["temperature"],
                    "humidity": feedback["humidity"],
                    "ph": feedback["ph"],
                    "rainfall": feedback["rainfall"],
                    "prediction": feedback["feedback"],
                }
            )
        return feedback_list


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
