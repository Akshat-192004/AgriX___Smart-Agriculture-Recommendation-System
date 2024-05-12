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
