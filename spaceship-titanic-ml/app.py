from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import pandas as pd
import joblib
from src.preprocess import feature_engineering

app = Flask(__name__)
api = Api(app, doc="/docs", title="Spaceship Titanic API",
          description="Predict if passengers are transported")

# Load model and preprocessor
model = joblib.load("models/model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# Define expected input model for Swagger
passenger_model = api.model('Passenger', {
    'PassengerId': fields.String(required=False, description="Optional, e.g., '0001_01'"),
    'Name': fields.String(required=False, description="Optional, passenger name"),
    'HomePlanet': fields.String(required=True),
    'CryoSleep': fields.Boolean(required=True),
    'Cabin': fields.String(required=True),
    'Destination': fields.String(required=True),
    'Age': fields.Float(required=True),
    'VIP': fields.Boolean(required=True),
    'RoomService': fields.Float(required=True),
    'FoodCourt': fields.Float(required=True),
    'ShoppingMall': fields.Float(required=True),
    'Spa': fields.Float(required=True),
    'VRDeck': fields.Float(required=True)
})

@api.route("/predict")
class Predict(Resource):
    @api.expect(passenger_model)
    def post(self):
        data = request.get_json()
        df = pd.DataFrame([data])

        # Feature engineering with safe defaults
        if "PassengerId" not in df.columns:
            df["PassengerId"] = "0000_00"
        if "Name" not in df.columns:
            df["Name"] = "Unknown"

        df = feature_engineering(df)

        # Ensure preprocessing columns exist (avoid missing column errors)
        for col in preprocessor.feature_names_in_:
            if col not in df.columns:
                df[col] = 0

        df_processed = preprocessor.transform(df)
        prediction = model.predict(df_processed)
        return {"Transported": bool(prediction[0])}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)