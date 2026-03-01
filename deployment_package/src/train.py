import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from preprocess import preprocess_and_split


def main():

    # Step 1: Preprocess data
    X_train, X_test, y_train, y_test = preprocess_and_split("data/train.csv")

    # Step 2: Train model
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=7,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Step 3: Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Accuracy:", acc)

    # Step 4: Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    # Step 5: Log with MLflow
    mlflow.set_experiment("Spaceship_Titanic_Academic")

    with mlflow.start_run():
        mlflow.log_param("n_estimators", 150)
        mlflow.log_param("max_depth", 7)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")


# ✅ This is the entry point
if __name__ == "__main__":
    main()