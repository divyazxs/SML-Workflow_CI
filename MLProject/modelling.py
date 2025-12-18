import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

# Load Data 
path_data = "loan_approval_dataset_preprocessing/loan_data_clean.csv"
df = pd.read_csv(path_data)

# Memisahkan Fitur (X) dan Target (y)
X = df.drop('loan_status', axis=1) 
y = df['loan_status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Set Experiment & Autolog
mlflow.set_experiment("CI_Experiment_Loan_Approval") 
mlflow.autolog()

# Training Model
with mlflow.start_run():
    # Log Parameter
    n_est = 100
    mlflow.log_param("n_estimators", n_est)

    # Train Model
    model = RandomForestClassifier(n_estimators=n_est)
    model.fit(X_train, y_train)
    
    # Evaluasi Model 
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log Metric dan Model
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Training Selesai. Akurasi: {acc}")