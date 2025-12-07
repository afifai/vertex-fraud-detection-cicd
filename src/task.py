# src/task.py
import pandas as pd
import json
import os
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import joblib
from google.cloud import storage

# Setup Environment
BUCKET_NAME = "usecase_fraud_detection" # Ganti dengan bucket Anda
DATA_PATH = "Bank_Transaction_Fraud_Detection.csv"
MODEL_DIR = os.getenv("AIP_MODEL_DIR", ".")

def download_data(bucket_name, source_blob_name, destination_file_name):
    # (Kode sama seperti sebelumnya...)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def train():
    # 1. Load & Preprocess (Kode sama seperti sebelumnya...)
    download_data(BUCKET_NAME, DATA_PATH, "data.csv")
    df = pd.read_csv("data.csv")
    
    # Cleanup & Encode
    drop_cols = ['Customer_ID', 'Customer_Name', 'Transaction_ID', 'Merchant_ID', 
                 'Transaction_Date', 'Transaction_Time', 'Customer_Contact', 
                 'Customer_Email', 'Transaction_Description']
    df = df.drop(columns=drop_cols, errors='ignore')
    
    le = LabelEncoder()
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
            
    X = df.drop('Is_Fraud', axis=1)
    y = df['Is_Fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Training
    # Parameter ini bisa kita ubah lewat argumen untuk testing hyperparameter
    # n_estimators = 100 
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    
    # 3. Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {acc}")
    print(f"F1 Score: {f1}")
    
    # 4. Save Artifacts
    # A. Save Model
    joblib.dump(model, "model.joblib")
    
    # B. Save Metrics ke JSON
    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        # "n_estimators": n_estimators
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    # 5. Upload to GCS (AIP_MODEL_DIR)
    if os.getenv("AIP_MODEL_DIR"):
        dest_path = os.getenv("AIP_MODEL_DIR")
        subprocess.check_call(['gsutil', 'cp', 'model.joblib', dest_path])
        # Kita simpan metrics.json di folder yang sama dengan model
        subprocess.check_call(['gsutil', 'cp', 'metrics.json', dest_path])

if __name__ == '__main__':
    train()
