# src/task.py
import pandas as pd
import json
import os
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import joblib
from google.cloud import storage

# Setup Environment
BUCKET_NAME = "sinarmas-vertex-training" # Ganti dengan bucket Anda
DATA_PATH = "data/Bank_Transaction_Fraud_Detection.csv"
MODEL_DIR = os.getenv("AIP_MODEL_DIR", ".")

def download_data(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name.replace("gs://", ""))
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def train():
    print("1. Loading Data...")
    download_data(BUCKET_NAME, DATA_PATH, "data.csv")
    df = pd.read_csv("data.csv")
    
    # --- CLEANING ---
    drop_cols = ['Customer_ID', 'Customer_Name', 'Transaction_ID', 'Merchant_ID', 
                 'Transaction_Date', 'Transaction_Time', 'Customer_Contact', 
                 'Customer_Email', 'Transaction_Description']
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # --- ENCODING ---
    le = LabelEncoder()
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # ==========================================
    # TRIK KHUSUS DEMO: BALANCING DATA (50:50)
    # ==========================================
    print("   Balancing dataset (Undersampling)...")
    
    # Pisahkan Fraud dan Non-Fraud
    df_fraud = df[df['Is_Fraud'] == 1]
    df_normal = df[df['Is_Fraud'] == 0]
    
    # Ambil data normal SEJUMLAH data fraud (biar seimbang 1 banding 1)
    # Ini akan membuang banyak data normal, tapi bagus agar model 'belajar' fraud
    df_normal_sample = df_normal.sample(n=len(df_fraud), random_state=42)
    
    # Gabung kembali
    df = pd.concat([df_fraud, df_normal_sample])
    print(f"   Data shape after balancing: {df.shape}")
    # ==========================================

    X = df.drop('Is_Fraud', axis=1)
    y = df['Is_Fraud']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Training
    print("2. Training Model...")
    # Kita pakai max_depth agar tidak overfit
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 3. Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"   Accuracy: {acc}")
    print(f"   F1 Score: {f1}")
    
    # 4. Save Artifacts
    joblib.dump(model, "model.joblib")
    
    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "n_estimators": 100
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    # 5. Upload to Vertex AI Output
    if os.getenv("AIP_MODEL_DIR"):
        dest_path = os.getenv("AIP_MODEL_DIR")
        print(f"3. Uploading artifacts to {dest_path}")
        subprocess.check_call(['gsutil', 'cp', 'model.joblib', dest_path])
        subprocess.check_call(['gsutil', 'cp', 'metrics.json', dest_path])

if __name__ == '__main__':
    train()