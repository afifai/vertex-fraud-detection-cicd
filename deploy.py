# deploy.py
from google.cloud import aiplatform, storage
import os
import json
import time

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
BUCKET_NAME = "gs://sinarmas-vertex-training"
REGION = "us-central1"
DISPLAY_NAME = "fraud-detection-model"

def download_json_from_gcs(gcs_uri):
    """Helper untuk download file JSON dari GCS URI"""
    try:
        storage_client = storage.Client()
        
        # Parse gs://bucket/path/to/file.json
        parts = gcs_uri.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1]
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        data = blob.download_as_text()
        return json.loads(data)
    except Exception as e:
        print(f"  [Info] Gagal download metrik dari {gcs_uri}: {e}")
        return None

def get_baseline_metrics():
    """
    Logika:
    1. Cari model dengan display_name tertentu di Registry.
    2. Ambil model yang paling baru dibuat (create_time desc).
    3. Cek lokasi GCS model tersebut (model.uri).
    4. Coba download 'metrics.json' dari lokasi tersebut.
    """
    print("1. Mencari baseline metrics dari Model Registry...")
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    try:
        # Ambil list model, urutkan dari yang terbaru
        models = aiplatform.Model.list(
            filter=f'display_name="{DISPLAY_NAME}"', 
            order_by="create_time desc"
        )
        
        if not models:
            print("  - Belum ada model di registry. Ini adalah run pertama.")
            return None
            
        latest_model = models[0]
        print(f"  - Model terakhir ditemukan: {latest_model.resource_name}")
        print(f"  - Model URI: {latest_model.uri}")
        
        # Vertex AI menyimpan model di folder. Kita asumsikan metrics.json ada di dalamnya.
        # Format model.uri biasanya: gs://bucket/output/dir/ (tanpa nama file)
        base_uri = latest_model.uri.rstrip("/")
        metrics_uri = f"{base_uri}/metrics.json"
        
        print(f"  - Mencoba fetch: {metrics_uri}")
        metrics = download_json_from_gcs(metrics_uri)
        
        if metrics:
            print(f"  - Baseline Metrics ditemukan: {metrics}")
            return metrics
        else:
            print("  - File metrics.json tidak ditemukan di artifact model terakhir.")
            return None
            
    except Exception as e:
        print(f"Warning: Error saat mengambil baseline: {e}")
        return None

def download_metrics(gcs_folder):
    """Download metrics.json dari hasil training baru"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME.replace("gs://", ""))
    
    # Path di GCS: bucket/model_output/model/metrics.json
    blob_path = f"{gcs_folder.replace(BUCKET_NAME + '/', '')}/model/metrics.json"
    blob = bucket.blob(blob_path)
    
    blob.download_to_filename("metrics.json")
    with open("metrics.json", "r") as f:
        return json.load(f)

def create_markdown_report(baseline, current):
    """Membuat tabel Markdown untuk Comment PR"""
    
    # Helper untuk panah naik/turun
    def diff_str(old, new):
        if not old: return ""
        diff = new - old
        icon = "jq " if diff > 0 else "kr "
        return f"({icon} {diff:+.4f})"

    # Jika baseline tidak ada (first run)
    if not baseline:
        baseline = {"accuracy": 0, "f1_score": 0}

    md = f"""
## 🤖 Vertex AI Training Report

**Job Status:** Success ✅
**Model:** {DISPLAY_NAME}

### 📊 Model Performance Comparison

| Metric | Main Branch (Baseline) | PR Branch (Candidate) | Diff |
|--------|------------------------|-----------------------|------|
| **Accuracy** | {baseline['accuracy']:.4f} | **{current['accuracy']:.4f}** | {diff_str(baseline['accuracy'], current['accuracy'])} |
| **F1 Score** | {baseline['f1_score']:.4f} | **{current['f1_score']:.4f}** | {diff_str(baseline['f1_score'], current['f1_score'])} |

### ⚙️ Parameters
- **n_estimators**: {current.get('n_estimators', 'N/A')}

_Report generated automatically by GitHub Actions_
    """
    
    with open("report.md", "w") as f:
        f.write(md)

def run_job():
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_NAME)

    # 1. Get Baseline
    baseline_metrics = get_baseline_metrics()
    
    # 2. Run Training
    job_id = f"fraud-job-{int(time.time())}"
    base_output_dir = f"{BUCKET_NAME}/model_output/{job_id}"
    
    job = aiplatform.CustomTrainingJob(
        display_name=DISPLAY_NAME,
        script_path="src/task.py",
        container_uri="us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-23:latest",
        requirements=["pandas", "google-cloud-storage", "scikit-learn"],
        model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/scikit-learn-cpu.0-23:latest"
    )

    print("Submitting Training Job...")
    model = job.run(
        machine_type="n1-standard-4",
        replica_count=1,
        base_output_dir=base_output_dir,
        sync=True
    )
    
    # 3. Fetch New Metrics
    current_metrics = download_metrics(base_output_dir)
    
    # 4. Generate Report
    create_markdown_report(baseline_metrics, current_metrics)
    print("Report generated: report.md")

if __name__ == "__main__":
    run_job()
