# deploy.py
from google.cloud import aiplatform, storage
import os
import json
import time
import datetime

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
BUCKET_NAME = "gs://usecase_fraud_detection"
REGION = "us-central1"
DISPLAY_NAME = f"fraud-detection-model-{str(datetime.datetime.now())}"

def get_baseline_metrics():
    """Mencari metrics dari model terakhir yang sukses dideploy (jika ada)"""
    aiplatform.init(project=PROJECT_ID, location=REGION)
    try:
        # Ambil model terbaru dari Registry
        models = aiplatform.Model.list(filter=f'display_name="{DISPLAY_NAME}"', order_by="create_time desc")
        if models:
            # Di scenario real, kita simpan metrics di Model Labels. 
            # Untuk demo, kita anggap baseline hardcoded atau fetch dari GCS run sebelumnya.
            # Mari kita simulasikan baseline untuk demo:
            return {"accuracy": 0.85, "f1_score": 0.80} 
    except Exception as e:
        print(f"Warning: Could not fetch baseline: {e}")
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
