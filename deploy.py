# deploy.py
from google.cloud import aiplatform, storage
import os
import json
import time

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
BUCKET_NAME = "gs://sinarmas-vertex-training"
REGION = "us-central1"
DISPLAY_NAME = "fraud-detection-model"


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


def create_markdown_report(current):
    """Membuat tabel Markdown untuk Comment PR hanya untuk PR Branch (Candidate)"""

    md = f"""
## 🤖 Vertex AI Training Report

**Job Status:** Success ✅
**Model:** {DISPLAY_NAME}

### 📊 Model Performance (PR Branch Candidate)

| Metric     | Value |
|------------|-------|
| **Accuracy** | {current['accuracy']:.4f} |
| **F1 Score** | {current['f1_score']:.4f} |

### ⚙️ Parameters
- **n_estimators**: {current.get('n_estimators', 'N/A')}

_Report generated automatically by GitHub Actions_
    """
    
    with open("report.md", "w") as f:
        f.write(md)


def run_job():
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_NAME)

    # 1. Run Training
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
    
    # 2. Fetch New Metrics
    current_metrics = download_metrics(base_output_dir)
    
    # 3. Generate Report (hanya candidate)
    create_markdown_report(current_metrics)
    print("Report generated: report.md")


if __name__ == "__main__":
    run_job()
