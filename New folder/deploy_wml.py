"""
deploy_wml.py — Optional: Deploy model to IBM Watson Machine Learning
======================================================================
This script only runs if a .env file exists in the same directory.
It does NOT need to be executed to use train.py or app.py.

Prerequisites:
    pip install ibm-watsonx-ai python-dotenv

Setup:
    1. Copy .env.example to .env
    2. Fill in your IBM Cloud API key, WML URL and Space ID.
    3. Run:  python deploy_wml.py
"""

import os
import sys

# ---------------------------------------------------------------------------
# Guard: only proceed if .env file exists
# ---------------------------------------------------------------------------
ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

if not os.path.exists(ENV_FILE):
    print("No .env file found — skipping Watson Machine Learning deployment.")
    print("To deploy, create a .env file using .env.example as a template.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE)
except ImportError:
    print("python-dotenv not installed. Run: pip install python-dotenv")
    sys.exit(1)

API_KEY   = os.getenv("IBM_API_KEY", "")
WML_URL   = os.getenv("IBM_WML_URL", "")
SPACE_ID  = os.getenv("IBM_SPACE_ID", "")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")

for name, val in [("IBM_API_KEY", API_KEY), ("IBM_WML_URL", WML_URL), ("IBM_SPACE_ID", SPACE_ID)]:
    if not val:
        print(f"Missing environment variable: {name}")
        sys.exit(1)

if not os.path.exists(MODEL_PATH):
    print("model.pkl not found. Run python train.py first.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Connect to Watson Machine Learning
# ---------------------------------------------------------------------------
try:
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference  # noqa (for reference)
except ImportError:
    print("ibm-watsonx-ai not installed. Run: pip install ibm-watsonx-ai")
    sys.exit(1)

import json

print("[WML] Connecting to IBM Watson Machine Learning ...")
credentials = Credentials(url=WML_URL, api_key=API_KEY)
client = APIClient(credentials=credentials)
client.set.default_space(SPACE_ID)
print("[WML] Connected.")

# ---------------------------------------------------------------------------
# Load feature / class metadata
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "features.json")) as f:
    features = json.load(f)
with open(os.path.join(BASE_DIR, "label_classes.json")) as f:
    label_classes = json.load(f)

# ---------------------------------------------------------------------------
# Store model in WML repository
# ---------------------------------------------------------------------------
print("[WML] Storing model in WML repository ...")

software_spec_uid = client.software_specifications.get_id_by_name("runtime-23.1-py3.10")

model_meta = {
    client.repository.ModelMetaNames.NAME:              "Power System Fault Classifier",
    client.repository.ModelMetaNames.TYPE:              "scikit-learn_1.1",
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: software_spec_uid,
}

stored_model = client.repository.store_model(
    model=MODEL_PATH,
    meta_props=model_meta,
)
model_uid = client.repository.get_model_id(stored_model)
print(f"[WML] Model stored. UID: {model_uid}")

# ---------------------------------------------------------------------------
# Create an online deployment
# ---------------------------------------------------------------------------
print("[WML] Creating online deployment ...")
deploy_meta = {
    client.deployments.ConfigurationMetaNames.NAME:           "fault-classifier-online",
    client.deployments.ConfigurationMetaNames.ONLINE:         {},
}
deployment = client.deployments.create(artifact_uid=model_uid, meta_props=deploy_meta)
deployment_uid = client.deployments.get_uid(deployment)
print(f"[WML] Deployment created. UID: {deployment_uid}")

# ---------------------------------------------------------------------------
# Test the deployed model with a sample payload
# ---------------------------------------------------------------------------
print("[WML] Running test scoring ...")
sample_row = {feat: stats["mean"] for feat, stats in features.items()}
sample_values = [[sample_row[feat] for feat in features.keys()]]

scoring_payload = {
    client.deployments.ScoringMetaNames.INPUT_DATA: [
        {"values": sample_values}
    ]
}
response = client.deployments.score(deployment_uid, scoring_payload)
print("[WML] Test scoring response:")
print(json.dumps(response, indent=2, default=str))

print("\n[WML] Deployment complete.")
print(f"  Model UID      : {model_uid}")
print(f"  Deployment UID : {deployment_uid}")
print(
    "\nTo use the deployed model via REST API, pass the deployment UID "
    "and your IBM Cloud IAM token to the WML scoring endpoint."
)
