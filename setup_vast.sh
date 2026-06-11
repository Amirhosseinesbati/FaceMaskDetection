#!/bin/bash

# ==========================================
# 1. Cleanup Function (Auto-Destroy)
# ==========================================
cleanup() {
    echo " Job finished or failed. Destroying this Vast.ai instance..."
    # Force Python to use UTF-8 to avoid 'charmap' decode errors on Windows/locale mismatches
    export PYTHONUTF8=1
    export PYTHONIOENCODING=utf-8
    export LC_ALL=C.UTF-8 || true
    export LANG=C.UTF-8 || true

    # Install/upgrade vastai CLI with minimal output
    pip install --upgrade --no-cache-dir vastai

    # استخراج فقط اعداد از نام کانتینر (keep only digits)
    INSTANCE_ID=${VAST_CONTAINERLABEL//[!0-9]/}

    echo " Target Instance ID to destroy: $INSTANCE_ID"

    # Run vastai CLI forcing UTF-8 env for this invocation as well
    PYTHONIOENCODING=utf-8 vastai destroy instance $INSTANCE_ID -y --api-key $VAST_API_KEY
}
#trap cleanup EXIT


echo "🚀 Starting Environment Setup..."

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

apt-get update && apt-get install -y git

echo "Cloning repository: $GIT_REPO_URL"
git clone -b $GIT_BRANCH $GIT_REPO_URL /workspace/project
cd /workspace/project

echo "📦 Installing dependencies with uv..."
uv sync



mkdir -p ~/.kaggle
echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

echo "🔗 Configuring ZenML Stack..."
uv run zenml init
uv run zenml integration install mlflow -y
uv run zenml experiment-tracker register dagshub_mlflow_tracker \
    --flavor=mlflow \
    --tracking_uri=$DAGSHUB_TRACKING_URI \
    --tracking_username=$DAGSHUB_USERNAME  \
    --tracking_password=$DAGSHUB_TOKEN \
    --tracking_token=$DAGSHUB_TOKEN

uv run zenml stack register vast_gpu_stack -o default -a default -e dagshub_mlflow_tracker
uv run zenml stack set vast_gpu_stack


# Fix for DagsHub HTML Redirect Issue: Force MLflow to use these credentials directly
export MLFLOW_TRACKING_USERNAME=$DAGSHUB_USERNAME
export MLFLOW_TRACKING_PASSWORD=$DAGSHUB_TOKEN
export MLFLOW_TRACKING_URI=$DAGSHUB_TRACKING_URI
export ZENML_STORE_API_KEY=$DAGSHUB_TOKEN

echo "🔥 Starting Training..."
uv run python -m src.pipeline.pipeline