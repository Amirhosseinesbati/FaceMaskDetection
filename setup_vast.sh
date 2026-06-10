#!/bin/bash

cleanup() {
    echo "========================================="
    echo "🚨 Pipeline finished or crashed!"
    echo "Destroying vast.ai instance $INSTANCE_ID to save money..."
    echo "========================================="
    pip install vastai --break-system-packages
    vastai set api-key $VAST_API_KEY
    # اضافه کردن yes برای رد کردن سوال "آیا مطمئن هستید؟"
    yes | vastai destroy instance $INSTANCE_ID
}

trap cleanup EXIT

echo "🚀 Starting Environment Setup..."
# نصب uv و اصلاح مسیر لینوکس (حل ارور اول)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

apt-get update && apt-get install -y git

echo "Cloning repository: $GIT_REPO_URL"
git clone -b $GIT_BRANCH $GIT_REPO_URL /workspace/project
cd /workspace/project

echo "📦 Installing dependencies with uv..."
uv sync


# کانفیگ کگل برای دانلود دیتاست
mkdir -p ~/.kaggle
echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

echo "🔗 Configuring ZenML Stack..."
uv run zenml init
uv run zenml integration install mlflow -y
uv run zenml experiment-tracker register vast_dagshub_tracker \
    --flavor=mlflow \
    --tracking_uri=$DAGSHUB_TRACKING_URI \
    --tracking_username=$DAGSHUB_USERNAME \
    --tracking_password=$DAGSHUB_TOKEN

uv run zenml stack register vast_gpu_stack -e vast_dagshub_tracker
uv run zenml stack set vast_gpu_stack

echo "🔥 Starting Training..."
uv run python -m src.pipeline.pipeline