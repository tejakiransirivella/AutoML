#!/bin/bash

# CONFIGURATION
VENV_PATH="$HOME/capstone/AutoML/automlvenv"
PROJECT_PATH="$HOME/capstone/AutoML"
PYTHONPATH_EXPORT="export PYTHONPATH=\"$PROJECT_PATH:\$PYTHONPATH\""
HEAD_IP="129.21.30.64"
HEAD_PORT="6379"
DASHBOARD_PORT="8265"
WORKER_NODES=("129.21.30.66" "129.21.30.65" "129.21.37.49" "129.21.22.196" "129.21.30.37" "129.21.34.80")

echo "▶️ Starting Ray HEAD node on $HEAD_IP..."

# Activate env and start Ray head
cd "$PROJECT_PATH"
source "$VENV_PATH/bin/activate"
eval "$PYTHONPATH_EXPORT"
ray stop
ray start --head \
  --port=$HEAD_PORT \
  --include-dashboard=true \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=$DASHBOARD_PORT

echo "✅ Ray HEAD started on $HEAD_IP:$HEAD_PORT"

# Loop through each worker node and start Ray worker
for WORKER_IP in "${WORKER_NODES[@]}"; do
    echo "Starting Ray WORKER on $WORKER_IP..."
    ssh $USER@$WORKER_IP << EOF
    cd "$PROJECT_PATH"
    source "$VENV_PATH/bin/activate"
    $PYTHONPATH_EXPORT
    ray stop
    ray start --address="$HEAD_IP:$HEAD_PORT" --node-ip-address=$WORKER_IP
EOF

    echo "✅ Ray worker started on $WORKER_IP"
done

echo "🚀 Ray cluster startup complete!"