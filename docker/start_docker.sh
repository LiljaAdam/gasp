#!/usr/bin/env bash
# Start the gasp dev container in the background. Run from gasp_release/: ./docker/start_docker.sh
# Set in docker/.env or export: WANDB_API_KEY, HF_TOKEN (Hugging Face token).

set -e
CONTAINER_NAME="gasp"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load .env if present (for WANDB_API_KEY, HF_TOKEN, etc.)
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  source "$SCRIPT_DIR/.env"
  set +a
fi

: "${WANDB_API_KEY:=}"
: "${HF_TOKEN:=}"

# PATH CONFIGURATION — edit to match your local dataset locations
DATASET_ROOT=${DATASET_ROOT:-"/datasets/"}
DINO_DIR=${DINO_DIR:-"/dino_features/"}
OUTPUT_DIR=${OUTPUT_DIR:-"/outputs/"}

base_run_cmd=(
  docker run --rm
  --gpus all
  --shm-size=64g
  --ipc=host
  --name "$CONTAINER_NAME"
  -e WANDB_API_KEY="$WANDB_API_KEY"
  -e HF_TOKEN="$HF_TOKEN"
  -v "$PROJECT_ROOT":/gasp/
  -v "$DATASET_ROOT":/datasets/
  -v "$DINO_DIR":/datasets/dino_features/
  -v "$OUTPUT_DIR":/outputs/
  -p 8082:8082
)

# If already running, don't try to start another one.
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Container '${CONTAINER_NAME}' is already running."
  echo "Attach with: ./docker/into_docker.sh"
  exit 0
fi

if [ $# -eq 0 ]; then
  # Start container in the background with a long-lived command.
  "${base_run_cmd[@]}" -d gasp:latest tail -f /dev/null
  echo "Started container '${CONTAINER_NAME}' in the background."
  echo "Attach with: ./docker/into_docker.sh"
else
  # If a command is provided, run it interactively (foreground) and exit when done.
  "${base_run_cmd[@]}" -it gasp:latest bash -c "$@"
fi
