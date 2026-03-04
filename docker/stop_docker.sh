#!/usr/bin/env bash
# Stop the gasp dev container. Run from gasp_release/: ./docker/stop_docker.sh

set -e
CONTAINER_NAME="gasp"

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  docker stop "$CONTAINER_NAME"
  echo "Stopped container '${CONTAINER_NAME}'."
else
  echo "Container '${CONTAINER_NAME}' is not running."
  exit 0
fi
