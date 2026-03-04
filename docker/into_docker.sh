#!/usr/bin/env bash
# Attach a shell to the running gasp container. Run from anywhere.
# If the container is not running, run ./docker/start_docker.sh first.

CONTAINER_NAME="gasp"

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Container '$CONTAINER_NAME' is not running."
  echo "Start it with: ./docker/start_docker.sh"
  exit 1
fi

docker exec -it "$CONTAINER_NAME" "${@:-bash}"
