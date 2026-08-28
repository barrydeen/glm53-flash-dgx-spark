#!/usr/bin/env bash
# Build the full GLM-5.3-Flash / DGX Spark image chain, in order.
# Each layer's FROM is the previous tag, so order is not optional.
#
# Every patch asserts the exact source text it expects and exits non-zero if
# the base image has drifted -- so a failure here is informative, not a flake.
set -euo pipefail

cd "$(dirname "$0")/../docker"

BASE="vllm/vllm-openai:glm53-flash-arm64-cu130"

if ! docker image inspect "$BASE" >/dev/null 2>&1; then
  echo "Pulling base image $BASE ..."
  docker pull "$BASE"
fi

for v in 1 2 3 4 5 6 7 8; do
  echo "==> building glm53:sm121-v$v"
  docker build -f "Dockerfile.sm121-v$v" -t "glm53:sm121-v$v" .
done

echo "==> building glm53:vllm-dflash2-v1"
docker build -f Dockerfile.vllm-dflash2 -t glm53:vllm-dflash2-v1 .

echo
echo "Built glm53:vllm-dflash2-v1"
echo
echo "Ship it to the worker node over the direct interconnect (not the LAN):"
echo "  docker save glm53:vllm-dflash2-v1 | ssh <worker-interconnect-ip> docker load"
