#!/bin/bash
DOCKER_BINARY="docker"
IMAGE_NAME="QSpec"
FILE_NAME="dockerfile"
DOCKER_BUILDKIT=1 ${DOCKER_BINARY} build -f ${FILE_NAME} -t ${IMAGE_NAME}:latest .