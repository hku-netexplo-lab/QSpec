DOCKER_BINARY="docker"
IMAGE_NAME="QSpec"
echo "check if mount_path.txt exists"
if [ ! -f "mount_path.txt" ]; then
    echo "mount_path.txt does not exist, please creating one"
    exit 1
fi
MOUNT_PATH=$(<mount_path.txt)
echo "The value of MOUNT_PATH is: $MOUNT_PATH"
DATA_MOUNT=" -v /PATH_TO_YOUR_DATA_DIR:/data "
${DOCKER_BINARY} run --gpus all --privileged --cap-add=SYS_ADMIN --ipc host -v ${MOUNT_PATH}:/workspace/${IMAGE_NAME} $DATA_MOUNT --name ${IMAGE_NAME} -it ${IMAGE_NAME}:latest 