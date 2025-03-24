echo "Installing QSpec"
echo "Installing VLLM for QSpec"
pip install -e .
echo "Installing QSpec dependencies"
pip install datasets
pip install -e third-party/ao
pip install -e third-party/fast-hadamard-transform
pip install -e third-party/kernels
pip install -e third-party/QuaRot
echo "QSpec installed"
CUDA_DEVICE_ORDER=PCI_BUS_ID

