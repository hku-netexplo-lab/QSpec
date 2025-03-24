echo "Installing QSpec"
echo "Installing VLLM for QSpec"
pip install -e .
echo "Installing QSpec dependencies"
pip install datasets
cd third-party
git clone https://github.com/NVIDIA/cutlass.git
cd ..
pip install -e third-party/ao
pip install -e third-party/fast-hadamard-transform
pip install -e third-party/kernels
pip install -e third-party/QuaRot
echo "QSpec installed"
CUDA_DEVICE_ORDER=PCI_BUS_ID

