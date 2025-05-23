echo "Installing QSpec"
echo "Installing VLLM for QSpec"
git checkout work
pip install -e .
echo "Installing QSpec dependencies"
pip install datasets
cd third-party
git clone https://github.com/NVIDIA/cutlass.git
cd ..
pip install -e third-party/ao
pip install -e third-party/fast-hadamard-transform
# pip install flash-attn --no-build-isolation
cd third-party/kernels
python setup.py install
cd ../..
pip install -e third-party/QuaRot
echo "QSpec installed"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

