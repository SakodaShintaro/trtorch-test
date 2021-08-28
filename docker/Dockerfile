FROM nvcr.io/nvidia/pytorch:21.07-py3

# get trtorch
WORKDIR /root
RUN wget https://github.com/NVIDIA/TRTorch/releases/download/v0.4.0/libtrtorch-v0.4.0-cudnn8.2-tensorrt8.0-cuda11.1-libtorch-1.9.0.tar.gz
RUN tar xvf libtrtorch-v0.4.0-cudnn8.2-tensorrt8.0-cuda11.1-libtorch-1.9.0.tar.gz .
