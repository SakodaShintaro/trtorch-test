FROM nvcr.io/nvidia/pytorch:20.10-py3

RUN apt-get update && apt-get install -y curl gnupg  && rm -rf /var/lib/apt/lists/*

RUN curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && \
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list

RUN apt-get update && apt-get install -y bazel-3.7.1 && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/bazel-3.7.1 /usr/bin/bazel

RUN pip install notebook

# trtorchの導入
WORKDIR /opt
RUN git clone https://github.com/NVIDIA/TRTorch trtorch
WORKDIR /opt/trtorch
RUN git checkout b228bf239aadd5f104af38ea64416bdda5f0aa57
RUN cp /opt/trtorch/docker/WORKSPACE.cu.docker /opt/trtorch/WORKSPACE

# Workaround for bazel expecting both static and shared versions, we only use shared libraries inside container
RUN cp /usr/lib/x86_64-linux-gnu/libnvinfer.so /usr/lib/x86_64-linux-gnu/libnvinfer_static.a

WORKDIR /opt/trtorch
RUN bazel build //:libtrtorch --compilation_mode opt

WORKDIR /opt/trtorch/py

RUN pip install ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension

# Locale is not set by default
RUN apt-get update && apt-get install -y locales ninja-build && rm -rf /var/lib/apt/lists/* && locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN python3 setup.py install --use-cxx11-abi

RUN conda init bash

ENV LD_LIBRARY_PATH /opt/conda/lib/python3.6/site-packages/torch/lib:$LD_LIBRARY_PATh


# extract trtorch
WORKDIR /root
RUN tar xvf /opt/trtorch/bazel-bin/libtrtorch.tar.gz .

# libtorch
RUN wget -P /root/ https://download.pytorch.org/libtorch/cu110/libtorch-shared-with-deps-1.7.1%2Bcu110.zip
RUN unzip -q libtorch-cxx11-abi-shared-with-deps-1.7.0.zip

# trtorch-test
RUN git clone https://github.com/SakodaShintaro/trtorch-test
WORKDIR /root/trtorch-test/cmake-build-debug
RUN cmake -DCMAKE_BUILD_TYPE=Release ..
RUN make -j$(nproc)
