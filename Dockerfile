ARG PYTORCH="1.3"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    sudo vim wget unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Install mmdetection
RUN conda clean --all
COPY . /ApulisVision
WORKDIR /ApulisVision
RUN pip install -r requirements/production.txt -i  https://mirrors.aliyun.com/pypi/simple
RUN pip install -r requirements/build.txt -i  https://mirrors.aliyun.com/pypi/simple
RUN pip install -r requirements/optional.txt -i  https://mirrors.aliyun.com/pypi/simple
ENV FORCE_CUDA="1"
RUN pip install --no-cache-dir -e .

RUN pip install kfserving==0.3.0 -i  https://mirrors.aliyun.com/pypi/simple six==1.14.0
WORKDIR /ApulisVision
COPY tools/model2pickle_kfserving.py tools/model2pickle_kfserving.py 
ENTRYPOINT ["python", "tools/model2pickle_kfserving.py"]
