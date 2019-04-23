FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
ARG DEBIAN_FRONTEND=noninteractive
ENV ANACONDA /opt/anaconda3
ENV CUDA_PATH /usr/local/cuda
ENV PATH ${ANACONDA}/bin:${CUDA_PATH}/bin:$PATH
ENV LD_LIBRARY_PATH ${ANACONDA}/lib:${CUDA_PATH}/bin64:$LD_LIBRARY_PATH
ENV C_INCLUDE_PATH ${CUDA_PATH}/include
RUN apt-get update && apt-get install -y --no-install-recommends \
         wget \
	 apt-utils \
	 axel 
#         imagemagick \
#         libopencv-dev \
#         python-opencv \
#         build-essential \
#         cmake \
#         git \
#         curl \
#         ca-certificates \
#         libjpeg-dev \
#         libpng-dev \
#         axel \
#         zip \
#         unzip \
RUN wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -P /tmp
WORKDIR /work
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
RUN axel http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
RUN DEBIAN_FRONTEND=noninteractive apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confnew" install ./cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
RUN axel http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
RUN apt-get install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
RUN apt-get update && apt-get install -y \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 
RUN apt-get update && apt-get install -y --allow-change-held-packages libcudnn7=7.2.1.38-1+cuda9.0 libnccl2=2.2.13-1+cuda9.0 cuda-command-line-tools-9-0
RUN bash /tmp/Anaconda3-5.0.1-Linux-x86_64.sh -b -p $ANACONDA
RUN rm /tmp/Anaconda3-5.0.1-Linux-x86_64.sh -rf
RUN conda install -y -c anaconda pip 
RUN pip install scikit-umfpack
RUN pip install cupy-cuda91
