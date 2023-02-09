
FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
WORKDIR /workspace
RUN apt-get update
RUN apt-get install -y wget
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
RUN bash Miniconda3-py39_4.12.0-Linux-x86_64.sh -b
RUN rm -rf Miniconda3-py39_4.12.0-Linux-x86_64.sh
ENV PATH "$PATH:/root/miniconda3/bin"
RUN pip install --upgrade pip
RUN pip install notebook
RUN pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install flax
CMD [ "/bin/bash" ]