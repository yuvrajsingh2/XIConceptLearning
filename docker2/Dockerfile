FROM nvidia/cuda:11.4.3-base-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive 
RUN ["apt", "update"]
RUN ["apt-get", "install", "-y", "pip"]
RUN ["apt-get", "install", "-y", "git"]

RUN ["apt-get", "update"]
#RUN pip install torch==1.11.0+cu115 torchvision==0.13.1 --find-links https://download.pytorch.org/whl/torch_stable.html 
RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Clone the repository
RUN git clone https://github.com/yuvrajsingh2/XIConceptLearning.git /workspace/repositories/XIConceptLearning

WORKDIR /workspace/repositories/XIConceptLearning/
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --upgrade
ENV PYTHONPATH "${PYTHONPATH}:./"