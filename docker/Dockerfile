FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:20.11-py3
RUN ["apt-get", "install", "-y", "linux-headers-$(uname -r)"]
RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb
RUN sudo dpkg -i cuda-keyring_1.0-1_all.deb
# Install NVIDIA driver on Ubuntu 
# sudo apt-get install linux-headers-$(uname -r)
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "cuda-drivers"]
RUN ["apt-get", "install", "-y", "zsh"]
RUN ["apt-get", "install", "-y", "git"]
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
COPY requirements.txt requirements.txt
RUN pip install setuptools Cython numpy pandas scipy statsmodels scikit-learn tensorboard_logger tqdm seaborn setproctitle tfrecord torch torchvision rtpt seaborn pytorch-lightning --upgrade
WORKDIR /workspace
ENV PYTHONPATH "${PYTHONPATH}:./"