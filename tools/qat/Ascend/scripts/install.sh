# sudo docker pull ubuntu:18.04
# sudo docker run --gpus all --shm-size=32g --privileged --name amct -v /data4:/data4 -it amct:3.7.5


apt-get update
apt-get install -y wget vim make zlib1g zlib1g-dev build-essential libbz2-dev libsqlite3-dev libssl-dev libxslt1-dev libffi-dev openssl python3-tk libjpeg-dev

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

conda create -n amct python=3.7.5
conda activate amct

# pip install -r requirements.txt
# pip install amct_pytorch/hotwheels_amct_pytorch+cu113-1.1.7-py3-none-linux_x86_64.tar.gz


docker start a66aba5f455e
docker exec -it a66aba5f455e /bin/bash
