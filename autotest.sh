#!/bin/bash
apt-get update -y && apt-get upgrade -y
apt-get install python3-pip ffmpeg libopenblas-dev -y
pip3 install gdown
gdown https://drive.google.com/uc?id=1fgckCX8O5R_HeQacWPzigxsXd2ytwBGN
pip3 install torch-1.8.0a0+37c1f4a-cp38-cp38-linux_aarch64.whl
rm torch-1.8.0a0+37c1f4a-cp38-cp38-linux_aarch64.whl
gdown https://drive.google.com/uc?id=15IRhB7yms6VYAJUfcVx9AN46oArFQUeu
pip3 install torchvision-0.9.0a0+01dfa8e-cp38-cp38-linux_aarch64.whl
rm torchvision-0.9.0a0+01dfa8e-cp38-cp38-linux_aarch64.whl
pip3 install pyyaml coloredlogs enlighten matplotlib seaborn tensorboard
ln -s /usr/bin/python3 /usr/bin/python