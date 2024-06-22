#!/bin/bash
pip install --upgrade pip
pip install -r requirements.txt

#pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu102
pip install torchsummary
pip install -U pip && pip install onnxsim
pip install protobuf==3.20.*
