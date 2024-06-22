#!/bin/bash
pip install --upgrade pip
pip install -r requirements.txt

pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
pip install torchsummary
