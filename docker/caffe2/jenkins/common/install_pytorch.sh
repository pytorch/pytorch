#!/bin/bash

set -ex

pip install numpy
git clone https://github.com/pytorch/pytorch.git --recursive
cd pytorch
git checkout 7fcfed19e7c4805405f3bec311fc056803ca7afb
pip install -r requirements.txt
/usr/bin/python3.6 setup.py install
cd ..

