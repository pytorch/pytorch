#!/bin/bash

sudo apt-get -qq update
sudo apt-get -qq install --allow-downgrades --allow-change-held-packages libnccl-dev=2.5.6-1+cuda10.1 libnccl2=2.5.6-1+cuda10.1
