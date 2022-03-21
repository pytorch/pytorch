# export PYTORCH_ROCM_ARCH=gfx1030
pip3 uninstall torchvision -y
cd /var/lib/jenkins/vision/
python setup.py install