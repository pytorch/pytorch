#!/bin/bash
conda_installation=''
environment_name=''
python_version=''
python_packages_to_install=''
pytorch_package_path=''
while getopts 'c:e:p:P:a:' flag; do
  case "${flag}" in
  c) conda_installation="${OPTARG}" ;;
  e) environment_name="${OPTARG}" ;;
  p) python_version="${OPTARG}" ;;
  P) python_packages_to_install="${OPTARG}" ;;
  a) pytorch_package_path="${OPTARG}" ;;
  esac
done
eval "$($conda_installation/bin/conda shell.bash hook)"
conda create -n $environment_name -y python=$python_version
conda activate $environment_name
pip install $python_packages_to_install
pip uninstall -y torch
pip install --upgrade --force-reinstall $pytorch_package_path