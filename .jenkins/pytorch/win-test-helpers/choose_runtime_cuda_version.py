import os
import subprocess


# The first argument should the CUDA version
subprocess.call('echo ' + os.environ['PATH'], shell=True)
subprocess.run(['echo ' + os.environ['CUDA_PATH'], shell=True)
os.environ['PATH'] = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v' + str(os.environ['1']) + '\\bin'
subprocess.call(os.environ['PATH'], shell=True)
