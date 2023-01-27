import os
import subprocess


# The first argument should the CUDA version
subprocess.run('echo ' + os.environ['PATH'], shell=True)
subprocess.run(['echo ' + os.environ['CUDA_PATH'], shell=True)
os.environ['PATH'] = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v' + str(os.environ['1']) + '\\bin'
subprocess.run(os.environ['PATH'], shell=True)
