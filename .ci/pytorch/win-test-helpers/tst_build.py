import subprocess

result = subprocess.run('build_pytorch.bat', shell=True)
result.check_returncode()
