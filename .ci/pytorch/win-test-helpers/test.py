import subprocess

try:
    print(subprocess.check_call('hdsjkgfjhsb --version', shell=True))
except Exception as e:

    subprocess.call('echo install dependencies failed', shell=True)
    subprocess.call('echo ' + str(e), shell=True)

print("continue")
#print(f'{username} home directory is {home_dir}')
