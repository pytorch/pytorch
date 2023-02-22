import os
import subprocess
import sys
import shutil
import contextlib
import pathlib


'''
:: This inflates our log size slightly, but it is REALLY useful to be
:: able to see what our cl.exe commands are (since you can actually
:: just copy-paste them into a local Windows setup to just rebuild a
:: single file.)
:: log sizes are too long, but leaving this here incase someone wants to use it locally
:: set CMAKE_VERBOSE_MAKEFILE=1
'''

subprocess.run(os.environ['INSTALLER_DIR'] + '\\install_mkl.bat', shell=True)
subprocess.run(os.environ['INSTALLER_DIR'] + '\\install_magma.bat', shell=True)
subprocess.run(os.environ['INSTALLER_DIR'] + '\\install_sccache.bat', shell=True)
subprocess.run(os.environ['INSTALLER_DIR'] + '\\activate_miniconda3.bat', shell=True)


'''
:: Miniconda has been installed as part of the Windows AMI with all the dependencies.
:: We just need to activate it here
'''

# subprocess.run(os.environ['INSTALLER_DIR'] + '\\conda_install.bat', shell=True, check=True)

os.system(str(pathlib.Path(__file__).parent.resolve()) + '\\tst_build.bat')
