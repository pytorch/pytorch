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

os.system(str(pathlib.Path(__file__).parent.resolve()) + '\\build_pytorch.bat')
