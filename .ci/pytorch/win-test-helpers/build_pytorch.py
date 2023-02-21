import os
import subprocess
import sys
import shutil
import contextlib
import pathlib


@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


def append_multiple_lines(file_name, lines_to_append):
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        appendEOL = False
        # Move read cursor to the start of file.
        file_object.seek(0)
        # Check if file is not empty
        data = file_object.read(100)
        if len(data) > 0:
            appendEOL = True
        # Iterate over each string in the list
        for line in lines_to_append:
            # If file is not empty then append '\n' before first line for
            # other lines always append '\n' before appending line
            if appendEOL == True:
                file_object.write("\n")
            else:
                appendEOL = True
            # Append element at the end of file
            file_object.write(line)


'''
:: This inflates our log size slightly, but it is REALLY useful to be
:: able to see what our cl.exe commands are (since you can actually
:: just copy-paste them into a local Windows setup to just rebuild a
:: single file.)
:: log sizes are too long, but leaving this here incase someone wants to use it locally
:: set CMAKE_VERBOSE_MAKEFILE=1
'''

subprocess.run('echo ' + os.environ['PATH'], shell=True)
subprocess.run(os.environ['INSTALLER_DIR'] + '\\install_mkl.bat', shell=True)
subprocess.run(os.environ['INSTALLER_DIR'] + '\\install_magma.bat', shell=True)
subprocess.run(os.environ['INSTALLER_DIR'] + '\\install_sccache.bat', shell=True)
# subprocess.run('python ' + os.environ['INSTALLER_DIR'] + '\\activate_miniconda3.py', shell=True)


'''
:: Miniconda has been installed as part of the Windows AMI with all the dependencies.
:: We just need to activate it here
'''

result = subprocess.run(os.environ['INSTALLER_DIR'] + '\\conda_install.bat', shell=True)
result.check_returncode()

# os.system(os.environ['INSTALLER_DIR'] + '\\conda_install.bat')

os.system(str(pathlib.Path(__file__).parent.resolve()) + '\\tst_build.bat')
