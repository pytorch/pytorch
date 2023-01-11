import os
import subprocess
import sys


tmp_win_dir = os.environ['TMP_DIR_WIN']

if 'REBUILD' not in os.environ:

    try:

        if 'BUILD_ENVIRONMENT' not in os.environ:

            subprocess.check_call('curl --retry 3 -k https://s3.amazonaws.com/ossci-windows/mkl_2020.2.254.7z ' +
                    '--output ' + tmp_win_dir + '\\mkl.7z', shell=True)

        else:

            subprocess.check_call('aws s3 cp s3://ossci-windows/mkl_2020.2.254.7z ' +
                tmp_win_dir + '\\mkl.7z --quiet', shell=True)

        subprocess.check_call('7z x -aoa ' + tmp_win_dir + '\\mkl.7z -o' + tmp_win_dir + '\\mkl', shell=True)

    except Exception as e:

        subprocess.check_call('echo install mkl failed', shell=True)
        subprocess.check_call('echo ' + str(e), shell=True)
        sys.exit()


os.environ['CMAKE_INCLUDE_PATH'] = tmp_win_dir + '\\mkl\\include'

if 'LIB' in os.environ:
    os.environ['LIB'] = tmp_win_dir + '\\mkl\\lib;' + os.environ['LIB']
else:
    os.environ['LIB'] = tmp_win_dir + '\\mkl\\lib'
