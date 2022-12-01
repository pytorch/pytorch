import os
import subprocess
import sys


cuda_version = os.environ['CUDA_VERSION']
tmp_win_dir = os.environ['TMP_DIR_WIN']
build_type = os.environ['BUILD_TYPE']


if cuda_version == "cpu":

    subprocess.run(['echo', 'skip magma installation for cpu builds'])
    sys.exit(0)


# remove dot in cuda_version, fox example 11.1 to 111


if os.environ['USE_CUDA'] != "1":

    sys.exit(0)


if not cuda_version.contains('.'):

    subprocess.run(['echo', 'CUDA version ' + cuda_version +
        'format isn\'t correct, which doesn\'t contain \'.\''])

    sys.exit(1)


os.environ['VERSION_SUFFIX'] = cuda_version.replace('.', '')
cuda_suffix = 'cuda' + cuda_version.replace('.', '')
os.environ['CUDA_SUFFIX'] = cuda_suffix


if cuda_suffix == '':

    subprocess.run(['echo', 'unknown CUDA version, please set \'CUDA_VERSION\' higher than 10.2'])

    sys.exit(1)


if 'REBUILD' not in os.environ:

    try:

        if 'BUILD_ENVIRONMENT' not in os.environ:

            subprocess.run(['curl', '--retry', '3', '-k',
                'https://s3.amazonaws.com/ossci-windows/magma_2.5.4_' + cuda_suffix +
                    '_' + build_type + '.7z', '--output', tmp_win_dir + '\\magma_2.5.4_'
                        + cuda_suffix + '_' + build_type + '.7z'])

        else:

            subprocess.run(['aws', 's3', 'cp', 's3://ossci-windows/magma_2.5.4_' +
                cuda_suffix + '_' + build_type + '.7z', tmp_win_dir + '\\magma_2.5.4_'
                    + cuda_suffix + '_' + build_type + '.7z', '--quiet'])

        subprocess.run(['7z', 'x', '-aoa', tmp_win_dir + '\\magma_2.5.4_' +
            cuda_suffix + '_' + build_type + '.7z', '-o' + tmp_win_dir + '\\magma'])

    except Exception as e:

        subprocess.run(['echo', 'install magma failed'])
        subprocess.run(['echo', e])
        sys.exit()


os.environ['MAGMA_HOME'] = tmp_win_dir + '\\magma'
