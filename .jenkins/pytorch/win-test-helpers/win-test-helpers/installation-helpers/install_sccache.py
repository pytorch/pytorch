import os
import subprocess


tmp_dir_win = os.environ['TMP_DIR_WIN']
directory = tmp_dir_win + '\\bin'
os.mkdir(directory, mode=0o666)

if 'REBUILD' not in os.environ:

    while True:

        try:
            result = subprocess.run(tmp_dir_win + '\\bin\\sccache.exe --show-stats', shell=True)
            result.check_returncode()

            break

        except Exception as e:

            subprocess.run('taskkill /im sccache.exe /f /t', shell=True)

            try:
                os.remove(tmp_dir_win + '\\bin\\sccache.exe')
            except Exception as e:
                pass

            try:
                os.remove(tmp_dir_win + '\\bin\\sccache-cl.exe')
            except Exception as e:
                pass


            if 'BUILD_ENVIRONMENT' not in os.environ:

                subprocess.run('curl --retry 3 -k https://s3.amazonaws.com/ossci-windows/sccache.exe ' +
                    '--output ' + tmp_dir_win + '\\bin\\sccache.exe', shell=True)

                subprocess.run('curl --retry 3 -k https://s3.amazonaws.com/ossci-windows/sccache-cl.exe' +
                    '--output ' + tmp_dir_win + '\\bin\\sccache-cl.exe', shell=True)

            else:

                subprocess.run('aws s3 cp s3://ossci-windows/sccache.exe ' + tmp_dir_win + '\\bin\\sccache.exe', shell=True)

                subprocess.run('aws s3 cp s3://ossci-windows/sccache-cl.exe ' + tmp_dir_win + '\\bin\\sccache-cl.exe', shell=True)
