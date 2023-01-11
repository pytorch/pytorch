import os
import subprocess


tmp_dir_win = os.environ['TMP_DIR_WIN']
directory = tmp_dir_win + '\\bin'
os.mkdir(directory, mode=0o666)

if 'REBUILD' not in os.environ:

    while True:

        try:
            subprocess.call(tmp_dir_win + '\\bin\\sccache.exe --show-stats', shell=True)
            break

        except Exception as e:

            try:
                subprocess.call('taskkill /im sccache.exe /f /t', shell=True)
            except Exception as e:
                pass

            try:
                os.remove(tmp_dir_win + '\\bin\\sccache.exe')
            except Exception as e:
                pass

            try:
                os.remove(tmp_dir_win + '\\bin\\sccache-cl.exe')
            except Exception as e:
                pass


            if 'BUILD_ENVIRONMENT' not in os.environ:

                subprocess.call('curl --retry 3 -k https://s3.amazonaws.com/ossci-windows/sccache.exe ' +
                    '--output ' + tmp_dir_win + '\\bin\\sccache.exe', shell=True)

                subprocess.call('curl --retry 3 -k https://s3.amazonaws.com/ossci-windows/sccache-cl.exe' +
                    '--output ' + tmp_dir_win + '\\bin\\sccache-cl.exe', shell=True)

            else:

                subprocess.call('aws s3 cp s3://ossci-windows/sccache.exe ' + tmp_dir_win + '\\bin\\sccache.exe', shell=True)

                subprocess.call('aws s3 cp s3://ossci-windows/sccache-cl.exe ' + tmp_dir_win + '\\bin\\sccache-cl.exe', shell=True)
