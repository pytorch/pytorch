import os
import subprocess


tmp_dir_win = os.environ['TMP_DIR_WIN']
directory = tmp_dir_win + '\\bin'
os.mkdir(directory, mode=0o666)

if os.environ['REBUILD'] == "":

    while True:

        try:
            subprocess.run([tmp_dir_win + '\\bin\\sccache.exe', '--show-stats'])
            break

        except Exception as e:

            try:
                subprocess.run(['taskkill', '/im', 'sccache.exe', '/f', '/t'])
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


            if os.environ['BUILD_ENVIRONMENT'] == "":

                subprocess.run(['curl', '--retry', '3', '-k',
                 'https://s3.amazonaws.com/ossci-windows/sccache.exe', '--output',
                    tmp_dir_win + '\\bin\\sccache.exe'])

                subprocess.run(['curl', '--retry', '3', '-k',
                 'https://s3.amazonaws.com/ossci-windows/sccache-cl.exe', '--output',
                    tmp_dir_win + '\\bin\\sccache-cl.exe'])

            else:

                subprocess.run(['aws', 's3', 'cp', 's3://ossci-windows/sccache.exe',
                    tmp_dir_win + '\\bin\\sccache.exe'])

                subprocess.run(['aws', 's3', 'cp', 's3://ossci-windows/sccache-cl.exe',
                    tmp_dir_win + '\\bin\\sccache-cl.exe'])
