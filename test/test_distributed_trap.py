import os
import tempfile
import sys
import random
import __test_main__

tmp_dir = tempfile.TemporaryDirectory()
os.environ["TEMP_DIR"] = tmp_dir.name
os.mkdir(os.path.join(tmp_dir.name, "barrier"))
os.mkdir(os.path.join(tmp_dir.name, "test_dir"))
init_dir_path = os.path.join(tmp_dir.name, "init_dir")
os.mkdir(init_dir_path)
init_method = os.environ.get('INIT_METHOD')
if init_method is not None and init_method == "zeus":
    os.environ['INIT_METHOD'] = 'zeus://unittest_' + \
        str(random.randint(1, 1000000000000))
else:
    os.environ['INIT_METHOD'] = 'file://' + \
        os.path.join(init_dir_path, 'shared_init_file')


if __name__ == '__main__':
    __test_main__.main(sys.argv)
