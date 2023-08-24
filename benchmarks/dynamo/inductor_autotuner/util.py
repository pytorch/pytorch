from os import listdir
from os.path import isdir, join


def kernel_iter(model_path, verbose=True):
    for kernel in sorted(listdir(model_path)):
        kernel_path = join(model_path, kernel)
        if not isdir(kernel_path):
            continue
        for py in listdir(kernel_path):
            py_path = join(kernel_path, py)
            if not py.endswith(".py"):
                continue

            # skip graph python file
            with open(py_path) as file:
                content = file.read()
                if "AsyncCompile()" in content:
                    if verbose:
                        print("Skip " + py_path + " GRAPH")
                    continue
            yield kernel, py, kernel_path, py_path
