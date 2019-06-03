import random
import os
import time
import datetime as dt

nfns = 4  # Functions per class
nargs = 4  # Arguments per function


def generate_dummy_code_pybind11(nclasses=10):
    decl = ""
    bindings = ""

    for cl in range(nclasses):
        decl += "class cl%03i;\n" % cl
    decl += '\n'

    for cl in range(nclasses):
        decl += "class cl%03i {\n" % cl
        decl += "public:\n"
        bindings += '    py::class_<cl%03i>(m, "cl%03i")\n' % (cl, cl)
        for fn in range(nfns):
            ret = random.randint(0, nclasses - 1)
            params  = [random.randint(0, nclasses - 1) for i in range(nargs)]
            decl += "    cl%03i *fn_%03i(" % (ret, fn)
            decl += ", ".join("cl%03i *" % p for p in params)
            decl += ");\n"
            bindings += '        .def("fn_%03i", &cl%03i::fn_%03i)\n' % \
                (fn, cl, fn)
        decl += "};\n\n"
        bindings += '        ;\n'

    result = "#include <pybind11/pybind11.h>\n\n"
    result += "namespace py = pybind11;\n\n"
    result += decl + '\n'
    result += "PYBIND11_MODULE(example, m) {\n"
    result += bindings
    result += "}"
    return result


def generate_dummy_code_boost(nclasses=10):
    decl = ""
    bindings = ""

    for cl in range(nclasses):
        decl += "class cl%03i;\n" % cl
    decl += '\n'

    for cl in range(nclasses):
        decl += "class cl%03i {\n" % cl
        decl += "public:\n"
        bindings += '    py::class_<cl%03i>("cl%03i")\n' % (cl, cl)
        for fn in range(nfns):
            ret = random.randint(0, nclasses - 1)
            params  = [random.randint(0, nclasses - 1) for i in range(nargs)]
            decl += "    cl%03i *fn_%03i(" % (ret, fn)
            decl += ", ".join("cl%03i *" % p for p in params)
            decl += ");\n"
            bindings += '        .def("fn_%03i", &cl%03i::fn_%03i, py::return_value_policy<py::manage_new_object>())\n' % \
                (fn, cl, fn)
        decl += "};\n\n"
        bindings += '        ;\n'

    result = "#include <boost/python.hpp>\n\n"
    result += "namespace py = boost::python;\n\n"
    result += decl + '\n'
    result += "BOOST_PYTHON_MODULE(example) {\n"
    result += bindings
    result += "}"
    return result


for codegen in [generate_dummy_code_pybind11, generate_dummy_code_boost]:
    print ("{")
    for i in range(0, 10):
        nclasses = 2 ** i
        with open("test.cpp", "w") as f:
            f.write(codegen(nclasses))
        n1 = dt.datetime.now()
        os.system("g++ -Os -shared -rdynamic -undefined dynamic_lookup "
            "-fvisibility=hidden -std=c++14 test.cpp -I include "
            "-I /System/Library/Frameworks/Python.framework/Headers -o test.so")
        n2 = dt.datetime.now()
        elapsed = (n2 - n1).total_seconds()
        size = os.stat('test.so').st_size
        print("   {%i, %f, %i}," % (nclasses * nfns, elapsed, size))
    print ("}")
