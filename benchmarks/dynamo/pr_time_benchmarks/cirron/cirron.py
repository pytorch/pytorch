import os
from ctypes import byref, c_int, c_uint64, CDLL, POINTER, Structure
from subprocess import call

from pkg_resources import resource_filename


class Counter(Structure):
    _fields_ = [
        ("time_enabled_ns", c_uint64),
        ("instruction_count", c_uint64),
        ("branch_misses", c_uint64),
        ("page_faults", c_uint64),
    ]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        repr = "Counter("

        for field, _ in Counter._fields_:
            repr += f"{field}={getattr(self, field)}, "
        repr = repr[:-2] + ")"

        return repr


lib_path = resource_filename(__name__, "cirronlib.so")
if not os.path.exists(lib_path):
    source_path = resource_filename(__name__, "cirronlib.cpp")
    print("cirronlib is not compiled yet, compiling it now...")
    print(
        f"calling c++ -std=c++17 -O3 -shared -fPIC -o {lib_path} {source_path}",
    )
    exit_status = call(
        f"c++ -std=c++17 -O3 -shared -fPIC -o {lib_path} {source_path}", shell=True
    )
    if exit_status != 0:
        raise Exception(
            "Failed to compile cirronlib.cpp, make sure you have 'c++' installed."
        )


overhead = {}


class Collector:
    cirron_lib = CDLL(lib_path)
    cirron_lib.start.argtypes = None
    cirron_lib.start.restype = c_int
    cirron_lib.end.argtypes = [c_int, POINTER(Counter)]
    cirron_lib.end.restype = None

    def __init__(self, measure_overhead=True):
        self._fd = None
        self.counters = Counter()
        self.measure_overhead = measure_overhead

        # We try to estimate what the overhead of the collector is, taking the minimum
        # of 10 runs.
        global overhead
        if measure_overhead and not overhead:
            for _ in range(10):
                with Collector(measure_overhead=False) as collector:
                    pass

                for field, _ in Counter._fields_:
                    if field not in overhead:
                        overhead[field] = getattr(collector.counters, field)
                    else:
                        overhead[field] = min(
                            overhead[field], getattr(collector.counters, field)
                        )

    def __enter__(self):
        ret_val = Collector.cirron_lib.start()
        if ret_val == -1:
            raise Exception(
                "Failed to start collector. Make sure you have the right permissions, you might need to use sudo."
            )
        self._fd = ret_val

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        ret_val = Collector.cirron_lib.end(self._fd, byref(self.counters))
        if ret_val == -1:
            raise Exception("Failed to end collector.")

        global overhead
        if self.measure_overhead and overhead:
            for field, _ in Counter._fields_:
                # Clamp the result of overhead substraction to 0.
                if getattr(self.counters, field) > overhead[field]:
                    setattr(
                        self.counters,
                        field,
                        getattr(self.counters, field) - overhead[field],
                    )
                else:
                    setattr(self.counters, field, 0)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return repr(self.counters)
