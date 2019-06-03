import multiprocessing
import threading
from pybind11_tests import gil_scoped as m


def _run_in_process(target, *args, **kwargs):
    """Runs target in process and returns its exitcode after 10s (None if still alive)."""
    process = multiprocessing.Process(target=target, args=args, kwargs=kwargs)
    process.daemon = True
    try:
        process.start()
        # Do not need to wait much, 10s should be more than enough.
        process.join(timeout=10)
        return process.exitcode
    finally:
        if process.is_alive():
            process.terminate()


def _python_to_cpp_to_python():
    """Calls different C++ functions that come back to Python."""
    class ExtendedVirtClass(m.VirtClass):
        def virtual_func(self):
            pass

        def pure_virtual_func(self):
            pass

    extended = ExtendedVirtClass()
    m.test_callback_py_obj(lambda: None)
    m.test_callback_std_func(lambda: None)
    m.test_callback_virtual_func(extended)
    m.test_callback_pure_virtual_func(extended)


def _python_to_cpp_to_python_from_threads(num_threads, parallel=False):
    """Calls different C++ functions that come back to Python, from Python threads."""
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=_python_to_cpp_to_python)
        thread.daemon = True
        thread.start()
        if parallel:
            threads.append(thread)
        else:
            thread.join()
    for thread in threads:
        thread.join()


def test_python_to_cpp_to_python_from_thread():
    """Makes sure there is no GIL deadlock when running in a thread.

    It runs in a separate process to be able to stop and assert if it deadlocks.
    """
    assert _run_in_process(_python_to_cpp_to_python_from_threads, 1) == 0


def test_python_to_cpp_to_python_from_thread_multiple_parallel():
    """Makes sure there is no GIL deadlock when running in a thread multiple times in parallel.

    It runs in a separate process to be able to stop and assert if it deadlocks.
    """
    assert _run_in_process(_python_to_cpp_to_python_from_threads, 8, parallel=True) == 0


def test_python_to_cpp_to_python_from_thread_multiple_sequential():
    """Makes sure there is no GIL deadlock when running in a thread multiple times sequentially.

    It runs in a separate process to be able to stop and assert if it deadlocks.
    """
    assert _run_in_process(_python_to_cpp_to_python_from_threads, 8, parallel=False) == 0


def test_python_to_cpp_to_python_from_process():
    """Makes sure there is no GIL deadlock when using processes.

    This test is for completion, but it was never an issue.
    """
    assert _run_in_process(_python_to_cpp_to_python) == 0
