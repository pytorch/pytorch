import math
import sys
import errno
import os
import ctypes
import torch
import gc
import time
import signal
import unittest
import itertools
import warnings
from torch import multiprocessing as mp
from torch.utils.data import _utils, Dataset, TensorDataset, DataLoader, ConcatDataset
from torch.utils.data._utils import ExceptionWrapper, MP_STATUS_CHECK_INTERVAL
from torch.utils.data.dataset import random_split
from common_utils import (TestCase, run_tests, TEST_NUMPY, IS_WINDOWS, PY3,
                          IS_PYTORCH_CI, NO_MULTIPROCESSING_SPAWN, skipIfRocm,
                          load_tests)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    err_msg = ("psutil not found. Some critical data loader tests relying on it "
               "(e.g., TestDataLoader.test_proper_exit) will not run.")
    if IS_PYTORCH_CI:
        raise ImportError(err_msg)
    else:
        warnings.warn(err_msg)

try:
    import faulthandler
    HAS_FAULTHANDLER = True
except ImportError:
    HAS_FAULTHANDLER = False
    err_msg = ("faulthandler not found. Some data loader tests use it for error "
               "reporting (e.g., TestDataLoader.test_proper_exit).")
    if IS_PYTORCH_CI:
        raise ImportError(err_msg)
    else:
        warnings.warn(err_msg)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

# We cannot import TEST_CUDA from common_cuda here, because if we do that,
# the TEST_CUDNN line from common_cuda will be executed multiple times
# as well during the execution of this test suite, and it will cause
# CUDA OOM error on Windows.
TEST_CUDA = torch.cuda.is_available()

if not NO_MULTIPROCESSING_SPAWN:
    # Get a multiprocessing context because some test / third party library will
    # set start_method when imported, and setting again triggers RuntimeError.
    mp = mp.get_context(method='spawn')


# 60s of timeout?
# Yes, in environments where physical CPU resources are shared, e.g., CI, the
# time for a inter-process communication can be highly varying.  With 15~17s of
# timeout, we have observed flakiness in some CI builds (see
# pytorch/pytorch#14501, pytorch/pytorch#16608).  We follow the CPython
# multiprocessing setup and set the timeout to 60s here:
#
# https://github.com/python/cpython/blob/e8113f51a8bdf33188ee30a1c038a298329e7bfa/Lib/test/_test_multiprocessing.py#L73
JOIN_TIMEOUT = 60.0  # seconds


# takes in dummy var so this can also be used as a `worker_init_fn`
def set_faulthander_if_available(_=None):
    if HAS_FAULTHANDLER:
        faulthandler.enable()
        if not IS_WINDOWS:
            # windows does not have faulthandler.register
            # chain=False prevents the default behavior of killing the process
            faulthandler.register(signal.SIGUSR1, chain=False)


# Process `pid` must have called `set_faulthander_if_available`
def print_traces_of_all_threads(pid):
    if HAS_FAULTHANDLER:
        if not IS_WINDOWS:
            # use the custom signal if available
            os.kill(pid, signal.SIGUSR1)
        else:
            # otherwise we can still use the handler given by faulthandler.enable()
            # at the cost of killing the process.
            os.kill(pid, signal.SIGSEGV)
    else:
        # if there is no faulthandler, use SIGINT otherwise and hope for the best
        os.kill(pid, signal.SIGINT)
    # wait in parent process to give subprocess some time to print
    time.sleep(5)


# Stores the first encountered exception in .exception.
# Inspired by https://stackoverflow.com/a/33599967
class ErrorTrackingProcess(mp.Process):

    # Why no *args?
    #   py2 doesn't support def fn(x, *args, key=val, **kwargs)
    # Setting disable_stderr=True may generate a lot of unrelated error outputs
    # but could be helpful for debugging.
    def __init__(self, disable_stderr=True, **kwargs):
        super(ErrorTrackingProcess, self).__init__(**kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None
        self.disable_stderr = disable_stderr

    def run(self):
        set_faulthander_if_available()
        if self.disable_stderr:
            # Disable polluting stderr with errors that are supposed to happen.
            sys.stderr = open(os.devnull, "w")
        try:
            super(ErrorTrackingProcess, self).run()
            self._cconn.send(None)
        except Exception:
            self._cconn.send(ExceptionWrapper(sys.exc_info()))
            raise

    def print_traces_of_all_threads(self):
        assert self.is_alive(), "can only use print_traces_of_all_threads if the process is alive"
        assert not self.disable_stderr, "do not disable stderr if you use print_traces_of_all_threads"
        # On platforms without `SIGUSR1`, `set_faulthander_if_available` sets
        # `faulthandler.enable()`, and `print_traces_of_all_threads` may kill
        # the process. So let's poll the exception first
        _ = self.exception
        print_traces_of_all_threads(self.pid)

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        if self._exception is None:
            return None
        else:
            return self._exception.exc_type(self._exception.exc_msg)

    # ESRCH means that os.kill can't finds alive proc
    def send_signal(self, signum, ignore_ESRCH=False):
        try:
            os.kill(self.pid, signum)
        except OSError as e:
            if not ignore_ESRCH or e.errno != errno.ESRCH:
                raise


class TestProperExitDataset(object):
    def __init__(self, size, error_event):
        self.size = size
        self.error_event = error_event

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.error_event is not None and self.error_event.is_set():
            raise RuntimeError('Worker error')
        return torch.tensor([idx])


# See TestDataLoader.test_proper_exit for usage
def _test_proper_exit(use_workers, pin_memory, exit_method, hold_iter_reference,
                      loader_setup_event, tester_setup_event):
    num_workers = 2 if use_workers else 0

    if exit_method == 'worker_error' or exit_method == 'worker_kill':
        assert use_workers is True

    if exit_method == 'worker_error':
        worker_error_event = mp.Event()
    else:
        worker_error_event = None

    ds = TestProperExitDataset(12, worker_error_event)

    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory,
                        worker_init_fn=set_faulthander_if_available)
    error_it = 2

    if use_workers:
        # 2 is the magical per-worker prefetch number...
        # FIXME: change this after the number becomes configurable.
        assert len(loader) > (error_it + 2 + 1) * num_workers

    it = iter(loader)
    if use_workers:
        workers = it.workers

    def kill_pid(pid):
        psutil_p = psutil.Process(pid)
        psutil_p.kill()
        psutil_p.wait(JOIN_TIMEOUT)
        assert not psutil_p.is_running()

    for i, _ in enumerate(it):
        if i == 0:
            if not hold_iter_reference:
                del it
            loader_setup_event.set()
            tester_setup_event.wait()
            # ensure that the workers are still alive
            if use_workers:
                for w in workers:
                    assert w.is_alive()
            if worker_error_event is not None:
                worker_error_event.set()

        if i == error_it:
            if exit_method == 'loader_error':
                raise RuntimeError('Loader error')
            elif exit_method == 'loader_kill':
                kill_pid(os.getpid())
            elif exit_method == 'worker_kill':
                kill_pid(workers[0].pid)

    if not hold_iter_reference:
        # Tries to trigger the __del__ clean-up rather than the automatic
        # exiting of daemonic children. Technically it should be automatically
        # triggered, but I don't want to rely on the implementation detail of
        # Python gc.
        gc.collect()


class TestDataLoader(TestCase):

    def setUp(self):
        super(TestDataLoader, self).setUp()
        self.data = torch.randn(100, 2, 3, 5)
        self.labels = torch.randperm(50).repeat(2)
        self.dataset = TensorDataset(self.data, self.labels)

    @skipIfRocm
    @unittest.skipIf(not HAS_PSUTIL, "psutil not found")
    def test_proper_exit(self):
        (r'''There might be ConnectionResetError or leaked semaphore warning '''
         r'''(due to dirty process exit), but they are all safe to ignore''')

        # TODO: test the case where the pin_memory_thread triggers an
        #       error/fatal signal. I haven't found out how to properly do that.

        for use_workers, pin_memory, hold_iter_reference in itertools.product([True, False], repeat=3):
            # `hold_iter_reference` specifies whether we hold a reference to the
            # iterator. This is interesting because Python3 error traces holds a
            # reference to the frames, which hold references to all the local
            # variables including the iterator, and then the iterator dtor may
            # not be called before process end. It is important to see that the
            # processes still exit in both cases.

            if pin_memory and (not TEST_CUDA or NO_MULTIPROCESSING_SPAWN or IS_WINDOWS):
                # Can't use CUDA without spawn
                # For windows, pin_memory sometimes causes CUDA oom.
                continue

            # `exit_method` controls the way the loader process ends.
            #   - `*_kill` means that `*` is killed by OS.
            #   - `*_error` means that `*` raises an error.
            #   - `None` means that no error happens.
            # In all cases, all processes should end properly.
            if use_workers:
                exit_methods = [None, 'loader_error', 'loader_kill', 'worker_error', 'worker_kill']
            else:
                exit_methods = [None, 'loader_error', 'loader_kill']

            for exit_method in exit_methods:
                desc = []
                desc.append('use_workers={}'.format(use_workers))
                desc.append('pin_memory={}'.format(pin_memory))
                desc.append('hold_iter_reference={}'.format(hold_iter_reference))
                desc.append('exit_method={}'.format(exit_method))
                desc = 'test_proper_exit with ' + ', '.join(desc)

                # Event that the loader process uses to signal testing process
                # that various things are setup, including that the worker pids
                # are specified in `worker_pids` array.
                loader_setup_event = mp.Event()

                # Event that this process has finished setting up, and the
                # loader process can now proceed to trigger error events or
                # finish normally.
                tester_setup_event = mp.Event()

                loader_p = ErrorTrackingProcess(target=_test_proper_exit,
                                                args=(use_workers, pin_memory, exit_method,
                                                      hold_iter_reference, loader_setup_event,
                                                      tester_setup_event),
                                                disable_stderr=False)
                loader_p.start()
                loader_psutil_p = psutil.Process(loader_p.pid)

                # Wait for loader process to set everything up, e.g., starting
                # workers.
                loader_setup_event.wait(timeout=JOIN_TIMEOUT)
                if not loader_setup_event.is_set():
                    fail_msg = desc + ': loader process failed to setup within given time'
                    if loader_p.exception is not None:
                        fail_msg += ', and had exception {}'.format(loader_p.exception)
                    elif not loader_p.is_alive():
                        fail_msg += ', and exited with code {} but had no exception'.format(loader_p.exitcode)
                    else:
                        fail_msg += ', and is still alive.'
                    if loader_p.is_alive():
                        # this may kill the process, needs to run after the above lines
                        loader_p.print_traces_of_all_threads()
                    self.fail(fail_msg)

                # We are certain that the workers have started now.
                worker_psutil_ps = loader_psutil_p.children()

                def fail(reason):
                    report_psutil_attrs = ['pid', 'name', 'cpu_times', 'io_counters',
                                           'memory_full_info', 'num_ctx_switches',
                                           'open_files', 'threads', 'status',
                                           'nice', 'ionice']
                    if reason is None:
                        err_msg = desc
                    else:
                        err_msg = '{}: {}'.format(desc, reason)
                    err_msg += '\nLoader info:\n\t'
                    if loader_psutil_p.is_running():
                        err_msg += str(loader_psutil_p.as_dict(attrs=report_psutil_attrs))
                        # this may kill the process, needs to run after the above line
                        loader_p.print_traces_of_all_threads()
                    else:
                        err_msg += 'exited with code {}'.format(loader_p.exitcode)
                    if use_workers:
                        err_msg += '\nWorker(s) info:'
                        for idx, worker_psutil_p in enumerate(worker_psutil_ps):
                            err_msg += '\n\tWorker {}:\n\t\t'.format(idx)
                            if worker_psutil_p.is_running():
                                err_msg += str(worker_psutil_p.as_dict(attrs=report_psutil_attrs))
                                # this may kill the process, needs to run after the above line
                                print_traces_of_all_threads(worker_psutil_p.pid)
                            else:
                                err_msg += 'exited with unknown code'
                    self.fail(err_msg)

                tester_setup_event.set()

                try:
                    loader_p.join(JOIN_TIMEOUT + MP_STATUS_CHECK_INTERVAL)
                    if loader_p.is_alive():
                        fail_reason = 'loader process did not terminate'
                        if loader_p.exception is not None:
                            fail(fail_reason + ', and had exception {}'.format(loader_p.exception))
                        else:
                            fail(fail_reason + ', and had no exception')
                    _, alive = psutil.wait_procs(worker_psutil_ps, timeout=(MP_STATUS_CHECK_INTERVAL + JOIN_TIMEOUT))
                    if len(alive) > 0:
                        self.fail(get_fail_msg('worker process (pid(s) {}) did not terminate'.format(
                            ', '.join(str(p.pid) for p in alive))))
                    if exit_method is None:
                        if loader_p.exitcode != 0:
                            fail('loader process had nonzero exitcode {}'.format(loader_p.exitcode))
                    else:
                        if loader_p.exitcode == 0:
                            fail('loader process had zero exitcode')
                        if exit_method == 'loader_error':
                            if not isinstance(loader_p.exception, RuntimeError) or \
                                    'Loader error' not in str(loader_p.exception):
                                fail('loader process did not raise expected exception, but had {}'.format(
                                    loader_p.exception))
                        elif exit_method == 'worker_kill':
                            if isinstance(loader_p.exception, RuntimeError):
                                if 'DataLoader worker (pid' not in str(loader_p.exception):
                                    fail('loader process did not raise expected exception, but had {}'.format(
                                        loader_p.exception))
                            elif PY3 and isinstance(loader_p.exception, ConnectionRefusedError):
                                # Sometimes, when the worker is being killed and is freeing its
                                # resources, the unpickling in loader process will be met an
                                # a `ConnectionRefusedError` as it can not open a socket to receive
                                # resource. In such cases, the worker may not have fully exited,
                                # and the loader can't know this via `is_alive` check or `SIGCHLD`
                                # handler. So we permit this as an allowed error as well.
                                # After all, we are happy as long as it terminates.
                                pass
                            elif not Py3 and isinstance(loader_p.exception, OSError):
                                # Same reasoning as the above if-block for Py2,
                                # where ConnectionRefusedError isn't a thing.
                                if loader_p.exception.errno != errno.ECONNREFUSED:
                                    fail('loader process did not raise expected exception, but had {}'.format(
                                        loader_p.exception))
                            else:
                                fail('loader process did not raise expected exception, but had {}'.format(
                                    loader_p.exception))
                        elif exit_method == 'worker_error':
                            if not isinstance(loader_p.exception, RuntimeError) or \
                                    'Worker error' not in str(loader_p.exception):
                                fail('loader process did not raise expected exception, but had {}'.format(
                                    loader_p.exception))
                finally:
                    loader_p.terminate()


if __name__ == '__main__':
    run_tests()
