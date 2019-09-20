#include <torch/csrc/DataLoader.h>

// Together with `torch/utils/data/_utils/signal_handling.py`, the following
// is an effort to do our best to provide some error message to users when a
// worker dies due to error / critical signals.
//
// See NOTE [ Signal handling in multiprocessing data loading ] for more details.

// TODO: The following don't work on Windows. Specifically, sigaction, waitid
// calls, and SIGCHLD handler. Currently, dummy implementations are provided
// for Windows.

#ifndef _WIN32

#include <atomic>
#include <map>
#include <set>
#include <csignal>
#include <sstream>
#include <sys/wait.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/python_numbers.h>

using namespace torch;

// Critical signal handlers should be registered on worker processes before
// doing work.
// The handler will raise default handler so that the kill information will be
// retrieved from main process.
// Python handle is _set_worker_signal_handlers().
#define SIGNAL_HANDLER(SIGNAL, HANDLER_NAME, ERROR_MSG)                       \
static void HANDLER_NAME(int sig, siginfo_t *info, void *ctx)                 \
{                                                                             \
  auto _w = write(STDERR_FILENO, ERROR_MSG, sizeof(ERROR_MSG) / sizeof(char));\
  (void)_w;                                                                   \
  struct sigaction sa{};                                                        \
  sa.sa_handler = SIG_DFL;                                                    \
  sa.sa_flags = 0;                                                            \
  if (sigemptyset(&sa.sa_mask) != 0 || sigaction(SIGNAL, &sa, nullptr) != 0) {   \
    _exit(EXIT_FAILURE);                                                      \
  } else {                                                                    \
    raise(SIGNAL);                                                            \
  }                                                                           \
}

// signal(2) is really not portable. So use sigaction.
// http://man7.org/linux/man-pages/man2/signal.2.html
static inline void setSignalHandler(int signal, void(*handler)(int, siginfo_t *, void *), struct sigaction *old_sa_ptr)
{
  struct sigaction sa{};
  sa.sa_sigaction = handler;
  sa.sa_flags = SA_RESTART|SA_SIGINFO|SA_NOCLDSTOP|SA_NODEFER;
  if (sigemptyset(&sa.sa_mask) != 0 || sigaction(signal, &sa, old_sa_ptr) != 0) {
    std::ostringstream oss;
    oss << "An error occurred while setting handler for " << strsignal(signal) << ".";
    throw std::runtime_error(oss.str());
  }
}

SIGNAL_HANDLER(SIGBUS, handler_SIGBUS, "ERROR: Unexpected bus error encountered in worker. "
  "This might be caused by insufficient shared memory (shm).\n");
SIGNAL_HANDLER(SIGSEGV, handler_SIGSEGV, "ERROR: Unexpected segmentation fault encountered in worker.\n");
SIGNAL_HANDLER(SIGFPE, handler_SIGFPE, "ERROR: Unexpected floating-point exception encountered in worker.\n");

// When an error happend in DataLoader methods and Python starts to exit, the
// error trace will keep the loader alive, and Python may kill the children
// processes first before deleting the loader object. Then the cleaning up
// methods in DataLoader.__del__ are not yet called, and SIGCHILD will print an
// error saying a worker is killed by SIGTERM. So we suppress SIGTERM from main
// loader process here to avoid this by _exit(EXIT_SUCCESS). Note that if we
// exit with nonzero code, the loader SIGCHLD handler may report RuntimeError
// again, and then it defeats the whole purpose.
static void handler_SIGTERM(int sig, siginfo_t *info, void *ctx)
{
  if (info->si_pid == getppid()) {
    _exit(EXIT_SUCCESS);
  }
  struct sigaction sa{};
  sa.sa_handler = SIG_DFL;
  sa.sa_flags = 0;
  if (sigemptyset(&sa.sa_mask) != 0 || sigaction(SIGTERM, &sa, nullptr) != 0) {
    _exit(EXIT_FAILURE);
  } else {
    raise(SIGTERM);
  }
}

static PyObject *THPModule_setWorkerSignalHandlers(PyObject *module, PyObject *arg) {
  HANDLE_TH_ERRORS
  setSignalHandler(SIGBUS, &handler_SIGBUS, nullptr);
  setSignalHandler(SIGSEGV, &handler_SIGSEGV, nullptr);
  setSignalHandler(SIGTERM, &handler_SIGTERM, nullptr);
  setSignalHandler(SIGFPE, &handler_SIGFPE, nullptr);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static std::map<int64_t, std::set<pid_t>> worker_pids = {};

static PyObject *THPModule_errorIfAnyWorkerFails(PyObject *module, PyObject *noargs) {
  HANDLE_TH_ERRORS
  int error;
  std::set<pid_t> *pid_set;
  pid_t worker_pid;
  siginfo_t infop;

  // Only check the pids we care about
  for (auto& w : worker_pids) {
    pid_set = &(w.second);
    for (auto pid_it = pid_set->begin(); pid_it != pid_set->end(); ++pid_it) {
      worker_pid = *pid_it;
      // Use waitid rather than waitpid so that we can set NOWAIT, and that Python
      // and other handlers can get whatever info they want about the child.
      infop.si_pid = 0;
      error = waitid(P_PID, worker_pid, &infop, WEXITED|WNOHANG|WNOWAIT);
      // ignore errors and case with no waitable child
      if (error < 0 || infop.si_pid == 0)
        continue;
      if (infop.si_code == CLD_EXITED && infop.si_status != EXIT_SUCCESS) {  // exit with error
        std::ostringstream oss;
        oss << "DataLoader worker (pid " << worker_pid << ") exited "
            << "unexpectedly with exit code " << infop.si_status << ". "
            << "Details are lost due to multiprocessing. Rerunning with "
            << "num_workers=0 may give better error trace.";
        // This is necessary. Otherwise, the runtime error will kill the other
        // workers, and trigger this again.
        pid_set->clear();
        throw std::runtime_error(oss.str());
      } else if (infop.si_code == CLD_KILLED || infop.si_code == CLD_DUMPED) {  // killed by signal
        std::ostringstream oss;
        oss << "DataLoader worker (pid " << worker_pid << ") is killed "
            << "by signal: " << strsignal(infop.si_status) << ". ";
        if (infop.si_status == SIGBUS) {
            oss << "It is possible that dataloader's workers are out of shared memory. "
                << "Please try to raise your shared memory limit.";
        }
        // This is necessary. Otherwise, the runtime error will kill the other
        // workers, and trigger this again.
        pid_set->clear();
        throw std::runtime_error(oss.str());
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// We don't want to exit on any SIGCHLD from any child. child_pids is a tuple
// of pids we are interested in.
static PyObject *THPModule_setWorkerPIDs(PyObject *module, PyObject *args) {
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 2) {
    throw TypeError("_set_worker_pids expects exactly 2 arguments.");
  }
  int64_t key = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
  if (worker_pids.find(key) != worker_pids.end()) {
    throw ValueError("_set_worker_pids should be called only once for each _DataLoaderIter.");
  }
  PyObject *child_pids = PyTuple_GET_ITEM(args, 1);
  if (!PyTuple_Check(child_pids)) {
    throw TypeError("_set_worker_pids expects a tuple for child_pids, but got %s.",
        Py_TYPE(child_pids)->tp_name);
  }

  std::set<pid_t> pids_set = {};
  auto size = PyTuple_GET_SIZE(child_pids);
  for (int idx = 0; idx < size; idx++) {
    PyObject* obj = PyTuple_GET_ITEM(child_pids, idx);
    pids_set.insert(static_cast<pid_t>(THPUtils_unpackLong(obj)));
  }

  worker_pids[key] = pids_set;

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject *THPModule_removeWorkerPIDs(PyObject *module, PyObject *loader_id) {
  HANDLE_TH_ERRORS

  int64_t key = THPUtils_unpackLong(loader_id);
  auto it = worker_pids.find(key);
  if (it == worker_pids.end()) {
    throw ValueError("Cannot find worker information for _DataLoaderIter with id %ld.", key);
  }
  worker_pids.erase(it);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

#undef SIGNAL_HANDLER

#else
// dummy implementations for windows

static PyObject *THPModule_setWorkerSignalHandlers(PyObject *module, PyObject *_ignored) {
  Py_RETURN_NONE;
}

static PyObject *THPModule_setWorkerPIDs(PyObject *module, PyObject *_ignored) {
  Py_RETURN_NONE;
}

static PyObject *THPModule_removeWorkerPIDs(PyObject *module, PyObject *_ignored) {
  Py_RETURN_NONE;
}

static PyObject *THPModule_errorIfAnyWorkerFails(PyObject *module, PyObject *_ignored) {
  Py_RETURN_NONE;
}

#endif

PyMethodDef DataLoaderMethods[] = {
  {"_set_worker_signal_handlers",  (PyCFunction)THPModule_setWorkerSignalHandlers,  METH_NOARGS,   nullptr},
  {"_set_worker_pids",             (PyCFunction)THPModule_setWorkerPIDs,            METH_VARARGS,  nullptr},
  {"_remove_worker_pids",          (PyCFunction)THPModule_removeWorkerPIDs,         METH_O,        nullptr},
  {"_error_if_any_worker_fails",   (PyCFunction)THPModule_errorIfAnyWorkerFails,    METH_NOARGS,   nullptr},
  {nullptr, nullptr, 0, nullptr}
};
