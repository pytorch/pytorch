#include "THP.h"

// In cases like DataLoader, if a worker process die due to bus error/segfault
// or just hang, the main process, if implemented with
// multiprocessing.queue.SimpleQueue, will hang waiting for data. This is
// difficult to avoid on PyTorch side as it can be caused by limited shm, or
// other libraries users call in the workers. The following methods is an effort
// to do our best provide some error message to users when such unfortunate
// events happen.

// TODO: The following don't work on Windows. Specifically, sigaction, waitid
// calls ,and SIGCHLD handler. Currently, dummy implementations are provided
// for Windows.

#ifndef _WIN32

#include <sys/wait.h>
#include <map>
#include <set>
#include <atomic>
#include <signal.h>


// Critical signal handlers should be registered on worker processes before
// doing work.
// The handler will raise default handler so that the kill information will be
// retrieved from main process.
// Python handle is _set_worker_signal_handlers().
#define SIGNAL_HANDLER(SIGNAL, HANDLER_NAME, ERROR_MSG)                       \
static void HANDLER_NAME(int sig, siginfo_t *info, void *ctx)                 \
{                                                                             \
  (void)write(STDERR_FILENO, ERROR_MSG, sizeof(ERROR_MSG) / sizeof(char));    \
  struct sigaction sa;                                                        \
  sa.sa_handler = SIG_DFL;                                                    \
  sa.sa_flags = 0;                                                            \
  if (sigemptyset(&sa.sa_mask) != 0 || sigaction(SIGNAL, &sa, NULL) != 0) {   \
    _exit(EXIT_FAILURE);                                                      \
  } else {                                                                    \
    raise(SIGNAL);                                                            \
  }                                                                           \
}

// signal(2) is really not portable. So use sigaction.
// http://man7.org/linux/man-pages/man2/signal.2.html
static void setSignalHandler(int signal, void(*handler)(int, siginfo_t *, void *), struct sigaction *old_sa_ptr)
{
  struct sigaction sa;
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

PyObject *THPModule_setWorkerSignalHandlers(PyObject *module, PyObject *arg) {
  HANDLE_TH_ERRORS
  setSignalHandler(SIGBUS, &handler_SIGBUS, NULL);
  setSignalHandler(SIGSEGV, &handler_SIGSEGV, NULL);
  Py_RETURN_TRUE;
  END_HANDLE_TH_ERRORS
}

static std::map<int64_t, std::set<pid_t>> worker_pids = {};

PyObject *THPModule_errorIfAnyWorkerFails(PyObject *module) {
  HANDLE_TH_ERRORS
  int error;
  std::set<pid_t> *pid_set;
  pid_t pid;
  siginfo_t infop;

  // Only check the pids we care about
  for (auto it = worker_pids.begin(); it != worker_pids.end(); ++it) {
    pid_set = &(it->second);
    for (auto pid_it = pid_set->begin(); pid_it != pid_set->end(); ++pid_it) {
      pid = *pid_it;
      // Use waitid rather than waitpid so that we can set NOWAIT, and that Python
      // and other handlers can get whatever info they want about the child.
      infop.si_pid = 0;
      error = waitid(P_PID, pid, &infop, WEXITED|WNOHANG|WNOWAIT);
      // ignore errors and case with no waitable child
      if (error < 0 || infop.si_pid == 0)
        continue;
      if (infop.si_code == CLD_EXITED && infop.si_status != 0) {  // exit with error
        std::ostringstream oss;
        oss << "DataLoader worker (pid " << pid << ") exited unexpectedly "
            << "with exit code " << infop.si_status << ".";
        // This is necessary. Otherwise, the runtime error will kill the other
        // workers, and trigger this again.
        pid_set->clear();
        throw std::runtime_error(oss.str());
      }  else if (infop.si_code == CLD_KILLED || infop.si_code == CLD_DUMPED) {  // killed by signal
        std::ostringstream oss;
        oss << "DataLoader worker (pid " << pid << ") is killed by signal: "
            << strsignal(infop.si_status) << ".";
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
PyObject *THPModule_updateWorkerPIDs(PyObject *module, PyObject *args) {
  HANDLE_TH_ERRORS
  Py_ssize_t num_args = args ? (Py_ssize_t) PyTuple_Size(args) : 0;
  THPUtils_assert(num_args == 2, "_update_worker_pids expectes exactly 2 arguments.");
  int64_t key = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
  THPUtils_assert(worker_pids.find(key) == worker_pids.end(), "_update_worker_pids "
        "should be called only once for each DataLoader.");
  PyObject *child_pids = PyTuple_GET_ITEM(args, 1);
  THPUtils_assert(PyTuple_Check(child_pids), "_update_worker_pids "
        "expects a tuple for child_pids, but got %s.", THPUtils_typename(child_pids));

  std::set<pid_t> pids_set = {};
  auto size = PyTuple_GET_SIZE(child_pids);
  for (int idx = 0; idx < size; idx++) {
    PyObject* obj = PyTuple_GET_ITEM(child_pids, idx);
    pids_set.insert((pid_t) THPUtils_unpackLong(obj));
  }

  worker_pids[key] = pids_set;

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject *THPModule_removeWorkerPIDs(PyObject *module, PyObject *loader_id) {
  HANDLE_TH_ERRORS

  int64_t key = THPUtils_unpackLong(loader_id);
  THPUtils_assert(worker_pids.find(key) != worker_pids.end(), "Cannot find worker "
        "information for DataLoader with id %ld.", key);

  worker_pids.erase(key);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

#undef SIGNAL_HANDLER

#else
// dummy implementations for windows

PyObject *THPModule_setWorkerSignalHandlers(PyObject *module, PyObject *_ignored) {
    Py_RETURN_TRUE;
}

PyObject *THPModule_updateWorkerPIDs(PyObject *module, PyObject *_ignored) {
    Py_RETURN_TRUE;
}

PyObject *THPModule_removeWorkerPIDs(PyObject *module, PyObject *_ignored) {
    Py_RETURN_NONE;
}

PyObject *THPModule_errorIfAnyWorkerFails(PyObject *module, PyObject *_ignored) {
    Py_RETURN_NONE;
}

#endif

PyMethodDef DataLoaderMethods[] = {
  {"_set_worker_signal_handlers",  (PyCFunction)THPModule_setWorkerSignalHandlers,  METH_NOARGS,   NULL},
  {"_update_worker_pids",          (PyCFunction)THPModule_updateWorkerPIDs,         METH_VARARGS,  NULL},
  {"_remove_worker_pids",          (PyCFunction)THPModule_removeWorkerPIDs,         METH_O,        NULL},
  {"_error_if_any_worker_fails",   (PyCFunction)THPModule_errorIfAnyWorkerFails,    METH_NOARGS,   NULL},
  {NULL, NULL, 0, NULL}
};
