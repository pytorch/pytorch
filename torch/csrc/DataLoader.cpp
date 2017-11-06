#include <sys/wait.h>
#include <set>
#include <signal.h>

#include "THP.h"

// In cases like data loader, if a worker process die due to bus error/segfault
// or just hang, the main process, if implemented with
// multiprocessing.queue.SimpleQueue, will hang waiting for data. This is
// difficult to avoid on PyTorch side as it can be caused by limited shm, or
// other libraries users call in the workers. The following methods is an effort
// to do our best provide some error message to users when such unfortunate
// events happen.

// TODO: The following don't work on Windows. Specifically, waitpid calls and
// SIGCHLD handler.

#ifndef _WIN32

// Critical signal handlers should be registered on worker processes before
// doing work.
// Python handle is _set_worker_signal_handlers().
#define SIGNAL_HANDLER(SIGNAL, HANDLER_NAME, ERROR_MSG)                       \
static void HANDLER_NAME(int sig)                                             \
{                                                                             \
    write(STDERR_FILENO, ERROR_MSG, sizeof(ERROR_MSG) / sizeof(char));        \
    _exit(EXIT_FAILURE);                                                      \
}

// signal(2) is really not portable. So use sigaction.
// http://man7.org/linux/man-pages/man2/signal.2.html
#define SET_SIGNAL_HANDLER(SIGNAL, HANDLER_NAME, ERROR_FLAG)                  \
{                                                                             \
  struct sigaction sa;                                                        \
  sa.sa_handler = HANDLER_NAME;                                               \
  sa.sa_flags = SA_RESTART;                                                   \
  sigemptyset(&sa.sa_mask);                                                   \
  ERROR_FLAG |= sigaction(SIGNAL, &sa, NULL) != 0;                            \
}

SIGNAL_HANDLER(SIGBUS, handler_SIGBUS, "ERROR: Unexpected bus error encountered in worker. "
  "This might be caused by insufficient shared memory (shm).\n");
SIGNAL_HANDLER(SIGSEGV, handler_SIGSEGV, "ERROR: Unexpected segmentation fault encountered in worker.\n");

static std::vector<pid_t> worker_pid_vec = {};
// The following are needed since std::vector is not asynchronous safe.
static std::atomic<pid_t *> worker_pids;
static std::atomic<size_t> num_worker_pids(0);
static std::atomic<size_t> num_dataloaders(0);

PyObject *THPModule_setWorkerSignalHandlers(PyObject *module, PyObject *arg) {
  HANDLE_TH_ERRORS
  int error = 0;
  SET_SIGNAL_HANDLER(SIGBUS, &handler_SIGBUS, error);
  SET_SIGNAL_HANDLER(SIGSEGV, &handler_SIGSEGV, error);
  if (error == 0) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

// SIGCHLD hander should be registered on main loader process to catch any
// worker failing. SIGALRM handler is needed for implementing timeout.
// Python handles are _set_main_signal_handers() and
// _remove_main_signal_handers().
static void handler_SIGCHLD_main(int sig) {
  int status;
  pid_t p;
  pid_t *pid_ptr = worker_pids;

  // Only check the pids we care about so that Python can see other processes'
  // status.
  for (size_t i = 0; i < num_worker_pids; i++) {
    // The flags and status checks ensure that we are really observing a child
    // exiting, rather than other cases such as SIGSTOP and SIGCONT.
    // https://stackoverflow.com/a/40707100
    p = waitpid(*pid_ptr, &status, WNOHANG|WUNTRACED|WCONTINUED);
    // Ignore errors / no changes
    if (p <= 0)
      continue;
    if (WIFCONTINUED(status) || WIFSTOPPED(status))
      continue;
    if (WIFEXITED(status) != 0 && WEXITSTATUS(status) == 0)
      continue;
    _exit(EXIT_FAILURE);
    pid_ptr++;
  }
}

// We don't want to exit on any SIGCHLD from any child. child_pids is a sequence
// of pids we are interested in.
PyObject *THPModule_setMainSignalHandlers(PyObject *module, PyObject *child_pids) {
  HANDLE_TH_ERRORS
  // assert these types are lock free, just to be safe
  THPUtils_assert(worker_pids.is_lock_free(), "worker_pids is not lock free");
  THPUtils_assert(num_worker_pids.is_lock_free(), "num_worker_pids is not lock free");

  THPUtils_assert(PyTuple_Check(child_pids), "_set_main_signal_handler "
        "expects a list, but got %s", THPUtils_typename(child_pids));

  num_dataloaders++;
  auto size = PyTuple_GET_SIZE(child_pids);
  for (int idx = 0; idx < size; idx++) {
    PyObject* obj = PyTuple_GET_ITEM(child_pids, idx);
    worker_pid_vec.push_back((pid_t) THPUtils_unpackLong(obj));
  }
  worker_pids = &worker_pid_vec[0];
  num_worker_pids = worker_pid_vec.size();

  int error = 0;
  SET_SIGNAL_HANDLER(SIGCHLD, &handler_SIGCHLD_main, error);
  if (error == 0) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject *THPModule_removeMainSignalHandlers(PyObject *module, PyObject *child_pids) {
  HANDLE_TH_ERRORS
  THPUtils_assert(PyTuple_Check(child_pids), "_remove_main_signal_handler "
        "expects a tuple or list, but got %s", THPUtils_typename(child_pids));

  auto size = PyTuple_GET_SIZE(child_pids);
  std::set<pid_t> pid_set;
  for (int idx = 0; idx < size; idx++) {
    PyObject* obj = PyTuple_GET_ITEM(child_pids, idx);
    pid_set.insert((pid_t) THPUtils_unpackLong(obj));
  }
  // During the following, worker_pids and num_worker_pids will likely not have
  // correct values. However, it is guaranteed that all values in this array are
  // child pids and contain all alive child pids.
  auto write_it = worker_pid_vec.begin();
  auto read_it = worker_pid_vec.begin();
  size_t remaining_num = 0;
  for (; read_it != worker_pid_vec.end(); read_it++) {
    if (pid_set.find(*read_it) == pid_set.end()) {
      *write_it = *read_it;
      write_it++;
      remaining_num++;
    }
  }
  worker_pids = &worker_pid_vec[0];
  num_worker_pids = remaining_num;
  worker_pid_vec.resize(remaining_num);

  int error = 0;

  // Need to restore original handler so that in case DataLoader errors, the
  // waitpids in hander won't block Python from updating Process.is_alive() and
  // Process.exitcode, etc.
  if (--num_dataloaders == 0)
    SET_SIGNAL_HANDLER(SIGCHLD, SIG_DFL, error);

  if (error == 0) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

#undef SIGNAL_HANDLER
#undef SET_SIGNAL_HANDLER

#else
// dummy implementations for eindows

PyObject *THPModule_setWorkerSignalHandlers(PyObject *module, PyObject *_ignored) {
    Py_RETURN_TRUE;
}

PyObject *THPModule_setMainSignalHandlers(PyObject *module, PyObject *_ignored) {
    Py_RETURN_TRUE;
}

PyObject *THPModule_removeMainSignalHandlers(PyObject *module, PyObject *_ignored) {
    Py_RETURN_TRUE;
}

#endif

PyMethodDef DataLoaderMethods[] = {
  {"_set_worker_signal_handlers",    (PyCFunction)THPModule_setWorkerSignalHandlers,    METH_NOARGS,  NULL},
  {"_set_main_signal_handlers",      (PyCFunction)THPModule_setMainSignalHandlers,      METH_O,       NULL},
  {"_remove_main_signal_handlers",   (PyCFunction)THPModule_removeMainSignalHandlers,   METH_O,       NULL},
  {NULL, NULL, 0, NULL}
};
