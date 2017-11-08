#include <sys/wait.h>
#include <set>
#include <atomic>
#include <signal.h>
#include "THP.h"

// In cases like DataLoader, if a worker process die due to bus error/segfault
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
static void HANDLER_NAME(int sig, siginfo_t *info, void *ctx)                 \
{                                                                             \
    write(STDERR_FILENO, ERROR_MSG, sizeof(ERROR_MSG) / sizeof(char));        \
    _exit(EXIT_FAILURE);                                                      \
}

// signal(2) is really not portable. So use sigaction.
// http://man7.org/linux/man-pages/man2/signal.2.html
static int setSignalHandler(int signal, void(*handler)(int, siginfo_t *, void *), struct sigaction *old_sa_ptr)
{
  struct sigaction sa;
  sa.sa_sigaction = handler;
  sa.sa_flags = SA_RESTART|SA_SIGINFO|SA_NOCLDSTOP;
  sigemptyset(&sa.sa_mask);
  return sigaction(signal, &sa, old_sa_ptr);
}

SIGNAL_HANDLER(SIGBUS, handler_SIGBUS, "ERROR: Unexpected bus error encountered in worker. "
  "This might be caused by insufficient shared memory (shm).\n");
SIGNAL_HANDLER(SIGSEGV, handler_SIGSEGV, "ERROR: Unexpected segmentation fault encountered in worker.\n");

PyObject *THPModule_setWorkerSignalHandlers(PyObject *module, PyObject *arg) {
  HANDLE_TH_ERRORS
  int error = 0;
  error |= setSignalHandler(SIGBUS, &handler_SIGBUS, NULL) != 0;
  error |= setSignalHandler(SIGSEGV, &handler_SIGSEGV, NULL) != 0;
  return PyBool_FromLong(!error);
  END_HANDLE_TH_ERRORS
}

static std::vector<pid_t> worker_pid_vec = {};
// The following are needed since std::vector is not asynchronous safe.
static std::atomic<pid_t *> worker_pids;
static std::atomic<size_t> num_worker_pids(0);
// Pipe used as a lock to avoid update of the above and SIGCHLD handler in parallel.
static int comm_pipe[2] = {-1, -1};

static int updatePIDsArray() {

  size_t new_size =  worker_pid_vec.size();
  auto new_ptr = (pid_t *)malloc(sizeof(pid_t) * new_size);
  for (size_t idx = 0; idx < new_size; idx++) {
    new_ptr[idx] = worker_pid_vec[idx];
  }

  // Block SIGCHLD handler for this thread so SIGCHLD handler can't interrupt
  // from this thread
  sigset_t sigset, old_sigset;
  sigemptyset(&sigset);
  sigaddset(&sigset, SIGCHLD);
  if (sigprocmask(SIG_BLOCK, &sigset, &old_sigset) != 0) {
    return -1;
  }
  // Acquire ``lock'' so handlers on other threads can't interrupt
  char c;
  read(comm_pipe[0], &c, 1);

  pid_t *old_ptr = worker_pids;
  if (new_size < num_worker_pids) {
    num_worker_pids = new_size;
    worker_pids = new_ptr;
  } else {
    worker_pids = new_ptr;
    num_worker_pids = new_size;
  }
  free(old_ptr);

  // Release ``lock''
  write(comm_pipe[1], &c, 1);
  // Restore handler for this thread.
  if (sigprocmask(SIG_SETMASK, &old_sigset, NULL) != 0) {
    return -1;
  }
  return 0;
}

static struct sigaction orig_SIGCHLD_sa;

// SIGCHLD hander should be registered on main loader process to catch any
// worker failing.
// Python handles are _set_main_signal_handers_for_workers() and
// _remove_main_signal_handers_for_workers().
static void handler_SIGCHLD_main(int sig, siginfo_t *info, void *ctx) {
  // Acquire ``lock'' so make sure that worker_pids won't change
  char c;
  read(comm_pipe[0], &c, 1);

  int error;
  siginfo_t infop;

  // Only check the pids we care about so that Python can see other processes'
  // status.
  for (size_t i = 0; i < num_worker_pids; i++) {
    // Use waitid rather than waitpid so that we can set NOWAIT, and that Python
    // can get whatever info it wants about the child process.
    error = waitid(P_PID, worker_pids[i], &infop, WEXITED|WNOHANG|WNOWAIT);
    if (error < 0)  // ignore errors
      continue;
    if ((infop.si_code == CLD_EXITED && infop.si_status != 0) ||  // exit with error
        (infop.si_code == CLD_KILLED) ||
        (infop.si_code == CLD_DUMPED) ||
        (infop.si_code == CLD_TRAPPED)) {
      _exit(EXIT_FAILURE);
    }
  }

  // Release ``lock''
  write(comm_pipe[1], &c, 1);

  // Call the overridden handler.
  if ((orig_SIGCHLD_sa.sa_flags | SA_SIGINFO) != 0) {
    // handler is sa_sigaction, this shouldn't really be SIG_IGN or SIG_DFL, but
    // they happen to be a union, and this fact is apparently used in Python,
    // so check here.
    if (orig_SIGCHLD_sa.sa_sigaction == (void (*)(int, siginfo_t *, void *)) SIG_IGN) {
      // SIG_IGN for SIGCHLD is to reap the child and do nothing else.
      while (waitpid(-1, 0, WNOHANG) > 0) {}
    } else if (orig_SIGCHLD_sa.sa_sigaction != (void (*)(int, siginfo_t *, void *)) SIG_DFL) {
      // SIG_DFL for SIGCHLD is to leave the child as a zombie (do nothing)
      orig_SIGCHLD_sa.sa_sigaction(sig, info, ctx);
    }
  } else {
    // handler is sa_handler
    // https://stackoverflow.com/a/24080440
    if (orig_SIGCHLD_sa.sa_handler == SIG_IGN) {
      while (waitpid(-1, 0, WNOHANG) > 0) {}
    } else if (orig_SIGCHLD_sa.sa_handler != SIG_DFL) {
      orig_SIGCHLD_sa.sa_handler(sig);
    }
  }
}

// returns -1 on error.
static int isSIGCHLDHanderSet() {
  struct sigaction sa;
  int error = sigaction(SIGCHLD, NULL, &sa);
  if (error == 0) {
    return ((sa.sa_flags | SA_SIGINFO) != 0) && (sa.sa_sigaction == &handler_SIGCHLD_main);
  } else {
    return -1;
  }
}

#define SAFECALL(CALL) if ((CALL) == -1) Py_RETURN_FALSE

// We don't want to exit on any SIGCHLD from any child. child_pids is a tuple
// of pids we are interested in.
PyObject *THPModule_setMainSignalHandlers(PyObject *module, PyObject *child_pids) {
  HANDLE_TH_ERRORS
  // assert these types are lock free, just to be safe
  THPUtils_assert(worker_pids.is_lock_free(), "worker_pids is not lock free");
  THPUtils_assert(num_worker_pids.is_lock_free(), "num_worker_pids is not lock free");

  THPUtils_assert(PyTuple_Check(child_pids), "_set_main_signal_handlers_for_workers "
        "expects a tuple, but got %s", THPUtils_typename(child_pids));

  if (comm_pipe[0] == -1) {
    // we have GIL here so we are fine
    SAFECALL(pipe(comm_pipe));
    char c = '_';
    write(comm_pipe[1], &c, 1);
  }

  auto size = PyTuple_GET_SIZE(child_pids);
  for (int idx = 0; idx < size; idx++) {
    PyObject* obj = PyTuple_GET_ITEM(child_pids, idx);
    worker_pid_vec.push_back((pid_t) THPUtils_unpackLong(obj));
  }
  SAFECALL(updatePIDsArray());

  // To avoid chain calling our handler, check if the current handler is already
  // set as ours.
  int set = 0;
  SAFECALL(set = isSIGCHLDHanderSet());
  if (set == 0) {
    SAFECALL(setSignalHandler(SIGCHLD, &handler_SIGCHLD_main, &orig_SIGCHLD_sa));
  }
  Py_RETURN_TRUE;
  END_HANDLE_TH_ERRORS
}

PyObject *THPModule_removeMainSignalHandlers(PyObject *module, PyObject *child_pids) {
  HANDLE_TH_ERRORS
  THPUtils_assert(PyTuple_Check(child_pids), "_remove_main_signal_handlers_for_workers "
        "expects a tuple, but got %s", THPUtils_typename(child_pids));

  auto size = PyTuple_GET_SIZE(child_pids);
  std::set<pid_t> pid_set;
  for (int idx = 0; idx < size; idx++) {
    PyObject* obj = PyTuple_GET_ITEM(child_pids, idx);
    pid_set.insert((pid_t) THPUtils_unpackLong(obj));
  }
  worker_pid_vec.erase(std::remove_if(worker_pid_vec.begin(), worker_pid_vec.end(),
     [&](pid_t p){return pid_set.find(p) != pid_set.end();}), worker_pid_vec.end());
  updatePIDsArray();

  int set = 0;
  SAFECALL(set = isSIGCHLDHanderSet());
  if (set == 1) {
    SAFECALL(sigaction(SIGCHLD, &orig_SIGCHLD_sa, NULL));
  }
  Py_RETURN_TRUE;
  END_HANDLE_TH_ERRORS
}

#undef SIGNAL_HANDLER

#else
// dummy implementations for windows

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
  {"_set_worker_signal_handlers",                (PyCFunction)THPModule_setWorkerSignalHandlers,    METH_NOARGS,  NULL},
  {"_set_main_signal_handlers_for_workers",      (PyCFunction)THPModule_setMainSignalHandlers,      METH_O,       NULL},
  {"_remove_main_signal_handlers_for_workers",   (PyCFunction)THPModule_removeMainSignalHandlers,   METH_O,       NULL},
  {NULL, NULL, 0, NULL}
};
