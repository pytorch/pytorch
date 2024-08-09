#include <torch/csrc/instruction_counter/Module.h>
#include <torch/csrc/utils/pybind.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace torch::instruction_counter {

long start() {
#if !defined(__linux__)
  throw std::runtime_error("This systems seems not to be Linux");
#else

  // Construct base perf_event_attr struct
  perf_event_attr attr{};
  memset(&attr, 0, sizeof(attr));
  attr.size = sizeof(attr);
  attr.exclude_kernel = 1;
  attr.disabled = 1;
  attr.exclude_hv = 1;
  attr.sample_period = 0;
  // Enable hardware counting
  attr.type = PERF_TYPE_HARDWARE;
  attr.config = PERF_COUNT_HW_INSTRUCTIONS;

  long fd = syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0);
  if (fd == -1) {
    fprintf(
        stderr,
        "Failed to open instruction count event: %s.\n",
        strerror(errno));
    return -1;
  }
  ioctl((int)fd, PERF_EVENT_IOC_RESET, 0); // Reset the counter
  ioctl((int)fd, PERF_EVENT_IOC_ENABLE, 0); // Enable the counter
  return fd;
#endif
}

uint64_t end(int fd) {
#if !defined(__linux__)
  throw std::runtime_error("This systems seems not to be Linux");
#else
  // Disable the event group
  if (ioctl(fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) == -1) {
    fprintf(
        stderr,
        "Error disabling perf event (fd: %d): %s\n",
        fd,
        strerror(errno));
    return -1;
  }

  uint64_t total_instructions = 0;

  // Read results
  long ret_val = read(fd, &total_instructions, sizeof(total_instructions));
  if (ret_val == -1) {
    fprintf(stderr, "Error reading perf event results: %s\n", strerror(errno));
    return -1;
  }

  close(fd);
  return total_instructions;
#endif
}

void initModule(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  auto instruction_counter = m.def_submodule(
      "_instruction_counter", "instruction_counter related pybind.");
  instruction_counter.def("start", start);
  instruction_counter.def("end", end);
}

} // namespace torch::instruction_counter
