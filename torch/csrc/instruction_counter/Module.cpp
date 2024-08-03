#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <torch/csrc/instruction_counter/Module.h>
#include <torch/csrc/utils/pybind.h>
#include <unistd.h>

#if defined(__linux__)
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#endif

namespace torch::instruction_counter {

struct read_format {
  uint64_t nr;
  uint64_t time_enabled;
  uint64_t time_running;
  struct {
    uint64_t value;
  } values[1];
};

int start() {
#if !defined(__linux__)
  printf("This systems seems not to be Linux");
  return -1;
#else

  // Construct base perf_event_attr struct
  struct perf_event_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.size = sizeof(attr);
  attr.exclude_kernel = 1;
  attr.disabled = 1;
  attr.exclude_hv = 1;
  attr.sample_period = 0;
  // Enable hardware counting
  attr.type = PERF_TYPE_HARDWARE;
  attr.config = PERF_COUNT_HW_INSTRUCTIONS;

  int fd = syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0);
  if (fd == -1) {
    fprintf(
        stderr,
        "Failed to open instruction count event: %s.\n",
        strerror(errno));
    return -1;
  }
  ioctl(fd, PERF_EVENT_IOC_RESET, 0); // Reset the counter
  ioctl(fd, PERF_EVENT_IOC_ENABLE, 0); // Enable the counter
  return fd;
#endif
}

uint64_t end(int fd) {
#if !defined(__linux__)
  "This systems seems not to be neither Linux" return -1;
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

  uint64_t total_instructions;

  // Read results
  int ret_val = read(fd, &total_instructions, sizeof(total_instructions));
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
