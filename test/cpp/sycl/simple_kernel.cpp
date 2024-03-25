#include <sycl/sycl.hpp>

class SimpleKer {
 public:
  SimpleKer(float* a) : a_(a) {}
  void operator()(sycl::item<1> item) const {
    a_[item] = item;
  }

 private:
  float* a_;
};

int enum_gpu_device(sycl::device& dev) {
  std::vector<sycl::device> root_devices;
  auto platform_list = sycl::platform::get_platforms();
  for (const auto& platform : platform_list) {
    if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
      continue;
    }
    auto device_list = platform.get_devices();
    for (const auto& device : device_list) {
      if (device.is_gpu()) {
        root_devices.push_back(device);
      }
    }
  }

  if (root_devices.empty()) {
    throw std::runtime_error(
        "test_sycl: simple_kernel: no GPU device found ...");
    return -1;
  }

  dev = root_devices[0];
  return 0;
}

void itoa(float* res, int numel) {
  sycl::device dev;
  if (enum_gpu_device(dev)) {
    return;
  }
  sycl::queue q = sycl::queue(dev, sycl::property_list());

  float* a = sycl::malloc_shared<float>(numel, q);
  auto cgf = [&](sycl::handler& cgh) {
    cgh.parallel_for<SimpleKer>(sycl::range<1>(numel), SimpleKer(a));
  };

  auto e = q.submit(cgf);
  e.wait();

  memcpy(res, a, numel * sizeof(float));
  sycl::free(a, q);

  return;
}
