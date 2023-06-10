#ifdef _WIN32
#include <wchar.h> // _wgetenv for nvtx
#endif
#include <nvToolsExt.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace cuda {
namespace shared {

#ifndef USE_ROCM

// registry idea from NVIDIA/nvtx-plugins  @jdekhtiar
class DomainCategoryRegistry {
 public:
  DomainCategoryRegistry() = default;

  ~DomainCategoryRegistry() {
    for (auto& domain : domains) {
      nvtxDomainDestroy(domain.second);
    }
  }

  nvtxDomainHandle_t getDomain(const std::string& domain_name) {
    if (const auto it = domains.find(domain_name); it != domains.end()) {
      return it->second;
    }

    const auto domain_handle = nvtxDomainCreateA(domain_name.c_str());
    domains[domain_name] = domain_handle;
    return domain_handle;
  }

  uint64_t getCategory(
      const std::string& category_name,
      std::optional<std::string> domain_name) {
    const auto id = category_name + "@" + domain_name.value_or("");
    if (const auto it = categories.find(id); it != categories.end()) {
      return it->second;
    }

    if (domain_name) {
      // category local to domain
      nvtxDomainNameCategoryA(
          getDomain(domain_name.value()),
          category_offset,
          category_name.c_str());
    } else {
      // category local to process
      nvtxNameCategoryA(category_offset, category_name.c_str());
    }

    categories[id] = category_offset;
    return category_offset++;
  }

 private:
  std::unordered_map<std::string, nvtxDomainHandle_t> domains;
  std::unordered_map<std::string, uint64_t> categories;
  uint64_t category_offset = 1337;
};

static DomainCategoryRegistry registry;

nvtxEventAttributes_t getAttributes(
    const std::string& msg,
    std::optional<std::string> domain,
    std::optional<std::string> category,
    std::optional<uint32_t> color) {
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;

  if (category) {
    eventAttrib.category = registry.getCategory(category.value(), domain);
  }

  if (color) {
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = color.value();
  }

  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = msg.c_str();
  return eventAttrib;
}

#endif

int rangePush(
    const std::string& msg,
    std::optional<std::string> domain,
    std::optional<std::string> category,
    std::optional<uint32_t> color) {
#ifdef USE_ROCM
  return roctxRangePushA(msg.c_str());
#else
  if (!domain && !category && !color) {
    return nvtxRangePushA(msg.c_str());
  }

  const auto eventAttrib = getAttributes(msg, domain, category, color);

  if (domain) {
    return nvtxDomainRangePushEx(
        registry.getDomain(domain.value()), &eventAttrib);
  }

  return nvtxRangePushEx(&eventAttrib);
#endif
}

int rangePop(std::optional<std::string> domain) {
#ifdef USE_ROCM
  return roctxRangePop();
#else
  if (domain) {
    return nvtxDomainRangePop(registry.getDomain(domain.value()));
  }

  return nvtxRangePop();
#endif
}

void mark(
    const std::string& msg,
    std::optional<std::string> domain,
    std::optional<std::string> category,
    std::optional<uint32_t> color) {
#ifdef USE_ROCM
  return roctxMarkA(msg.c_str());
#else
  if (!domain && !category && !color) {
    return nvtxMarkA(msg.c_str());
  }

  const auto eventAttrib = getAttributes(msg, domain, category, color);

  if (domain) {
    return nvtxDomainMarkEx(registry.getDomain(domain.value()), &eventAttrib);
  }

  nvtxMarkEx(&eventAttrib);
#endif
}

uint64_t rangeStart(
    const std::string& msg,
    std::optional<std::string> domain,
    std::optional<std::string> category,
    std::optional<uint32_t> color) {
#ifdef USE_ROCM
  return roctxRangeStartA(msg.c_str());
#else
  if (!domain && !category && !color) {
    return nvtxRangeStartA(msg.c_str());
  }

  const auto eventAttrib = getAttributes(msg, domain, category, color);

  if (domain) {
    return nvtxDomainRangeStartEx(
        registry.getDomain(domain.value()), &eventAttrib);
  }

  return nvtxRangeStartEx(&eventAttrib);
#endif
}

void rangeEnd(const uint64_t& range_id, std::optional<std::string> domain) {
#ifdef USE_ROCM
  return roctxRangeStop(range_id);
#else
  if (domain) {
    return nvtxDomainRangeEnd(registry.getDomain(domain.value()), range_id);
  }

  nvtxRangeEnd(range_id);
#endif
}

void initNvtxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto nvtx = m.def_submodule("_nvtx", "libNvToolsExt.so bindings");
  nvtx.def(
      "rangePush",
      rangePush,
      "NVTX range push",
      py::arg("message"),
      py::arg("domain"),
      py::arg("category"),
      py::arg("color"));
  nvtx.def("rangePop", rangePop, "NVTX range pop", py::arg("domain"));
  nvtx.def(
      "rangeStart",
      rangeStart,
      "NVTX range start",
      py::arg("message"),
      py::arg("domain"),
      py::arg("category"),
      py::arg("color"));
  nvtx.def(
      "rangeEnd", rangeEnd, "NVTX range end", py::arg("id"), py::arg("domain"));
  nvtx.def(
      "mark",
      mark,
      "NVTX mark",
      py::arg("message"),
      py::arg("domain"),
      py::arg("category"),
      py::arg("color"));
}

} // namespace shared
} // namespace cuda
} // namespace torch
