#include "torch/csrc/jit/fusers/cpu/fusion_compiler.h"

#include "torch/csrc/jit/fusers/interface.h"
#include "torch/csrc/jit/fusers/common/fusion_handle_impl.h"

#include "torch/csrc/jit/passes/shape_analysis.h" // EraseShapeInformation
#include "torch/csrc/utils/functional.h" //fmap
#include "torch/csrc/jit/ivalue.h" // IValue
#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/jit/assertions.h"

#include <cstdlib>
#include <string>
#include <sstream>
#include <tuple>

#if ((__GLIBC__ == 2) && (__GLIBC_MINOR__ >= 2)) || (__GLIBC__ > 2)
#include <spawn.h>
#ifdef POSIX_SPAWN_USEVFORK
#define USE_POSIX_SPAWN
#include <sys/wait.h>
extern char **environ;
#endif
#endif

#ifndef USE_POSIX_SPAWN
#ifndef __APPLE__
#include <malloc.h>
#define USE_MALLOC_TRIM
#endif
#endif

namespace torch { namespace jit { namespace cpufuser {

CPUFusionCompiler& getFusionCompiler() {
  static CPUFusionCompiler compiler;
  return compiler;
}

#ifdef USE_POSIX_SPAWN
struct PosixSpawnAttrVfork {
  explicit PosixSpawnAttrVfork() {
    int err = posix_spawnattr_init(&attr);
    if (err != 0) {
      AT_ERROR("posix_spawnattr_init: ", strerror(err));
    }

    err = posix_spawnattr_setflags(&attr, POSIX_SPAWN_USEVFORK);
    if (err != 0) {
      AT_ERROR("posix_spawnattr_setflags: ", strerror(err));
    }
  }
  ~PosixSpawnAttrVfork() {
    int err = posix_spawnattr_destroy(&attr);
    if (err != 0) {
      AT_WARN("posix_spawnattr_destroy: ", strerror(err));
    }
  }
  posix_spawnattr_t attr;
};
#endif // USE_POSIX_SPAWN

int runCommand(const std::string& command) {
#ifdef USE_POSIX_SPAWN
  // NB: Even with copy-on-write, fork can fail if the parent process's resident
  // memory usage is more than half the total RAM. Because of this, we avoid
  // system() (that calls fork) and try to use posix_spawn whenever possible.
  PosixSpawnAttrVfork attr;
  pid_t pid;

  // Run sh to run the provided command
  const char* cmd = command.c_str();
  std::vector<char> cmd_copy(cmd, cmd + command.size() + 1);
  char* argv[] = {"sh", "-c", cmd_copy.data(), NULL};

  int status = posix_spawn(&pid, "/bin/sh", NULL, &attr.attr, argv, environ);
  if (status != 0) {
    AT_ERROR("posix_spawn: ", strerror(status));
  }
  if (waitpid(pid, &status, 0) == -1) {
    AT_ERROR("waitpid: ", strerror(errno));
  }
  return WEXITSTATUS(status);
#else

  // No posix_spawn with vfork found on the system: try malloc_trim + system
#ifdef USE_MALLOC_TRIM
  malloc_trim(/*pad=*/0);
#endif // USE_MALLOC_TRIM
  int retval = system(command.c_str());
  if (retval == -1) {
    AT_ERROR("system(): ", strerror(errno));
  } else if (retval == 127) {
    AT_ERROR("system(): shell could not be executed in child process");
  }
  return retval;
#endif // USE_POSIX_SPAWN
}

static const std::string check_exists_string = "which '${program}' > /dev/null";
static bool programExists(const std::string& program) {
  TemplateEnv env;
  env.s("program", program);
  std::string cmd = format(check_exists_string, env);
  return 0 == runCommand(cmd);
}

CPUFusionCompiler::CPUFusionCompiler() {
  const char* cxx_env = getenv("CXX");
  if (cxx_env != nullptr) {
    config_.cxx = cxx_env;
  }

  if (!programExists(config_.cxx)) {
    config_.cxx = "";
  }

  const char* debug_env = getenv("PYTORCH_FUSION_DEBUG");
  config_.debug = debug_env && atoi(debug_env) != 0;
}

std::shared_ptr<FusionHandle> CPUFusionCompiler::getFusionHandle(Node* fusion_group) {
  int device = fusion_group->i(attr::device);
  JIT_ASSERT(device == kCPUDevice);
  auto graph = fusion_group->g(attr::Subgraph)->copy();
  EraseShapeInformation(*graph);
  std::stringstream key;
  key << "device " << device << "\n";
  key << *graph << "\n";
  std::string key_ = key.str();
  auto it = cache_map.find(key_);
  if (it == cache_map.end()) {
    std::tie(it, std::ignore) = cache_map.emplace(key_, std::make_shared<FusionHandleImpl>(graph, device));
  }
  return it->second;
}

std::vector<at::Tensor> CPUFusionCompiler::debugLaunchGraph(
  Graph& graph
, int device
, at::ArrayRef<at::Tensor> inputs) {
  auto wrapper_graph = std::make_shared<Graph>();
  Node* fusion_group = wrapper_graph->insertNode(wrapper_graph->createFusionGroup(device));
  fusion_group->g_(attr::Subgraph, graph.copy());
  for (size_t i = 0; i < graph.inputs().size(); ++i) {
    fusion_group->addInput(wrapper_graph->addInput());
  }
  for (size_t i = 0; i < graph.outputs().size(); ++i) {
    wrapper_graph->registerOutput(fusion_group->addOutput());
  }
  auto cache = getFusionHandle(fusion_group);
  Stack stack = fmap<IValue>(inputs);
  cache->run(stack);
  return fmap(stack, [](const IValue& iv) { return iv.toTensor(); });
}



} // namespace cpufuser
} // namespace jit
} // namespace torch
#ifdef USE_POSIX_SPAWN
#undef USE_POSIX_SPAWN
#endif
#ifdef USE_MALLOC_TRIM
#undef USE_MALLOC_TRIM
#endif
