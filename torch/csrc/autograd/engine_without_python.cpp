#include "torch/csrc/autograd/engine_without_python.h"

#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/utils/auto_gpu.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <typeinfo>
#include <sstream>
#include <queue>
#include <TH/TH.h>

#ifdef WITH_CUDA
#include <cuda.h>
#include <THC/THC.h>
#endif
namespace torch{ namespace autograd{
    
Engine& Engine::getDefaultEngine(){
    static Engine engine;
    return engine;
}

}
}
