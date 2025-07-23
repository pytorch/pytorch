#include <moodycamel/concurrentqueue.h>
#include <torch/nativert/executor/ExecutorConfig.h>
#include <torch/nativert/executor/ParallelGraphExecutor.h>

namespace {

#define WITH_LOCK(m, block)               \
  {                                       \
    std::unique_lock<decltype(m)> lk_(m); \
    block                                 \
  }

} // namespace

namespace torch::nativert {

ThreadPoolExecutor::ThreadPoolExecutor()
    : work_(std::make_unique<moodycamel::ConcurrentQueue<Work>>()) {}

ThreadPoolExecutor::~ThreadPoolExecutor() {
  stop();
}

C10_ALWAYS_INLINE moodycamel::ProducerToken& ThreadPoolExecutor::ptok() {
  thread_local moodycamel::ProducerToken ptok(*work_);
  return ptok;
}

C10_ALWAYS_INLINE moodycamel::ConsumerToken& ThreadPoolExecutor::ctok() {
  thread_local moodycamel::ConsumerToken ctok(*work_);
  return ctok;
}

void ThreadPoolExecutor::execute_inline(SessionState* session, WorkUnit* unit) {
  session->addWork();
  unit->run(this, session);
}

void ThreadPoolExecutor::start(int32_t numThreads) {
  stopped_ = false;
  for (int32_t i = 0; i < numThreads; ++i) {
    threads_.emplace_back(std::thread(&ThreadPoolExecutor::loop, this));
  }
}

void ThreadPoolExecutor::loop() {
  while (true) {
    Work unit;

    sem_->acquire();

    if (stopped_) {
      return;
    }

    while (!work_->try_dequeue(ctok(), unit)) {
    };

    unit();
  }
}

void ThreadPoolExecutor::add(SessionState* session, WorkUnit* unit) {
  session->addWork();
  work_->enqueue(ptok(), std::bind(&WorkUnit::run, unit, this, session));
  sem_->release();
}

void ThreadPoolExecutor::add(
    SessionState* session,
    std::vector<WorkUnit*>::const_iterator&& begin,
    const std::vector<WorkUnit*>::const_iterator&& end) {
  const auto count = end - begin;

  switch (count) {
    case 0: {
      return;
    }
    case 1: {
      return add(session, *begin);
    }
  }

  session->addWork(count);

  std::vector<Work> runnables;
  runnables.reserve(count);
  for (; begin != end; ++begin) {
    runnables.push_back(std::bind(&WorkUnit::run, *begin, this, session));
  }

  work_->enqueue_bulk(ptok(), runnables.begin(), count);
  sem_->release(count);
}

void ThreadPoolExecutor::stop() {
  stopped_ = true;
  sem_->release(threads_.size());

  std::for_each(threads_.begin(), threads_.end(), [](auto& t) { t.join(); });
  threads_.clear();

  {
    // reset sem
    auto tmp = std::make_unique<c10::Semaphore>();
    sem_.swap(tmp);
  }

  {
    // flush queue
    auto tmp = moodycamel::ConcurrentQueue<Work>();
    work_->swap(tmp);
  }
}

void ThreadPoolExecutor::run(
    SessionState& session,
    const std::vector<WorkUnit*>& roots) {
  // case where thread ptok exists but work_ was swapped
  if (auto& tok = ptok(); C10_UNLIKELY(!tok.valid())) {
    moodycamel::ProducerToken tmp(*work_);
    tok.swap(tmp);
  }

  const auto rootCount = roots.size();

  if (C10_UNLIKELY(rootCount == 0)) {
    return;
  } else if (C10_LIKELY(rootCount > 1)) {
    add(&session, roots.begin() + 1, roots.end());
  }

  execute_inline(&session, roots[0]);

  session.wait();
}

void WorkUnit::run(ThreadPoolExecutor* executor, SessionState* session) {
  thread_local std::vector<WorkUnit*> newWorkUnits;
  thread_local c10::InferenceMode mode;

  WorkUnit* unit = this;

  while (true) {
    unit->kernel->compute(session->frame());

    for (auto* user : unit->users) {
      if (session->decrementProducers(user->node)) {
        newWorkUnits.push_back(user);
      }
    }

    switch (newWorkUnits.size()) {
      case 0: {
        return session->removeWork();
      }
      case 1: {
        break;
      }
      case 2: {
        executor->add(session, newWorkUnits[1]);
        break;
      }
      default: {
        executor->add(session, newWorkUnits.begin() + 1, newWorkUnits.end());
        break;
      }
    }

    unit = newWorkUnits[0];
    newWorkUnits.clear();
  }
}

ParallelGraphExecutor::ParallelGraphExecutor(
    const Graph& graph,
    std::vector<std::unique_ptr<OpKernel>> nodeKernels,
    const ExecutorConfig& executorConfig)
    : GraphExecutorBase(graph, std::move(nodeKernels), executorConfig),
      workUnits_(
          graph.nodes().size() - 2 /* no need for prim.Input or Prim.Output */),
      graph_(graph) {
  auto& nodes = graph_.nodes();

  auto input = &*nodes.begin();
  auto output = &*nodes.rbegin();

  {
    // get rid of prim.Input and prim.Output kernels
    // since we won't be needing them
    nodeKernels_.erase(nodeKernels_.begin());
    nodeKernels_.pop_back();
  }

  size_t idx = 0;
  for (const auto& node : nodes) {
    if (&node == input || &node == output) {
      continue;
    }
    auto& workUnit =
        nodeToWorkUnit_.insert_or_assign(&node, &workUnits_[idx]).first->second;
    workUnit->node = &node;
    workUnit->kernel = nodeKernels_[idx++].get();
    producers_.insert({&node, 0});
  }

  for (auto& unit : workUnits_) {
    for (const auto* dep : unit.node->users()) {
      if (dep != output) {
        unit.users.push_back(nodeToWorkUnit_[dep]);
        producers_[dep] += 1;
      }
    }
  }

  for (auto& [node, p] : producers_) {
    if (p == 0) {
      inputWorkUnits_.push_back(nodeToWorkUnit_[node]);
    }
  }

  executor_.start(executorConfig.maxParallelOps);
}

std::vector<c10::IValue> ParallelGraphExecutor::execute(
    ExecutionFrame& executionFrame,
    std::vector<c10::IValue> inputs) {
  fillUserInputs(executionFrame, std::move(inputs));
  return executeWithPrefilledFrame(executionFrame);
}

std::vector<c10::IValue> ParallelGraphExecutor::executeWithPrefilledFrame(
    ExecutionFrame& executionFrame) {
  auto session = SessionState(executionFrame, producers_);
  executor_.run(session, inputWorkUnits_);

  return executionFrame.tryMoveUserOutputs();
}

} // namespace torch::nativert
