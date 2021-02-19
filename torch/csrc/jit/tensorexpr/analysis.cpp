#include <torch/csrc/jit/tensorexpr/analysis.h>

#include <queue>

namespace torch {
namespace jit {
namespace tensorexpr {

class BufDepTracker : public IRVisitor {
 public:
  std::unordered_set<Tensor*> findUsedTensors(
      Tensor* tensor,
      std::unordered_map<const Buf*, Tensor*>& bufs_to_tensors) {
    used_bufs.clear();
    used_tensors.clear();
    tensor->stmt()->accept(this);
    // For all the used buffers collected, look up the corresponding
    // tensors from the given map.
    for (auto b : used_bufs) {
      auto t = bufs_to_tensors[b];
      if (t) {
        used_tensors.insert(t);
      }
    }
    return used_tensors;
  }

 private:
  void visit(const Load* l) override {
    // The buffer being loaded from is "used" here.
    used_bufs.insert(l->buf());
  }

  void visit(const FunctionCall* f) override {
    // The tensor that is called is "used" here.
    // Save the tensor and recurse on its stmt.
    used_tensors.insert(const_cast<Tensor*>(f->tensor()));
    f->tensor()->stmt()->accept(this);
  }

  std::unordered_set<const Buf*> used_bufs;
  std::unordered_set<Tensor*> used_tensors;
};

std::vector<Tensor*> findDependentTensors(
    const std::vector<Tensor*>& non_output_tensors,
    const std::vector<Tensor*>& output_tensors) {
  // Build a map from buffers to tensors for all non-output tensors.
  // This is required to look up the tensors for the used buffers.
  std::unordered_map<const Buf*, Tensor*> bufs_to_tensors;
  for (auto t : non_output_tensors) {
    bufs_to_tensors[t->buf()] = t;
  }

  BufDepTracker d;
  std::queue<Tensor*> q;
  std::unordered_set<Tensor*> queued;
  std::vector<Tensor*> result;
  std::unordered_set<Tensor*> processed;
  for (Tensor* t : output_tensors) {
    if (queued.insert(t).second) {
      q.push(t);
    }
  }
  while (!q.empty()) {
    Tensor* t = q.front();
    q.pop();
    queued.erase(t);
    auto deps = d.findUsedTensors(t, bufs_to_tensors);
    bool all_processed = true;
    for (Tensor* dep : deps) {
      if (!processed.count(dep)) {
        if (queued.insert(dep).second) {
          q.push(dep);
        }
        all_processed = false;
      }
    }
    if (all_processed) {
      result.push_back(t);
      if (processed.count(t)) {
        throw malformed_input("failure to find all processed Tensors");
      }

      processed.insert(t);
    } else {
      if (queued.count(t)) {
        throw malformed_input("failure to find all queued Tensors");
      }

      q.push(t);
      queued.insert(t);
    }
  }

  return result;
}

class DepTracker : public IRVisitor {
 public:
  std::vector<Tensor*> findUsedTensors(Tensor* tensor) {
    used_tensors.clear();
    tensor->stmt()->accept(this);
    return used_tensors;
  }

 private:
  void visit(const FunctionCall* v) override {
    used_tensors.push_back(const_cast<Tensor*>(v->tensor())); // NOLINT
  }

  std::vector<Tensor*> used_tensors;
};

std::vector<Tensor*> findAllNeededTensors(const std::vector<Tensor*>& tensors) {
  DepTracker d;
  std::queue<Tensor*> q;
  std::unordered_set<Tensor*> queued;
  std::vector<Tensor*> result;
  std::unordered_set<Tensor*> processed;
  for (Tensor* t : tensors) {
    if (queued.insert(t).second) {
      q.push(t);
    }
  }
  while (!q.empty()) {
    Tensor* t = q.front();
    q.pop();
    queued.erase(t);
    std::vector<Tensor*> deps = d.findUsedTensors(t);
    bool all_processed = true;
    for (Tensor* dep : deps) {
      if (!processed.count(dep)) {
        if (queued.insert(dep).second) {
          q.push(dep);
        }
        all_processed = false;
      }
    }
    if (all_processed) {
      result.push_back(t);
      if (processed.count(t)) {
        throw malformed_input("failure to find all processed Tensors");
      }

      processed.insert(t);
    } else {
      if (queued.count(t)) {
        throw malformed_input("failure to find all queued Tensors");
      }

      q.push(t);
      queued.insert(t);
    }
  }

  return result;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
