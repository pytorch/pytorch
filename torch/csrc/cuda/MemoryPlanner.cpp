// Offline memory planner for CUDA graphs (used by
// torch.cuda.MemoryPlannedGraph).
//
// Input: the allocations of a captured graph, in allocation order, with the
// host-order times of their allocation and free, and the positions in the
// captured graph (capture frontiers) at which they were allocated and freed.
// Output: an offset for each allocation in one arena such that allocations
// that may be in use on the GPU at the same time never overlap.
//
// Allocations i < j (in allocation order) conflict, i.e. may not share memory,
// unless i was freed before j was allocated (host order) and every node of i's
// free frontier is an ancestor of (or equal to) a node of j's allocation
// frontier.
//
// The conflicts are computed once, in parallel. A fixed portfolio of
// strategies (first and best fit over a few orders) then runs in parallel; the
// plan used is validated against the rule above, re-evaluated from the inputs
// (not from the conflict matrix).
//
// The strategy interface below is internal. A public API for custom planning
// strategies (e.g. from C++ extensions) is planned as a follow-up; the
// interface is kept so that it can be exposed later.

#include <ATen/Parallel.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <torch/csrc/utils/pybind.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace torch::cuda::memory_planner {

// every offset is a multiple of this (cudaMalloc's alignment)
constexpr int64_t kAlignment = 256;

struct ProblemInput {
  // per allocation, in allocation order (alloc_time strictly increasing)
  std::vector<int64_t> size;
  std::vector<int64_t> alloc_time;
  std::vector<int64_t> free_time; // < 0: live at the end of the capture
  // allocation frontiers, as node indices of the captured graph; CSR (n + 1
  // pointers)
  std::vector<int64_t> alloc_nodes_ptr;
  std::vector<int64_t> alloc_nodes;
  // free frontiers; CSR (empty for an allocation live at the end)
  std::vector<int64_t> free_nodes_ptr;
  std::vector<int64_t> free_nodes;
  // the captured graph: number of nodes, and edges as (src, dst) node pairs
  int64_t num_nodes = 0;
  std::vector<int64_t> edges;
};

class GraphMemoryStrategy;

struct StrategyResult {
  std::string name;
  int64_t arena = 0;
  std::vector<int64_t> offsets;
};

// What a strategy sees: the allocations and which pairs conflict.
class PlanningProblem {
 public:
  // Computes the conflicts in parallel (the intra-op thread count, at most
  // 32 threads).
  explicit PlanningProblem(ProblemInput input);
  ~PlanningProblem();
  PlanningProblem(const PlanningProblem&) = delete;
  PlanningProblem& operator=(const PlanningProblem&) = delete;

  int64_t numAllocations() const;
  const std::vector<int64_t>& sizes() const;
  const std::vector<int64_t>& allocTimes() const;
  const std::vector<int64_t>& freeTimes() const; // < 0: live at the end
  int64_t alignment() const; // of every offset: 256 bytes
  // whether allocations i and j may be in use at the same time
  bool conflict(int64_t i, int64_t j) const;
  // the conflicts of i as a bitset over j, in words of 64
  const uint64_t* conflictRow(int64_t i) const;

  // Runs the strategies in parallel and returns the plan with the smallest
  // arena, ties going to the earlier strategy, after validating it (throws if
  // it is invalid).
  StrategyResult plan(
      const std::vector<std::shared_ptr<GraphMemoryStrategy>>& strategies)
      const;
  // Checks offsets (one per allocation, aligned, and no two conflicting
  // allocations overlap); throws with the first violation found. Returns the
  // arena size.
  int64_t validate(const std::vector<int64_t>& offsets) const;

  // weight of a clique of the conflict graph: no valid plan is smaller
  int64_t lowerBound() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend class FirstFit;
  friend class BestFit;
};

// A planning algorithm. plan() returns one offset per allocation, each a
// multiple of the alignment; it runs without the GIL, possibly concurrently
// with other strategies on the same problem, and the plan used is validated.
class GraphMemoryStrategy {
 public:
  virtual ~GraphMemoryStrategy() = default;
  virtual std::string name() const = 0;
  virtual std::vector<int64_t> plan(const PlanningProblem& problem) const = 0;
};

// The orders in which the built-in strategies place allocations.
enum class Order : int8_t {
  Size, // largest first
  // largest first, then the largest total size of conflicting allocations
  SizeWDegree,
  Breadth, // largest total size live during its lifetime first
};
Order parseOrder(const std::string& name);
const char* orderName(Order order);

// Places the allocations one at a time in `order`, each at the lowest offset
// that does not overlap a placed allocation it conflicts with.
class FirstFit : public GraphMemoryStrategy {
 public:
  explicit FirstFit(const std::string& order = "size");
  Order order() const {
    return order_;
  }
  std::string name() const override;
  std::vector<int64_t> plan(const PlanningProblem& problem) const override;

 private:
  Order order_;
};

// Like FirstFit, but each allocation takes the smallest gap that holds it.
class BestFit : public GraphMemoryStrategy {
 public:
  explicit BestFit(const std::string& order = "size");
  Order order() const {
    return order_;
  }
  std::string name() const override;
  std::vector<int64_t> plan(const PlanningProblem& problem) const override;

 private:
  Order order_;
};

namespace {

inline int64_t alignUp(int64_t x, int64_t a) {
  return (x + a - 1) / a * a;
}

inline bool testBit(const uint64_t* words, int64_t b) {
  return (words[b >> 6] >> (b & 63)) & 1;
}

inline void setBit(uint64_t* words, int64_t b) {
  words[b >> 6] |= uint64_t(1) << (b & 63);
}

// sets bits [a, b)
void setBits(uint64_t* words, int64_t a, int64_t b) {
  while (a < b && (a & 63)) {
    setBit(words, a++);
  }
  while (a + 64 <= b) {
    words[a >> 6] = ~uint64_t(0);
    a += 64;
  }
  while (a < b) {
    setBit(words, a++);
  }
}

// Runs fn(task) for task in [0, num_tasks) on up to num_threads threads,
// handing out tasks dynamically (they may differ in cost). Plain threads: this
// file is not compiled with OpenMP, which at::parallel_for would need here.
void parallelTasks(
    int64_t num_tasks,
    int64_t num_threads,
    const std::function<void(int64_t)>& fn) {
  const int64_t workers =
      std::max<int64_t>(1, std::min<int64_t>(num_threads, num_tasks));
  std::atomic<int64_t> next{0};
  std::exception_ptr error;
  std::mutex error_mutex;
  auto work = [&] {
    try {
      for (int64_t t = next++; t < num_tasks; t = next++) {
        fn(t);
      }
    } catch (...) {
      std::lock_guard<std::mutex> guard(error_mutex);
      if (!error) {
        error = std::current_exception();
      }
      next = num_tasks;
    }
  };
  std::vector<std::thread> threads;
  threads.reserve(workers - 1);
  for (int64_t w = 1; w < workers; w++) {
    threads.emplace_back(work);
  }
  work();
  for (auto& t : threads) {
    t.join();
  }
  if (error) {
    std::rethrow_exception(error);
  }
}

constexpr std::pair<const char*, Order> kOrders[] = {
    {"size", Order::Size},
    {"size_wdegree", Order::SizeWDegree},
    {"breadth", Order::Breadth},
};

// Placed allocations sorted by (offset, end), so that a placement walks them
// in address order.
struct Placed {
  std::vector<int64_t> id, lo, hi;
  explicit Placed(int64_t n) {
    id.reserve(n);
    lo.reserve(n);
    hi.reserve(n);
  }
};

} // namespace

Order parseOrder(const std::string& name) {
  for (const auto& [n, o] : kOrders) {
    if (name == n) {
      return o;
    }
  }
  std::ostringstream valid;
  for (const auto& [n, o] : kOrders) {
    valid << " " << n;
  }
  TORCH_CHECK_VALUE(
      false,
      "unknown memory planning order '",
      name,
      "'; valid orders:",
      valid.str());
}

const char* orderName(Order order) {
  for (const auto& [n, o] : kOrders) {
    if (o == order) {
      return n;
    }
  }
  return "";
}

struct PlanningProblem::Impl {
  ProblemInput in;
  int64_t n = 0;
  int64_t threads = 1;
  // closures of the allocation frontiers, as bitsets over the nodes that
  // appear in a free frontier ("columns"); n x closure_words
  int64_t closure_words = 0;
  std::vector<uint64_t> closure;
  // free frontiers as column indices; CSR
  std::vector<int64_t> free_col_ptr, free_col;
  // dense conflict bit matrix, n x words
  int64_t words = 0;
  std::vector<uint64_t> bits;
  // the host time after the last event
  int64_t end_time = 0;
  mutable c10::once_flag wdegree_once, breadth_once, bound_once;
  mutable std::vector<int64_t> wdegree, breadth;
  mutable int64_t bound = 0;

  const uint64_t* row(int64_t i) const {
    return bits.data() + i * words;
  }

  int64_t freeTime(int64_t i) const {
    return in.free_time[i] < 0 ? end_time : in.free_time[i];
  }

  void check();
  void computeClosures();
  void computeConflicts();
  bool conflictDirect(int64_t i, int64_t j) const;
  void ensureWDegree() const;
  void ensureBreadth() const;
  std::vector<int64_t> visitOrder(Order order) const;
  int64_t place(
      const std::vector<int64_t>& order,
      bool best_fit,
      std::vector<int64_t>& offsets) const;
  int64_t validate(
      const std::vector<int64_t>& offsets,
      int64_t num_threads,
      const std::string& what) const;
};

void PlanningProblem::Impl::check() {
  n = static_cast<int64_t>(in.size.size());
  TORCH_CHECK(in.num_nodes >= 0, "negative number of graph nodes");
  for (auto* v : {&in.alloc_time, &in.free_time}) {
    TORCH_CHECK(
        static_cast<int64_t>(v->size()) == n,
        "every per-allocation input must have one entry per allocation");
  }
  TORCH_CHECK(
      static_cast<int64_t>(in.alloc_nodes_ptr.size()) == n + 1 &&
          static_cast<int64_t>(in.free_nodes_ptr.size()) == n + 1,
      "frontier pointers must have n + 1 entries");
  auto check_csr = [&](const std::vector<int64_t>& ptr,
                       const std::vector<int64_t>& idx) {
    TORCH_CHECK(
        ptr[0] == 0 && ptr[n] == static_cast<int64_t>(idx.size()),
        "invalid frontier pointers");
    for (int64_t i = 0; i < n; i++) {
      TORCH_CHECK(ptr[i] <= ptr[i + 1], "invalid frontier pointers");
      for (int64_t e = ptr[i]; e < ptr[i + 1]; e++) {
        TORCH_CHECK(
            idx[e] >= 0 && idx[e] < in.num_nodes, "frontier node out of range");
      }
    }
  };
  check_csr(in.alloc_nodes_ptr, in.alloc_nodes);
  check_csr(in.free_nodes_ptr, in.free_nodes);
  for (int64_t i = 0; i < n; i++) {
    TORCH_CHECK(in.size[i] > 0, "allocation sizes must be positive");
    TORCH_CHECK(
        i == 0 || in.alloc_time[i] > in.alloc_time[i - 1],
        "allocations must be given in allocation order");
    TORCH_CHECK(
        in.free_time[i] < 0 || in.free_time[i] > in.alloc_time[i],
        "an allocation must be freed after it is allocated");
    end_time = std::max({end_time, in.alloc_time[i] + 1, in.free_time[i] + 1});
  }
  TORCH_CHECK(in.edges.size() % 2 == 0, "edges must be pairs");
  for (const int64_t v : in.edges) {
    TORCH_CHECK(v >= 0 && v < in.num_nodes, "edge out of range");
  }
}

void PlanningProblem::Impl::computeClosures() {
  // columns: the distinct nodes that appear in a free frontier
  std::vector<int64_t> column(in.num_nodes, -1);
  int64_t num_columns = 0;
  free_col_ptr.assign(n + 1, 0);
  free_col.clear();
  free_col.reserve(in.free_nodes.size());
  for (int64_t i = 0; i < n; i++) {
    for (int64_t e = in.free_nodes_ptr[i]; e < in.free_nodes_ptr[i + 1]; e++) {
      int64_t& c = column[in.free_nodes[e]];
      if (c < 0) {
        c = num_columns++;
      }
      free_col.push_back(c);
    }
    free_col_ptr[i + 1] = static_cast<int64_t>(free_col.size());
  }
  closure_words = (num_columns + 63) / 64;
  closure.assign(n * closure_words, 0);
  if (closure_words == 0) {
    return;
  }
  // per node: bitset of the columns that are the node itself or an ancestor
  const int64_t nodes = in.num_nodes;
  const int64_t w = closure_words;
  std::vector<uint64_t> node_bits(nodes * w, 0);
  const auto m = static_cast<int64_t>(in.edges.size() / 2);
  std::vector<int64_t> child_ptr(nodes + 1, 0), child(m), indegree(nodes, 0);
  for (int64_t e = 0; e < m; e++) {
    child_ptr[in.edges[2 * e] + 1]++;
    indegree[in.edges[2 * e + 1]]++;
  }
  std::partial_sum(child_ptr.begin(), child_ptr.end(), child_ptr.begin());
  std::vector<int64_t> fill(child_ptr.begin(), child_ptr.end() - 1);
  for (int64_t e = 0; e < m; e++) {
    child[fill[in.edges[2 * e]]++] = in.edges[2 * e + 1];
  }
  std::vector<int64_t> ready;
  for (int64_t v = 0; v < nodes; v++) {
    if (indegree[v] == 0) {
      ready.push_back(v);
    }
  }
  int64_t visited = 0;
  // Kahn: a node is visited after all of its parents, which have ORed their
  // bits into it
  while (!ready.empty()) {
    const int64_t v = ready.back();
    ready.pop_back();
    visited++;
    uint64_t* bv = node_bits.data() + v * w;
    if (column[v] >= 0) {
      setBit(bv, column[v]);
    }
    for (int64_t e = child_ptr[v]; e < child_ptr[v + 1]; e++) {
      const int64_t c = child[e];
      uint64_t* bc = node_bits.data() + c * w;
      for (int64_t k = 0; k < w; k++) {
        bc[k] |= bv[k];
      }
      if (--indegree[c] == 0) {
        ready.push_back(c);
      }
    }
  }
  TORCH_CHECK(visited == nodes, "the captured graph has a cycle");
  parallelTasks((n + 255) / 256, threads, [&](int64_t block) {
    for (int64_t i = block * 256; i < std::min(n, (block + 1) * 256); i++) {
      uint64_t* ci = closure.data() + i * w;
      for (int64_t e = in.alloc_nodes_ptr[i]; e < in.alloc_nodes_ptr[i + 1];
           e++) {
        const uint64_t* bv = node_bits.data() + in.alloc_nodes[e] * w;
        for (int64_t k = 0; k < w; k++) {
          ci[k] |= bv[k];
        }
      }
    }
  });
}

bool PlanningProblem::Impl::conflictDirect(int64_t i, int64_t j) const {
  if (i == j) {
    return false;
  }
  if (i > j) {
    std::swap(i, j);
  }
  // i was allocated first
  if (in.free_time[i] < 0 || in.free_time[i] > in.alloc_time[j]) {
    return true;
  }
  const uint64_t* cj = closure.data() + j * closure_words;
  for (int64_t e = free_col_ptr[i]; e < free_col_ptr[i + 1]; e++) {
    if (!testBit(cj, free_col[e])) {
      return true;
    }
  }
  return false;
}

void PlanningProblem::Impl::computeConflicts() {
  words = (n + 63) / 64;
  bits.assign(static_cast<size_t>(n) * words, 0);
  // upper triangle: row i owns bits j > i. Rows are paired (i, n - 1 - i) so
  // that tasks cost about the same.
  auto upper = [&](int64_t i) {
    uint64_t* r = bits.data() + i * words;
    if (in.free_time[i] < 0) {
      setBits(r, i + 1, n);
      return;
    }
    // allocated before i is freed: live at the same time
    const int64_t host_end = std::lower_bound(
                                 in.alloc_time.begin() + i + 1,
                                 in.alloc_time.end(),
                                 in.free_time[i]) -
        in.alloc_time.begin();
    setBits(r, i + 1, host_end);
    for (int64_t j = host_end; j < n; j++) {
      if (conflictDirect(i, j)) {
        setBit(r, j);
      }
    }
  };
  parallelTasks((n + 1) / 2, threads, [&](int64_t t) {
    upper(t);
    if (n - 1 - t != t) {
      upper(n - 1 - t);
    }
  });
  // mirror into the lower triangle: row j reads bit (i, j) of rows i < j
  parallelTasks((n + 63) / 64, threads, [&](int64_t block) {
    for (int64_t j = block * 64; j < std::min(n, (block + 1) * 64); j++) {
      uint64_t* r = bits.data() + j * words;
      for (int64_t i = 0; i < j; i++) {
        if (testBit(row(i), j)) {
          setBit(r, i);
        }
      }
    }
  });
}

// For each allocation, the total size of the allocations it conflicts with.
void PlanningProblem::Impl::ensureWDegree() const {
  c10::call_once(wdegree_once, [&] {
    wdegree.assign(n, 0);
    parallelTasks((n + 63) / 64, threads, [&](int64_t block) {
      for (int64_t i = block * 64; i < std::min(n, (block + 1) * 64); i++) {
        const uint64_t* r = row(i);
        int64_t w = 0;
        for (int64_t k = 0; k < words; k++) {
          for (uint64_t x = r[k]; x; x &= x - 1) {
            w += in.size[k * 64 + std::countr_zero(x)];
          }
        }
        wdegree[i] = w;
      }
    });
  });
}

// For each allocation, the largest total size live (in host order) at any time
// during its lifetime (TFLite's "greedy by breadth").
void PlanningProblem::Impl::ensureBreadth() const {
  c10::call_once(breadth_once, [&] {
    std::vector<int64_t> times;
    times.reserve(2 * n);
    for (int64_t i = 0; i < n; i++) {
      times.push_back(in.alloc_time[i]);
      times.push_back(freeTime(i));
    }
    std::sort(times.begin(), times.end());
    times.erase(std::unique(times.begin(), times.end()), times.end());
    const auto m = static_cast<int64_t>(times.size());
    auto index = [&](int64_t t) {
      return std::lower_bound(times.begin(), times.end(), t) - times.begin();
    };
    std::vector<int64_t> live(m + 1, 0);
    std::vector<int64_t> lo(n), hi(n);
    for (int64_t i = 0; i < n; i++) {
      lo[i] = index(in.alloc_time[i]);
      hi[i] = index(freeTime(i));
      live[lo[i]] += in.size[i];
      live[hi[i]] -= in.size[i];
    }
    std::partial_sum(live.begin(), live.end(), live.begin());
    // sparse table for range maxima
    std::vector<std::vector<int64_t>> table{std::vector<int64_t>(
        live.begin(), live.begin() + std::max<int64_t>(m, 1))};
    for (int64_t k = 1; (int64_t(1) << k) <= m; k++) {
      const auto& prev = table.back();
      std::vector<int64_t> next(m - (int64_t(1) << k) + 1);
      for (int64_t x = 0; x < static_cast<int64_t>(next.size()); x++) {
        next[x] = std::max(prev[x], prev[x + (int64_t(1) << (k - 1))]);
      }
      table.push_back(std::move(next));
    }
    breadth.assign(n, 0);
    for (int64_t i = 0; i < n; i++) {
      const int64_t a = lo[i], b = std::max(hi[i], lo[i] + 1); // [a, b)
      const int k = std::bit_width(static_cast<uint64_t>(b - a)) - 1;
      breadth[i] = std::max(table[k][a], table[k][b - (int64_t(1) << k)]);
    }
  });
}

std::vector<int64_t> PlanningProblem::Impl::visitOrder(Order order) const {
  std::vector<int64_t> o(n);
  std::iota(o.begin(), o.end(), 0);
  const auto& S = in.size;
  const auto& A = in.alloc_time;
  auto sort_by = [&](auto key) {
    std::stable_sort(o.begin(), o.end(), [&](int64_t a, int64_t b) {
      return key(a) < key(b);
    });
  };
  switch (order) {
    case Order::Size:
      sort_by([&](int64_t i) { return std::make_tuple(-S[i], A[i]); });
      break;
    case Order::SizeWDegree:
      ensureWDegree();
      sort_by(
          [&](int64_t i) { return std::make_tuple(-S[i], -wdegree[i], A[i]); });
      break;
    case Order::Breadth:
      ensureBreadth();
      sort_by(
          [&](int64_t i) { return std::make_tuple(-breadth[i], -S[i], A[i]); });
      break;
  }
  return o;
}

// One greedy pass over `order`. First fit: the lowest offset that does not
// overlap a placed allocation it conflicts with. Best fit: the smallest such
// gap that holds it (the space above everything counts as an infinite gap).
int64_t PlanningProblem::Impl::place(
    const std::vector<int64_t>& order,
    bool best_fit,
    std::vector<int64_t>& offsets) const {
  const int64_t align = kAlignment;
  offsets.assign(n, 0);
  Placed p(n);
  int64_t arena = 0;
  for (const int64_t k : order) {
    const int64_t sz = in.size[k];
    const uint64_t* r = row(k);
    const auto m = static_cast<int64_t>(p.id.size());
    int64_t offset = 0;
    if (!best_fit) {
      for (int64_t q = 0; q < m; q++) {
        if (offset + sz <= p.lo[q]) {
          break; // later ones start at or above p.lo[q]
        }
        if (p.hi[q] > offset && testBit(r, p.id[q])) {
          offset = alignUp(p.hi[q], align);
        }
      }
    } else {
      int64_t cur = 0, best_gap = std::numeric_limits<int64_t>::max(),
              best_off = -1;
      for (int64_t q = 0; q < m; q++) {
        if (!testBit(r, p.id[q])) {
          continue;
        }
        if (p.lo[q] > cur) {
          const int64_t gap = p.lo[q] - cur;
          if (gap >= sz && gap < best_gap) {
            best_gap = gap;
            best_off = cur;
          }
        }
        cur = std::max(cur, alignUp(p.hi[q], align));
      }
      offset = best_off >= 0 ? best_off : cur;
    }
    offsets[k] = offset;
    const int64_t end = offset + sz;
    auto pos =
        std::lower_bound(p.lo.begin(), p.lo.end(), offset) - p.lo.begin();
    while (pos < m && p.lo[pos] == offset && p.hi[pos] < end) {
      pos++;
    }
    p.id.insert(p.id.begin() + pos, k);
    p.lo.insert(p.lo.begin() + pos, offset);
    p.hi.insert(p.hi.begin() + pos, end);
    arena = std::max(arena, end);
  }
  return arena;
}

int64_t PlanningProblem::Impl::validate(
    const std::vector<int64_t>& offsets,
    int64_t num_threads,
    const std::string& what) const {
  TORCH_CHECK(
      static_cast<int64_t>(offsets.size()) == n,
      what,
      ": ",
      offsets.size(),
      " offsets for ",
      n,
      " allocations");
  int64_t arena = 0;
  for (int64_t i = 0; i < n; i++) {
    TORCH_CHECK(
        offsets[i] >= 0 && offsets[i] % kAlignment == 0,
        what,
        ": offset ",
        offsets[i],
        " of allocation ",
        i,
        " is negative or not a multiple of ",
        kAlignment);
    arena = std::max(arena, offsets[i] + in.size[i]);
  }
  std::vector<int64_t> by_offset(n);
  std::iota(by_offset.begin(), by_offset.end(), 0);
  std::sort(by_offset.begin(), by_offset.end(), [&](int64_t a, int64_t b) {
    return std::make_pair(offsets[a], a) < std::make_pair(offsets[b], b);
  });
  // every pair that overlaps in address is found from the one that starts
  // first
  std::atomic<bool> failed{false};
  std::mutex mutex;
  std::pair<int64_t, int64_t> bad{-1, -1};
  parallelTasks((n + 63) / 64, num_threads, [&](int64_t block) {
    for (int64_t p = block * 64; p < std::min(n, (block + 1) * 64); p++) {
      if (failed.load(std::memory_order_relaxed)) {
        return;
      }
      const int64_t k = by_offset[p];
      const int64_t end = offsets[k] + in.size[k];
      for (int64_t q = p + 1; q < n && offsets[by_offset[q]] < end; q++) {
        const int64_t j = by_offset[q];
        if (conflictDirect(k, j)) {
          std::lock_guard<std::mutex> guard(mutex);
          const auto pair = std::make_pair(std::min(k, j), std::max(k, j));
          if (!failed || pair < bad) {
            bad = pair;
          }
          failed = true;
          return;
        }
      }
    }
  });
  if (failed) {
    const auto [i, j] = bad;
    TORCH_CHECK(
        false,
        what,
        ": allocations ",
        i,
        " [",
        offsets[i],
        ", ",
        offsets[i] + in.size[i],
        ") and ",
        j,
        " [",
        offsets[j],
        ", ",
        offsets[j] + in.size[j],
        ") overlap but may be in use at the same time");
  }
  return arena;
}

PlanningProblem::PlanningProblem(ProblemInput input)
    : impl_(std::make_unique<Impl>()) {
  impl_->in = std::move(input);
  // the phases take milliseconds: more threads than this only add overhead
  constexpr int64_t kMaxThreads = 32;
  impl_->threads = std::min<int64_t>(at::get_num_threads(), kMaxThreads);
  impl_->check();
  impl_->computeClosures();
  impl_->computeConflicts();
}

PlanningProblem::~PlanningProblem() = default;

int64_t PlanningProblem::numAllocations() const {
  return impl_->n;
}

const std::vector<int64_t>& PlanningProblem::sizes() const {
  return impl_->in.size;
}

const std::vector<int64_t>& PlanningProblem::allocTimes() const {
  return impl_->in.alloc_time;
}

const std::vector<int64_t>& PlanningProblem::freeTimes() const {
  return impl_->in.free_time;
}

int64_t PlanningProblem::alignment() const {
  return kAlignment;
}

bool PlanningProblem::conflict(int64_t i, int64_t j) const {
  TORCH_CHECK(i >= 0 && i < impl_->n && j >= 0 && j < impl_->n);
  return testBit(impl_->row(i), j);
}

const uint64_t* PlanningProblem::conflictRow(int64_t i) const {
  TORCH_CHECK(i >= 0 && i < impl_->n);
  return impl_->row(i);
}

StrategyResult PlanningProblem::plan(
    const std::vector<std::shared_ptr<GraphMemoryStrategy>>& strategies) const {
  TORCH_CHECK(!strategies.empty(), "no memory planning strategy");
  const auto& d = *impl_;
  std::vector<StrategyResult> results(strategies.size());
  const auto num = static_cast<int64_t>(strategies.size());
  parallelTasks(num, d.threads, [&](int64_t s) {
    auto& r = results[s];
    r.name = strategies[s]->name();
    r.offsets = strategies[s]->plan(*this);
    TORCH_CHECK(
        static_cast<int64_t>(r.offsets.size()) == d.n,
        "invalid memory plan from strategy '",
        r.name,
        "': ",
        r.offsets.size(),
        " offsets for ",
        d.n,
        " allocations");
    for (int64_t i = 0; i < d.n; i++) {
      r.arena = std::max(r.arena, r.offsets[i] + d.in.size[i]);
    }
  });
  // the first of the smallest
  auto best = std::min_element(
      results.begin(), results.end(), [](const auto& a, const auto& b) {
        return a.arena < b.arena;
      });
  // only the plan used is validated
  d.validate(
      best->offsets,
      d.threads,
      "invalid memory plan from strategy '" + best->name + "'");
  return std::move(*best);
}

namespace {

// "<order>" is FirstFit(order) and "<order>/best" is BestFit(order).
std::shared_ptr<GraphMemoryStrategy> makeStrategy(const std::string& name) {
  const auto slash = name.find('/');
  const std::string order = name.substr(0, slash);
  if (slash == std::string::npos) {
    return std::make_shared<FirstFit>(order);
  }
  TORCH_CHECK_VALUE(
      name.substr(slash) == "/best",
      "unknown memory planning strategy '",
      name,
      "'; expected '<order>' or '<order>/best'");
  return std::make_shared<BestFit>(order);
}

// The strategies MemoryPlannedGraph runs. On 105 plans recorded from training
// steps of Llama 3 (70B, 405B), Qwen 3.5 (27B, 397B) and DeepSeek V3/V4
// models, these give the best arena of 22 strategies (first and best fit over
// eleven orders) on every plan.
std::vector<std::shared_ptr<GraphMemoryStrategy>> defaultStrategies() {
  std::vector<std::shared_ptr<GraphMemoryStrategy>> strategies;
  for (const char* name :
       {"size", "size/best", "size_wdegree", "breadth", "breadth/best"}) {
    strategies.push_back(makeStrategy(name));
  }
  return strategies;
}

} // namespace

int64_t PlanningProblem::validate(const std::vector<int64_t>& offsets) const {
  return impl_->validate(offsets, impl_->threads, "invalid memory plan");
}

// A clique of the conflict graph: its members must be pairwise disjoint, so
// their total size bounds any plan from below. Seeds are the sets live at the
// heaviest host times (pairwise conflicting by the host-order rule), each grown
// greedily, largest first, by allocations that conflict with every member.
int64_t PlanningProblem::lowerBound() const {
  auto& d = *impl_;
  c10::call_once(d.bound_once, [&] {
    const int64_t n = d.n;
    std::vector<std::pair<int64_t, int64_t>> events; // (time, delta)
    for (int64_t i = 0; i < n; i++) {
      events.emplace_back(d.in.alloc_time[i], d.in.size[i]);
      if (d.in.free_time[i] >= 0) {
        events.emplace_back(d.in.free_time[i], -d.in.size[i]);
      }
    }
    std::sort(events.begin(), events.end());
    // live bytes right after each allocation
    std::vector<std::pair<int64_t, int64_t>> live_at; // (-live, time)
    int64_t live = 0;
    for (size_t e = 0; e < events.size(); e++) {
      live += events[e].second;
      if (events[e].second > 0 &&
          (e + 1 == events.size() || events[e + 1].first != events[e].first)) {
        live_at.emplace_back(-live, events[e].first);
      }
    }
    std::sort(live_at.begin(), live_at.end());
    const int64_t seeds = std::min<int64_t>(64, live_at.size());
    std::vector<int64_t> by_size(n);
    std::iota(by_size.begin(), by_size.end(), 0);
    std::stable_sort(by_size.begin(), by_size.end(), [&](int64_t a, int64_t b) {
      return d.in.size[a] > d.in.size[b];
    });
    std::vector<int64_t> weight(seeds, 0);
    parallelTasks(seeds, d.threads, [&](int64_t s) {
      const int64_t t = live_at[s].second;
      std::vector<uint64_t> cand(d.words, ~uint64_t(0));
      if (n % 64) {
        cand.back() = (uint64_t(1) << (n % 64)) - 1;
      }
      int64_t w = 0;
      auto add = [&](int64_t k) {
        w += d.in.size[k];
        const uint64_t* r = d.row(k);
        for (int64_t q = 0; q < d.words; q++) {
          cand[q] &= r[q];
        }
      };
      std::vector<int64_t> members;
      for (int64_t i = 0; i < n && d.in.alloc_time[i] <= t; i++) {
        if (d.in.free_time[i] < 0 || d.in.free_time[i] > t) {
          members.push_back(i);
        }
      }
      for (const int64_t k : members) {
        add(k);
      }
      for (const int64_t k : by_size) {
        if (testBit(cand.data(), k)) {
          add(k);
        }
      }
      weight[s] = w;
    });
    d.bound = seeds ? *std::max_element(weight.begin(), weight.end()) : 0;
  });
  return d.bound;
}

FirstFit::FirstFit(const std::string& order) : order_(parseOrder(order)) {}

std::string FirstFit::name() const {
  return orderName(order_);
}

std::vector<int64_t> FirstFit::plan(const PlanningProblem& problem) const {
  std::vector<int64_t> offsets;
  problem.impl_->place(problem.impl_->visitOrder(order_), false, offsets);
  return offsets;
}

BestFit::BestFit(const std::string& order) : order_(parseOrder(order)) {}

std::string BestFit::name() const {
  return std::string(orderName(order_)) + "/best";
}

std::vector<int64_t> BestFit::plan(const PlanningProblem& problem) const {
  std::vector<int64_t> offsets;
  problem.impl_->place(problem.impl_->visitOrder(order_), true, offsets);
  return offsets;
}

// Python binding (private): torch._C._cuda_PlanningProblem
// NOLINTNEXTLINE(misc-use-internal-linkage)
void initMemoryPlannerBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<PlanningProblem>(m, "_cuda_PlanningProblem")
      .def(
          py::init([](std::vector<int64_t> size,
                      std::vector<int64_t> alloc_time,
                      std::vector<int64_t> free_time,
                      std::vector<int64_t> alloc_nodes_ptr,
                      std::vector<int64_t> alloc_nodes,
                      std::vector<int64_t> free_nodes_ptr,
                      std::vector<int64_t> free_nodes,
                      int64_t num_nodes,
                      std::vector<int64_t> edges) {
            ProblemInput in;
            in.size = std::move(size);
            in.alloc_time = std::move(alloc_time);
            in.free_time = std::move(free_time);
            in.alloc_nodes_ptr = std::move(alloc_nodes_ptr);
            in.alloc_nodes = std::move(alloc_nodes);
            in.free_nodes_ptr = std::move(free_nodes_ptr);
            in.free_nodes = std::move(free_nodes);
            in.num_nodes = num_nodes;
            in.edges = std::move(edges);
            py::gil_scoped_release no_gil;
            return std::make_unique<PlanningProblem>(std::move(in));
          }),
          py::arg("size"),
          py::arg("alloc_time"),
          py::arg("free_time"),
          py::arg("alloc_nodes_ptr"),
          py::arg("alloc_nodes"),
          py::arg("free_nodes_ptr"),
          py::arg("free_nodes"),
          py::arg("num_nodes"),
          py::arg("edges"))
      .def(
          "plan",
          // the default strategies, or one ("<order>" or "<order>/best");
          // returns (strategy, arena, offsets)
          [](const PlanningProblem& p, std::optional<std::string> strategy) {
            std::vector<std::shared_ptr<GraphMemoryStrategy>> strategies;
            if (strategy) {
              strategies.push_back(makeStrategy(*strategy));
            } else {
              strategies = defaultStrategies();
            }
            StrategyResult r;
            {
              py::gil_scoped_release no_gil;
              r = p.plan(strategies);
            }
            return py::make_tuple(r.name, r.arena, py::cast(r.offsets));
          },
          py::arg("strategy") = py::none())
      .def_static(
          "strategies",
          [] {
            std::vector<std::string> names;
            for (const auto& s : defaultStrategies()) {
              names.push_back(s->name());
            }
            return names;
          })
      .def(
          "validate",
          [](const PlanningProblem& p, const std::vector<int64_t>& offsets) {
            py::gil_scoped_release no_gil;
            return p.validate(offsets);
          },
          py::arg("offsets"))
      .def("conflict", &PlanningProblem::conflict)
      .def_property_readonly(
          "num_allocations", &PlanningProblem::numAllocations)
      .def_property_readonly("lower_bound", [](const PlanningProblem& p) {
        py::gil_scoped_release no_gil;
        return p.lowerBound();
      });
}

} // namespace torch::cuda::memory_planner
