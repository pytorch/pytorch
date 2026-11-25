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
// The conflicts are computed once, in parallel. A portfolio of strategies
// (first and best fit over a few orders) then runs in parallel; the smallest
// plan is used, ties going to the earlier strategy, and it is validated
// against the rule above, re-evaluated from the inputs (not from the conflict
// matrix).
//
// The strategy interface below is internal; a public API for custom planning
// strategies (e.g. from C++ extensions) could build on it.

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
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
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

class PlanningProblem;

// The orders in which the built-in strategies place allocations.
enum class Order : int8_t {
  Size, // largest first
  // largest first, then the largest total size of conflicting allocations
  SizeWDegree,
  Breadth, // largest total size live during its lifetime first
};
Order parseOrder(const std::string& name);
const char* orderName(Order order);

// A planning algorithm. plan() returns one offset per allocation, each a
// multiple of the alignment; it runs without the GIL, possibly concurrently
// with other strategies on the same problem, and the plan used is validated.
class GraphMemoryStrategy {
 public:
  virtual ~GraphMemoryStrategy() = default;
  virtual std::string name() const = 0;
  virtual std::vector<int64_t> plan(const PlanningProblem& problem) const = 0;
};

// Places the allocations one at a time in `order`, each at the lowest offset
// that does not overlap a placed allocation it conflicts with.
class FirstFit : public GraphMemoryStrategy {
 public:
  explicit FirstFit(const std::string& order = "size")
      : order_(parseOrder(order)) {}
  std::string name() const override {
    return orderName(order_);
  }
  std::vector<int64_t> plan(const PlanningProblem& problem) const override;

 private:
  Order order_;
};

// Like FirstFit, but each allocation takes the smallest gap that holds it.
class BestFit : public GraphMemoryStrategy {
 public:
  explicit BestFit(const std::string& order = "size")
      : order_(parseOrder(order)) {}
  std::string name() const override {
    return std::string(orderName(order_)) + "/best";
  }
  std::vector<int64_t> plan(const PlanningProblem& problem) const override;

 private:
  Order order_;
};

using Strategies = std::vector<std::shared_ptr<GraphMemoryStrategy>>;

struct Plan {
  std::string strategy;
  int64_t arena = 0;
  std::vector<int64_t> offsets;
};

// The allocations and which pairs conflict.
class PlanningProblem {
 public:
  // Computes the conflicts in parallel (the intra-op thread count, at most
  // 32 threads).
  explicit PlanningProblem(ProblemInput input);

  int64_t numAllocations() const {
    return n;
  }
  // whether allocations i and j may be in use at the same time
  bool conflict(int64_t i, int64_t j) const;

  // Runs the strategies in parallel and keeps the plan with the smallest
  // arena, ties going to the earlier strategy. Validates the plan returned
  // (throws if it is invalid).
  Plan plan(const Strategies& strategies) const;
  // One greedy pass in `order` (FirstFit and BestFit).
  std::vector<int64_t> greedy(Order order, bool best_fit) const;
  // Checks offsets (one per allocation, aligned, and no two conflicting
  // allocations overlap); throws with the first violation found. Returns the
  // arena size.
  int64_t validate(const std::vector<int64_t>& offsets) const {
    return validateWith(offsets, threads, "invalid memory plan");
  }

 private:
  ProblemInput in;
  int64_t n = 0;
  int64_t threads = 1;
  // the closures of the allocation frontiers over the nodes that appear in a
  // free frontier ("columns"), stored by column: bit j of column c's row is
  // set if c is an ancestor of (or in) allocation j's frontier;
  // num_columns x words
  int64_t num_columns = 0;
  std::vector<uint64_t> closure;
  // free frontiers as column indices; CSR
  std::vector<int64_t> free_col_ptr, free_col;
  // dense conflict bit matrix, n x words
  int64_t words = 0;
  std::vector<uint64_t> bits;
  // the host time after the last event
  int64_t end_time = 0;
  mutable c10::once_flag wdegree_once, breadth_once;
  mutable std::vector<int64_t> wdegree, breadth;

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
  int64_t validateWith(
      const std::vector<int64_t>& offsets,
      int64_t num_threads,
      const std::string& what) const;
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

// Transposes a 64 x 64 bit matrix in place: bit c of a[r] becomes bit r of
// a[c].
inline void transpose64(uint64_t* a) {
  uint64_t m = 0x00000000FFFFFFFFULL;
  for (int j = 32; j != 0; j >>= 1, m ^= (m << j)) {
    for (int k = 0; k < 64; k = ((k | j) + 1) & ~j) {
      const uint64_t t = ((a[k] >> j) ^ a[k | j]) & m;
      a[k] ^= t << j;
      a[k | j] ^= t;
    }
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

void PlanningProblem::check() {
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

void PlanningProblem::computeClosures() {
  // columns: the distinct nodes that appear in a free frontier
  std::vector<int64_t> column(in.num_nodes, -1);
  num_columns = 0;
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
  words = (n + 63) / 64;
  closure.assign(num_columns * words, 0);
  if (num_columns == 0) {
    return;
  }
  // per node: bitset of the columns that are the node itself or an ancestor
  const int64_t nodes = in.num_nodes;
  const int64_t w = (num_columns + 63) / 64;
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
  // per block of 64 allocations: their closures over the columns (the union
  // of their frontier nodes' bits), transposed 64 x 64 bits at a time into
  // the columns' rows (word `block` of each)
  parallelTasks(words, threads, [&](int64_t block) {
    const int64_t first = block * 64, count = std::min<int64_t>(64, n - first);
    std::vector<uint64_t> rows(64 * w, 0);
    for (int64_t r = 0; r < count; r++) {
      const int64_t i = first + r;
      uint64_t* ci = rows.data() + r * w;
      for (int64_t e = in.alloc_nodes_ptr[i]; e < in.alloc_nodes_ptr[i + 1];
           e++) {
        const uint64_t* bv = node_bits.data() + in.alloc_nodes[e] * w;
        for (int64_t k = 0; k < w; k++) {
          ci[k] |= bv[k];
        }
      }
    }
    uint64_t a[64];
    for (int64_t k = 0; k < w; k++) {
      for (int64_t r = 0; r < 64; r++) {
        a[r] = rows[r * w + k];
      }
      transpose64(a);
      for (int64_t c = 0; c < 64 && k * 64 + c < num_columns; c++) {
        closure[(k * 64 + c) * words + block] = a[c];
      }
    }
  });
}

bool PlanningProblem::conflictDirect(int64_t i, int64_t j) const {
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
  for (int64_t e = free_col_ptr[i]; e < free_col_ptr[i + 1]; e++) {
    if (!testBit(closure.data() + free_col[e] * words, j)) {
      return true;
    }
  }
  return false;
}

void PlanningProblem::computeConflicts() {
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
    // later: j conflicts unless every node of i's free frontier is in j's
    // closure (conflictDirect), a word of j's at a time
    if (free_col_ptr[i] == free_col_ptr[i + 1] || host_end >= n) {
      return;
    }
    for (int64_t k = host_end >> 6; k < words; k++) {
      uint64_t x = 0;
      for (int64_t e = free_col_ptr[i]; e < free_col_ptr[i + 1]; e++) {
        x |= ~closure[free_col[e] * words + k];
      }
      if (k == (host_end >> 6)) {
        x &= ~uint64_t(0) << (host_end & 63);
      }
      if (k == words - 1 && (n & 63)) {
        x &= (uint64_t(1) << (n & 63)) - 1;
      }
      r[k] |= x;
    }
  };
  parallelTasks((n + 1) / 2, threads, [&](int64_t t) {
    upper(t);
    if (n - 1 - t != t) {
      upper(n - 1 - t);
    }
  });
  // mirror into the lower triangle, 64 x 64 bits at a time: the rows of
  // block jb get word ib <= jb from word jb of block ib's rows (which are only
  // read at words > their own block, or by this task on the diagonal)
  parallelTasks(words, threads, [&](int64_t jb) {
    uint64_t a[64];
    for (int64_t ib = 0; ib <= jb; ib++) {
      for (int64_t r = 0; r < 64; r++) {
        const int64_t i = ib * 64 + r;
        a[r] = i < n ? bits[i * words + jb] : 0;
      }
      transpose64(a);
      for (int64_t c = 0; c < 64 && jb * 64 + c < n; c++) {
        bits[(jb * 64 + c) * words + ib] |= a[c];
      }
    }
  });
}

// For each allocation, the total size of the allocations it conflicts with.
void PlanningProblem::ensureWDegree() const {
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
void PlanningProblem::ensureBreadth() const {
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

std::vector<int64_t> PlanningProblem::visitOrder(Order order) const {
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

std::vector<int64_t> PlanningProblem::greedy(Order order, bool best_fit) const {
  std::vector<int64_t> offsets;
  place(visitOrder(order), best_fit, offsets);
  return offsets;
}

// One greedy pass over `order`. First fit: the lowest offset that does not
// overlap a placed allocation it conflicts with. Best fit: the smallest such
// gap that holds it (the space above everything counts as an infinite gap).
int64_t PlanningProblem::place(
    const std::vector<int64_t>& order,
    bool best_fit,
    std::vector<int64_t>& offsets) const {
  const int64_t align = kAlignment;
  offsets.assign(n, 0);
  // the placed allocations: a bitset, and their intervals in address order
  std::vector<uint64_t> placed(words, 0);
  Placed p(n);
  std::vector<std::pair<int64_t, int64_t>> conflicting; // (lo, hi)
  int64_t arena = 0;
  // the fits, over the placed allocations k conflicts with in address order
  // (they do not depend on how intervals that start at the same address are
  // ordered)
  auto first_fit = [&](int64_t sz, auto&& each) {
    int64_t offset = 0;
    each([&](int64_t lo, int64_t hi) {
      if (offset + sz <= lo) {
        return false; // later ones start at or above lo
      }
      if (hi > offset) {
        offset = alignUp(hi, align);
      }
      return true;
    });
    return offset;
  };
  auto best = [&](int64_t sz, auto&& each) {
    int64_t cur = 0, best_gap = std::numeric_limits<int64_t>::max(),
            best_off = -1;
    each([&](int64_t lo, int64_t hi) {
      if (lo > cur) {
        const int64_t gap = lo - cur;
        if (gap >= sz && gap < best_gap) {
          best_gap = gap;
          best_off = cur;
        }
      }
      cur = std::max(cur, alignUp(hi, align));
      return true;
    });
    return best_off >= 0 ? best_off : cur;
  };
  for (const int64_t k : order) {
    const int64_t sz = in.size[k];
    const uint64_t* r = row(k);
    int64_t c = 0;
    for (int64_t w = 0; w < words; w++) {
      c += std::popcount(r[w] & placed[w]);
    }
    // The conflicting placed allocations can be gathered from k's row and
    // sorted (about c log c), or filtered from all the placed intervals in
    // address order (up to all of them; first fit stops at the first gap that
    // fits). First fit tries the filter within the gather's budget.
    const int64_t gather_cost =
        words + 4 * c * (64 - std::countl_zero(static_cast<uint64_t>(c)));
    const auto m = static_cast<int64_t>(p.id.size());
    auto filtered = [&](int64_t budget, auto&& visit) {
      for (int64_t q = 0; q < m; q++) {
        if (q >= budget) {
          return false;
        }
        if (testBit(r, p.id[q]) && !visit(p.lo[q], p.hi[q])) {
          break;
        }
      }
      return true;
    };
    auto gathered = [&](auto&& visit) {
      conflicting.clear();
      for (int64_t w = 0; w < words; w++) {
        for (uint64_t x = r[w] & placed[w]; x; x &= x - 1) {
          const int64_t j = w * 64 + std::countr_zero(x);
          conflicting.emplace_back(offsets[j], offsets[j] + in.size[j]);
        }
      }
      std::sort(conflicting.begin(), conflicting.end());
      for (const auto& [lo, hi] : conflicting) {
        if (!visit(lo, hi)) {
          break;
        }
      }
    };
    int64_t offset = 0;
    if (!best_fit) {
      bool done = false;
      offset = first_fit(
          sz, [&](auto&& visit) { done = filtered(gather_cost, visit); });
      if (!done) {
        offset = first_fit(sz, gathered);
      }
    } else if (gather_cost < m) {
      offset = best(sz, gathered);
    } else {
      offset = best(sz, [&](auto&& visit) {
        filtered(std::numeric_limits<int64_t>::max(), visit);
      });
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
    setBit(placed.data(), k);
    arena = std::max(arena, end);
  }
  return arena;
}

int64_t PlanningProblem::validateWith(
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

PlanningProblem::PlanningProblem(ProblemInput input) : in(std::move(input)) {
  // the phases take milliseconds: more threads than this only add overhead
  constexpr int64_t kMaxThreads = 32;
  threads = std::min<int64_t>(at::get_num_threads(), kMaxThreads);
  check();
  computeClosures();
  computeConflicts();
}

bool PlanningProblem::conflict(int64_t i, int64_t j) const {
  TORCH_CHECK(i >= 0 && i < n && j >= 0 && j < n);
  return testBit(row(i), j);
}

Plan PlanningProblem::plan(const Strategies& strategies) const {
  TORCH_CHECK(!strategies.empty(), "no memory planning strategy");
  std::vector<Plan> results(strategies.size());
  const auto num = static_cast<int64_t>(strategies.size());
  parallelTasks(num, threads, [&](int64_t s) {
    auto& r = results[s];
    r.strategy = strategies[s]->name();
    r.offsets = strategies[s]->plan(*this);
    TORCH_CHECK(
        static_cast<int64_t>(r.offsets.size()) == n,
        "invalid memory plan from strategy '",
        r.strategy,
        "': ",
        r.offsets.size(),
        " offsets for ",
        n,
        " allocations");
    for (int64_t i = 0; i < n; i++) {
      r.arena = std::max(r.arena, r.offsets[i] + in.size[i]);
    }
  });
  // the first of the smallest
  auto best = std::min_element(
      results.begin(), results.end(), [](const auto& a, const auto& b) {
        return a.arena < b.arena;
      });
  // only the plan used is validated
  validateWith(
      best->offsets,
      threads,
      "invalid memory plan from strategy '" + best->strategy + "'");
  return std::move(*best);
}

std::vector<int64_t> FirstFit::plan(const PlanningProblem& problem) const {
  return problem.greedy(order_, /*best_fit=*/false);
}

std::vector<int64_t> BestFit::plan(const PlanningProblem& problem) const {
  return problem.greedy(order_, /*best_fit=*/true);
}

namespace {

constexpr std::pair<const char*, Order> kOrders[] = {
    {"size", Order::Size},
    {"size_wdegree", Order::SizeWDegree},
    {"breadth", Order::Breadth},
};

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

// The strategies MemoryPlannedGraph runs. On the plans recorded from training
// steps of Llama 3 (70B, 405B), Qwen 3.5 (27B, 397B) and DeepSeek V3/V4
// models, these give the best arena of 22 strategies (first and best fit over
// eleven orders) on every plan; "size" alone is at most 0.48% larger, and
// "size" with "breadth/best" at most 0.02%.
Strategies defaultStrategies() {
  Strategies strategies;
  for (const char* name :
       {"size", "size/best", "size_wdegree", "breadth", "breadth/best"}) {
    strategies.push_back(makeStrategy(name));
  }
  return strategies;
}

} // namespace

Order parseOrder(const std::string& name) {
  for (const auto& [n, o] : kOrders) {
    if (name == n) {
      return o;
    }
  }
  std::string valid;
  for (const auto& [n, o] : kOrders) {
    valid += std::string(" ") + n;
  }
  TORCH_CHECK_VALUE(
      false,
      "unknown memory planning order '",
      name,
      "'; valid orders:",
      valid);
}

const char* orderName(Order order) {
  for (const auto& [n, o] : kOrders) {
    if (o == order) {
      return n;
    }
  }
  return "";
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
            const Strategies strategies = strategy
                ? Strategies{makeStrategy(*strategy)}
                : defaultStrategies();
            Plan r;
            {
              py::gil_scoped_release no_gil;
              r = p.plan(strategies);
            }
            return py::make_tuple(r.strategy, r.arena, py::cast(r.offsets));
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
          "num_allocations", &PlanningProblem::numAllocations);
}

} // namespace torch::cuda::memory_planner
