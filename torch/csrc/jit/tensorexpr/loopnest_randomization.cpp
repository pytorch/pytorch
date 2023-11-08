#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/jit_opt_limit.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/loopnest_randomization.h>

namespace torch::jit::tensorexpr {

namespace randomization_helper {

static int64_t max_transformations(int n_max_transforms) {
  // Reuse the env variable PYTORCH_JIT_OPT_LIMIT to control the max number of
  // transformations.  Example - set the env variable
  // PYTORCH_JIT_OPT_LIMIT="loopnest_randomization=10" to set max
  // transformations to 10.  This can be helpful in gradually reducing the
  // number of transformations when we see an error.
  if (!JIT_OPT_ALLOWED) {
    return n_max_transforms;
  }
  int max_transforms = 1;
  while (JIT_OPT_ALLOWED && max_transforms < n_max_transforms) {
    max_transforms++;
  }
  return max_transforms;
}

static std::vector<std::vector<ForPtr>> GetAllPerfectlyNestedLoopNests(
    std::vector<ForPtr> loops) {
  // Find the first set of loops that can be reordered
  std::vector<std::vector<ForPtr>> all_nested_loops;
  std::vector<ForPtr> nested_loops;
  if (loops.empty()) {
    return all_nested_loops;
  }
  nested_loops.push_back(loops[0]);
  for (size_t i = 1; i < loops.size(); i++) {
    auto last_loop = nested_loops.back();
    auto next_loop = loops[i];
    if (last_loop->body()->nstmts() == 1 &&
        last_loop->body()->front() == next_loop) {
      nested_loops.push_back(next_loop);
    } else {
      if (nested_loops.size() > 1) {
        all_nested_loops.push_back(nested_loops);
      }
      nested_loops.clear();
      nested_loops.push_back(next_loop);
    }
  }
  return all_nested_loops;
}

template <typename T>
std::tuple<std::vector<T>, std::vector<int>> select_n_randomly(
    std::vector<T>& objects,
    int n,
    std::default_random_engine& random_engine) {
  std::vector<int> indices(objects.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), random_engine);

  std::vector<T> selected_objects;
  std::vector<int> selected_indices;
  if (static_cast<int>(indices.size()) < n) {
    return std::make_tuple(selected_objects, selected_indices);
  }
  for (int i = 0; i < n; i++) {
    int index = indices[i];
    selected_indices.push_back(index);
    selected_objects.push_back(objects[index]);
  }
  return std::make_tuple(selected_objects, selected_indices);
}

static int find_factor(ForPtr loop) {
  // Find valid factors
  ExprPtr loop_stop = loop->stop();
  auto loop_imm = intValue(loop_stop);
  if (loop_imm) {
    int loop_bound = *loop_imm;
    int factor = rand() % (loop_bound - 1) + 1;
    return factor;
  }
  return -1;
}

static void printHistory(int index, std::string message) {
  message = "Random Transform Sequence - Transformations[" +
      std::to_string(index) + "] = " + message;
  GRAPH_DEBUG(message);
}

template <typename T>
std::string join(std::vector<T> indices, char sep = ',') {
  std::string s = "";
  for (const auto& index : indices) {
    s += std::to_string(index) + sep;
  }
  return s;
}

static std::string join(std::vector<std::string> indices, char sep = ',') {
  std::string s = "";
  for (const auto& index : indices) {
    s += index + sep;
  }
  return s;
}
template <typename T>
std::string indexOf(const std::vector<T>& objects, const T& object) {
  return std::to_string(std::distance(
      objects.begin(), std::find(objects.begin(), objects.end(), object)));
}

} // namespace randomization_helper

void loopnestRandomization(int64_t seed, LoopNest& l) {
  // This is to help with deterministic testing of randomized infrastructure.
  // When seed value is 1, we perform preset loop transformations. This allows
  // testing of interface.
  if (seed == 1) {
    l.simplify();
    return;
  }

  std::default_random_engine random_engine(seed);
  std::srand(seed);
  // Set the maximum allowed number of transformations beyond which it is hard
  // to track and debug. Arbitrarily choosing 20 as maximum number.
  int max_allowed_transformations = 20;
  int n_transforms = randomization_helper::max_transformations(
      std::rand() % max_allowed_transformations);
  std::string message = "";
  // clang-format off
  //   Transformations list:
  //
  //       StmtPtr simplify();
  //       bool computeInline(BufPtr b);
  //       void inlineIntermediateBufs(bool allow_duplicated_work);
  //       bool optimizeConditionals();
  //       static void splitWithTail(ForPtr f, int factor);
  //       static void splitWithMask(ForPtr f, int factor);
  //       static std::vector<ForPtr> distributeLoop(ForPtr loop, const std::unordered_set<StmtPtr>& pivots);
  //       static std::vector<ForPtr> distributeLoop(ForPtr loop);
  //       static std::vector<ForPtr> distributeLoopAndParents(ForPtr loop);
  //       static std::vector<ForPtr> distributeLoopOverInnerLoops(ForPtr loop);
  //       static std::vector<ForPtr> distributeLoopAndParentsOverInnerLoops(ForPtr loop);
  //       static bool fuseLoops(const std::vector<ForPtr>& loops, ForPtr* fused);
  //       static void reorderAxis(ForPtr a, ForPtr b);
  //       static std::vector<ForPtr> reorder(const std::vector<ForPtr>& loops, const std::vector<size_t>& permutation);
  //       ForPtr tile(ForPtr x, ForPtr y, int x_factor, int y_factor);
  //       static void fullUnroll(ForPtr f);
  //       static bool normalize(ForPtr f);
  //       static bool flatten(const std::vector<ForPtr>& f, ForPtr* flattened);
  //       static void compressBuffer(BufPtr buf, StmtPtr stmt);
  //       static void compressAllBuffers(StmtPtr stmt);
  //       static void sliceHead(ForPtr f, int factor, ForPtr* head, ForPtr* tail);
  //       static void sliceHead(ForPtr f, int factor);
  //       static void sliceTail(ForPtr f, int factor, ForPtr* head, ForPtr* tail);
  //       static void sliceTail(ForPtr f, int factor);
  //       static AccessResult cacheAccesses(BufPtr producer, const std::string& name, StmtPtr consumer);
  //       static void computeAt(StmtPtr s, ForPtr at);
  //       static bool rfactor(StmtPtr s, ForPtr outer_reduction_for);
  //       static bool vectorize(ForPtr);
  //       void vectorizeInnerLoops();
  //       void eliminateDeadStores();
  //       void prepareForCodegen();
  // clang-format on
  enum TransformKind {
    SIMPLIFY = 0,
    COMPUTE_INLINE,
    INLINE_ALL,
    OPT_COND,
    SPLIT_TAIL,
    SPLIT_MASK,
    DIST1,
    DIST2,
    DIST3,
    DIST4,
    DIST5,
    FUSE_LOOPS,
    REORDER_AXIS,
    REORDER,
    TILE,
    FULL_UNROLL,
    NORMALIZE,
    FLATTEN,
    COMPRESS_BUFFER,
    COMPRESS_ALL_BUFFERS,
    SLICE_HEAD,
    SLICE_TAIL,
    CACHE_ACCESSES,
    COMPUTE_AT,
    RFACTOR,
    VECTORIZE,
    VECTORIZE_INNER_LOOPS,
    ELIMINATE_DEAD_STORES,
    MAX_TRANSFORM,
  };
  bool can_inline = true;
  try {
    for (int n_transform = 0; n_transform < n_transforms; n_transform++) {
      int transform = std::rand() % MAX_TRANSFORM;
      switch (transform) {
        case SIMPLIFY: {
          message = "simplify();\n";
          randomization_helper::printHistory(n_transform, message);
          l.simplify();
          break;
        }
        case COMPUTE_INLINE: {
          if (can_inline) {
            auto bufs = NodeFinder<Buf>::find(l.root_stmt());
            if (!bufs.empty()) {
              int buf_number = std::rand() % (int)bufs.size();
              message =
                  "computeInline(" + bufs[buf_number]->name_hint() + ");\n";
              randomization_helper::printHistory(n_transform, message);
              l.computeInline(bufs[buf_number]);
            }
          }
          break;
        }
        case INLINE_ALL: {
          if (can_inline) {
            int allow_dup = std::rand() % 2;
            message =
                "inlineIntermediateBufs(" + std::to_string(allow_dup) + ");\n";
            randomization_helper::printHistory(n_transform, message);
            l.inlineIntermediateBufs(allow_dup);
            can_inline = false;
          }
          break;
        }
        case OPT_COND: {
          message = "optimizeConditionals();\n";
          randomization_helper::printHistory(n_transform, message);
          l.optimizeConditionals();
          break;
        }
        case SPLIT_TAIL: {
          auto loops = NodeFinder<For>::find(l.root_stmt());
          if (loops.empty()) {
            break;
          }
          int loop_n = std::rand() % (int)loops.size();
          auto loop = loops[loop_n];
          int factor = (std::rand() % 20) + 1;
          message = "splitWithTail(loops[" + std::to_string(loop_n) + "], " +
              std::to_string(factor) + ");\n";
          randomization_helper::printHistory(n_transform, message);
          l.splitWithTail(loop, factor);
          break;
        }
        case SPLIT_MASK: {
          auto loops = NodeFinder<For>::find(l.root_stmt());
          if (loops.empty()) {
            break;
          }
          int loop_n = std::rand() % (int)loops.size();
          auto loop = loops[loop_n];
          int factor = (std::rand() % 20) + 1;
          message = "splitWithMask(loops[" + std::to_string(loop_n) + "], " +
              std::to_string(factor) + ")\n";
          randomization_helper::printHistory(n_transform, message);
          l.splitWithMask(loop, factor);
          break;
        }
        case DIST1: {
          auto loops = NodeFinder<For>::find(l.root_stmt());
          if (loops.empty()) {
            break;
          }
          int loop_n = std::rand() % (int)loops.size();
          auto loop = loops[loop_n];
          std::vector<StmtPtr> stmts(
              loop->body()->begin(), loop->body()->end());
          if (stmts.empty()) {
            break;
          }
          int n_pivots = (std::rand() % (int)stmts.size()) + 1;
          std::vector<StmtPtr> pivots;
          std::vector<int> chosen_indices;
          std::tie(pivots, chosen_indices) =
              randomization_helper::select_n_randomly<StmtPtr>(
                  stmts, n_pivots, random_engine);
          std::unordered_set<StmtPtr> pivots_set(pivots.begin(), pivots.end());
          message = "distributeLoop(loops[" + std::to_string(loop_n) +
              "], pivots=stmts(" + randomization_helper::join(chosen_indices) +
              "))\n";
          randomization_helper::printHistory(n_transform, message);
          l.distributeLoop(loop, pivots_set);
          break;
        }
        case DIST2: {
          auto loops = NodeFinder<For>::find(l.root_stmt());

          if (loops.empty()) {
            break;
          }
          int loop_n = std::rand() % (int)loops.size();
          auto loop = loops[loop_n];

          message = "distributeLoop(loops[" + std::to_string(loop_n) + "])\n";
          randomization_helper::printHistory(n_transform, message);
          l.distributeLoop(loop);
          break;
        }
        case DIST3: {
          auto loops = NodeFinder<For>::find(l.root_stmt());

          if (loops.empty()) {
            break;
          }
          int loop_n = std::rand() % (int)loops.size();
          auto loop = loops[loop_n];

          message = "distributeLoopAndParents(loops[" + std::to_string(loop_n) +
              "])\n";
          randomization_helper::printHistory(n_transform, message);
          l.distributeLoopAndParents(loop);
          break;
        }
        case DIST4: {
          auto loops = NodeFinder<For>::find(l.root_stmt());

          if (loops.empty()) {
            break;
          }
          int loop_n = std::rand() % (int)loops.size();
          auto loop = loops[loop_n];

          message = "distributeLoopOverInnerLoops(loops[" +
              std::to_string(loop_n) + "])\n";
          randomization_helper::printHistory(n_transform, message);
          l.distributeLoopOverInnerLoops(loop);
          break;
        }
        case DIST5: {
          auto loops = NodeFinder<For>::find(l.root_stmt());

          if (loops.empty()) {
            break;
          }
          int loop_n = std::rand() % (int)loops.size();
          auto loop = loops[loop_n];

          message = "distributeLoopAndParentsOverInnerLoops(loops[" +
              std::to_string(loop_n) + "])\n";
          randomization_helper::printHistory(n_transform, message);
          l.distributeLoopAndParentsOverInnerLoops(loop);
          break;
        }
        case FUSE_LOOPS: {
          // Get all the loops
          auto loops = NodeFinder<For>::find(l.root_stmt());
          if (loops.size() <= 1) {
            break;
          }

          // Find a random number of loops to fuse
          int num_loops_to_fuse =
              std::max(2, (int)(std::rand() % (int)loops.size()));

          std::vector<ForPtr> loops_to_fuse;
          std::vector<int> chosen_indices;
          std::tie(loops_to_fuse, chosen_indices) =
              randomization_helper::select_n_randomly<ForPtr>(
                  loops, num_loops_to_fuse, random_engine);

          message = "fuseLoops(loops[" +
              randomization_helper::join(chosen_indices) + "], &fused_loop);\n";
          randomization_helper::printHistory(n_transform, message);
          // Fuse the loops
          ForPtr fused_loop;
          l.fuseLoops(loops_to_fuse, &fused_loop);
          break;
        }

        case REORDER_AXIS: {
          // Get all the loops
          auto loops = NodeFinder<For>::find(l.root_stmt());
          if (loops.size() <= 1) {
            break;
          }

          // Find pairs of axes that can be reordered
          std::vector<std::pair<ForPtr, ForPtr>> valid_pairs;
          for (const auto i : c10::irange(loops.size())) {
            for (const auto j : c10::irange(i + 1, loops.size())) {
              if (LoopNest::findOuterFor(loops[i], loops[j])) {
                valid_pairs.emplace_back(loops[i], loops[j]);
              }
            }
          }

          // Choose a pair randomly
          if (valid_pairs.empty()) {
            break;
          }
          int valid_pair_n = std::rand() % (int)valid_pairs.size();
          auto loop_pair = valid_pairs.at(valid_pair_n);
          auto first_loop = std::get<0>(loop_pair);
          auto second_loop = std::get<1>(loop_pair);

          std::string first_index =
              randomization_helper::indexOf(loops, first_loop);
          std::string second_index =
              randomization_helper::indexOf(loops, second_loop);
          message = "reorderAxis(loops[";
          message += first_index;
          message += "], loops[";
          message += second_index + "]);\n";
          randomization_helper::printHistory(n_transform, message);
          // reorder the axis
          l.reorderAxis(first_loop, second_loop);
          break;
        }

        case REORDER: {
          // Get all the loops
          auto loops = NodeFinder<For>::find(l.root_stmt());
          if (loops.size() <= 1) {
            break;
          }

          // Find all perfectly nested loop nests
          auto all_nested_loops =
              randomization_helper::GetAllPerfectlyNestedLoopNests(loops);
          if (all_nested_loops.empty()) {
            break;
          }

          // Randomly pick a set of consecutive loops to reorder
          int index = rand() % (int)all_nested_loops.size();
          auto nested_loops = all_nested_loops.at(index);

          // Create a random permutation for reordering
          std::vector<size_t> permutation(nested_loops.size());
          std::iota(permutation.begin(), permutation.end(), 0);
          std::shuffle(permutation.begin(), permutation.end(), random_engine);

          // Generate a good history message
          std::vector<std::string> indices;
          indices.reserve(nested_loops.size());
          for (const auto& l : nested_loops) {
            indices.push_back(randomization_helper::indexOf(loops, l));
          }
          message = "reorder(loops[" + randomization_helper::join(indices) +
              "], permutation=[" + randomization_helper::join(permutation) +
              "]);\n";
          randomization_helper::printHistory(n_transform, message);
          // reorder
          l.reorder(nested_loops, permutation);
          break;
        }

        case TILE: {
          // Get all the loops
          auto loops = NodeFinder<For>::find(l.root_stmt());
          if (loops.size() <= 1) {
            break;
          }

          // Tile needs two perfectly nested loops. To find such loops, we find
          // all perfectly nested loop nests, randomly pick one of them, and
          // randomly pick 2 consecutive loops in that loop nest.
          // Find all perfectly nested loop nests
          auto all_nested_loops =
              randomization_helper::GetAllPerfectlyNestedLoopNests(loops);
          if (all_nested_loops.empty()) {
            break;
          }

          int index = rand() % (int)all_nested_loops.size();
          auto nested_loops = all_nested_loops.at(index);
          if (nested_loops.size() < 2) {
            break;
          }
          int loop_number = rand() % ((int)nested_loops.size() - 1);
          auto x_loop = nested_loops.at(loop_number);
          auto y_loop = nested_loops.at(loop_number + 1);

          int x_factor = randomization_helper::find_factor(x_loop);
          int y_factor = randomization_helper::find_factor(y_loop);
          if (x_factor == -1 || y_factor == -1) {
            break;
          }

          std::string x_loop_index =
              randomization_helper::indexOf(loops, x_loop);
          std::string y_loop_index =
              randomization_helper::indexOf(loops, y_loop);
          message = "tile(loops[";
          message += x_loop_index;
          message += "], loops[";
          message += y_loop_index + "], ";
          message += std::to_string(x_factor);
          message += ", " + std::to_string(y_factor) + ");\n";
          randomization_helper::printHistory(n_transform, message);
          // tile
          l.tile(x_loop, y_loop, x_factor, y_factor);
          break;
        }

        case FULL_UNROLL: {
          auto loops = NodeFinder<For>::find(l.root_stmt());
          if (loops.empty()) {
            break;
          }
          int loop_n = std::rand() % (int)loops.size();
          auto loop = loops[loop_n];

          message = "fullUnroll(loops[" + std::to_string(loop_n) + "]);\n";
          randomization_helper::printHistory(n_transform, message);
          LoopNest::fullUnroll(loop);
          break;
        }

        case NORMALIZE: {
          auto loops = NodeFinder<For>::find(l.root_stmt());
          if (loops.empty()) {
            break;
          }
          int loop_n = std::rand() % (int)loops.size();
          auto loop = loops[loop_n];

          message = "normalize(loops[" + std::to_string(loop_n) + "]);\n";
          randomization_helper::printHistory(n_transform, message);
          l.normalize(loop);
          break;
        }

        case FLATTEN: {
          // Get all the loops
          auto loops = NodeFinder<For>::find(l.root_stmt());
          if (loops.size() <= 1) {
            break;
          }

          // Find all perfectly nested loop nests
          auto all_nested_loops =
              randomization_helper::GetAllPerfectlyNestedLoopNests(loops);
          if (all_nested_loops.empty()) {
            break;
          }

          // Randomly pick a set of consecutive loops to flatten
          int index = rand() % (int)all_nested_loops.size();
          auto nested_loops = all_nested_loops.at(index);

          // Generate a good history message
          std::vector<std::string> indices;
          indices.reserve(nested_loops.size());
          for (const auto& l : nested_loops) {
            indices.push_back(randomization_helper::indexOf(loops, l));
          }
          message =
              "flatten(loops[" + randomization_helper::join(indices) + "]);\n";
          randomization_helper::printHistory(n_transform, message);
          // flatten
          l.flatten(nested_loops);
          break;
        }

        case COMPRESS_BUFFER: {
          auto buffers = NodeFinder<Buf>::find(l.root_stmt());
          int buffer_n = std::rand() % (int)buffers.size();
          auto buffer = buffers[buffer_n];

          message = "compressBuffer(buffers[" + std::to_string(buffer_n) +
              "], l.root_stmt());\n";
          randomization_helper::printHistory(n_transform, message);
          l.compressBuffer(buffer, l.root_stmt());
          break;
        }

        case COMPRESS_ALL_BUFFERS: {
          message = "compressAllBuffers(l.root_stmt());\n";
          randomization_helper::printHistory(n_transform, message);
          l.compressAllBuffers(l.root_stmt());
          break;
        }

        case SLICE_HEAD: {
          // Get all the loops
          auto loops = NodeFinder<For>::find(l.root_stmt());
          if (loops.empty()) {
            break;
          }
          int loop_n = std::rand() % (int)loops.size();
          auto loop = loops[loop_n];

          int factor = randomization_helper::find_factor(loop);
          if (factor == -1) {
            break;
          }
          message = "sliceHead(loops[" + std::to_string(loop_n) + "]);\n";
          randomization_helper::printHistory(n_transform, message);
          l.sliceHead(loop, factor);
          break;
        }

        case SLICE_TAIL: {
          // Get all the loops
          auto loops = NodeFinder<For>::find(l.root_stmt());
          if (loops.empty()) {
            break;
          }
          int loop_n = std::rand() % (int)loops.size();
          auto loop = loops[loop_n];

          int factor = randomization_helper::find_factor(loop);
          if (factor == -1) {
            break;
          }
          message = "sliceTail(loops[" + std::to_string(loop_n) + "]);\n";
          randomization_helper::printHistory(n_transform, message);
          l.sliceTail(loop, factor);
          break;
        }

        case CACHE_ACCESSES: {
          // TODO - Implement cache_access
          break;
        }

        case COMPUTE_AT: {
          // To find valid compute at pairs, we need to collect the producer
          // consumer pairs. For now, we do not collect all such pairs for
          // simplicity. For now, we collect producer and the immediate parent
          // loop of the consumer. We could collect all the consumer enclosing
          // loops, but then we will have to clean up the ones that are shared
          // with the producer encloser loop. Currently, we only test on the
          // immediate parent loop.
          auto buffers = BufFinder::find(l.root_stmt());
          std::vector<std::pair<StmtPtr, ForPtr>> producer_consumer_pairs;

          for (const auto& buffer : buffers) {
            auto producers = l.getAllWritesToBuf(buffer);
            auto consumers = StmtsReadingBuf::find(l.root_stmt(), buffer);
            if (producers.size() != 1 || consumers.empty()) {
              continue;
            }

            for (const auto& producer : producers) {
              for (const auto& consumer : consumers) {
                auto parent_loop = LoopNest::getParentLoop(consumer);
                auto pc_pair = std::make_pair(producer, parent_loop);
                producer_consumer_pairs.push_back(pc_pair);
              }
            }
          }

          if (producer_consumer_pairs.empty()) {
            break;
          }

          // Choose a random pair
          int pair_n = std::rand() % (int)producer_consumer_pairs.size();
          auto pc_pair = producer_consumer_pairs.at(pair_n);
          auto store = std::get<0>(pc_pair);
          auto for_ptr = std::get<1>(pc_pair);

          // TODO - come up with better message
          message = "computeAt(....);\n";
          randomization_helper::printHistory(n_transform, message);
          l.computeAt(store, for_ptr);
          break;
        }

        case RFACTOR: {
          // TODO - Implement rfactor
          break;
        }

        case VECTORIZE: {
          auto loops = NodeFinder<For>::find(l.root_stmt());
          std::vector<ForPtr> innermost_loops;

          for (const auto& loop : loops) {
            bool containsSubLoops = false;
            if (BlockPtr body = to<Block>(loop->body())) {
              for (const StmtPtr& stmt : *body) {
                if (ForPtr f2 = to<For>(stmt)) {
                  containsSubLoops = true;
                }
              }
            }

            if (!containsSubLoops) {
              innermost_loops.push_back(loop);
            }
          }

          if (innermost_loops.empty()) {
            break;
          }
          int loop_n = std::rand() % (int)innermost_loops.size();
          auto loop = innermost_loops[loop_n];

          message = "vectorize(loops[" + std::to_string(loop_n) + "]);\n";
          randomization_helper::printHistory(n_transform, message);
          l.vectorize(loop);
          break;
        }

        case VECTORIZE_INNER_LOOPS: {
          message = "vectorizeInnerLoops();\n";
          randomization_helper::printHistory(n_transform, message);
          l.vectorizeInnerLoops();
          break;
        }

        case ELIMINATE_DEAD_STORES: {
          message = "eliminateDeadStores();\n";
          randomization_helper::printHistory(n_transform, message);
          l.eliminateDeadStores();
          break;
        }

        // TODO: Add remaining transforms
        default:
          break;
      }
    }
  } catch (...) {
    std::cout << "EXCEPTION THROWN!\n";
    std::cout << "SEED: " << seed << "\n";
    throw std::runtime_error("Random test failed");
  }
  message = "End of transformations;\n";
  randomization_helper::printHistory(n_transforms, message);
  return;
}

} // namespace torch::jit::tensorexpr
