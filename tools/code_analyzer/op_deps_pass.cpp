// This LLVM pass takes LLVM bitcode / assembly as input and generates
// dependency graph among aten ops. From a set of root ops used by a model, we
// can calculate transitive closure of all dependent ops, then we can produce a
// custom LibTorch library with optimal build size which only registers and
// contains ops needed by the specific model - unregistered / unused ops can be
// stripped at link time.
//
// [Approach]
// To generate the dependency graph it searches for 3 types of connections in
// LLVM bitcode / assembly:
//  1) op registration: op name (schema string literal) -> registered function;
//  2) regular function call: function -> function;
//  3) op invocation: function -> op name (schema string literal)
//
// For #2 it uses similar algorithm as llvm::LazyCallGraph - not only looks into
// call/invoke instructions but also recursively searches for function pointers
// in each instruction's operands.
//
// For #1 and #3 it searches for connections between operator name string
// literals / function pointers and c10 op registration/invocation API calls in
// LLVM IR graph via "use" edges (bi-directional):
// 1. llvm::Value has "users()" method to get other llvm::Value nodes that use
//    the value;
// 2. most of types derive from llvm::User which has "operands()" method to get
//    other llvm::Value nodes being used by the value;
//
// [Limitation]
// For now the search doesn't go beyond the function boundary because the
// reference to op name string literals and c10 op registration/invocation
// APIs are almost always in the same function. If we create helper function
// around c10 API, we could simply add them to the regular expression used to identify c10 API.
//
// [Example]
// In the following example, it finds out:
//  1) the registered function for "quantized:add" operator;
//  2) one possible call path to at::empty() function;
//  3) the called operator name "aten::empty":
//
// - quantized::add
// - c10::detail::wrap_kernel_functor_unboxed_<at::native::(anonymous
//   namespace)::QAdd<false>, at::Tensor (at::Tensor, at::Tensor, double,
//   long)>::call(c10::OperatorKernel*, at::Tensor, at::Tensor, double, long)
// - at::native::(anonymous namespace)::QAdd<false>::operator()(at::Tensor,
//   at::Tensor, double, long)
// - void at::native::DispatchStub<void (*)(at::Tensor&, at::Tensor const&,
//   at::Tensor const&), at::native::qadd_stub>::operator()<at::Tensor&,
//   at::Tensor const&, at::Tensor const&>(c10::DeviceType, at::Tensor&,
//   at::Tensor const&, at::Tensor const&)
// - at::native::DispatchStub<void (*)(at::Tensor&, at::Tensor const&,
//   at::Tensor const&), at::native::qadd_stub>::choose_cpu_impl()
// - void at::native::(anonymous namespace)::qadd_kernel<false>(at::Tensor&,
//   at::Tensor const&, at::Tensor const&)
// - at::TensorIterator::binary_op(at::Tensor&, at::Tensor const&, at::Tensor
//   const&, bool)
// - at::TensorIterator::build()
// - at::TensorIterator::fast_set_up()
// - at::empty(c10::ArrayRef<long>, c10::TensorOptions const&,
//   c10::optional<c10::MemoryFormat>)
// - aten::empty

#include <deque>
#include <iostream>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "llvm/Demangle/Demangle.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

struct RegexOpt {
  std::shared_ptr<Regex> pattern;

  void operator=(const std::string& val) {
    if (val.empty()) {
      return;
    }
    pattern = std::make_shared<Regex>(val);
    std::string regexError;
    if (!pattern->isValid(regexError)) {
      report_fatal_error(
          "Invalid regular expression param: '" + val + "' err: " + regexError,
          false);
    }
  };
};

class RegexOptParser : public cl::basic_parser<RegexOpt> {
public:
  RegexOptParser(cl::Option& O) : basic_parser(O) {}
  virtual ~RegexOptParser() = default;

  // parse - Return true on error.
  bool parse(cl::Option&, StringRef, StringRef Arg, RegexOpt& Value) {
    Value = Arg.str();
    return false;
  }
  StringRef getValueName() const override {
    return "RegexOpt";
  }
};

RegexOpt FunctionSchemaPatternLoc;
cl::opt<RegexOpt, true, cl::parser<std::string>> FunctionSchemaPattern(
    "op_schema_pattern",
    cl::desc("Regular expression used to identify aten op schema strings. "
             "Example: -op_schema_pattern '^(aten|quantized)::[^ ]+'"),
    cl::location(FunctionSchemaPatternLoc),
    cl::Required,
    cl::ValueRequired);

RegexOpt OpRegistrationPatternLoc;
cl::opt<RegexOpt, true, cl::parser<std::string>> OpRegistrationPattern(
    "op_register_pattern",
    cl::desc("Regular expression used to identify c10 op registration API. "
             "Example: -op_register_pattern 'c10::RegisterOperators::op'"),
    cl::location(OpRegistrationPatternLoc),
    cl::Required,
    cl::ValueRequired);

RegexOpt OpInvocationPatternLoc;
cl::opt<RegexOpt, true, cl::parser<std::string>> OpInvocationPattern(
    "op_invoke_pattern",
    cl::desc("Regular expression used to identify c10 op invocation API. "
             "Example: -op_invoke_pattern 'c10::Dispatcher::findSchema'"),
    cl::location(OpInvocationPatternLoc),
    cl::Required,
    cl::ValueRequired);

// The `root_symbol_pattern` is used to specify the seeds of C++ symbols
// from which it searches for transitively reachable ops which need to be
// kept for these C++ APIs to be able to run.
//
// Why not dump ops that are reachable from any visible C++ symbols? Why
// limit it to a subset of root symbols?
// Because op registration callsites in static initializer are visible root
// symbols, too. It will dump ALL the registered ops without any filtering.
//
// Can we use some fixed entry point like `main()`?
// The target to be analyzed can be DSO that doesn't have a `main()`. And
// sometimes we want to get ops that could (but not yet) be called.
//
// This temporary flag will be deprecated by better alternatives in the future.
RegexOpt RootSymbolPatternLoc;
cl::opt<RegexOpt, true, cl::parser<std::string>> RootSymbolPattern(
    "root_symbol_pattern",
    cl::desc("Regular expression used to identify root symbols. It will insert "
             "an entry to the output graph with key = `__ROOT__` and value = "
             "set of ops reachable from root symbols, if the pattern is set. "
             "Example: -root_symbol_pattern 'torch::jit'"),
    cl::location(RootSymbolPatternLoc));

cl::list<RegexOpt, bool, RegexOptParser> TorchLibraryInitPattern(
    "torch_library_init_pattern",
    cl::desc("Regular expression used to identify TorchLibraryInit symbols "
             "that are generated by `TORCH_LIBRARY` macro. The first capturing "
             "group is used to extract namespace string. "
             "Example: -torch_library_init_pattern "
             "'^.*TORCH_LIBRARY_init_([^(]+)(\\(.*)?$'"),
    cl::ZeroOrMore);

cl::opt<int> Verbose(
    "v",
    cl::desc("Verbose level"),
    cl::Hidden,
    cl::init(0));

cl::opt<bool> DebugPath(
    "debug_path",
    cl::desc("Output path between two nodes."),
    cl::init(false));

using SET = std::set<std::string>;
using GRAPH = std::unordered_map<std::string, std::set<std::string>>;
using VALUE_MAP = std::unordered_map<Value*, Value*>;
using VALUE_SET = std::unordered_set<Value*>;

// SRC -> Inverse "tree" from all reachable destinations back to SRC, e.g.:
// (DEST-1 -> PREV_11, PREV_11 -> PREV_12, ..., PREV_1n -> SRC)
// (DEST-2 -> PREV_21, PREV_21 -> PREV_22, ..., PREV_2n -> SRC)
using PATH = std::unordered_map<std::string,
                                std::unordered_map<std::string, std::string>>;

// Referenced the logic in llvm-cxxfilt.cpp.
// Starting from LLVM 9 it provides a `demangle()` API. Here we keep our ad-hoc
// version for backward compatibility.
std::string _demangle(const std::string& mangled) {
  int status;
  const char* decorated = mangled.c_str();
  size_t decoratedLength = mangled.length();

  char *undecorated = itaniumDemangle(decorated, nullptr, nullptr, &status);

  if (!undecorated &&
      (decoratedLength > 6 && strncmp(decorated, "__imp_", 6) == 0)) {
    undecorated = itaniumDemangle(decorated + 6, nullptr, nullptr, &status);
  }
  std::string result(undecorated ? undecorated : mangled);
  free(undecorated);
  return result;
}

// LLVM_DEBUG needs opt to be built with debug support.
template<
    typename T,
    typename std::enable_if<std::is_base_of<Value, T>::value, int>::type = 0>
std::ostream& operator<<(std::ostream& out, T& I) {
  std::string str;
  raw_string_ostream O(str);
  O << I;
  return out << str;
}

class OpDependency : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid

  OpDependency() : ModulePass(ID) {}
  ~OpDependency() = default;

  bool runOnModule(Module& M) override {
    // Scan all functions and instructions to construct function -> function
    // dependency graph and to find out:
    // - visible functions matching `root_symbol_pattern` option;
    // - instructions that might register or invoke operators, respectively.
    GRAPH deps;
    VALUE_SET visibleFuncs, opRegistrationInsts, opInvocationInsts;
    scanAllFunctions(
        M, &deps, &visibleFuncs, &opRegistrationInsts, &opInvocationInsts);

    // "Key nodes" are nodes we want to keep in output graph. They are usually
    // op-schema strings.
    SET keyNodes;

    // Insert a dummy root node with links to function nodes matching the
    // "root symbol" regex pattern and with default visibility. The goal is to
    // find aten ops that are possibly called via torch C++ APIs.
    insertRoot(visibleFuncs, &deps, &keyNodes);

    // Scan op registration/invocation API calls to construct the link between
    // op name (a.k.a op schema string) and related functions.
    // Dump the op-schema -> function and function -> op-schema mappings into
    // the same `deps` graph with function -> function mappings as they will
    // be processed together next.
    scanOpRegistration(opRegistrationInsts, &keyNodes, &deps);
    scanOpInvocation(opInvocationInsts, &keyNodes, &deps);

    // Shrink the graph by removing intermediate nodes (functions) while
    // maintaining transitive dependency between operators (schema strings).
    GRAPH result;
    std::shared_ptr<PATH> path = DebugPath ? std::make_shared<PATH>() : nullptr;
    simplifyGraph(deps, keyNodes, &result, path.get());

    printAsYAML(std::cout, keyNodes, result, path.get());
    return false;
  }

private:
  static void insertRoot(
      const VALUE_SET& visibleFuncs, GRAPH* deps, SET* keyNodes) {
    if (!RootSymbolPatternLoc.pattern) {
      return;
    }
    SET roots;
    for (const auto& F : visibleFuncs) {
      std::string name = F->getName();
      auto demangled = _demangle(name);
      if (RootSymbolPatternLoc.pattern->match(demangled)) {
        roots.insert(name);
        if (Verbose) {
          std::cerr << "[DEBUG][ROOT_FUNC] " << demangled << std::endl;
        }
      }
    }

    static const std::string ROOT_NODE{"__ROOT__"};
    deps->emplace(ROOT_NODE, std::move(roots));
    keyNodes->insert(ROOT_NODE);
  }

  // Scan the entire IR graph to construct function -> function dependency graph
  // as well as instructions that might register or invoke operators.
  static void scanAllFunctions(
      Module& M, GRAPH* deps, VALUE_SET* visibleFuncs,
      VALUE_SET* opRegistrationInsts, VALUE_SET* opInvocationInsts) {
    for (Function& F : M) {
      if (F.hasDefaultVisibility()) {
        visibleFuncs->insert(&F);
      }
      std::string caller = F.getName();
      std::string callerDemangled = _demangle(caller);
      for (BasicBlock& BB : F) {
        for (Instruction& I : BB) {
          scanReferredFunctions(I, [&](Function* func) -> void {
            std::string callee = func->getName();
            std::string calleeDemangled = _demangle(callee);
            (*deps)[caller].insert(callee);
            if (Verbose > 1) {
              std::cerr << "[DEBUG][FUNC_CALL] " << callerDemangled << " => "
                        << calleeDemangled << std::endl;
            }
            // One registration/invocation API might call another registration/
            // invocation API in which case we can skip processing the nested
            // call. This is a simple trick to avoid "cannot find registered/
            // invoked op" warning and doesn't affect correctness, because
            // later in scanOpRegistration we'll walk the transitively reachable
            // IR graph again from each registration instance.
            if (!OpRegistrationPatternLoc.pattern->match(callerDemangled) &&
                OpRegistrationPatternLoc.pattern->match(calleeDemangled)) {
              (*opRegistrationInsts).insert(&I);
            }
            if (!OpInvocationPatternLoc.pattern->match(callerDemangled) &&
                OpInvocationPatternLoc.pattern->match(calleeDemangled)) {
              (*opInvocationInsts).insert(&I);
            }
          });
        }
      }
    }
  }

  // llvm::CallGraph only searches for functions referenced by "CallSites" (i.e.
  // by call/invoke instructions). However functions can be referenced by
  // non-call/invoke instructions as well (being passed as function pointer),
  // e.g.:
  // ```
  // store i64 ptrtoint (void (%"class.at::Tensor"*, %"class.at::Tensor"*)*
  //                     @at::foo_op(at::Tensor const&) to i64), i64* %14, ...
  // ```
  // "@at::foo_op" is a operand of "ptrtoint", which in turn is a constant
  // operand of "store" instruction. The stored function pointer can be called
  // indirectly later on.
  //
  // Sometimes directly called functions can be in ConstExpr as well, e.g.:
  // ```
  // invoke void bitcast (
  //    void (ty1*, ...)* @c10::Dispatcher::findSchema(...) to
  //    void (ty2*, ...)*)(...)
  // ```
  // In above case, "CallSite(I).getCalledFunction()" won't return "findSchema"
  // as it's nested in "bitcast" instruction.
  //
  // To cover these cases this method recursively traverses all operands of the
  // input instruction "I" to search for directly/indirectly referenced function
  // pointers by the instruction. The referenced functions might NOT actually be
  // called (which is fine for our use case). llvm::LazyCallGraph has similar
  // logic and we reuse its "visitReferences" method to traverse all operands.
  static void scanReferredFunctions(
      Instruction& I, const std::function<void(Function*)>& CB) {
    SmallVector<Constant*, 16> worklist;
    SmallPtrSet<Constant*, 16> visited;

    if (auto CS = CallSite(&I)) {
      Function* callee = CS.getCalledFunction();
      if (callee && !callee->isIntrinsic() && visited.insert(callee).second) {
        CB(callee);
      }
    }

    for (Value* op : I.operand_values()) {
      Constant* C = dyn_cast<Constant>(op);
      if (C && visited.insert(C).second) {
        worklist.push_back(C);
      }
    }

    LazyCallGraph::visitReferences(worklist, visited, [&](Function& F) {
      if (!F.isIntrinsic()) {
        CB(&F);
      }
    });
  }

  // Naive connectivity analysis to find out all nodes that are reachable from a
  // specific node in IR graph by following each node's "use" edges (link to its
  // operands and users).
  // This is the core algorithm we use to find the connection between op name
  // string literals and registered/invoked functions - there should be a path
  // to connect them to the c10 op registration/invocation APIs.
  // For now the search doesn't go beyond the function boundary because the
  // reference to op name string literals and c10 op registration/invocation
  // APIs are almost always in the same function.
  static void scanConnectedNodes(
      Value* src,
      VALUE_SET blocked,
      const std::function<void(Value*)>& CB, VALUE_MAP* debugPath) {
    std::deque<Value*> worklist;
    SmallPtrSet<Value*, 16> visited;

    auto insert = [&](Value* cur, Value* parent) -> void {
      if (!blocked.count(cur) && visited.insert(cur).second) {
        worklist.push_back(cur);
        if (debugPath) {
          (*debugPath).emplace(cur, parent);
        }
      }
    };

    auto expandOperands = [&](Value* V) -> void {
      // Stops if it doesn't have operands (!isa<User>) or it is a function.
      if (!isa<User>(V) || isa<Function>(V)) {
        return;
      }
      auto node = dyn_cast<User>(V);
      for (auto& O : node->operands()) {
        insert(O, node);
      }
    };

    auto blockSiblingOperands = [&](User* U, Value* V) -> void {
      // This is to handle a special case only appears in LLVM 9 (not in 5 - 8
      // and 10), where it can falsely associate unrelated PyTorch op
      // registrations.
      //
      // If the value `V` is used by a PHI-node `U`, then we should stop
      // crawling `U`'s operands, i.e. `V`'s siblings in `U`. E.g.:
      //
      //   114:                                            ; preds = %111, %109
      //     %115 = phi i32 [ %110, %109 ], [ %112, %111 ]
      //
      // `%115` might take the value of `%110` or `%112`, depending on from
      // which label it comes. Assuming `V` is `%110` and `U` is `%115`, we can
      // continue to scan `%115` but should not crawl `%112` as it does not
      // directly pass data from `%110` to `%112` (and vice versa).
      //
      // NB: we probably should do the same for other LLVM instructions with
      // this kind of selective semantics. But for the purpose of analyzing
      // PyTorch registrations it seems to be sufficent for now.
      if (isa<PHINode>(U)) {
        for (auto& S : U->operands()) {
          blocked.insert(S);
        }
      }
    };

    auto expandUsers = [&](Value* V) -> void {
      // If the value is not constant, then the user of the value might pass
      // other value into it, e.g.:
      //   store @.str.15, %10
      //   invoke @c10.reg_op, %10, @foo
      // The store instruction, which is the user of "%10", passes "@.str.15" to
      // "%10" which in turn is passed to "@c10.reg_op" API function.
      // Users of constants are not interesting as they cannot change the state
      // of the constant. We skip users of functions as well assuming
      // interesting values (op names and function pointers) are not set via
      // other invocations of the function.
      if (!isa<User>(V) || isa<Constant>(V) || isa<Function>(V)) {
        return;
      }
      for (auto U : V->users()) {
        insert(U, V);
        blockSiblingOperands(U, V);
      }
    };

    auto expand = [&](Value* V) -> void {
      expandOperands(V);
      expandUsers(V);
    };

    expand(src);
    while (!worklist.empty()) {
      auto cur = worklist.front();
      worklist.pop_front();
      expand(cur);

      if (isa<Function>(cur) || isa<Constant>(cur)) {
        CB(cur);
      }
    }
  }

  // Calculate transitive closure and remove intermediate (non-key) nodes.
  // Note that there are two type of nodes in the dependency graph:
  // 1) String literals in source files, e.g.:
  //    "aten::cos_(Tensor(a!) self) -> Tensor(a!)", which represents operator
  //    "schema";
  // 2) Function symbols in object files, e.g.:
  //    "at::CPUType::(anonymous namespace)::cos_(at::Tensor&)";
  // Both of them are added to the dependency graph as std::string. Ultimately
  // we only care about #1 as that's what we use to prune registered ops via
  // codegen, then #2 will be stripped by linker automatically. So the goal is
  // to remove #2 from the graph while maintaining the transitive dependency
  // between #1. #1 is called "key nodes" in this method.
  static void simplifyGraph(
      const GRAPH& input, SET& keyNodes, GRAPH* output, PATH* path) {
    // Starting from every key node, use BFS to traverse all nodes that are
    // transitively reachable from the node in the sparse graph.
    for (auto& key : keyNodes) {
      std::deque<std::string> queue;
      SET visited;  // has some runtime issue with std::unordered_set
      auto expand = [&](const std::string& curNode) -> void {
        auto it = input.find(curNode);
        if (it == input.end()) {
          return;
        }
        for (const auto& next : it->second) {
          if (!visited.insert(next).second) {
            continue;
          }
          queue.push_back(next);
          if (path) {
            (*path)[key].emplace(next, curNode);
          }
        }
      };

      expand(key);
      while (!queue.empty()) {
        auto curNode = queue.front();
        queue.pop_front();
        if (keyNodes.count(curNode)) {
          // Output links between key nodes.
          (*output)[key].insert(curNode);
          // Stop expanding key nodes.
          continue;
        }
        expand(curNode);
      }
    }
  }

  // Find out operator names and function pointers that are transitively
  // connected to the same 'src' instruction.
  static void scanOpSchemaStrAndFunction(
      Instruction* src, const VALUE_SET& blocked,
      const std::string& contextualNamespace,
      SET* visitedOps, SET* visitedFunctions) {
    std::shared_ptr<VALUE_MAP> debugPath =
        (Verbose > 2 ? std::make_shared<VALUE_MAP>() : nullptr);
    auto callback = [&](Value* V) -> void {
      if (auto schemaStr = extractOpSchema(contextualNamespace, V)) {
        if (visitedOps) {
          // NB: Some debug string constants might be connected to the
          // registration instruction, e.g.: "Lambda". Since we have factored
          // out namespace from op schema string, there is no longer a simple
          // way to identify these fake ops. For now we simply take the first
          // instance as the real op name is closest to the seed instruction
          // in BFS order.
          if (!visitedOps->empty()) {
            if (Verbose) {
              std::cerr << "[INFO] ignore extra op schema str: " << *schemaStr
                        << " in: " << _demangle(src->getFunction()->getName())
                        << ", because already found valid op schema str: "
                        << *visitedOps->begin() << std::endl;
            }
          } else {
            (*visitedOps).insert(*schemaStr);
          }
        }
        if (Verbose > 1) {
          std::cerr << "[DEBUG][OP_SCHEMA] " << *schemaStr << std::endl;
          printDebugPath(debugPath.get(), src, V);
        }
      } else if (auto F = dyn_cast<Function>(V)) {
        if (F->isIntrinsic()) {
          return;
        }
        if (visitedFunctions) {
          (*visitedFunctions).insert(F->getName());
        }
        if (Verbose > 1) {
          std::cerr << "[DEBUG][FUNC] " << _demangle(F->getName()) << std::endl;
          printDebugPath(debugPath.get(), src, V);
        }
      }
    };
    scanConnectedNodes(src, blocked, callback, debugPath.get());
  }

  // This method looks for op schema strings and function pointers that connect
  // to the same c10 op registration API call via "use" edges (bi-directional)
  // in IR graph - exploring both nodes being used (operands) by the node and
  // nodes using (users) the node.
  //
  // It assumes that the function pointers are needed (registered) for the op.
  //
  // For example, from op name "aten::add" to registration API call:
  // [OP_SCHEMA] aten::add
  // [PATH][1][CONST] [70 x i8] c"aten::add.Scalar(Tensor self...\00"
  // [PATH][2][CONST] @.str.55.20575 = private unnamed_addr constant [70 x i8]
  //                  c"aten::add.Scalar(Tensor self, ...\00", align 1
  // [PATH][3][CONST] i8* getelementptr inbounds ([70 x i8], [70 x i8]*
  //                  @.str.55.20575, i64 0, i64 0)
  // [PATH][4][INST]  invoke void @std::basic_string<...>::basic_string(...)
  //                  (%"class.std::basic_string"* ... %1477,
  //                  i8* getelementptr ... @.str.55.20575 ...)
  // [PATH][5][INST]  %1477 = alloca %"class.std::basic_string" ...
  // [PATH][6][INST]  %4086 = invoke ...
  //                  @c10::RegisterOperators::Options::schema(... %1477)
  // [PATH][7][INST]  %4088 = invoke ... @...catchAllKernel...(... %4086, ...
  //                  @at::TypeDefault::add(at::Tensor const&...))
  // [PATH][8][INST]  %4090 = invoke ...
  //                  &&(%"class.c10::RegisterOperators::Options"*... %4088 ...)
  // [PATH][9][INST]  invoke void
  //                  @c10::RegisterOperators::checkSchemaAndRegisterOp_(...
  //                  %"class.c10::RegisterOperators::Options"* ... %4090)
  //
  // From function pointer to registration API call:
  // [FUNC] at::TypeDefault::add(at::Tensor const&, c10::Scalar, c10::Scalar)
  // [PATH][1][FUNC]  at::TypeDefault::add(at::Tensor const&...)
  // [PATH][2][INST]  %4088 = invoke ... @...catchAllKernel...(... %4086, ...
  //                  @at::TypeDefault::add(at::Tensor const&...))
  // [PATH][3][INST]  %4090 = invoke ...
  //                  &&(%"class.c10::RegisterOperators::Options"*... %4088 ...)
  // [PATH][4][INST]  invoke void
  //                  @c10::RegisterOperators::checkSchemaAndRegisterOp_(...
  //                  %"class.c10::RegisterOperators::Options"* ... %4090)
  static void scanOpRegistration(
      VALUE_SET& instructions, SET* opSchemaStrs, GRAPH* schemaStrToFunctions) {
    for (auto V : instructions) {
      auto I = dyn_cast<Instruction>(V);
      // We only need to process call/invoke instructions.
      if (!I || !CallSite(I)) {
        continue;
      }
      auto contextualNamespace = inferContextualNamespace(I);
      if (Verbose && !contextualNamespace.empty()) {
        std::cerr << "[DEBUG][REG][NAMESPACE] " << contextualNamespace
                  << std::endl;
      }
      if (Verbose > 2) {
        std::cerr << "[DEBUG][REG][INST] " << *I << std::endl;
      }
      SET visitedOps, visitedFunctions;
      // Pass in "instructions" set as "blocked" set - all operator registration
      // calls are connected to global op registry object so we should avoid
      // going from one op registration call to another op registration call via
      // the global registry object.
      scanOpSchemaStrAndFunction(
          I, instructions, contextualNamespace, &visitedOps, &visitedFunctions);
      if (visitedOps.size() != 1) {
        std::cerr << "[WARNING] found " << visitedOps.size() << " ops ( ";
        for (auto& op : visitedOps) {
          std::cerr << op << " ";
        }
        std::cerr << ") in a registration call in function: "
                  << _demangle(I->getFunction()->getName())
                  << " contextualNamespace: " << contextualNamespace
                  << std::endl;
      }
      for (const auto& op : visitedOps) {
        opSchemaStrs->insert(op);
        if (visitedFunctions.empty()) {
          std::cerr << "[WARNING] could not find registered function for op: "
                    << op << " in function: "
                    << _demangle(I->getFunction()->getName())
                    << " contextualNamespace: " << contextualNamespace
                    << std::endl;
        }
        for (const auto& func : visitedFunctions) {
          (*schemaStrToFunctions)[op].insert(func);
          if (Verbose) {
            std::cerr << "[DEBUG][OP_REG] " << op << " => "
                      << _demangle(func) << std::endl;
          }
        }
      }
    }
  }

  static std::string inferContextualNamespace(Instruction* I) {
    auto functionName = _demangle(I->getFunction()->getName());
    for (auto& pattern : TorchLibraryInitPattern) {
      if (!pattern.pattern->match(functionName)) {
        continue;
      }
      if (Verbose) {
        std::cerr << "[DEBUG][REG][INIT_FUNC] " << functionName << std::endl;
      }
      return pattern.pattern->sub("\\1", functionName) + "::";
    }
    return {};
  }

  // Similar as scanOpRegistration - it searches for op schema strings that
  // connect to c10 op invocation API call and assume the parent function of the
  // API call invokes the operator.
  //
  // For example, from op name "aten::empty" to invocation API call:
  // [OP_SCHEMA] aten::empty
  // [PATH][1][CONST] [12 x i8] c"aten::empty\00"
  // [PATH][2][CONST] @.str.69.1990 = private unnamed_addr constant [12 x i8]
  //                  c"aten::empty\00", align 1
  // [PATH][3][CONST] i8* getelementptr inbounds ([12 x i8], [12 x i8]*
  //                  @.str.69.1990, i64 0, i64 0)
  // [PATH][4][INST]  invoke void @std::basic_string<...>::basic_string(...
  //                  (%"class.std::basic_string"* nonnull %19,
  //                  i8* getelementptr inbounds ([12 x i8], [12 x i8]*
  //                  @.str.69.1990, i64 0, i64 0) ...
  // [PATH][5][INST]  %19 = alloca %"class.std::basic_string", align 8
  // [PATH][6][INST]  %53 = bitcast %"class.std::basic_string"* %19 to i64*
  // [PATH][7][INST]  %54 = load i64, i64* %53, align 8, !tbaa !4
  // [PATH][8][INST]  store i64 %54, i64* %55, align 8, !tbaa !4
  // [PATH][9][INST]  %55 = bitcast %"struct.c10::OperatorName"* %18 to i64*
  // [PATH][10][INST] %18 = alloca %"struct.c10::OperatorName", align 8
  // [PATH][11][INST] invoke void @c10::Dispatcher::findSchema(c10::OperatorName
  //                  const&)(%"class.c10::optional.105"* nonnull sret %17,
  //                  %"class.c10::Dispatcher.6320"* nonnull %45,
  //                  %"struct.c10::OperatorName"* nonnull dereferenceable(16)
  //                  %18)
  static void scanOpInvocation(
      VALUE_SET& instructions, SET* opSchemaStrs, GRAPH* functionToSchemaStrs) {
    for (auto V : instructions) {
      auto I = dyn_cast<Instruction>(V);
      // We only need to process call/invoke instructions.
      if (!I || !CallSite(I)) {
        continue;
      }
      if (Verbose > 2) {
        std::cerr << "[DEBUG][CALL][INST] " << *I << std::endl;
      }
      std::string caller = I->getFunction()->getName();
      SET visitedOps;
      scanOpSchemaStrAndFunction(I, {}, {}, &visitedOps, nullptr);
      if (visitedOps.size() != 1) {
        std::cerr << "[WARNING] found " << visitedOps.size() << " ops ( ";
        for (auto& op : visitedOps) {
          std::cerr << op << " ";
        }
        std::cerr << ") in a invocation call in function: "
                  << _demangle(caller) << std::endl;
      }
      for (const auto& op : visitedOps) {
        opSchemaStrs->insert(op);
        (*functionToSchemaStrs)[caller].insert(op);
        if (Verbose) {
          std::cerr << "[DEBUG][OP_CALL] " << _demangle(caller) << " => "
                    << op << std::endl;
        }
      }
    }
  }

  static void extractStringValue(
      Value* V, const std::function<void(const std::string&)>& CB) {
    if (auto array = dyn_cast<ConstantDataArray>(V)) {
      // Normal case for c-style string literal and "std::basic_string".
      if (array->isCString()) {
        CB(array->getAsCString().str());
      } else if (array->isString()) {
        std::cerr << "[WARNING] ignore non-C string: "
                  << array->getAsString().str() << std::endl;
      }
    } else if (auto CI = dyn_cast<ConstantInt>(V)) {
      // Short string literal might be encoded into constant integer, e.g.:
      // "aten::AA" => 4702103508586165345 (0x41413A3A6E657461)
      // This can be tricky as it depends on consistent endianness/size.
      // Seen this case for "std::__1::basic_string" ABI.
      uint64_t intValue = CI->getZExtValue();
      auto data = reinterpret_cast<const char*>(&intValue);
      CB({data, data + sizeof(uint64_t)/sizeof(char)});
    } else if (auto C = dyn_cast<Constant>(V)) {
      // Short string literal might be in a constant vector, e.g.:
      // store <2 x i64> <i64 8, i64 4702103508586165345>, <2 x i64>* %25
      // Recursively extract each element to cover this case.
      // Seen this case for "std::__cxx11::basic_string" ABI.
      for (unsigned i = 0; auto elem = C->getAggregateElement(i); ++i) {
        extractStringValue(elem, CB);
      }
    }
  }

  static std::shared_ptr<std::string> extractOpSchema(
      const std::string& contextualNamespace, Value* V) {
    std::vector<std::string> schemaStrs;
    extractStringValue(V, [&](const std::string& str) {
      // NB: some operator names might contain namespace. If this occurs, we
      // MUST NOT use the contextual namespace. Fortunately, it's easy to tell
      // if a namespace is included: a double colon will be present.
      // In particular, this will occur with TORCH_SELECTIVE_NAME.
      const std::string& schemaStr =
          (contextualNamespace.empty() || str.find("::") != std::string::npos)
          ? str : contextualNamespace + str;
      if (FunctionSchemaPatternLoc.pattern->match(schemaStr)) {
        schemaStrs.push_back(schemaStr);
      }
    });
    if (schemaStrs.empty()) {
      return {};
    }
    if (schemaStrs.size() > 1) {
      std::cerr << "[WARNING] found " << schemaStrs.size()
                << " op schema strings in one value!" << std::endl;
    }
    const std::string schemaStr = schemaStrs[0];
    auto pos = schemaStr.find_first_of(".(");
    return std::make_shared<std::string>(
        pos == std::string::npos ? schemaStr : schemaStr.substr(0, pos));
  }

  static void printDebugPath(
      const VALUE_MAP* debugPath, Value* src, Value* dest) {
    if (!debugPath) {
      return;
    }
    int depth = 0;
    for (auto N = dest; ; N = debugPath->at(N)) {
      std::cerr << "[DEBUG][PATH][" << ++depth << "]";
      printDebugValue(N);
      std::cerr << std::endl;
      if (N == src) {
        break;
      }
    }
  }

  static void printDebugValue(Value* V) {
    if (auto F = dyn_cast<Function>(V)) {
      std::cerr << "[FUNC] " << _demangle(F->getName());
    } else if (isa<Constant>(V)) {
      std::cerr << "[CONST] " << *V;
    } else if (isa<Instruction>(V)) {
      std::cerr << "[INST] " << *V;
    } else if (V) {
      std::cerr << "[VALUE] " << *V;
    } else {
      std::cerr << "NULL";
    }
  }

  static void printAsYAML(
      std::ostream& out, const SET& keys, const GRAPH& graph,
      const PATH* path) {
    for (const auto& K : keys) {
      out << "- name: " << _demangle(K) << std::endl;
      auto it = graph.find(K);
      if (it == graph.end() || it->second.empty()) {
        continue;
      }
      out << "  depends:" << std::endl;
      for (const auto& value : it->second) {
        out << "  - name: " << _demangle(value) << std::endl;
        if (path) {
          std::vector<std::string> rpath;
          for (std::string prev = value;
               rpath.push_back(prev), prev != K;
               prev = path->at(K).at(prev));
          out << "    path:" << std::endl;
          for (auto pit = rpath.rbegin(); pit != rpath.rend(); ++pit) {
            out << "    - " << _demangle(*pit) << std::endl;
          }
        }
      }
    }
  }
};

} // namespace

char OpDependency::ID = 0;
static RegisterPass<OpDependency> X("op_dependency", "Op Dependency Pass");
