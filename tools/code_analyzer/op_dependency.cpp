// This LLVM pass takes LLVM bitcode / assembly as input and generates
// dependency graph among aten ops. From a set of root ops used by a model, we
// can calculate transitive closure of all dependent ops, then we can produce a
// custom LibTorch library with optimal build size which only registers and
// contains ops needed by the specific model - unregistered / unused ops can be
// stripped at link time.

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
    if (val.empty()) return;
    pattern = std::make_shared<Regex>(val);
    std::string regexError;
    if (!pattern->isValid(regexError)) {
      report_fatal_error(
          "Invalid regular expression param: '" + val + "' err: " + regexError,
          false);
    }
  };
};

static RegexOpt FunctionSchemaPatternLoc;
static cl::opt<RegexOpt, true, cl::parser<std::string>> FunctionSchemaPattern(
    "op_schema_pattern",
    cl::desc("Op schema regex pattern. "
             "Example: -op_schema_pattern "
             "'(^aten::[^ ]+)|(^quantized::[^ ]+)'"),
    cl::location(FunctionSchemaPatternLoc),
    cl::Required,
    cl::ValueRequired);

static RegexOpt OpRegistrationPatternLoc;
static cl::opt<RegexOpt, true, cl::parser<std::string>> OpRegistrationPattern(
    "op_register_pattern",
    cl::desc("Op registration signature regex pattern. "
             "Example: -op_register_pattern "
             "'^c10::RegisterOperators::op'"),
    cl::location(OpRegistrationPatternLoc),
    cl::Required,
    cl::ValueRequired);

static RegexOpt OpInvocationPatternLoc;
static cl::opt<RegexOpt, true, cl::parser<std::string>> OpInvocationPattern(
    "op_invoke_pattern",
    cl::desc("Op invocation signature regex pattern. "
             "Example: -op_invoke_pattern "
             "'^c10::Dispatcher::findSchema'"),
    cl::location(OpInvocationPatternLoc),
    cl::Required,
    cl::ValueRequired);

enum OutputFormatType { Dot, YAML };
static cl::opt<OutputFormatType> OutputFormat(
    "format",
    cl::desc("Output format."),
    cl::values(clEnumValN(Dot, "dot", "print as dot"),
               clEnumValN(YAML, "yaml", "print as yaml")));

static cl::opt<bool> TransitiveClosure(
    "closure",
    cl::desc("Output transitive closure."),
    cl::init(true));

static cl::opt<int> Verbose(
    "v",
    cl::desc("Verbose level"),
    cl::Hidden,
    cl::init(0));

static cl::opt<bool> DebugPath(
    "debug_path",
    cl::desc("Output path between two nodes."),
    cl::init(false));

typedef std::set<std::string> SET;
typedef std::unordered_map<std::string, std::set<std::string>> GRAPH;
typedef std::unordered_map<Value*, Value*> LINK;
typedef std::unordered_set<Value*> VALUE_SET;

// SRC -> (DEST -> PREV)
typedef std::unordered_map<std::string,
                           std::unordered_map<std::string, std::string>> PATH;

// Referenced the logic in llvm-cxxfilt.cpp.
std::string demangle(const std::string& mangled) {
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
    // dependency graph and to find out instructions that might register or
    // invoke operators, respectively.
    GRAPH deps;
    VALUE_SET opRegistrationInsts, opInvocationInsts;
    for (Function& F : M) {
      std::string caller = F.getName();
      std::string callerDemangled = demangle(caller);
      for (BasicBlock& BB : F) {
        for (Instruction& I : BB) {
          scanReferredFunctions(I, [&](Function* func) -> void {
            std::string callee = func->getName();
            std::string calleeDemangled = demangle(callee);
            deps[caller].insert(callee);
            if (Verbose > 1) {
              std::cerr << "[DEBUG][FUNC_CALL] " << callerDemangled << " => "
                        << calleeDemangled << std::endl;
            }
            // One registration/invocation API might call another registration/
            // invocation API in which case we can skip processing the nested
            // call. This is a simple trick to avoid "cannot find registered/
            // invoked op" warning and doesn't affect correctness.
            if (!OpRegistrationPatternLoc.pattern->match(callerDemangled) &&
                OpRegistrationPatternLoc.pattern->match(calleeDemangled)) {
              opRegistrationInsts.insert(&I);
            }
            if (!OpInvocationPatternLoc.pattern->match(callerDemangled) &&
                OpInvocationPatternLoc.pattern->match(calleeDemangled)) {
              opInvocationInsts.insert(&I);
            }
          });
        }
      }
    }

    // Scan op registration/invocation API calls to construct the link between
    // op name (a.k.a op schema string) and related functions.
    SET opSchemaStrs;
    scanOpRegistration(opRegistrationInsts, &opSchemaStrs, &deps);
    scanOpInvocation(opInvocationInsts, &opSchemaStrs, &deps);

    // Shrink the graph by removing intermediate nodes (functions) while
    // maintaining transitive dependency between operators (schema strings).
    GRAPH result;
    std::shared_ptr<PATH> path = DebugPath ? std::make_shared<PATH>() : nullptr;
    simplifyGraph(deps, opSchemaStrs, &result, path.get());

    if (OutputFormat == OutputFormatType::Dot) {
      printAsDot(std::cout, opSchemaStrs, result);
    } else if (OutputFormat == OutputFormatType::YAML) {
      printAsYAML(std::cout, opSchemaStrs, result, path);
    }

    return false;
  }

private:
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
      if (!F.isIntrinsic()) CB(&F);
    });
  }

  // Naive connectivity analysis to find out all nodes that are reachable from a
  // specific node in IR graph by following each node's "use" edges (to its
  // operands and users).
  static void scanConnectedNodes(
      Value* src,
      const VALUE_SET& blocked,
      const std::function<void(Value*)>& CB, LINK* debugLink) {
    std::deque<Value*> queue;
    SmallPtrSet<Value*, 16> visited;

    auto insert = [&](Value* cur, Value* parent) -> void {
      if (!blocked.count(cur) && visited.insert(cur).second) {
        queue.push_back(cur);
        if (debugLink) (*debugLink).emplace(cur, parent);
      }
    };

    auto expandOperands = [&](Value* V) -> void {
      if (!isa<User>(V) || isa<Function>(V)) return;
      auto node = dyn_cast<User>(V);
      for (auto& O : node->operands()) {
        insert(O, node);
      }
    };

    auto expandUsers = [&](Value* V) -> void {
      if (!isa<User>(V) || isa<Constant>(V) || isa<Function>(V)) return;
      for (auto U : V->users()) {
        insert(U, V);
      }
    };

    auto expand = [&](Value* V) -> void {
      expandOperands(V);
      expandUsers(V);
    };

    expand(src);
    while (!queue.empty()) {
      auto cur = queue.front();
      queue.pop_front();
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
      GRAPH& input, SET& keyNodes, GRAPH* output, PATH* path) {
    // Starting from every key node, use BFS to traverse all nodes that are
    // transitively reachable from the node in the sparse graph.
    for (auto& key : keyNodes) {
      std::deque<std::string> queue;
      SET visited;  // has some runtime issue with std::unordered_set
      auto expand = [&](const std::string& curNode) -> void {
        for (auto& next : input[curNode]) {
          if (!visited.insert(next).second) continue;
          queue.push_back(next);
          if (path) (*path)[key].emplace(next, curNode);
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
          if (!TransitiveClosure) continue;
        }
        expand(curNode);
      }
    }
  }

  // Find out operator names and function pointers that are transitively
  // connected to the same 'src' value.
  static void scanOpSchemaStrAndFunction(
      Value* src, const VALUE_SET& blocked,
      SET* visitedOps, SET* visitedFunctions) {
    std::shared_ptr<LINK> debugLink =
        (Verbose > 2 ? std::make_shared<LINK>() : nullptr);
    auto callback = [&](Value* V) -> void {
      if (auto schemaStr = extractOpSchema(V)) {
        if (visitedOps) (*visitedOps).insert(*schemaStr);
        if (Verbose > 1) {
          std::cerr << "[DEBUG][OP_SCHEMA] " << *schemaStr << std::endl;
          printDebugPath(debugLink.get(), src, V);
        }
      } else if (auto F = dyn_cast<Function>(V)) {
        if (F->isIntrinsic()) return;
        if (visitedFunctions) (*visitedFunctions).insert(F->getName());
        if (Verbose > 1) {
          std::cerr << "[DEBUG][FUNC] " << demangle(F->getName()) << std::endl;
          printDebugPath(debugLink.get(), src, V);
        }
      }
    };
    scanConnectedNodes(src, blocked, callback, debugLink.get());
  }

  static void scanOpRegistration(
      VALUE_SET& instructions, SET* opSchemaStrs, GRAPH* schemaStrToFunctions) {
    for (auto V : instructions) {
      auto I = dyn_cast<Instruction>(V);
      if (!I || !CallSite(I)) continue;
      if (Verbose > 2) {
        std::cerr << "[DEBUG][REG][INST] " << *I << std::endl;
      }
      SET visitedOps, visitedFunctions;
      scanOpSchemaStrAndFunction(
          I, instructions, &visitedOps, &visitedFunctions);
      if (visitedOps.size() != 1) {
        std::cerr << "[WARNING] found " << visitedOps.size() << " ops ( ";
        for (auto& op : visitedOps) {
          std::cerr << op << " ";
        }
        std::cerr << ") in a registration call in function: "
                  << demangle(I->getFunction()->getName()) << std::endl;
      }
      for (const auto& op : visitedOps) {
        opSchemaStrs->insert(op);
        if (visitedFunctions.empty()) {
          std::cerr << "[WARNING] could not find registered function for op: "
                    << op << std::endl;
        }
        for (const auto& func : visitedFunctions) {
          (*schemaStrToFunctions)[op].insert(func);
          if (Verbose) {
            std::cerr << "[DEBUG][OP_REG] " << op << " => "
                      << demangle(func) << std::endl;
          }
        }
      }
    }
  }

  static void scanOpInvocation(
      VALUE_SET& instructions, SET* opSchemaStrs, GRAPH* functionToSchemaStrs) {
    for (auto V : instructions) {
      auto I = dyn_cast<Instruction>(V);
      if (!I || !CallSite(I)) continue;
      if (Verbose > 2) {
        std::cerr << "[DEBUG][CALL][INST] " << *I << std::endl;
      }
      std::string caller = I->getFunction()->getName();
      SET visitedOps;
      scanOpSchemaStrAndFunction(I, {}, &visitedOps, nullptr);
      if (visitedOps.empty()) {
        std::cerr << "[WARNING] could not find called op in function: "
                  << demangle(caller) << std::endl;
      }
      for (const auto& op : visitedOps) {
        opSchemaStrs->insert(op);
        (*functionToSchemaStrs)[caller].insert(op);
        if (Verbose) {
          std::cerr << "[DEBUG][OP_CALL] " << demangle(caller) << " => "
                    << op << std::endl;
        }
      }
    }
  }

  static std::string extractStringValue(Value* V) {
    if (auto array = dyn_cast<ConstantDataArray>(V)) {
      if (array->isCString()) {
        return array->getAsCString().str();
      } else if (array->isString()) {
        std::cerr << "[WARNING] ignore non-C string: "
                  << array->getAsString().str() << std::endl;
      }
    } else if (ConstantInt* CI = dyn_cast<ConstantInt>(V)) {
      // Short string literal might be encoded into constant integer, e.g.:
      // "aten::AA" => 4702103508586165345 (0x41413A3A6E657461)
      // This can be tricky as it depends on consistent endianness/size.
      int64_t intValue = CI->getZExtValue();
      auto data = reinterpret_cast<const char*>(&intValue);
      return {data, data + sizeof(int64_t)/sizeof(char)};
    }
    return {};
  }

  static std::shared_ptr<std::string> extractOpSchema(Value* V) {
    const std::string& schemaStr = extractStringValue(V);
    if (!FunctionSchemaPatternLoc.pattern->match(schemaStr)) return {};
    auto pos = schemaStr.find_first_of(".(");
    return std::make_shared<std::string>(
        pos == std::string::npos ? schemaStr : schemaStr.substr(0, pos));
  }

  static void printDebugPath(LINK* debugLink, Value* src, Value* dest) {
    if (!debugLink) return;
    int depth = 0;
    for (auto N = dest; ; N = (*debugLink)[N]) {
      std::cerr << "[DEBUG][PATH][" << ++depth << "]";
      printDebugValue(N);
      if (N == src) break;
    }
  }

  static void printDebugValue(Value* V) {
    if (auto F = dyn_cast<Function>(V)) {
      std::cerr << "[FUNC] " << demangle(F->getName()) << std::endl;
    } else if (isa<Constant>(V)) {
      std::cerr << "[CONST] " << *V << std::endl;
    } else if (isa<Instruction>(V)) {
      std::cerr << "[INST] " << *V << std::endl;
    } else {
      std::cerr << "[VALUE] " << *V << std::endl;
    }
  }

  static void printAsDot(std::ostream& out, SET& keys, GRAPH& graph) {
    out << "digraph {" << std::endl;
    out << "layout=\"circo\";" << std::endl;
    for (const auto& K : keys) {
      auto key = demangle(K);
      for (const auto& value : graph[K]) {
        out << '"' << key << '"'
            << " -> "
            << '"' << demangle(value) << "\";"
            << std::endl;
      }
    }
    out << "}" << std::endl;
  }

  static void printAsYAML(std::ostream& out, SET& keys, GRAPH& graph,
      std::shared_ptr<PATH> path) {
    for (const auto& K : keys) {
      out << "- name: " << demangle(K) << std::endl;
      auto& values = graph[K];
      if (values.empty()) continue;
      out << "  depends:" << std::endl;
      for (const auto& value : values) {
        out << "  - name: " << demangle(value) << std::endl;
        if (path) {
          std::vector<std::string> rpath;
          for (std::string prev = value;
               rpath.push_back(prev), prev != K;
               prev = (*path)[K][prev]);
          out << "    path:" << std::endl;
          for (auto it = rpath.rbegin(); it != rpath.rend(); ++it) {
            out << "    - " << demangle(*it) << std::endl;
          }
        }
      }
    }
  }
};

} // namespace

char OpDependency::ID = 0;
static RegisterPass<OpDependency> X("op_dependency", "Op Dependency Pass");
