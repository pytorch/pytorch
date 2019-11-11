#include <deque>
#include <iostream>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

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

static cl::list<std::string> DebugFilters(
    "df",
    cl::desc("Debug filter patterns. Example: -df CPUType,at::native"),
    cl::ZeroOrMore,
    cl::CommaSeparated);

namespace {

typedef std::set<std::string> SET;
typedef std::unordered_map<std::string, std::set<std::string>> GRAPH;

// Referenced the logic in llvm-cxxfilt.cpp.
std::string demangle(const std::string& mangled) {
  int status;
  const char* decorated = mangled.c_str();
  size_t decoratedLength = strlen(decorated);

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
    // Key nodes are the nodes we keep in the result graph. They are global
    // strings that match function schema pattern. Function names that match
    // DebugFilters are also counted as key nodes.
    SET keyNodes;

    // Find global strings matching function schema pattern. Then use fuzz
    // matcher to find functions related to these function schema strings.
    GRAPH schemaStrToFunctions, functionsToSchemaStrs;
    scanGlobalsForFunctionSchema(
        M, &keyNodes, &schemaStrToFunctions, &functionsToSchemaStrs);

    // Find all function names that match DebugFilters.
    if (!DebugFilters.empty()) {
      scanFunctionsForDebugFilters(M, &keyNodes);
    }

    // Extended llvm::CallGraph analysis - all references are counted.
    GRAPH functionToFunctions;
    scanReferenceGraph(M, &functionToFunctions);

    // Simply dump all types of edges into one input graph.
    GRAPH input, result;
    mergeGraph(schemaStrToFunctions, &input);
    mergeGraph(functionsToSchemaStrs, &input);
    mergeGraph(functionToFunctions, &input);

    // Calculate transitive closure and remove non-key nodes.
    simplifyGraph(input, keyNodes, &result);

    if (OutputFormat == OutputFormatType::Dot) {
      printAsDot(std::cout, keyNodes, result);
    } else if (OutputFormat == OutputFormatType::YAML) {
      printAsYAML(std::cout, keyNodes, result);
    }

    return false;
  }

private:
  // Searching for global string constants that look like function schema, e.g.:
  // "aten::relu". Then search for places using the function schema string
  // constant to 1) register op; 2) invoke op.
  static void scanGlobalsForFunctionSchema(
      Module& M, SET* keyNodes,
      GRAPH* schemaStrToFunctions, GRAPH* functionsToSchemaStrs) {
    for (GlobalVariable& GV : M.globals()) {
      if (!GV.hasInitializer()) continue;
      Constant* gvInit = GV.getInitializer();
      if (gvInit->isNullValue()) continue;
      ConstantDataArray* gvInitArray = dyn_cast<ConstantDataArray>(gvInit);
      if (!gvInitArray || !gvInitArray->isCString()) continue;
      std::string gvInitStr = gvInitArray->getAsCString().str();
      if (!isFunctionSchemaString(gvInitStr)) continue;
      std::string schemaStr = truncateFunctionSchemaString(gvInitStr);

      // Track all seen schema strings. Useful for checking whether we miss any
      // dependency in the logic below.
      keyNodes->insert(schemaStr);

      // Search for GV users.
      for (auto gvUser : GV.users()) {
        // Search for "i8* getelementptr inbounds" ConstantExpr.
        ConstantExpr* CE = dyn_cast<ConstantExpr>(gvUser);
        if (!CE) continue;

        // Search for CE users.
        for (auto ceUser : CE->users()) {
          // Search for "std::basic_string" constructor.
          Instruction* ceUserInst = dyn_cast<Instruction>(ceUser);
          if (!ceUserInst) continue;
          analyzeFunctionSchemaUser(
              schemaStr, *ceUserInst, schemaStrToFunctions,
              functionsToSchemaStrs);
        }
      }
    }
  }

  // Use fuzzy match to find all llvm::Function instances related to the
  // function schema string. There are two types of relationship:
  // 1) function schema to function (op registration);
  // 2) function to function schema (op invocation);
  //
  // For op invocation, simply check whether the parent function calls
  // 'c10::Dispatcher::findSchema' since the use of the schema string.
  //
  // For op registration, collect all function references between two following
  // "c10::RegisterOperators::Options::schema" calls. The function references
  // can be direct function call, function pointer or template argument.
  //
  // Note that these rules highly depend on how LLVM emit IRs. If it changes
  // the sequence of schema string reference and c10::RegisterOperators API call
  // then we might need use more sophisticated alias analysis, analyze clang-AST
  // instead or simplify leverage the connection between function schema string
  // and related function name (as they are generated by script).
  static void analyzeFunctionSchemaUser(
      const std::string& schemaStr,
      Instruction& I,
      GRAPH* schemaStrToFunctions,
      GRAPH* functionsToSchemaStrs) {
    std::string parentFunctionName = I.getFunction()->getName();
    bool seenOptionsSchemaCall = false;
    for (Instruction* cur = getNextInstruction(I);
         cur;
         cur = getNextInstruction(*cur)) {
      // Check pattern to call registered ops.
      if (checkInstructionCalledFunction(*cur, "c10::Dispatcher::findSchema")) {
        (*functionsToSchemaStrs)[parentFunctionName].insert(schemaStr);
        continue;
      }

      // Check pattern to register ops.
      if (checkInstructionCalledFunction(
          *cur, "c10::RegisterOperators::Options::schema")) {
        if (seenOptionsSchemaCall) break;
        seenOptionsSchemaCall = true;
        continue;
      }

      // Collect function references between two op registration (between two
      // "Options::schema" calls).
      if (seenOptionsSchemaCall) {
        auto cb = [&](Function* func) -> void {
          (*schemaStrToFunctions)[schemaStr].insert(func->getName());
          if (Verbose) {
            std::cerr << "[DEBUG] " << schemaStr << " => "
                      << demangle(func->getName()) << std::endl;
          }
        };
        if (Verbose) {
          std::cerr << "[DEBUG] " << schemaStr << " [INST] "
                    << *cur << std::endl;
        }
        scanReferredFunctionsFromOperands(*cur, cb);
      }
    }
  }

  // llvm::CallGraph only checks CallSites. This method also checks constant
  // function pointer references for better recall - the function might not
  // actually be called.
  static void scanReferenceGraph(Module& M, GRAPH* functionToFunctions) {
    for (Function& F : M) {
      std::string name = F.getName();
      scanReferredFunctionsFromOperandsInFunction(F,
          [&](Function* func) -> void {
            (*functionToFunctions)[name].insert(func->getName());
      });
    }
  }

  static void scanFunctionsForDebugFilters(Module& M, SET* keyNodes) {
    for (Function& F : M) {
      std::string name = F.getName();
      if (matchDebugFilters(name)) {
        keyNodes->insert(name);
      }
    }
  }

  // Calculate transitive closure and remove non-key nodes.
  static void simplifyGraph(GRAPH& input, SET& keyNodes, GRAPH* output) {
    // Starting from every key node, use BFS to traverse all nodes that are
    // transitively reachable from the node in the sparse graph.
    for (auto& key : keyNodes) {
      std::deque<std::string> queue;
      SET expanded;  // has some runtime issue with std::unordered_set
      auto expand = [&](const std::string& curNode) -> void {
        if (!expanded.insert(curNode).second) return;
        for (auto& next : input[curNode]) {
          queue.emplace_back(next);
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
        // Expand node.
        expand(curNode);
      }
    }
  }

  static bool isFunctionSchemaString(const std::string& str) {
    return str.find("aten::") == 0 || str.find("quantized::") == 0;
  }

  static std::string truncateFunctionSchemaString(
      const std::string& schemaStr) {
    auto pos = schemaStr.find_first_of(".(");
    return pos == std::string::npos ? schemaStr : schemaStr.substr(0, pos);
  }

  static bool matchDebugFilters(const std::string& node) {
    if (DebugFilters.empty()) return false;
    auto str = demangle(node);
    for (const auto& debug : DebugFilters) {
      if (str.find(debug) != std::string::npos) {
        return true;
      }
    }
    return false;
  }

  static Instruction* getNextInstruction(Instruction& I) {
    Instruction* next = I.getNextNonDebugInstruction();
    if (next) return next;
    auto parentBlock = I.getParent();
    if (!parentBlock) return nullptr;
    auto nextBlock = parentBlock->getNextNode();
    return nextBlock ? &nextBlock->front() : nullptr;
  }

  // Referenced the logic in llvm::CallGraph.
  static Function* getCalledFunction(Instruction& I) {
    auto CS = CallSite(&I);
    if (!CS) return nullptr;
    Function* callee = CS.getCalledFunction();
    if (!callee || callee->isIntrinsic()) {
      return nullptr;
    }
    return callee;
  }

  // CallGraph only searches for CallSites (call/invoke instructions). However
  // functions can be referenced in other instructions as well (being passed
  // as function pointer).
  // This method recursively traverses all operands to search for function
  // pointers, e.g.:
  // ```
  // store i64 ptrtoint (void (%"class.at::Tensor"*, %"class.at::Tensor"*)*
  //                     @at::foo_op(at::Tensor const&) to i64), i64* %14, ...
  // ```
  // "@at::foo_op" is a operand of "ptrtoint", which in turn is a constant
  // operand of "store" instruction.
  static void scanReferredFunctionsFromOperands(
      User& U, const std::function<void(Function*)>& CB) {
    for (auto& O : U.operands()) {
      Function* func = dyn_cast<Function>(O);
      if (func && !func->isIntrinsic()) {
        CB(func);
      }
      // Recursively scans constant operand. Operands that are instructions
      // should already be scanned when it scans the entire function.
      Constant* C = dyn_cast<Constant>(O);
      if (C) {
        scanReferredFunctionsFromOperands(*C, CB);
      }
    }
  }

  static void scanReferredFunctionsFromOperandsInFunction(
      Function& F, const std::function<void(Function*)>& CB) {
    for (BasicBlock& BB : F) {
      for (Instruction& I : BB) {
        scanReferredFunctionsFromOperands(I, CB);
      }
    }
  }

  static bool checkInstructionCalledFunction(
      Instruction& I, const std::string& pattern) {
    Function* callee = getCalledFunction(I);
    return callee
        && demangle(callee->getName()).find(pattern) != std::string::npos;
  }

  static void mergeGraph(GRAPH& src, GRAPH* dest) {
    for (auto& S : src) {
      for (auto& E : S.second) {
        (*dest)[S.first].insert(E);
      }
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

  static void printAsYAML(std::ostream& out, SET& keys, GRAPH& graph) {
    for (const auto& K : keys) {
      out << "- name: " << demangle(K) << std::endl;
      auto& values = graph[K];
      if (values.empty()) continue;
      out << "  depends:" << std::endl;
      for (const auto& value : values) {
        out << "  - name: " << demangle(value) << std::endl;
      }
    }
  }
};

} // namespace

char OpDependency::ID = 0;
static RegisterPass<OpDependency> X("op_dependency", "Op Dependency Pass");
