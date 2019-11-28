// This LLVM pass takes LLVM bitcode / assembly as input and generates
// dependency graph among aten ops. From a set of root ops used by a model, we
// can calculate transitive closure of all dependent ops, then we can produce a
// custom LibTorch library with optimal build size which only registers and
// contains ops needed by the specific model - unregistered / unused ops can be
// stripped at link time.
//
// 1. Function Schema String -> Function (op registration)
//
// The analysis starts from searching for global string constants that look like
// function schema, e.g.: "aten::AA(Tensor self) -> Tensor" (from which we can
// obtain operator name "aten::AA"). Then search around the places using the
// function schema string constants. We use fuzzy match to find all
// llvm::Function instances which are likely to be registered as the
// implementation of the function schema. For example, for the following code
// snippet that registers function schemas to function pointers:
//
// auto registerer = torch::RegisterOperators()
//   .op(torch::RegisterOperators::options()
//     .schema("aten::AA(Tensor self) -> Tensor")
//     .kernel<decltype(AA_op), &AA_op>(TensorTypeId::CPUTensorId)
//     .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
//   .op(torch::RegisterOperators::options()
//     .schema("aten::BB(Tensor self) -> Tensor")
//     .catchAllKernel<decltype(BB_op), &BB_op>()
//     .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
//   ...
//
// It emits non-trivial LLVM assembly code to handle these registration
// patterns, but we figured that at high level it follows the similar sequence
// as in source code:
//
// Reference to function schema string "aten::AA(Tensor self) -> Tensor"
// Call "c10::RegisterOperators::Options::schema()"
// Reference to Function AA_op()
// ...
// Reference to function schema string "aten::BB(Tensor self) -> Tensor"
// Call "c10::RegisterOperators::Options::schema()"
// Reference to Function BB_op()
// ...
//
// So our analysis pass will try to collect all function references between two
// consecutive "c10::RegisterOperators::Options::schema()" calls and associate
// them with the function schema string.
//
// The function references can be function pointer or template argument, which
// are not direct function call/invoke instructions. But eventually registered
// function get passed as some operand directly / as operand in a constant
// expression. For example:
//
// %11 = tail call i8* @operator new(unsigned long)(i64 16) #37, !noalias !349
// ...
// %13 = getelementptr inbounds i8, i8* %11, i64 8
// %14 = bitcast i8* %13 to i64*
// store i64 ptrtoint (void (%"class.at::Tensor"*, %"class.at::Tensor"*)*
//   @at::EE_op(at::Tensor const&) to i64), i64* %14, align 8, !tbaa !333, ...
//
// Above "@at::EE_op" is an operand of "ptrtoint", which in turn is a constant
// operand of "store" instruction. So we make it to recursively traverse all
// constant operands of an instruction and collect all llvm::Function type
// objects it encounters.
//
// 2. Function -> Function
//
// LLVM has a standard “basiccg” analysis pass to dump call graph. However,
// llvm::CallGraph only checks CallSites (call/invoke instructions only). It
// doesn’t cover calling via lambda function / function pointer unless they are
// inlined. We modified the method a bit to count function pointer references as
// well for better recall (which is also consistent with the analysis we did
// above for op registration via function pointer) - but the function might not
// actually be called. For example:
//
// static std::function<Tensor()> helper(const Tensor& self) {
//   return [&]() {
//     return call_AA_op(self);
//   };
// }
//
// Tensor helper_call_AA_op(const Tensor& self) {
//   return helper(self)();
// }
//
// Tensor helper_not_call_AA_op(const Tensor& self) {
//   helper(self);
//   return self;
// }
//
// The first helper calls the AA op while the second helper only accesses its
// reference without actually calling. llvm::CallGraph analysis won’t count
// either of them. We count both of them as edges in dependency graph. Again, in
// our use case false positive is better than false negative.
//
// 3. Function -> Function Schema String (op invocation)
//
// One op can call another op and this is typically done via a static inline
// function, e.g.:
//
// static inline Tensor call_AA_op(const Tensor& self) {
//   static c10::OperatorHandle op = c10::Dispatcher::singleton()
//       .findSchema({"aten::AA", "out"}).value();
//   return c10::Dispatcher::singleton().callUnboxedOnly<Tensor, const Tensor&>(
//       op, self, self);
// }
//
// Corresponding LLVM assembly:
//
// @.str.3.347 = private unnamed_addr constant [9 x i8] c"aten::AA\00", align 1
// ...
// invoke void @std::basic_string<char, std::char_traits<char>,
//   std::allocator<char> >:: basic_string(char const*, std::allocator<char>
//   const&)(%"class.std::basic_string"* nonnull %14, i8* getelementptr inbounds
//   ([9 x i8], [9 x i8]* @.str.3.347, i64 0, i64 0), %"class.std::allocator.8"*
//   nonnull dereferenceable(1) %15)
// ...
// invoke void @c10::Dispatcher::findSchema(c10::OperatorName const&)(
//   %"class.c10::optional.92"* nonnull sret %12, %"class.c10::Dispatcher"*
//   nonnull %28, %"struct.c10::OperatorName"* nonnull dereferenceable(16) %13)
// ...
//
// We detect op invocation in a similar way as op registration - starting from
// global string constants that look like function schema / op name (e.g.:
// "aten::AA"), then searching around places using them. In the example above
// the function schema string is first used to initialize std::basic_string,
// which in turn is used to initialize c10::OperatorName struct, which
// eventually gets passed into findSchema. It’s NOT trivial to analyze the data
// flow accurately as it involves several c++ structures. So we simply search
// for “c10::Dispatcher::findSchema” function calls within the scope of parent
// function of the instruction that “uses” the global string “aten::AA”. It
// might generate false positive but we don’t care.
//
// Note that although we try to make these rules as general as possible it still
// depends on how LLVM emit IRs to some extent. For example, if it changes the
// sequence of schema string reference and c10::RegisterOperators API call to
// the extent that they no longer interlace with each other as in source code,
// then we might need use more sophisticated alias analysis, analyze clang-AST
// instead or simplify leverage the connection between function schema string
// and related function name (as in most cases they are generated by script).

#include <deque>
#include <iostream>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "llvm/Demangle/Demangle.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
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
             "'^c10::RegisterOperators::Options::schema'"),
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
    // There are two type of nodes in the dependency graph:
    // 1) String constants in source files, e.g.:
    //    "aten::cos_(Tensor(a!) self) -> Tensor(a!)", which represents operator
    //    function schemas;
    // 2) Function symbols in object files, e.g.:
    //    "at::CPUType::(anonymous namespace)::cos_(at::Tensor&)";
    // Both of them are added to the dependency graph as std::strings and are
    // called as "nodes".
    // "Key nodes" are the nodes we keep in the result graph. Ultimately we only
    // care about #1 as that's what we use to prune registered ops via codegen,
    // (then #2 will be stripped by linker automatically), so #1 is counted as
    // key nodes.
    SET keyNodes;

    // Find global strings matching function schema pattern. Then use fuzz
    // matcher to find functions related to these function schema strings.
    GRAPH schemaStrToFunctions, functionsToSchemaStrs;
    scanGlobalsForFunctionSchema(
        M, &keyNodes, &schemaStrToFunctions, &functionsToSchemaStrs);

    // Extended llvm::CallGraph analysis - all references are counted.
    GRAPH functionToFunctions;
    scanReferenceGraph(M, &functionToFunctions);

    // Simply dump all types of edges into one input graph.
    GRAPH input, result;
    mergeGraph(schemaStrToFunctions, &input);
    mergeGraph(functionsToSchemaStrs, &input);
    mergeGraph(functionToFunctions, &input);

    // Calculate transitive closure and remove non-key nodes.
    std::shared_ptr<PATH> path = DebugPath ? std::make_shared<PATH>() : nullptr;
    simplifyGraph(input, keyNodes, &result, path);

    if (OutputFormat == OutputFormatType::Dot) {
      printAsDot(std::cout, keyNodes, result);
    } else if (OutputFormat == OutputFormatType::YAML) {
      printAsYAML(std::cout, keyNodes, result, path);
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
      if (Verbose) {
        std::cerr << "[DEBUG][SCHEMA_STR] " << gvInitStr << " => "
                  << schemaStr << std::endl;
      }

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
          // Usually the use is to invoke "std::basic_string" constructor, but
          // this should be able to handle general uses as well.
          // TODO: maybe handle ConstantExpr as well?
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
  // For op invocation, simply check whether the parent function calls function
  // matching `OpInvocationPattern` (e.g.: 'c10::Dispatcher::findSchema')
  // since the use of the schema string.
  //
  // For op registration, collect all function references between two following
  // calls to function matching `OpRegistrationPattern` (e.g.:
  // "c10::RegisterOperators::Options::schema"). The function references
  // can be direct function call, function pointer or template argument.
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
      if (Verbose > 2) {
        std::cerr << "[DEBUG][INST] " << schemaStr << " => "
                  << *cur << std::endl;
      }
      // Check pattern to call registered ops.
      if (checkInstructionCalledFunction(
              *cur, *OpInvocationPatternLoc.pattern)) {
        (*functionsToSchemaStrs)[parentFunctionName].insert(schemaStr);
        if (Verbose) {
          std::cerr << "[DEBUG][OP_CALL] " << demangle(parentFunctionName)
                    << " => " << schemaStr << std::endl;
        }
        continue;
      }

      // Check pattern to register ops.
      if (checkInstructionCalledFunction(
              *cur, *OpRegistrationPatternLoc.pattern)) {
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
            std::cerr << "[DEBUG][OP_REG] " << schemaStr << " => "
                      << demangle(func->getName()) << std::endl;
          }
        };
        scanReferredFunctions(*cur, cb);
      }
    }
  }

  static bool isFunctionSchemaString(const std::string& str) {
    return FunctionSchemaPatternLoc.pattern->match(str);
  }

  static std::string truncateFunctionSchemaString(
      const std::string& schemaStr) {
    auto pos = schemaStr.find_first_of(".(");
    return pos == std::string::npos ? schemaStr : schemaStr.substr(0, pos);
  }

  static Instruction* getNextInstruction(Instruction& I) {
    Instruction* next = I.getNextNonDebugInstruction();
    if (next) return next;
    auto parentBlock = I.getParent();
    if (!parentBlock) return nullptr;
    auto nextBlock = parentBlock->getNextNode();
    return nextBlock ? &nextBlock->front() : nullptr;
  }

  static bool checkInstructionCalledFunction(Instruction& I, Regex& pattern) {
    bool called = false;
    scanReferredFunctions(I, [&](Function* func) -> void {
      called |= pattern.match(demangle(func->getName()));
    });
    return called;
  }

  // This method constructs function -> function reference graph. It inserts an
  // edge from function A to function B if A *might* call B.
  static void scanReferenceGraph(Module& M, GRAPH* functionToFunctions) {
    for (Function& F : M) {
      std::string name = F.getName();
      scanReferredFunctionsInFunction(F,
          [&](Function* func) -> void {
            (*functionToFunctions)[name].insert(func->getName());
          if (Verbose > 1) {
            std::cerr << "[DEBUG][FUNC_CALL] " << demangle(name) << " => "
                      << demangle(func->getName()) << std::endl;
          }
      });
    }
  }

  static void scanReferredFunctionsInFunction(
      Function& F, const std::function<void(Function*)>& CB) {
    for (BasicBlock& BB : F) {
      for (Instruction& I : BB) {
        scanReferredFunctions(I, CB);
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
  // logic.
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

    LazyCallGraph::visitReferences(worklist, visited, [&](Function &F) {
      CB(&F);
    });
  }

  static void mergeGraph(GRAPH& src, GRAPH* dest) {
    for (auto& S : src) {
      for (auto& E : S.second) {
        (*dest)[S.first].insert(E);
      }
    }
  }

  // Calculate transitive closure and remove non-key nodes.
  static void simplifyGraph(GRAPH& input, SET& keyNodes, GRAPH* output,
      std::shared_ptr<PATH> path) {
    // Starting from every key node, use BFS to traverse all nodes that are
    // transitively reachable from the node in the sparse graph.
    for (auto& key : keyNodes) {
      std::deque<std::string> queue;
      SET expanded;  // has some runtime issue with std::unordered_set
      auto expand = [&](const std::string& curNode) -> void {
        if (!expanded.insert(curNode).second) return;
        for (auto& next : input[curNode]) {
          queue.emplace_back(next);
          if (path) (*path)[key].emplace(next, curNode); // don't replace
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
               rpath.emplace_back(prev), prev != K;
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
