#include <torch/csrc/jit/testing/module_differ.h>

#include <torch/csrc/jit/mobile/interpreter.h>

#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {

template <typename It>
bool ivalueListEquals(
    It lbegin,
    It lend,
    It rbegin,
    It rend,
    bool print,
    int print_indent) {
  int i = 0;
  const std::string indent(print_indent, '\t');
  for (; lbegin != lend && rbegin != rend; ++lbegin, ++rbegin, ++i) {
    if (!ivalueEquals(*lbegin, *rbegin, print, print_indent + 1)) {
      std::cout << indent << "list element differs at position " << i
                << std::endl;
      return false;
    }
  }
  return true;
}

bool ivalueEquals(
    const IValue& lhs,
    const IValue& rhs,
    bool print,
    int print_indent) {
  const std::string indent(print_indent, '\t');
  if (lhs.tagKind() != rhs.tagKind()) {
    if (print) {
      std::cout << indent << "lhs is type: " << lhs.tagKind()
                << "rhs is type: " << rhs.tagKind() << std::endl;
    }
    return false;
  }
  if (lhs.isCapsule()) {
    return true;
  }

  if (lhs.isDouble() || lhs.isComplexDouble() || lhs.isInt() || lhs.isBool() ||
      lhs.isString() || lhs.isDevice() || lhs.isCapsule() || lhs.isRRef() ||
      lhs.isEnum() || lhs.isIntList() || lhs.isDoubleList() ||
      lhs.isBoolList() || lhs.isNone()) {
    // operator == should do what we want
    if (lhs != rhs) {
      if (print) {
        std::cout << indent << "lhs is " << lhs << " ||  rhs is " << rhs
                  << std::endl;
      }
      return false;
    }
    return true;
  }

  if (lhs.isTensor()) {
    const auto& lt = lhs.toTensor();
    const auto& rt = rhs.toTensor();
    std::stringstream lsize;
    std::stringstream rsize;
    for (const auto x : lt.sizes()) {
      lsize << x << ",";
    }
    for (const auto x : rt.sizes()) {
      rsize << x << ",";
    }
    if (lsize.str() != lsize.str()) {
      if (print) {
        std::cout << indent << "left tensor is of shape " << lsize.str()
                  << "but right tensor is of shape " << rsize.str()
                  << std::endl;
      }
      return false;
    }
    if (lt.allclose(rt)) {
      return true;
    } else {
      if (print) {
        std::cout << indent << "rhs and lhs has are not close" << std::endl;
      }
      return false;
    }
  }

  if (lhs.isGenericDict()) {
    const auto& ldict = lhs.toGenericDict();
    const auto& rdict = rhs.toGenericDict();
    if (ldict.size() != rdict.size()) {
      if (print) {
        std::cout << indent << "lhs and rhs are dicts of different sizes: "
                  << ldict.size() << " vs. " << rdict.size() << std::endl;
      }
      return false;
    }

    for (const auto& kv : ldict) {
      auto rhs_iter = rdict.find(kv.key());
      if (rhs_iter == rdict.end()) {
        if (print) {
          std::cout << indent << "rhs missing key: " << kv.key() << std::endl;
        }
      }
      if (!ivalueEquals(
              kv.value(), rhs_iter->value(), print, print_indent + 1)) {
        if (print) {
          std::cout << indent << "for key: " << kv.key() << " value differs."
                    << std::endl;
        }
        return false;
      }
    }
    return true;
  } else if (lhs.isTensorList() || lhs.isList()) {
    const auto& vec = lhs.toList();
    const auto& rvec = rhs.toList();
    return ivalueListEquals(
        vec.begin(), vec.end(), rvec.begin(), rvec.end(), print, print_indent);
  } else if (lhs.isTuple()) {
    const auto vec = lhs.toTuple()->elements();
    const auto rvec = rhs.toTuple()->elements();
    return ivalueListEquals(
        vec.begin(), vec.end(), rvec.begin(), rvec.end(), print, print_indent);
  } else if (lhs.isObject()) {
    auto lobj = lhs.toObject();
    auto robj = rhs.toObject();
    auto ltype = lobj->type();
    auto rtype = robj->type();

    if (ltype->name() != rtype->name()) {
      if (print) {
        std::cerr << indent << "left object is of type: "
                  << ltype->name()->qualifiedName()
                  << " but right obj is of type: "
                  << rtype->name()->qualifiedName() << std::endl;
      }
      return false;
    }

    auto getstate = ltype->findMethod("__getstate__");
    if (getstate != nullptr) {
      return ivalueEquals(
          (*getstate)({lobj}), (*getstate)({robj}), print, print_indent + 1);
    }

    for (int i = 0; i < ltype->numAttributes(); i++) {
      if (!ivalueEquals(
              lobj->getSlot(i), robj->getSlot(i), print, print_indent + 1)) {
        std::cout << "attribute differs at position " << i << std::endl;
        return false;
      }
    }
    return true;
  }
  std::cerr << " I am here and should not be: " << rhs.tagKind() << std::endl;
  return false;
}

template <typename T, typename COMP, typename PRINTER>
bool vectorEqual(
    const std::vector<T>& lhs,
    const std::vector<T>& rhs,
    bool print,
    COMP comparator,
    PRINTER printer) {
  if (lhs.size() != rhs.size()) {
    if (print) {
      std::cout << "lhs and rhs has different size: " << lhs.size() << "vs. "
                << rhs.size() << std::endl;
    }
    return false;
  }

  for (int i = 0; i < lhs.size(); i++) {
    if (!comparator(lhs[i], rhs[i])) {
      if (print) {
        std::cout << i << "th element of lhs and rhs differs \n lhs is "
                  << printer(lhs[i]) << " rhs is " << printer(rhs[i])
                  << std::endl;
      }
      return false;
    }
  }

  return true;
}

bool moduleFunctionEquals(
    const mobile::Function& lhs,
    const mobile::Function& rhs,
    bool print) {
  const auto* lhs_code = lhs.get_code().get();
  const auto* rhs_code = rhs.get_code().get();

  // instructions
  if (print) {
    std::cout << "> Diffing instructions..." << std::endl;
  }
  auto ins_equal = [](Instruction lins, Instruction rins) -> bool {
    return (lins.op == rins.op && lins.N == rins.N && lins.X == rins.X);
  };
  auto id = [](auto ins) {
    return ins; // operator << works for it already
  };
  if (vectorEqual(
          lhs_code->instructions_,
          rhs_code->instructions_,
          true,
          ins_equal,
          id)) {
    std::cout << "     pass." << std::endl;
  } else {
    std::cout << "     fail." << std::endl;
    return false;
  }

  // constants

  if (print) {
    std::cout << "> Diffing constants..." << std::endl;
  }
  if (ivalueListEquals(
          lhs_code->constants_.begin(),
          lhs_code->constants_.end(),
          rhs_code->constants_.begin(),
          rhs_code->constants_.end(),
          true,
          2)) {
    std::cout << "        pass" << std::endl;
  } else {
    std::cout << "        fail" << std::endl;
    return false;
  }

  // diffing operators
  if (print) {
    std::cout << "> Diffing operators ..." << std::endl;
  }
  auto equals = [](auto op1, auto op2) -> bool { return op1 == op2; };
  if (vectorEqual(lhs_code->op_names_, rhs_code->op_names_, true, equals, id)) {
    std::cout << "     pass." << std::endl;
  } else {
    std::cout << "     fail." << std::endl;
    return false;
  }

  if (lhs_code->register_size_ != rhs_code->register_size_) {
    std::cout << "Register size differs: " << lhs_code->register_size_
              << " vs. " << rhs_code->register_size_ << std::endl;
    return false;
  }

  // debug handles
  if (print) {
    std::cout << "> Diffing debug handles..." << std::endl;
  }
  if (vectorEqual(
          lhs_code->debug_handles_,
          rhs_code->debug_handles_,
          true,
          equals,
          id)) {
    std::cout << "     pass." << std::endl;
  } else {
    std::cout << "     fail." << std::endl;
    return false;
  }

  // types
  auto type_eq = [](auto t1, auto t2) { return t1->str() == t2->str(); };
  auto type_print = [](auto t1) { return t1->str(); };

  if (print) {
    std::cout << "> Diffing types..." << std::endl;
  }
  if (vectorEqual(
          lhs_code->types_, rhs_code->types_, true, type_eq, type_print)) {
    std::cout << "     pass." << std::endl;
  } else {
    std::cout << "     fail." << std::endl;
    return false;
  }

  if (print) {
    std::cout << "> Diffing schema..." << std::endl;
  }
  // NOTE: Schema has Argument; which has TypePtr. operator== of
  // TypePtr is pointer identity. This behavior is not suitable for
  // our use case.
  if (toString(lhs.getSchema()) == toString(rhs.getSchema())) {
    std::cout << "     pass." << std::endl;
  } else {
    std::cout << "      lhs is " << lhs.getSchema() << std::endl;
    std::cout << "      rhs is " << rhs.getSchema() << std::endl;
    std::cout << "     fail." << std::endl;
    return false;
  }

  return true;
}

bool moduleEquals(const mobile::Module& lhs, const mobile::Module& rhs) {
  std::unordered_map<std::string, const mobile::Function*> lhs_name_to_func;
  std::unordered_map<std::string, const mobile::Function*> rhs_name_to_func;

  for (const auto& func : lhs.compilation_unit().methods()) {
    lhs_name_to_func[func->name()] = func.get();
  }
  for (const auto& func : rhs.compilation_unit().methods()) {
    rhs_name_to_func[func->name()] = func.get();
  }

  for (const auto& name_func : lhs_name_to_func) {
    auto rhs_func = rhs_name_to_func.find(name_func.first);
    if (rhs_func == rhs_name_to_func.end()) {
      std::cout << "Method with name: " << name_func.first
                << " only exists in lhs";
    }
    std::cout << "comparing method with name " << name_func.first << std::endl;
    if (moduleFunctionEquals(*name_func.second, *rhs_func->second, true)) {
      std::cout << "pass" << std::endl;
    } else {
      std::cout << "fail" << std::endl;
      return false;
    }
  }

  std::cout << "Diffing m._ivalue()..." << std::endl;
  if (ivalueEquals(lhs._ivalue(), rhs._ivalue(), true, 0)) {
    std::cout << "  pass." << std::endl;
  } else {
    std::cout << "  fail." << std::endl;
    return false;
  }
  return true;
}

} // namespace jit
} // namespace torch
