#pragma once
#include <iostream>
#include <string>
#include <vector>

struct ASTExpr {
  std::string name = "";
  std::vector<ASTExpr*> children;
  bool isCallFlag = false;
  bool starInputsFlag = false;

  ~ASTExpr() {
    for (ASTExpr* e : children)
      delete e;
  }
  bool isCall() const {
    return isCallFlag;
  }
  bool starInputs() const {
    return starInputsFlag;
  }
  void dump(int level = 0) const {
    for (int i = 0; i < level; i++)
      std::cout << "  ";
    if (!isCall())
      std::cout << "Var: " << name << std::endl;
    else {
      std::cout << "Function: " << name << ", args: " << std::endl;
      for (auto* e : children) {
        e->dump(level + 1);
      }
    }
  }
};

struct ASTStmt {
  std::vector<std::string> lhs;
  ASTExpr* rhs = NULL;

  ~ASTStmt() {
    delete rhs;
  }
  void dump(int level = 0) const {
    for (int i = 0; i < level; i++)
      std::cout << "  ";
    std::cout << "LHS:" << std::endl;
    for (auto s : lhs) {
      for (int i = 0; i < level + 1; i++)
        std::cout << "  ";
      std::cout << s << std::endl;
    }
    rhs->dump(level);
  }
};

struct ASTGraph {
  std::string name;
  std::vector<ASTStmt*> stmts;
  ~ASTGraph() {
    for (auto s : stmts)
      delete s;
  }
  void dump() const {
    std::cout << "GRAPH: " << name << std::endl;
    for (auto s : stmts)
      s->dump(1);
  }
};

extern std::vector<void*> tokens;
extern std::vector<void*> tokenVectors;

std::string* allocString();
std::vector<void*>* allocVector();

void parseString(const char*, ASTGraph*);
void parseFile(const char*, ASTGraph*);
