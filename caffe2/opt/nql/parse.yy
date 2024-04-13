%code provides {

#define YY_DECL \
  int yylex(ASTGraph* graph)
extern YY_DECL;

}

%{
#include <stdio.h>
#include <fstream>
#include <streambuf>
#include <string>
#include <unordered_map>
#include <vector>
#include "caffe2/opt/nql/ast.h"

extern int yyparse(ASTGraph* graph);
void yyerror(ASTGraph* graph, const char* s) {
  printf("ERROR: %s\n", s);
}

typedef struct yy_buffer_state* YY_BUFFER_STATE;
extern YY_BUFFER_STATE yy_scan_string(const char* str);
extern void yy_delete_buffer(YY_BUFFER_STATE buffer);

std::unordered_map<std::string, ASTExpr*> vmap;
std::vector<void*> tokens;
std::vector<void*> tokenVectors;

std::string* allocString() {
  std::string* n = new std::string();
  tokens.push_back(n);
  return n;
}

std::vector<void*>* allocVector() {
  std::vector<void*>* n = new std::vector<void*>();
  tokenVectors.push_back(n);
  return n;
}

ASTExpr* getStarExpr() {
  ASTExpr* new_ex = new ASTExpr;
  new_ex->name = "*";
  new_ex->isCallFlag = false;
  new_ex->starInputsFlag = true;
  return new_ex;
}

ASTExpr* getOrCreateVar(std::string * name) {
  if (*name == "*") {
    return getStarExpr();
  }
  if (vmap.count(*name)) {
    return vmap[*name];
  }
  ASTExpr* new_ex = new ASTExpr;
  new_ex->name = *name;
  new_ex->isCallFlag = false;
  return new_ex;
}

ASTExpr* processCall(std::string * name, std::vector<void*> * children) {
  ASTExpr* new_ex = new ASTExpr;
  new_ex->name = *name;
  new_ex->isCallFlag = true;
  if (children) {
    for (auto* e : *children) {
      new_ex->children.push_back((ASTExpr*)e);
    }
  }
  return new_ex;
}

ASTGraph* processGraph(
    ASTGraph * graph, std::string * name, std::vector<void*> * stmts) {
  graph->name = *name;
  if (stmts) {
    for (auto* e : *stmts) {
      graph->stmts.push_back((ASTStmt*)e);
    }
  }
  return graph;
}

ASTStmt* processStmt(std::vector<void*> * outputs, ASTExpr * r) {
  ASTStmt* new_st = new ASTStmt;
  if (outputs) {
    for (auto* e : *outputs) {
      new_st->lhs.push_back(*((std::string*)e));
    }
  }
  new_st->rhs = r;
  return new_st;
}

void parseString(const char* str, ASTGraph* graph) {
  YY_BUFFER_STATE buffer = yy_scan_string(str);
  yyparse(graph);
  yy_delete_buffer(buffer);
}

void parseFile(const char* fname, ASTGraph* graph) {
  std::ifstream t(fname);
  std::string str(
      (std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
  if (!str.size()) {
    return;
  }
  const char* c_str = str.c_str();
  parseString(c_str, graph);
}

%}

%lex-param { ASTGraph* graph }
%parse-param { ASTGraph* graph }

%union {
  ASTGraph* graph;
  std::string *pstr;
  int token;
  ASTExpr* expr;
  ASTStmt* stmt;
  std::vector<void*> *pvec;
}

%token <pstr> TIDENTIFIER TINTEGER TDOUBLE TVAR TSTAR
%token <token> TCEQ TCNE TCLT TCLE TCGT TCGE TEQUAL TPLUS TMINUS TDIV
%token <token> TLPAREN TRPAREN TLBRACE TRBRACE TCOMMA TDOT TCOLON TSEMICOLON TDEF

%type <pstr> ident var
%type <expr> expr
%type <stmt> stmt
%type <pvec> stmts vars exprs

/* Operator precedence for mathematical operators */
%left TPLUS TMINUS
%left TDIV

%start graph

%%
graph : TDEF ident TLBRACE stmts TRBRACE { processGraph(graph, $2, $4); }
      | error { }
      ;

stmts : stmt { $$ = allocVector(); $$->push_back((void*)$1); }
      | stmts stmt { $$->push_back($2); }
      ;

stmt : expr { $$ = processStmt(NULL, $1); }
     | vars TEQUAL expr { $$ = processStmt($1, $3); }
     ;

exprs : expr { $$ = allocVector(); $$->push_back((void*)$1); }
      | exprs TCOMMA expr { $$->push_back($3); }
      ;

expr : var { $$ = getOrCreateVar($1); }
     | ident TLPAREN exprs TRPAREN { $$ = processCall($1, $3); }
     ;
vars : var { $$ = allocVector(); $$->push_back((void*)$1); }
     | vars TCOMMA var { $$->push_back($3); }
     ;

var  : TVAR { $$ = $1; }
     | TSTAR { $$ = $1; }
     ;

ident : TIDENTIFIER { $$ = $1; }
      ;
%%
