%{
#include <string>
#include "caffe2/caffe2/opt/nql/ast.h"
#include "caffe2/caffe2/opt/nql/parse.yy.h"

#define SAVE_TOKEN yylval.pstr = allocString(); *yylval.pstr = std::string(yytext, yyleng)
#define SAVE_STR yylval.pstr = allocString(); *yylval.pstr = std::string(yytext, yyleng)
#define TOKEN(t) (yylval.token = t)

extern "C" int yywrap() { return 1; }

%}

%%

#.*\n                   ;
[ \t\n]                 ;
"def"                   return TOKEN(TDEF);
[a-zA-Z_][a-zA-Z0-9_]*  SAVE_STR; return TIDENTIFIER;
%[a-z_][a-z0-9_]*       SAVE_STR; return TVAR;
[0-9]+.[0-9]*           SAVE_STR; return TDOUBLE;
[0-9]+                  SAVE_STR; return TINTEGER;
"*"                     SAVE_STR; return TSTAR;
"="                     return TOKEN(TEQUAL);
"=="                    return TOKEN(TCEQ);
"!="                    return TOKEN(TCNE);
"<"                     return TOKEN(TCLT);
"<="                    return TOKEN(TCLE);
">"                     return TOKEN(TCGT);
">="                    return TOKEN(TCGE);
"("                     return TOKEN(TLPAREN);
")"                     return TOKEN(TRPAREN);
"{"                     return TOKEN(TLBRACE);
"}"                     return TOKEN(TRBRACE);
"."                     return TOKEN(TDOT);
","                     return TOKEN(TCOMMA);
"+"                     return TOKEN(TPLUS);
"-"                     return TOKEN(TMINUS);
"/"                     return TOKEN(TDIV);
":"                     return TOKEN(TCOLON);
";"                     return TOKEN(TSEMICOLON);
.                       printf("Unknown token!\n"); yyterminate();

%%
