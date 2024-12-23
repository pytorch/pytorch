grammar Autolev;

options {
        language = Python3;
}

prog:   stat+;

stat:   varDecl
    |   functionCall
    |   codeCommands
    |   massDecl
    |   inertiaDecl
    |   assignment
    |   settings
    ;

assignment:   vec equals expr #vecAssign
          |   ID '[' index ']' equals expr #indexAssign
          |   ID diff? equals expr #regularAssign;

equals:   ('='|'+='|'-='|':='|'*='|'/='|'^=');

index:   expr (',' expr)* ;

diff:   ('\'')+;

functionCall:   ID '(' (expr (',' expr)*)? ')'
            |   (Mass|Inertia) '(' (ID (',' ID)*)? ')';

varDecl:   varType varDecl2 (',' varDecl2)*;

varType:   Newtonian|Frames|Bodies|Particles|Points|Constants
       |   Specifieds|Imaginary|Variables ('\'')*|MotionVariables ('\'')*;

varDecl2:   ID ('{' INT ',' INT '}')? (('{' INT ':' INT (',' INT ':' INT)* '}'))? ('{' INT '}')? ('+'|'-')? ('\'')* ('=' expr)?;

ranges:   ('{' INT ':' INT (',' INT ':' INT)* '}');

massDecl:   Mass massDecl2 (',' massDecl2)*;

massDecl2:   ID '=' expr;

inertiaDecl:   Inertia ID ('(' ID ')')? (',' expr)+;

matrix:   '[' expr ((','|';') expr)* ']';
matrixInOutput:   (ID (ID '=' (FLOAT|INT)?))|FLOAT|INT;

codeCommands:   units
            |   inputs
            |   outputs
            |   codegen
            |   commands;

settings:   ID (EXP|ID|FLOAT|INT)?;

units:     UnitSystem ID (',' ID)*;
inputs:    Input inputs2 (',' inputs2)*;
id_diff:   ID diff?;
inputs2:   id_diff '=' expr expr?;
outputs:   Output outputs2 (',' outputs2)*;
outputs2:  expr expr?;
codegen:   ID functionCall ('['matrixInOutput (',' matrixInOutput)*']')? ID'.'ID;

commands:  Save ID'.'ID
        |  Encode ID (',' ID)*;

vec:  ID ('>')+
   |  '0>'
   |  '1>>';

expr:   expr '^'<assoc=right> expr  # Exponent
    |   expr ('*'|'/') expr         # MulDiv
    |   expr ('+'|'-') expr         # AddSub
    |   EXP                         # exp
    |   '-' expr                    # negativeOne
    |   FLOAT                       # float
    |   INT                         # int
    |   ID('\'')*                   # id
    |   vec                         # VectorOrDyadic
    |   ID '['expr (',' expr)* ']'  # Indexing
    |   functionCall                # function
    |   matrix                      # matrices
    |   '(' expr ')'                # parens
    |   expr '=' expr               # idEqualsExpr
    |   expr ':' expr               # colon
    |   ID? ranges ('\'')*          # rangess
    ;

// These are to take care of the case insensitivity of Autolev.
Mass: ('M'|'m')('A'|'a')('S'|'s')('S'|'s');
Inertia: ('I'|'i')('N'|'n')('E'|'e')('R'|'r')('T'|'t')('I'|'i')('A'|'a');
Input: ('I'|'i')('N'|'n')('P'|'p')('U'|'u')('T'|'t')('S'|'s')?;
Output: ('O'|'o')('U'|'u')('T'|'t')('P'|'p')('U'|'u')('T'|'t');
Save: ('S'|'s')('A'|'a')('V'|'v')('E'|'e');
UnitSystem: ('U'|'u')('N'|'n')('I'|'i')('T'|'t')('S'|'s')('Y'|'y')('S'|'s')('T'|'t')('E'|'e')('M'|'m');
Encode: ('E'|'e')('N'|'n')('C'|'c')('O'|'o')('D'|'d')('E'|'e');
Newtonian: ('N'|'n')('E'|'e')('W'|'w')('T'|'t')('O'|'o')('N'|'n')('I'|'i')('A'|'a')('N'|'n');
Frames: ('F'|'f')('R'|'r')('A'|'a')('M'|'m')('E'|'e')('S'|'s')?;
Bodies: ('B'|'b')('O'|'o')('D'|'d')('I'|'i')('E'|'e')('S'|'s')?;
Particles: ('P'|'p')('A'|'a')('R'|'r')('T'|'t')('I'|'i')('C'|'c')('L'|'l')('E'|'e')('S'|'s')?;
Points: ('P'|'p')('O'|'o')('I'|'i')('N'|'n')('T'|'t')('S'|'s')?;
Constants: ('C'|'c')('O'|'o')('N'|'n')('S'|'s')('T'|'t')('A'|'a')('N'|'n')('T'|'t')('S'|'s')?;
Specifieds: ('S'|'s')('P'|'p')('E'|'e')('C'|'c')('I'|'i')('F'|'f')('I'|'i')('E'|'e')('D'|'d')('S'|'s')?;
Imaginary: ('I'|'i')('M'|'m')('A'|'a')('G'|'g')('I'|'i')('N'|'n')('A'|'a')('R'|'r')('Y'|'y');
Variables: ('V'|'v')('A'|'a')('R'|'r')('I'|'i')('A'|'a')('B'|'b')('L'|'l')('E'|'e')('S'|'s')?;
MotionVariables: ('M'|'m')('O'|'o')('T'|'t')('I'|'i')('O'|'o')('N'|'n')('V'|'v')('A'|'a')('R'|'r')('I'|'i')('A'|'a')('B'|'b')('L'|'l')('E'|'e')('S'|'s')?;

fragment DIFF:   ('\'')*;
fragment DIGIT:  [0-9];
INT:   [0-9]+ ;         // match integers
FLOAT:  DIGIT+ '.' DIGIT*
     |  '.' DIGIT+;
EXP:   FLOAT 'E' INT
|      FLOAT 'E' '-' INT;
LINE_COMMENT : '%' .*? '\r'? '\n' -> skip ;
ID:   [a-zA-Z][a-zA-Z0-9_]*;
WS:   [ \t\r\n&]+ -> skip ; // toss out whitespace
