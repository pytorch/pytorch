% ruletest7.al
VARIABLES X', Y'
E = COS(X) + SIN(X) + TAN(X)&
+ COSH(X) + SINH(X) + TANH(X)&
+ ACOS(X) + ASIN(X) + ATAN(X)&
+ LOG(X) + EXP(X) + SQRT(X)&
+ FACTORIAL(X) + CEIL(X) +&
FLOOR(X) + SIGN(X)

E = SQR(X) + LOG10(X)

A = ABS(-1) + INT(1.5) + ROUND(1.9)

E1 = 2*X + 3*Y
E2 = X + Y

AM = COEF([E1;E2], [X,Y])
B = COEF(E1, X)
C = COEF(E2, Y)
D1 = EXCLUDE(E1, X)
D2 = INCLUDE(E1, X)
FM = ARRANGE([E1,E2],2,X)
F = ARRANGE(E1, 2, Y)
G = REPLACE(E1, X=2*X)
GM = REPLACE([E1;E2], X=3)

FRAMES A, B
VARIABLES THETA
SIMPROT(A,B,3,THETA)
V1> = 2*A1> - 3*A2> + A3>
V2> = B1> + B2> + B3>
A = DOT(V1>, V2>)
BM = DOT(V1>, [V2>;2*V2>])
C> = CROSS(V1>,V2>)
D = MAG(2*V1>) + MAG(3*V1>)
DYADIC>> = 3*A1>*A1> + A2>*A2> + 2*A3>*A3>
AM = MATRIX(B, DYADIC>>)
M = [1;2;3]
V> = VECTOR(A, M)
