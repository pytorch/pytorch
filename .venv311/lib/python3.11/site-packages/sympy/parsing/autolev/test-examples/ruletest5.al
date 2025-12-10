% ruletest5.al
VARIABLES X', Y'

E1 = (X+Y)^2 + (X-Y)^3
E2 = (X-Y)^2
E3 = X^2 + Y^2 + 2*X*Y

M1 = [E1;E2]
M2 = [(X+Y)^2,(X-Y)^2]
M3 = M1 + [X;Y]

AM = EXPAND(M1)
CM = EXPAND([(X+Y)^2,(X-Y)^2])
EM = EXPAND(M1 + [X;Y])
F = EXPAND(E1)
G = EXPAND(E2)

A = FACTOR(E3, X)
BM = FACTOR(M1, X)
CM = FACTOR(M1 + [X;Y], X)

A = D(E3, X)
B = D(E3, Y)
CM = D(M2, X)
DM = D(M1 + [X;Y], X)
FRAMES A, B
A_B = [1,0,0;1,0,0;1,0,0]
V1> = X*A1> + Y*A2> + X*Y*A3>
E> = D(V1>, X, B)
FM = DT(M1)
GM = DT([(X+Y)^2,(X-Y)^2])
H> = DT(V1>, B)
