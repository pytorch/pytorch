% ruletest9.al
NEWTONIAN N
FRAMES A
A> = 0>
D>> = EXPRESS(1>>, A)

POINTS PO{2}
PARTICLES P{2}
MOTIONVARIABLES' C{3}'
BODIES R
P_P1_PO2> = C1*A1>
V> = 2*P_P1_PO2> + C2*A2>

W_A_N> = C3*A3>
V> = 2*W_A_N> + C2*A2>
W_R_N> = C3*A3>
V> = 2*W_R_N> + C2*A2>

ALF_A_N> = DT(W_A_N>, A)
V> = 2*ALF_A_N> + C2*A2>

V_P1_A> = C1*A1> + C3*A2>
A_RO_N> = C2*A2>
V_A> = CROSS(A_RO_N>, V_P1_A>)

X_B_C> = V_A>
X_B_D> = 2*X_B_C>
A_B_C_D_E> = X_B_D>*2

A_B_C = 2*C1*C2*C3
A_B_C += 2*C1
A_B_C := 3*C1

MOTIONVARIABLES' Q{2}', U{2}'
Q1' = U1
Q2' = U2

VARIABLES X'', Y''
SPECIFIED YY
Y'' = X*X'^2 + 1
YY = X*X'^2 + 1

M[1] = 2*X
M[2] = 2*Y
A = 2*M[1]

M = [1,2,3;4,5,6;7,8,9]
M[1, 2] = 5
A = M[1, 2]*2

FORCE_RO> = Q1*N1>
TORQUE_A> = Q2*N3>
FORCE_RO> = Q2*N2>
F> = FORCE_RO>*2
