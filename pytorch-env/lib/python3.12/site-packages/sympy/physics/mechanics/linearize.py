__all__ = ['Linearizer']

from sympy import Matrix, eye, zeros
from sympy.core.symbol import Dummy
from sympy.utilities.iterables import flatten
from sympy.physics.vector import dynamicsymbols
from sympy.physics.mechanics.functions import msubs, _parse_linear_solver

from collections import namedtuple
from collections.abc import Iterable


class Linearizer:
    """This object holds the general model form for a dynamic system. This
    model is used for computing the linearized form of the system, while
    properly dealing with constraints leading to  dependent coordinates and
    speeds. The notation and method is described in [1]_.

    Attributes
    ==========

    f_0, f_1, f_2, f_3, f_4, f_c, f_v, f_a : Matrix
        Matrices holding the general system form.
    q, u, r : Matrix
        Matrices holding the generalized coordinates, speeds, and
        input vectors.
    q_i, u_i : Matrix
        Matrices of the independent generalized coordinates and speeds.
    q_d, u_d : Matrix
        Matrices of the dependent generalized coordinates and speeds.
    perm_mat : Matrix
        Permutation matrix such that [q_ind, u_ind]^T = perm_mat*[q, u]^T

    References
    ==========

    .. [1] D. L. Peterson, G. Gede, and M. Hubbard, "Symbolic linearization of
           equations of motion of constrained multibody systems," Multibody
           Syst Dyn, vol. 33, no. 2, pp. 143-161, Feb. 2015, doi:
           10.1007/s11044-014-9436-5.

    """

    def __init__(self, f_0, f_1, f_2, f_3, f_4, f_c, f_v, f_a, q, u, q_i=None,
                 q_d=None, u_i=None, u_d=None, r=None, lams=None,
                 linear_solver='LU'):
        """
        Parameters
        ==========

        f_0, f_1, f_2, f_3, f_4, f_c, f_v, f_a : array_like
            System of equations holding the general system form.
            Supply empty array or Matrix if the parameter
            does not exist.
        q : array_like
            The generalized coordinates.
        u : array_like
            The generalized speeds
        q_i, u_i : array_like, optional
            The independent generalized coordinates and speeds.
        q_d, u_d : array_like, optional
            The dependent generalized coordinates and speeds.
        r : array_like, optional
            The input variables.
        lams : array_like, optional
            The lagrange multipliers
        linear_solver : str, callable
            Method used to solve the several symbolic linear systems of the
            form ``A*x=b`` in the linearization process. If a string is
            supplied, it should be a valid method that can be used with the
            :meth:`sympy.matrices.matrixbase.MatrixBase.solve`. If a callable is
            supplied, it should have the format ``x = f(A, b)``, where it
            solves the equations and returns the solution. The default is
            ``'LU'`` which corresponds to SymPy's ``A.LUsolve(b)``.
            ``LUsolve()`` is fast to compute but will often result in
            divide-by-zero and thus ``nan`` results.

        """
        self.linear_solver = _parse_linear_solver(linear_solver)

        # Generalized equation form
        self.f_0 = Matrix(f_0)
        self.f_1 = Matrix(f_1)
        self.f_2 = Matrix(f_2)
        self.f_3 = Matrix(f_3)
        self.f_4 = Matrix(f_4)
        self.f_c = Matrix(f_c)
        self.f_v = Matrix(f_v)
        self.f_a = Matrix(f_a)

        # Generalized equation variables
        self.q = Matrix(q)
        self.u = Matrix(u)
        none_handler = lambda x: Matrix(x) if x else Matrix()
        self.q_i = none_handler(q_i)
        self.q_d = none_handler(q_d)
        self.u_i = none_handler(u_i)
        self.u_d = none_handler(u_d)
        self.r = none_handler(r)
        self.lams = none_handler(lams)

        # Derivatives of generalized equation variables
        self._qd = self.q.diff(dynamicsymbols._t)
        self._ud = self.u.diff(dynamicsymbols._t)
        # If the user doesn't actually use generalized variables, and the
        # qd and u vectors have any intersecting variables, this can cause
        # problems. We'll fix this with some hackery, and Dummy variables
        dup_vars = set(self._qd).intersection(self.u)
        self._qd_dup = Matrix([var if var not in dup_vars else Dummy() for var
                               in self._qd])

        # Derive dimesion terms
        l = len(self.f_c)
        m = len(self.f_v)
        n = len(self.q)
        o = len(self.u)
        s = len(self.r)
        k = len(self.lams)
        dims = namedtuple('dims', ['l', 'm', 'n', 'o', 's', 'k'])
        self._dims = dims(l, m, n, o, s, k)

        self._Pq = None
        self._Pqi = None
        self._Pqd = None
        self._Pu = None
        self._Pui = None
        self._Pud = None
        self._C_0 = None
        self._C_1 = None
        self._C_2 = None
        self.perm_mat = None

        self._setup_done = False

    def _setup(self):
        # Calculations here only need to be run once. They are moved out of
        # the __init__ method to increase the speed of Linearizer creation.
        self._form_permutation_matrices()
        self._form_block_matrices()
        self._form_coefficient_matrices()
        self._setup_done = True

    def _form_permutation_matrices(self):
        """Form the permutation matrices Pq and Pu."""

        # Extract dimension variables
        l, m, n, o, s, k = self._dims
        # Compute permutation matrices
        if n != 0:
            self._Pq = permutation_matrix(self.q, Matrix([self.q_i, self.q_d]))
            if l > 0:
                self._Pqi = self._Pq[:, :-l]
                self._Pqd = self._Pq[:, -l:]
            else:
                self._Pqi = self._Pq
                self._Pqd = Matrix()
        if o != 0:
            self._Pu = permutation_matrix(self.u, Matrix([self.u_i, self.u_d]))
            if m > 0:
                self._Pui = self._Pu[:, :-m]
                self._Pud = self._Pu[:, -m:]
            else:
                self._Pui = self._Pu
                self._Pud = Matrix()
        # Compute combination permutation matrix for computing A and B
        P_col1 = Matrix([self._Pqi, zeros(o + k, n - l)])
        P_col2 = Matrix([zeros(n, o - m), self._Pui, zeros(k, o - m)])
        if P_col1:
            if P_col2:
                self.perm_mat = P_col1.row_join(P_col2)
            else:
                self.perm_mat = P_col1
        else:
            self.perm_mat = P_col2

    def _form_coefficient_matrices(self):
        """Form the coefficient matrices C_0, C_1, and C_2."""

        # Extract dimension variables
        l, m, n, o, s, k = self._dims
        # Build up the coefficient matrices C_0, C_1, and C_2
        # If there are configuration constraints (l > 0), form C_0 as normal.
        # If not, C_0 is I_(nxn). Note that this works even if n=0
        if l > 0:
            f_c_jac_q = self.f_c.jacobian(self.q)
            self._C_0 = (eye(n) - self._Pqd *
                         self.linear_solver(f_c_jac_q*self._Pqd,
                                            f_c_jac_q))*self._Pqi
        else:
            self._C_0 = eye(n)
        # If there are motion constraints (m > 0), form C_1 and C_2 as normal.
        # If not, C_1 is 0, and C_2 is I_(oxo). Note that this works even if
        # o = 0.
        if m > 0:
            f_v_jac_u = self.f_v.jacobian(self.u)
            temp = f_v_jac_u * self._Pud
            if n != 0:
                f_v_jac_q = self.f_v.jacobian(self.q)
                self._C_1 = -self._Pud * self.linear_solver(temp, f_v_jac_q)
            else:
                self._C_1 = zeros(o, n)
            self._C_2 = (eye(o) - self._Pud *
                         self.linear_solver(temp, f_v_jac_u))*self._Pui
        else:
            self._C_1 = zeros(o, n)
            self._C_2 = eye(o)

    def _form_block_matrices(self):
        """Form the block matrices for composing M, A, and B."""

        # Extract dimension variables
        l, m, n, o, s, k = self._dims
        # Block Matrix Definitions. These are only defined if under certain
        # conditions. If undefined, an empty matrix is used instead
        if n != 0:
            self._M_qq = self.f_0.jacobian(self._qd)
            self._A_qq = -(self.f_0 + self.f_1).jacobian(self.q)
        else:
            self._M_qq = Matrix()
            self._A_qq = Matrix()
        if n != 0 and m != 0:
            self._M_uqc = self.f_a.jacobian(self._qd_dup)
            self._A_uqc = -self.f_a.jacobian(self.q)
        else:
            self._M_uqc = Matrix()
            self._A_uqc = Matrix()
        if n != 0 and o - m + k != 0:
            self._M_uqd = self.f_3.jacobian(self._qd_dup)
            self._A_uqd = -(self.f_2 + self.f_3 + self.f_4).jacobian(self.q)
        else:
            self._M_uqd = Matrix()
            self._A_uqd = Matrix()
        if o != 0 and m != 0:
            self._M_uuc = self.f_a.jacobian(self._ud)
            self._A_uuc = -self.f_a.jacobian(self.u)
        else:
            self._M_uuc = Matrix()
            self._A_uuc = Matrix()
        if o != 0 and o - m + k != 0:
            self._M_uud = self.f_2.jacobian(self._ud)
            self._A_uud = -(self.f_2 + self.f_3).jacobian(self.u)
        else:
            self._M_uud = Matrix()
            self._A_uud = Matrix()
        if o != 0 and n != 0:
            self._A_qu = -self.f_1.jacobian(self.u)
        else:
            self._A_qu = Matrix()
        if k != 0 and o - m + k != 0:
            self._M_uld = self.f_4.jacobian(self.lams)
        else:
            self._M_uld = Matrix()
        if s != 0 and o - m + k != 0:
            self._B_u = -self.f_3.jacobian(self.r)
        else:
            self._B_u = Matrix()

    def linearize(self, op_point=None, A_and_B=False, simplify=False):
        """Linearize the system about the operating point. Note that
        q_op, u_op, qd_op, ud_op must satisfy the equations of motion.
        These may be either symbolic or numeric.

        Parameters
        ==========
        op_point : dict or iterable of dicts, optional
            Dictionary or iterable of dictionaries containing the operating
            point conditions for all or a subset of the generalized
            coordinates, generalized speeds, and time derivatives of the
            generalized speeds. These will be substituted into the linearized
            system before the linearization is complete. Leave set to ``None``
            if you want the operating point to be an arbitrary set of symbols.
            Note that any reduction in symbols (whether substituted for numbers
            or expressions with a common parameter) will result in faster
            runtime.
        A_and_B : bool, optional
            If A_and_B=False (default), (M, A, B) is returned and of
            A_and_B=True, (A, B) is returned. See below.
        simplify : bool, optional
            Determines if returned values are simplified before return.
            For large expressions this may be time consuming. Default is False.

        Returns
        =======
        M, A, B : Matrices, ``A_and_B=False``
            Matrices from the implicit form:
                ``[M]*[q', u']^T = [A]*[q_ind, u_ind]^T + [B]*r``
        A, B : Matrices, ``A_and_B=True``
            Matrices from the explicit form:
                ``[q_ind', u_ind']^T = [A]*[q_ind, u_ind]^T + [B]*r``

        Notes
        =====

        Note that the process of solving with A_and_B=True is computationally
        intensive if there are many symbolic parameters. For this reason, it
        may be more desirable to use the default A_and_B=False, returning M, A,
        and B. More values may then be substituted in to these matrices later
        on. The state space form can then be found as A = P.T*M.LUsolve(A), B =
        P.T*M.LUsolve(B), where P = Linearizer.perm_mat.

        """

        # Run the setup if needed:
        if not self._setup_done:
            self._setup()

        # Compose dict of operating conditions
        if isinstance(op_point, dict):
            op_point_dict = op_point
        elif isinstance(op_point, Iterable):
            op_point_dict = {}
            for op in op_point:
                op_point_dict.update(op)
        else:
            op_point_dict = {}

        # Extract dimension variables
        l, m, n, o, s, k = self._dims

        # Rename terms to shorten expressions
        M_qq = self._M_qq
        M_uqc = self._M_uqc
        M_uqd = self._M_uqd
        M_uuc = self._M_uuc
        M_uud = self._M_uud
        M_uld = self._M_uld
        A_qq = self._A_qq
        A_uqc = self._A_uqc
        A_uqd = self._A_uqd
        A_qu = self._A_qu
        A_uuc = self._A_uuc
        A_uud = self._A_uud
        B_u = self._B_u
        C_0 = self._C_0
        C_1 = self._C_1
        C_2 = self._C_2

        # Build up Mass Matrix
        #     |M_qq    0_nxo   0_nxk|
        # M = |M_uqc   M_uuc   0_mxk|
        #     |M_uqd   M_uud   M_uld|
        if o != 0:
            col2 = Matrix([zeros(n, o), M_uuc, M_uud])
        if k != 0:
            col3 = Matrix([zeros(n + m, k), M_uld])
        if n != 0:
            col1 = Matrix([M_qq, M_uqc, M_uqd])
            if o != 0 and k != 0:
                M = col1.row_join(col2).row_join(col3)
            elif o != 0:
                M = col1.row_join(col2)
            else:
                M = col1
        elif k != 0:
            M = col2.row_join(col3)
        else:
            M = col2
        M_eq = msubs(M, op_point_dict)

        # Build up state coefficient matrix A
        #     |(A_qq + A_qu*C_1)*C_0       A_qu*C_2|
        # A = |(A_uqc + A_uuc*C_1)*C_0    A_uuc*C_2|
        #     |(A_uqd + A_uud*C_1)*C_0    A_uud*C_2|
        # Col 1 is only defined if n != 0
        if n != 0:
            r1c1 = A_qq
            if o != 0:
                r1c1 += (A_qu * C_1)
            r1c1 = r1c1 * C_0
            if m != 0:
                r2c1 = A_uqc
                if o != 0:
                    r2c1 += (A_uuc * C_1)
                r2c1 = r2c1 * C_0
            else:
                r2c1 = Matrix()
            if o - m + k != 0:
                r3c1 = A_uqd
                if o != 0:
                    r3c1 += (A_uud * C_1)
                r3c1 = r3c1 * C_0
            else:
                r3c1 = Matrix()
            col1 = Matrix([r1c1, r2c1, r3c1])
        else:
            col1 = Matrix()
        # Col 2 is only defined if o != 0
        if o != 0:
            if n != 0:
                r1c2 = A_qu * C_2
            else:
                r1c2 = Matrix()
            if m != 0:
                r2c2 = A_uuc * C_2
            else:
                r2c2 = Matrix()
            if o - m + k != 0:
                r3c2 = A_uud * C_2
            else:
                r3c2 = Matrix()
            col2 = Matrix([r1c2, r2c2, r3c2])
        else:
            col2 = Matrix()
        if col1:
            if col2:
                Amat = col1.row_join(col2)
            else:
                Amat = col1
        else:
            Amat = col2
        Amat_eq = msubs(Amat, op_point_dict)

        # Build up the B matrix if there are forcing variables
        #     |0_(n + m)xs|
        # B = |B_u        |
        if s != 0 and o - m + k != 0:
            Bmat = zeros(n + m, s).col_join(B_u)
            Bmat_eq = msubs(Bmat, op_point_dict)
        else:
            Bmat_eq = Matrix()

        # kwarg A_and_B indicates to return  A, B for forming the equation
        # dx = [A]x + [B]r, where x = [q_indnd, u_indnd]^T,
        if A_and_B:
            A_cont = self.perm_mat.T * self.linear_solver(M_eq, Amat_eq)
            if Bmat_eq:
                B_cont = self.perm_mat.T * self.linear_solver(M_eq, Bmat_eq)
            else:
                # Bmat = Matrix([]), so no need to sub
                B_cont = Bmat_eq
            if simplify:
                A_cont.simplify()
                B_cont.simplify()
            return A_cont, B_cont
        # Otherwise return M, A, B for forming the equation
        # [M]dx = [A]x + [B]r, where x = [q, u]^T
        else:
            if simplify:
                M_eq.simplify()
                Amat_eq.simplify()
                Bmat_eq.simplify()
            return M_eq, Amat_eq, Bmat_eq


def permutation_matrix(orig_vec, per_vec):
    """Compute the permutation matrix to change order of
    orig_vec into order of per_vec.

    Parameters
    ==========

    orig_vec : array_like
        Symbols in original ordering.
    per_vec : array_like
        Symbols in new ordering.

    Returns
    =======

    p_matrix : Matrix
        Permutation matrix such that orig_vec == (p_matrix * per_vec).
    """
    if not isinstance(orig_vec, (list, tuple)):
        orig_vec = flatten(orig_vec)
    if not isinstance(per_vec, (list, tuple)):
        per_vec = flatten(per_vec)
    if set(orig_vec) != set(per_vec):
        raise ValueError("orig_vec and per_vec must be the same length, "
                         "and contain the same symbols.")
    ind_list = [orig_vec.index(i) for i in per_vec]
    p_matrix = zeros(len(orig_vec))
    for i, j in enumerate(ind_list):
        p_matrix[i, j] = 1
    return p_matrix
