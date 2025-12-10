from .cartan_type import CartanType
from sympy.core.basic import Atom

class RootSystem(Atom):
    """Represent the root system of a simple Lie algebra

    Every simple Lie algebra has a unique root system.  To find the root
    system, we first consider the Cartan subalgebra of g, which is the maximal
    abelian subalgebra, and consider the adjoint action of g on this
    subalgebra.  There is a root system associated with this action. Now, a
    root system over a vector space V is a set of finite vectors Phi (called
    roots), which satisfy:

    1.  The roots span V
    2.  The only scalar multiples of x in Phi are x and -x
    3.  For every x in Phi, the set Phi is closed under reflection
        through the hyperplane perpendicular to x.
    4.  If x and y are roots in Phi, then the projection of y onto
        the line through x is a half-integral multiple of x.

    Now, there is a subset of Phi, which we will call Delta, such that:
    1.  Delta is a basis of V
    2.  Each root x in Phi can be written x = sum k_y y for y in Delta

    The elements of Delta are called the simple roots.
    Therefore, we see that the simple roots span the root space of a given
    simple Lie algebra.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Root_system
    .. [2] Lie Algebras and Representation Theory - Humphreys

    """

    def __new__(cls, cartantype):
        """Create a new RootSystem object

        This method assigns an attribute called cartan_type to each instance of
        a RootSystem object.  When an instance of RootSystem is called, it
        needs an argument, which should be an instance of a simple Lie algebra.
        We then take the CartanType of this argument and set it as the
        cartan_type attribute of the RootSystem instance.

        """
        obj = Atom.__new__(cls)
        obj.cartan_type = CartanType(cartantype)
        return obj

    def simple_roots(self):
        """Generate the simple roots of the Lie algebra

        The rank of the Lie algebra determines the number of simple roots that
        it has.  This method obtains the rank of the Lie algebra, and then uses
        the simple_root method from the Lie algebra classes to generate all the
        simple roots.

        Examples
        ========

        >>> from sympy.liealgebras.root_system import RootSystem
        >>> c = RootSystem("A3")
        >>> roots = c.simple_roots()
        >>> roots
        {1: [1, -1, 0, 0], 2: [0, 1, -1, 0], 3: [0, 0, 1, -1]}

        """
        n = self.cartan_type.rank()
        roots = {i: self.cartan_type.simple_root(i) for i in range(1, n+1)}
        return roots


    def all_roots(self):
        """Generate all the roots of a given root system

        The result is a dictionary where the keys are integer numbers.  It
        generates the roots by getting the dictionary of all positive roots
        from the bases classes, and then taking each root, and multiplying it
        by -1 and adding it to the dictionary.  In this way all the negative
        roots are generated.

        """
        alpha = self.cartan_type.positive_roots()
        keys = list(alpha.keys())
        k = max(keys)
        for val in keys:
            k += 1
            root = alpha[val]
            newroot = [-x for x in root]
            alpha[k] = newroot
        return alpha

    def root_space(self):
        """Return the span of the simple roots

        The root space is the vector space spanned by the simple roots, i.e. it
        is a vector space with a distinguished basis, the simple roots.  This
        method returns a string that represents the root space as the span of
        the simple roots, alpha[1],...., alpha[n].

        Examples
        ========

        >>> from sympy.liealgebras.root_system import RootSystem
        >>> c = RootSystem("A3")
        >>> c.root_space()
        'alpha[1] + alpha[2] + alpha[3]'

        """
        n = self.cartan_type.rank()
        rs = " + ".join("alpha["+str(i) +"]" for i in range(1, n+1))
        return rs

    def add_simple_roots(self, root1, root2):
        """Add two simple roots together

        The function takes as input two integers, root1 and root2.  It then
        uses these integers as keys in the dictionary of simple roots, and gets
        the corresponding simple roots, and then adds them together.

        Examples
        ========

        >>> from sympy.liealgebras.root_system import RootSystem
        >>> c = RootSystem("A3")
        >>> newroot = c.add_simple_roots(1, 2)
        >>> newroot
        [1, 0, -1, 0]

        """

        alpha = self.simple_roots()
        if root1 > len(alpha) or root2 > len(alpha):
            raise ValueError("You've used a root that doesn't exist!")
        a1 = alpha[root1]
        a2 = alpha[root2]
        newroot = [_a1 + _a2 for _a1, _a2 in zip(a1, a2)]
        return newroot

    def add_as_roots(self, root1, root2):
        """Add two roots together if and only if their sum is also a root

        It takes as input two vectors which should be roots.  It then computes
        their sum and checks if it is in the list of all possible roots.  If it
        is, it returns the sum.  Otherwise it returns a string saying that the
        sum is not a root.

        Examples
        ========

        >>> from sympy.liealgebras.root_system import RootSystem
        >>> c = RootSystem("A3")
        >>> c.add_as_roots([1, 0, -1, 0], [0, 0, 1, -1])
        [1, 0, 0, -1]
        >>> c.add_as_roots([1, -1, 0, 0], [0, 0, -1, 1])
        'The sum of these two roots is not a root'

        """
        alpha = self.all_roots()
        newroot = [r1 + r2 for r1, r2 in zip(root1, root2)]
        if newroot in alpha.values():
            return newroot
        else:
            return "The sum of these two roots is not a root"


    def cartan_matrix(self):
        """Cartan matrix of Lie algebra associated with this root system

        Examples
        ========

        >>> from sympy.liealgebras.root_system import RootSystem
        >>> c = RootSystem("A3")
        >>> c.cartan_matrix()
        Matrix([
            [ 2, -1,  0],
            [-1,  2, -1],
            [ 0, -1,  2]])
        """
        return self.cartan_type.cartan_matrix()

    def dynkin_diagram(self):
        """Dynkin diagram of the Lie algebra associated with this root system

        Examples
        ========

        >>> from sympy.liealgebras.root_system import RootSystem
        >>> c = RootSystem("A3")
        >>> print(c.dynkin_diagram())
        0---0---0
        1   2   3
        """
        return self.cartan_type.dynkin_diagram()
