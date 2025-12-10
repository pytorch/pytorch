from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.named_groups import SymmetricGroup, AlternatingGroup, DihedralGroup
from sympy.matrices import Matrix

def test_pc_presentation():
    Groups = [SymmetricGroup(3), SymmetricGroup(4), SymmetricGroup(9).sylow_subgroup(3),
         SymmetricGroup(9).sylow_subgroup(2), SymmetricGroup(8).sylow_subgroup(2), DihedralGroup(10)]

    S = SymmetricGroup(125).sylow_subgroup(5)
    G = S.derived_series()[2]
    Groups.append(G)

    G = SymmetricGroup(25).sylow_subgroup(5)
    Groups.append(G)

    S = SymmetricGroup(11**2).sylow_subgroup(11)
    G = S.derived_series()[2]
    Groups.append(G)

    for G in Groups:
        PcGroup = G.polycyclic_group()
        collector = PcGroup.collector
        pc_presentation = collector.pc_presentation

        pcgs = PcGroup.pcgs
        free_group = collector.free_group
        free_to_perm = {}
        for s, g in zip(free_group.symbols, pcgs):
            free_to_perm[s] = g

        for k, v in pc_presentation.items():
            k_array = k.array_form
            if v != ():
                v_array = v.array_form

            lhs = Permutation()
            for gen in k_array:
                s = gen[0]
                e = gen[1]
                lhs = lhs*free_to_perm[s]**e

            if v == ():
                assert lhs.is_identity
                continue

            rhs = Permutation()
            for gen in v_array:
                s = gen[0]
                e = gen[1]
                rhs = rhs*free_to_perm[s]**e

            assert lhs == rhs


def test_exponent_vector():

    Groups = [SymmetricGroup(3), SymmetricGroup(4), SymmetricGroup(9).sylow_subgroup(3),
         SymmetricGroup(9).sylow_subgroup(2), SymmetricGroup(8).sylow_subgroup(2)]

    for G in Groups:
        PcGroup = G.polycyclic_group()
        collector = PcGroup.collector

        pcgs = PcGroup.pcgs
        # free_group = collector.free_group

        for gen in G.generators:
            exp = collector.exponent_vector(gen)
            g = Permutation()
            for i in range(len(exp)):
                g = g*pcgs[i]**exp[i] if exp[i] else g
            assert g == gen


def test_induced_pcgs():
    G = [SymmetricGroup(9).sylow_subgroup(3), SymmetricGroup(20).sylow_subgroup(2), AlternatingGroup(4),
    DihedralGroup(4), DihedralGroup(10), DihedralGroup(9), SymmetricGroup(3), SymmetricGroup(4)]

    for g in G:
        PcGroup = g.polycyclic_group()
        collector = PcGroup.collector
        gens = list(g.generators)
        ipcgs = collector.induced_pcgs(gens)
        m = []
        for i in ipcgs:
            m.append(collector.exponent_vector(i))
        assert Matrix(m).is_upper
