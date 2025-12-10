r"""
This module contains the functionality to arrange the nodes of a
diagram on an abstract grid, and then to produce a graphical
representation of the grid.

The currently supported back-ends are Xy-pic [Xypic].

Layout Algorithm
================

This section provides an overview of the algorithms implemented in
:class:`DiagramGrid` to lay out diagrams.

The first step of the algorithm is the removal composite and identity
morphisms which do not have properties in the supplied diagram.  The
premises and conclusions of the diagram are then merged.

The generic layout algorithm begins with the construction of the
"skeleton" of the diagram.  The skeleton is an undirected graph which
has the objects of the diagram as vertices and has an (undirected)
edge between each pair of objects between which there exist morphisms.
The direction of the morphisms does not matter at this stage.  The
skeleton also includes an edge between each pair of vertices `A` and
`C` such that there exists an object `B` which is connected via
a morphism to `A`, and via a morphism to `C`.

The skeleton constructed in this way has the property that every
object is a vertex of a triangle formed by three edges of the
skeleton.  This property lies at the base of the generic layout
algorithm.

After the skeleton has been constructed, the algorithm lists all
triangles which can be formed.  Note that some triangles will not have
all edges corresponding to morphisms which will actually be drawn.
Triangles which have only one edge or less which will actually be
drawn are immediately discarded.

The list of triangles is sorted according to the number of edges which
correspond to morphisms, then the triangle with the least number of such
edges is selected.  One of such edges is picked and the corresponding
objects are placed horizontally, on a grid.  This edge is recorded to
be in the fringe.  The algorithm then finds a "welding" of a triangle
to the fringe.  A welding is an edge in the fringe where a triangle
could be attached.  If the algorithm succeeds in finding such a
welding, it adds to the grid that vertex of the triangle which was not
yet included in any edge in the fringe and records the two new edges in
the fringe.  This process continues iteratively until all objects of
the diagram has been placed or until no more weldings can be found.

An edge is only removed from the fringe when a welding to this edge
has been found, and there is no room around this edge to place
another vertex.

When no more weldings can be found, but there are still triangles
left, the algorithm searches for a possibility of attaching one of the
remaining triangles to the existing structure by a vertex.  If such a
possibility is found, the corresponding edge of the found triangle is
placed in the found space and the iterative process of welding
triangles restarts.

When logical groups are supplied, each of these groups is laid out
independently.  Then a diagram is constructed in which groups are
objects and any two logical groups between which there exist morphisms
are connected via a morphism.  This diagram is laid out.  Finally,
the grid which includes all objects of the initial diagram is
constructed by replacing the cells which contain logical groups with
the corresponding laid out grids, and by correspondingly expanding the
rows and columns.

The sequential layout algorithm begins by constructing the
underlying undirected graph defined by the morphisms obtained after
simplifying premises and conclusions and merging them (see above).
The vertex with the minimal degree is then picked up and depth-first
search is started from it.  All objects which are located at distance
`n` from the root in the depth-first search tree, are positioned in
the `n`-th column of the resulting grid.  The sequential layout will
therefore attempt to lay the objects out along a line.

References
==========

.. [Xypic] https://xy-pic.sourceforge.net/

"""
from sympy.categories import (CompositeMorphism, IdentityMorphism,
                              NamedMorphism, Diagram)
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on

from itertools import chain


__doctest_requires__ = {('preview_diagram',): 'pyglet'}


class _GrowableGrid:
    """
    Holds a growable grid of objects.

    Explanation
    ===========

    It is possible to append or prepend a row or a column to the grid
    using the corresponding methods.  Prepending rows or columns has
    the effect of changing the coordinates of the already existing
    elements.

    This class currently represents a naive implementation of the
    functionality with little attempt at optimisation.
    """
    def __init__(self, width, height):
        self._width = width
        self._height = height

        self._array = [[None for j in range(width)] for i in range(height)]

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def __getitem__(self, i_j):
        """
        Returns the element located at in the i-th line and j-th
        column.
        """
        i, j = i_j
        return self._array[i][j]

    def __setitem__(self, i_j, newvalue):
        """
        Sets the element located at in the i-th line and j-th
        column.
        """
        i, j = i_j
        self._array[i][j] = newvalue

    def append_row(self):
        """
        Appends an empty row to the grid.
        """
        self._height += 1
        self._array.append([None for j in range(self._width)])

    def append_column(self):
        """
        Appends an empty column to the grid.
        """
        self._width += 1
        for i in range(self._height):
            self._array[i].append(None)

    def prepend_row(self):
        """
        Prepends the grid with an empty row.
        """
        self._height += 1
        self._array.insert(0, [None for j in range(self._width)])

    def prepend_column(self):
        """
        Prepends the grid with an empty column.
        """
        self._width += 1
        for i in range(self._height):
            self._array[i].insert(0, None)


class DiagramGrid:
    r"""
    Constructs and holds the fitting of the diagram into a grid.

    Explanation
    ===========

    The mission of this class is to analyse the structure of the
    supplied diagram and to place its objects on a grid such that,
    when the objects and the morphisms are actually drawn, the diagram
    would be "readable", in the sense that there will not be many
    intersections of moprhisms.  This class does not perform any
    actual drawing.  It does strive nevertheless to offer sufficient
    metadata to draw a diagram.

    Consider the following simple diagram.

    >>> from sympy.categories import Object, NamedMorphism
    >>> from sympy.categories import Diagram, DiagramGrid
    >>> from sympy import pprint
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> diagram = Diagram([f, g])

    The simplest way to have a diagram laid out is the following:

    >>> grid = DiagramGrid(diagram)
    >>> (grid.width, grid.height)
    (2, 2)
    >>> pprint(grid)
    A  B
    <BLANKLINE>
       C

    Sometimes one sees the diagram as consisting of logical groups.
    One can advise ``DiagramGrid`` as to such groups by employing the
    ``groups`` keyword argument.

    Consider the following diagram:

    >>> D = Object("D")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> h = NamedMorphism(D, A, "h")
    >>> k = NamedMorphism(D, B, "k")
    >>> diagram = Diagram([f, g, h, k])

    Lay it out with generic layout:

    >>> grid = DiagramGrid(diagram)
    >>> pprint(grid)
    A  B  D
    <BLANKLINE>
       C

    Now, we can group the objects `A` and `D` to have them near one
    another:

    >>> grid = DiagramGrid(diagram, groups=[[A, D], B, C])
    >>> pprint(grid)
    B     C
    <BLANKLINE>
    A  D

    Note how the positioning of the other objects changes.

    Further indications can be supplied to the constructor of
    :class:`DiagramGrid` using keyword arguments.  The currently
    supported hints are explained in the following paragraphs.

    :class:`DiagramGrid` does not automatically guess which layout
    would suit the supplied diagram better.  Consider, for example,
    the following linear diagram:

    >>> E = Object("E")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> h = NamedMorphism(C, D, "h")
    >>> i = NamedMorphism(D, E, "i")
    >>> diagram = Diagram([f, g, h, i])

    When laid out with the generic layout, it does not get to look
    linear:

    >>> grid = DiagramGrid(diagram)
    >>> pprint(grid)
    A  B
    <BLANKLINE>
       C  D
    <BLANKLINE>
          E

    To get it laid out in a line, use ``layout="sequential"``:

    >>> grid = DiagramGrid(diagram, layout="sequential")
    >>> pprint(grid)
    A  B  C  D  E

    One may sometimes need to transpose the resulting layout.  While
    this can always be done by hand, :class:`DiagramGrid` provides a
    hint for that purpose:

    >>> grid = DiagramGrid(diagram, layout="sequential", transpose=True)
    >>> pprint(grid)
    A
    <BLANKLINE>
    B
    <BLANKLINE>
    C
    <BLANKLINE>
    D
    <BLANKLINE>
    E

    Separate hints can also be provided for each group.  For an
    example, refer to ``tests/test_drawing.py``, and see the different
    ways in which the five lemma [FiveLemma] can be laid out.

    See Also
    ========

    Diagram

    References
    ==========

    .. [FiveLemma] https://en.wikipedia.org/wiki/Five_lemma
    """
    @staticmethod
    def _simplify_morphisms(morphisms):
        """
        Given a dictionary mapping morphisms to their properties,
        returns a new dictionary in which there are no morphisms which
        do not have properties, and which are compositions of other
        morphisms included in the dictionary.  Identities are dropped
        as well.
        """
        newmorphisms = {}
        for morphism, props in morphisms.items():
            if isinstance(morphism, CompositeMorphism) and not props:
                continue
            elif isinstance(morphism, IdentityMorphism):
                continue
            else:
                newmorphisms[morphism] = props
        return newmorphisms

    @staticmethod
    def _merge_premises_conclusions(premises, conclusions):
        """
        Given two dictionaries of morphisms and their properties,
        produces a single dictionary which includes elements from both
        dictionaries.  If a morphism has some properties in premises
        and also in conclusions, the properties in conclusions take
        priority.
        """
        return dict(chain(premises.items(), conclusions.items()))

    @staticmethod
    def _juxtapose_edges(edge1, edge2):
        """
        If ``edge1`` and ``edge2`` have precisely one common endpoint,
        returns an edge which would form a triangle with ``edge1`` and
        ``edge2``.

        If ``edge1`` and ``edge2`` do not have a common endpoint,
        returns ``None``.

        If ``edge1`` and ``edge`` are the same edge, returns ``None``.
        """
        intersection = edge1 & edge2
        if len(intersection) != 1:
            # The edges either have no common points or are equal.
            return None

        # The edges have a common endpoint.  Extract the different
        # endpoints and set up the new edge.
        return (edge1 - intersection) | (edge2 - intersection)

    @staticmethod
    def _add_edge_append(dictionary, edge, elem):
        """
        If ``edge`` is not in ``dictionary``, adds ``edge`` to the
        dictionary and sets its value to ``[elem]``.  Otherwise
        appends ``elem`` to the value of existing entry.

        Note that edges are undirected, thus `(A, B) = (B, A)`.
        """
        if edge in dictionary:
            dictionary[edge].append(elem)
        else:
            dictionary[edge] = [elem]

    @staticmethod
    def _build_skeleton(morphisms):
        """
        Creates a dictionary which maps edges to corresponding
        morphisms.  Thus for a morphism `f:A\rightarrow B`, the edge
        `(A, B)` will be associated with `f`.  This function also adds
        to the list those edges which are formed by juxtaposition of
        two edges already in the list.  These new edges are not
        associated with any morphism and are only added to assure that
        the diagram can be decomposed into triangles.
        """
        edges = {}
        # Create edges for morphisms.
        for morphism in morphisms:
            DiagramGrid._add_edge_append(
                edges, frozenset([morphism.domain, morphism.codomain]), morphism)

        # Create new edges by juxtaposing existing edges.
        edges1 = dict(edges)
        for w in edges1:
            for v in edges1:
                wv = DiagramGrid._juxtapose_edges(w, v)
                if wv and wv not in edges:
                    edges[wv] = []

        return edges

    @staticmethod
    def _list_triangles(edges):
        """
        Builds the set of triangles formed by the supplied edges.  The
        triangles are arbitrary and need not be commutative.  A
        triangle is a set that contains all three of its sides.
        """
        triangles = set()

        for w in edges:
            for v in edges:
                wv = DiagramGrid._juxtapose_edges(w, v)
                if wv and wv in edges:
                    triangles.add(frozenset([w, v, wv]))

        return triangles

    @staticmethod
    def _drop_redundant_triangles(triangles, skeleton):
        """
        Returns a list which contains only those triangles who have
        morphisms associated with at least two edges.
        """
        return [tri for tri in triangles
                if len([e for e in tri if skeleton[e]]) >= 2]

    @staticmethod
    def _morphism_length(morphism):
        """
        Returns the length of a morphism.  The length of a morphism is
        the number of components it consists of.  A non-composite
        morphism is of length 1.
        """
        if isinstance(morphism, CompositeMorphism):
            return len(morphism.components)
        else:
            return 1

    @staticmethod
    def _compute_triangle_min_sizes(triangles, edges):
        r"""
        Returns a dictionary mapping triangles to their minimal sizes.
        The minimal size of a triangle is the sum of maximal lengths
        of morphisms associated to the sides of the triangle.  The
        length of a morphism is the number of components it consists
        of.  A non-composite morphism is of length 1.

        Sorting triangles by this metric attempts to address two
        aspects of layout.  For triangles with only simple morphisms
        in the edge, this assures that triangles with all three edges
        visible will get typeset after triangles with less visible
        edges, which sometimes minimizes the necessity in diagonal
        arrows.  For triangles with composite morphisms in the edges,
        this assures that objects connected with shorter morphisms
        will be laid out first, resulting the visual proximity of
        those objects which are connected by shorter morphisms.
        """
        triangle_sizes = {}
        for triangle in triangles:
            size = 0
            for e in triangle:
                morphisms = edges[e]
                if morphisms:
                    size += max(DiagramGrid._morphism_length(m)
                                for m in morphisms)
            triangle_sizes[triangle] = size
        return triangle_sizes

    @staticmethod
    def _triangle_objects(triangle):
        """
        Given a triangle, returns the objects included in it.
        """
        # A triangle is a frozenset of three two-element frozensets
        # (the edges).  This chains the three edges together and
        # creates a frozenset from the iterator, thus producing a
        # frozenset of objects of the triangle.
        return frozenset(chain(*tuple(triangle)))

    @staticmethod
    def _other_vertex(triangle, edge):
        """
        Given a triangle and an edge of it, returns the vertex which
        opposes the edge.
        """
        # This gets the set of objects of the triangle and then
        # subtracts the set of objects employed in ``edge`` to get the
        # vertex opposite to ``edge``.
        return list(DiagramGrid._triangle_objects(triangle) - set(edge))[0]

    @staticmethod
    def _empty_point(pt, grid):
        """
        Checks if the cell at coordinates ``pt`` is either empty or
        out of the bounds of the grid.
        """
        if (pt[0] < 0) or (pt[1] < 0) or \
           (pt[0] >= grid.height) or (pt[1] >= grid.width):
            return True
        return grid[pt] is None

    @staticmethod
    def _put_object(coords, obj, grid, fringe):
        """
        Places an object at the coordinate ``cords`` in ``grid``,
        growing the grid and updating ``fringe``, if necessary.
        Returns (0, 0) if no row or column has been prepended, (1, 0)
        if a row was prepended, (0, 1) if a column was prepended and
        (1, 1) if both a column and a row were prepended.
        """
        (i, j) = coords
        offset = (0, 0)
        if i == -1:
            grid.prepend_row()
            i = 0
            offset = (1, 0)
            for k in range(len(fringe)):
                ((i1, j1), (i2, j2)) = fringe[k]
                fringe[k] = ((i1 + 1, j1), (i2 + 1, j2))
        elif i == grid.height:
            grid.append_row()

        if j == -1:
            j = 0
            offset = (offset[0], 1)
            grid.prepend_column()
            for k in range(len(fringe)):
                ((i1, j1), (i2, j2)) = fringe[k]
                fringe[k] = ((i1, j1 + 1), (i2, j2 + 1))
        elif j == grid.width:
            grid.append_column()

        grid[i, j] = obj
        return offset

    @staticmethod
    def _choose_target_cell(pt1, pt2, edge, obj, skeleton, grid):
        """
        Given two points, ``pt1`` and ``pt2``, and the welding edge
        ``edge``, chooses one of the two points to place the opposing
        vertex ``obj`` of the triangle.  If neither of this points
        fits, returns ``None``.
        """
        pt1_empty = DiagramGrid._empty_point(pt1, grid)
        pt2_empty = DiagramGrid._empty_point(pt2, grid)

        if pt1_empty and pt2_empty:
            # Both cells are empty.  Of these two, choose that cell
            # which will assure that a visible edge of the triangle
            # will be drawn perpendicularly to the current welding
            # edge.

            A = grid[edge[0]]

            if skeleton.get(frozenset([A, obj])):
                return pt1
            else:
                return pt2
        if pt1_empty:
            return pt1
        elif pt2_empty:
            return pt2
        else:
            return None

    @staticmethod
    def _find_triangle_to_weld(triangles, fringe, grid):
        """
        Finds, if possible, a triangle and an edge in the ``fringe`` to
        which the triangle could be attached.  Returns the tuple
        containing the triangle and the index of the corresponding
        edge in the ``fringe``.

        This function relies on the fact that objects are unique in
        the diagram.
        """
        for triangle in triangles:
            for (a, b) in fringe:
                if frozenset([grid[a], grid[b]]) in triangle:
                    return (triangle, (a, b))
        return None

    @staticmethod
    def _weld_triangle(tri, welding_edge, fringe, grid, skeleton):
        """
        If possible, welds the triangle ``tri`` to ``fringe`` and
        returns ``False``.  If this method encounters a degenerate
        situation in the fringe and corrects it such that a restart of
        the search is required, it returns ``True`` (which means that
        a restart in finding triangle weldings is required).

        A degenerate situation is a situation when an edge listed in
        the fringe does not belong to the visual boundary of the
        diagram.
        """
        a, b = welding_edge
        target_cell = None

        obj = DiagramGrid._other_vertex(tri, (grid[a], grid[b]))

        # We now have a triangle and an edge where it can be welded to
        # the fringe.  Decide where to place the other vertex of the
        # triangle and check for degenerate situations en route.

        if (abs(a[0] - b[0]) == 1) and (abs(a[1] - b[1]) == 1):
            # A diagonal edge.
            target_cell = (a[0], b[1])
            if grid[target_cell]:
                # That cell is already occupied.
                target_cell = (b[0], a[1])

                if grid[target_cell]:
                    # Degenerate situation, this edge is not
                    # on the actual fringe.  Correct the
                    # fringe and go on.
                    fringe.remove((a, b))
                    return True
        elif a[0] == b[0]:
            # A horizontal edge.  We first attempt to build the
            # triangle in the downward direction.

            down_left = a[0] + 1, a[1]
            down_right = a[0] + 1, b[1]

            target_cell = DiagramGrid._choose_target_cell(
                down_left, down_right, (a, b), obj, skeleton, grid)

            if not target_cell:
                # No room below this edge.  Check above.
                up_left = a[0] - 1, a[1]
                up_right = a[0] - 1, b[1]

                target_cell = DiagramGrid._choose_target_cell(
                    up_left, up_right, (a, b), obj, skeleton, grid)

                if not target_cell:
                    # This edge is not in the fringe, remove it
                    # and restart.
                    fringe.remove((a, b))
                    return True
        elif a[1] == b[1]:
            # A vertical edge.  We will attempt to place the other
            # vertex of the triangle to the right of this edge.
            right_up = a[0], a[1] + 1
            right_down = b[0], a[1] + 1

            target_cell = DiagramGrid._choose_target_cell(
                right_up, right_down, (a, b), obj, skeleton, grid)

            if not target_cell:
                # No room to the left.  See what's to the right.
                left_up = a[0], a[1] - 1
                left_down = b[0], a[1] - 1

                target_cell = DiagramGrid._choose_target_cell(
                    left_up, left_down, (a, b), obj, skeleton, grid)

                if not target_cell:
                    # This edge is not in the fringe, remove it
                    # and restart.
                    fringe.remove((a, b))
                    return True

        # We now know where to place the other vertex of the
        # triangle.
        offset = DiagramGrid._put_object(target_cell, obj, grid, fringe)

        # Take care of the displacement of coordinates if a row or
        # a column was prepended.
        target_cell = (target_cell[0] + offset[0],
                       target_cell[1] + offset[1])
        a = (a[0] + offset[0], a[1] + offset[1])
        b = (b[0] + offset[0], b[1] + offset[1])

        fringe.extend([(a, target_cell), (b, target_cell)])

        # No restart is required.
        return False

    @staticmethod
    def _triangle_key(tri, triangle_sizes):
        """
        Returns a key for the supplied triangle.  It should be the
        same independently of the hash randomisation.
        """
        objects = sorted(
            DiagramGrid._triangle_objects(tri), key=default_sort_key)
        return (triangle_sizes[tri], default_sort_key(objects))

    @staticmethod
    def _pick_root_edge(tri, skeleton):
        """
        For a given triangle always picks the same root edge.  The
        root edge is the edge that will be placed first on the grid.
        """
        candidates = [sorted(e, key=default_sort_key)
                      for e in tri if skeleton[e]]
        sorted_candidates = sorted(candidates, key=default_sort_key)
        # Don't forget to assure the proper ordering of the vertices
        # in this edge.
        return tuple(sorted(sorted_candidates[0], key=default_sort_key))

    @staticmethod
    def _drop_irrelevant_triangles(triangles, placed_objects):
        """
        Returns only those triangles whose set of objects is not
        completely included in ``placed_objects``.
        """
        return [tri for tri in triangles if not placed_objects.issuperset(
            DiagramGrid._triangle_objects(tri))]

    @staticmethod
    def _grow_pseudopod(triangles, fringe, grid, skeleton, placed_objects):
        """
        Starting from an object in the existing structure on the ``grid``,
        adds an edge to which a triangle from ``triangles`` could be
        welded.  If this method has found a way to do so, it returns
        the object it has just added.

        This method should be applied when ``_weld_triangle`` cannot
        find weldings any more.
        """
        for i in range(grid.height):
            for j in range(grid.width):
                obj = grid[i, j]
                if not obj:
                    continue

                # Here we need to choose a triangle which has only
                # ``obj`` in common with the existing structure.  The
                # situations when this is not possible should be
                # handled elsewhere.

                def good_triangle(tri):
                    objs = DiagramGrid._triangle_objects(tri)
                    return obj in objs and \
                        placed_objects & (objs - {obj}) == set()

                tris = [tri for tri in triangles if good_triangle(tri)]
                if not tris:
                    # This object is not interesting.
                    continue

                # Pick the "simplest" of the triangles which could be
                # attached.  Remember that the list of triangles is
                # sorted according to their "simplicity" (see
                # _compute_triangle_min_sizes for the metric).
                #
                # Note that ``tris`` are sequentially built from
                # ``triangles``, so we don't have to worry about hash
                # randomisation.
                tri = tris[0]

                # We have found a triangle which could be attached to
                # the existing structure by a vertex.

                candidates = sorted([e for e in tri if skeleton[e]],
                                    key=lambda e: FiniteSet(*e).sort_key())
                edges = [e for e in candidates if obj in e]

                # Note that a meaningful edge (i.e., and edge that is
                # associated with a morphism) containing ``obj``
                # always exists.  That's because all triangles are
                # guaranteed to have at least two meaningful edges.
                # See _drop_redundant_triangles.

                # Get the object at the other end of the edge.
                edge = edges[0]
                other_obj = tuple(edge - frozenset([obj]))[0]

                # Now check for free directions.  When checking for
                # free directions, prefer the horizontal and vertical
                # directions.
                neighbours = [(i - 1, j), (i, j + 1), (i + 1, j), (i, j - 1),
                              (i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)]

                for pt in neighbours:
                    if DiagramGrid._empty_point(pt, grid):
                        # We have a found a place to grow the
                        # pseudopod into.
                        offset = DiagramGrid._put_object(
                            pt, other_obj, grid, fringe)

                        i += offset[0]
                        j += offset[1]
                        pt = (pt[0] + offset[0], pt[1] + offset[1])
                        fringe.append(((i, j), pt))

                        return other_obj

        # This diagram is actually cooler that I can handle.  Fail cowardly.
        return None

    @staticmethod
    def _handle_groups(diagram, groups, merged_morphisms, hints):
        """
        Given the slightly preprocessed morphisms of the diagram,
        produces a grid laid out according to ``groups``.

        If a group has hints, it is laid out with those hints only,
        without any influence from ``hints``.  Otherwise, it is laid
        out with ``hints``.
        """
        def lay_out_group(group, local_hints):
            """
            If ``group`` is a set of objects, uses a ``DiagramGrid``
            to lay it out and returns the grid.  Otherwise returns the
            object (i.e., ``group``).  If ``local_hints`` is not
            empty, it is supplied to ``DiagramGrid`` as the dictionary
            of hints.  Otherwise, the ``hints`` argument of
            ``_handle_groups`` is used.
            """
            if isinstance(group, FiniteSet):
                # Set up the corresponding object-to-group
                # mappings.
                for obj in group:
                    obj_groups[obj] = group

                # Lay out the current group.
                if local_hints:
                    groups_grids[group] = DiagramGrid(
                        diagram.subdiagram_from_objects(group), **local_hints)
                else:
                    groups_grids[group] = DiagramGrid(
                        diagram.subdiagram_from_objects(group), **hints)
            else:
                obj_groups[group] = group

        def group_to_finiteset(group):
            """
            Converts ``group`` to a :class:``FiniteSet`` if it is an
            iterable.
            """
            if iterable(group):
                return FiniteSet(*group)
            else:
                return group

        obj_groups = {}
        groups_grids = {}

        # We would like to support various containers to represent
        # groups.  To achieve that, before laying each group out, it
        # should be converted to a FiniteSet, because that is what the
        # following code expects.

        if isinstance(groups, (dict, Dict)):
            finiteset_groups = {}
            for group, local_hints in groups.items():
                finiteset_group = group_to_finiteset(group)
                finiteset_groups[finiteset_group] = local_hints
                lay_out_group(group, local_hints)
            groups = finiteset_groups
        else:
            finiteset_groups = []
            for group in groups:
                finiteset_group = group_to_finiteset(group)
                finiteset_groups.append(finiteset_group)
                lay_out_group(finiteset_group, None)
            groups = finiteset_groups

        new_morphisms = []
        for morphism in merged_morphisms:
            dom = obj_groups[morphism.domain]
            cod = obj_groups[morphism.codomain]
            # Note that we are not really interested in morphisms
            # which do not employ two different groups, because
            # these do not influence the layout.
            if dom != cod:
                # These are essentially unnamed morphisms; they are
                # not going to mess in the final layout.  By giving
                # them the same names, we avoid unnecessary
                # duplicates.
                new_morphisms.append(NamedMorphism(dom, cod, "dummy"))

        # Lay out the new diagram.  Since these are dummy morphisms,
        # properties and conclusions are irrelevant.
        top_grid = DiagramGrid(Diagram(new_morphisms))

        # We now have to substitute the groups with the corresponding
        # grids, laid out at the beginning of this function.  Compute
        # the size of each row and column in the grid, so that all
        # nested grids fit.

        def group_size(group):
            """
            For the supplied group (or object, eventually), returns
            the size of the cell that will hold this group (object).
            """
            if group in groups_grids:
                grid = groups_grids[group]
                return (grid.height, grid.width)
            else:
                return (1, 1)

        row_heights = [max(group_size(top_grid[i, j])[0]
                           for j in range(top_grid.width))
                       for i in range(top_grid.height)]

        column_widths = [max(group_size(top_grid[i, j])[1]
                             for i in range(top_grid.height))
                         for j in range(top_grid.width)]

        grid = _GrowableGrid(sum(column_widths), sum(row_heights))

        real_row = 0
        real_column = 0
        for logical_row in range(top_grid.height):
            for logical_column in range(top_grid.width):
                obj = top_grid[logical_row, logical_column]

                if obj in groups_grids:
                    # This is a group.  Copy the corresponding grid in
                    # place.
                    local_grid = groups_grids[obj]
                    for i in range(local_grid.height):
                        for j in range(local_grid.width):
                            grid[real_row + i,
                                real_column + j] = local_grid[i, j]
                else:
                    # This is an object.  Just put it there.
                    grid[real_row, real_column] = obj

                real_column += column_widths[logical_column]
            real_column = 0
            real_row += row_heights[logical_row]

        return grid

    @staticmethod
    def _generic_layout(diagram, merged_morphisms):
        """
        Produces the generic layout for the supplied diagram.
        """
        all_objects = set(diagram.objects)
        if len(all_objects) == 1:
            # There only one object in the diagram, just put in on 1x1
            # grid.
            grid = _GrowableGrid(1, 1)
            grid[0, 0] = tuple(all_objects)[0]
            return grid

        skeleton = DiagramGrid._build_skeleton(merged_morphisms)

        grid = _GrowableGrid(2, 1)

        if len(skeleton) == 1:
            # This diagram contains only one morphism.  Draw it
            # horizontally.
            objects = sorted(all_objects, key=default_sort_key)
            grid[0, 0] = objects[0]
            grid[0, 1] = objects[1]

            return grid

        triangles = DiagramGrid._list_triangles(skeleton)
        triangles = DiagramGrid._drop_redundant_triangles(triangles, skeleton)
        triangle_sizes = DiagramGrid._compute_triangle_min_sizes(
            triangles, skeleton)

        triangles = sorted(triangles, key=lambda tri:
                           DiagramGrid._triangle_key(tri, triangle_sizes))

        # Place the first edge on the grid.
        root_edge = DiagramGrid._pick_root_edge(triangles[0], skeleton)
        grid[0, 0], grid[0, 1] = root_edge
        fringe = [((0, 0), (0, 1))]

        # Record which objects we now have on the grid.
        placed_objects = set(root_edge)

        while placed_objects != all_objects:
            welding = DiagramGrid._find_triangle_to_weld(
                triangles, fringe, grid)

            if welding:
                (triangle, welding_edge) = welding

                restart_required = DiagramGrid._weld_triangle(
                    triangle, welding_edge, fringe, grid, skeleton)
                if restart_required:
                    continue

                placed_objects.update(
                    DiagramGrid._triangle_objects(triangle))
            else:
                # No more weldings found.  Try to attach triangles by
                # vertices.
                new_obj = DiagramGrid._grow_pseudopod(
                    triangles, fringe, grid, skeleton, placed_objects)

                if not new_obj:
                    # No more triangles can be attached, not even by
                    # the edge.  We will set up a new diagram out of
                    # what has been left, laid it out independently,
                    # and then attach it to this one.

                    remaining_objects = all_objects - placed_objects

                    remaining_diagram = diagram.subdiagram_from_objects(
                        FiniteSet(*remaining_objects))
                    remaining_grid = DiagramGrid(remaining_diagram)

                    # Now, let's glue ``remaining_grid`` to ``grid``.
                    final_width = grid.width + remaining_grid.width
                    final_height = max(grid.height, remaining_grid.height)
                    final_grid = _GrowableGrid(final_width, final_height)

                    for i in range(grid.width):
                        for j in range(grid.height):
                            final_grid[i, j] = grid[i, j]

                    start_j = grid.width
                    for i in range(remaining_grid.height):
                        for j in range(remaining_grid.width):
                            final_grid[i, start_j + j] = remaining_grid[i, j]

                    return final_grid

                placed_objects.add(new_obj)

            triangles = DiagramGrid._drop_irrelevant_triangles(
                triangles, placed_objects)

        return grid

    @staticmethod
    def _get_undirected_graph(objects, merged_morphisms):
        """
        Given the objects and the relevant morphisms of a diagram,
        returns the adjacency lists of the underlying undirected
        graph.
        """
        adjlists = {obj: [] for obj in objects}

        for morphism in merged_morphisms:
            adjlists[morphism.domain].append(morphism.codomain)
            adjlists[morphism.codomain].append(morphism.domain)

        # Assure that the objects in the adjacency list are always in
        # the same order.
        for obj in adjlists.keys():
            adjlists[obj].sort(key=default_sort_key)

        return adjlists

    @staticmethod
    def _sequential_layout(diagram, merged_morphisms):
        r"""
        Lays out the diagram in "sequential" layout.  This method
        will attempt to produce a result as close to a line as
        possible.  For linear diagrams, the result will actually be a
        line.
        """
        objects = diagram.objects
        sorted_objects = sorted(objects, key=default_sort_key)

        # Set up the adjacency lists of the underlying undirected
        # graph of ``merged_morphisms``.
        adjlists = DiagramGrid._get_undirected_graph(objects, merged_morphisms)

        root = min(sorted_objects, key=lambda x: len(adjlists[x]))
        grid = _GrowableGrid(1, 1)
        grid[0, 0] = root

        placed_objects = {root}

        def place_objects(pt, placed_objects):
            """
            Does depth-first search in the underlying graph of the
            diagram and places the objects en route.
            """
            # We will start placing new objects from here.
            new_pt = (pt[0], pt[1] + 1)

            for adjacent_obj in adjlists[grid[pt]]:
                if adjacent_obj in placed_objects:
                    # This object has already been placed.
                    continue

                DiagramGrid._put_object(new_pt, adjacent_obj, grid, [])
                placed_objects.add(adjacent_obj)
                placed_objects.update(place_objects(new_pt, placed_objects))

                new_pt = (new_pt[0] + 1, new_pt[1])

            return placed_objects

        place_objects((0, 0), placed_objects)

        return grid

    @staticmethod
    def _drop_inessential_morphisms(merged_morphisms):
        r"""
        Removes those morphisms which should appear in the diagram,
        but which have no relevance to object layout.

        Currently this removes "loop" morphisms: the non-identity
        morphisms with the same domains and codomains.
        """
        morphisms = [m for m in merged_morphisms if m.domain != m.codomain]
        return morphisms

    @staticmethod
    def _get_connected_components(objects, merged_morphisms):
        """
        Given a container of morphisms, returns a list of connected
        components formed by these morphisms.  A connected component
        is represented by a diagram consisting of the corresponding
        morphisms.
        """
        component_index = {}
        for o in objects:
            component_index[o] = None

        # Get the underlying undirected graph of the diagram.
        adjlist = DiagramGrid._get_undirected_graph(objects, merged_morphisms)

        def traverse_component(object, current_index):
            """
            Does a depth-first search traversal of the component
            containing ``object``.
            """
            component_index[object] = current_index
            for o in adjlist[object]:
                if component_index[o] is None:
                    traverse_component(o, current_index)

        # Traverse all components.
        current_index = 0
        for o in adjlist:
            if component_index[o] is None:
                traverse_component(o, current_index)
                current_index += 1

        # List the objects of the components.
        component_objects = [[] for i in range(current_index)]
        for o, idx in component_index.items():
            component_objects[idx].append(o)

        # Finally, list the morphisms belonging to each component.
        #
        # Note: If some objects are isolated, they will not get any
        # morphisms at this stage, and since the layout algorithm
        # relies, we are essentially going to lose this object.
        # Therefore, check if there are isolated objects and, for each
        # of them, provide the trivial identity morphism.  It will get
        # discarded later, but the object will be there.

        component_morphisms = []
        for component in component_objects:
            current_morphisms = {}
            for m in merged_morphisms:
                if (m.domain in component) and (m.codomain in component):
                    current_morphisms[m] = merged_morphisms[m]

            if len(component) == 1:
                # Let's add an identity morphism, for the sake of
                # surely having morphisms in this component.
                current_morphisms[IdentityMorphism(component[0])] = FiniteSet()

            component_morphisms.append(Diagram(current_morphisms))

        return component_morphisms

    def __init__(self, diagram, groups=None, **hints):
        premises = DiagramGrid._simplify_morphisms(diagram.premises)
        conclusions = DiagramGrid._simplify_morphisms(diagram.conclusions)
        all_merged_morphisms = DiagramGrid._merge_premises_conclusions(
            premises, conclusions)
        merged_morphisms = DiagramGrid._drop_inessential_morphisms(
            all_merged_morphisms)

        # Store the merged morphisms for later use.
        self._morphisms = all_merged_morphisms

        components = DiagramGrid._get_connected_components(
            diagram.objects, all_merged_morphisms)

        if groups and (groups != diagram.objects):
            # Lay out the diagram according to the groups.
            self._grid = DiagramGrid._handle_groups(
                diagram, groups, merged_morphisms, hints)
        elif len(components) > 1:
            # Note that we check for connectedness _before_ checking
            # the layout hints because the layout strategies don't
            # know how to deal with disconnected diagrams.

            # The diagram is disconnected.  Lay out the components
            # independently.
            grids = []

            # Sort the components to eventually get the grids arranged
            # in a fixed, hash-independent order.
            components = sorted(components, key=default_sort_key)

            for component in components:
                grid = DiagramGrid(component, **hints)
                grids.append(grid)

            # Throw the grids together, in a line.
            total_width = sum(g.width for g in grids)
            total_height = max(g.height for g in grids)

            grid = _GrowableGrid(total_width, total_height)
            start_j = 0
            for g in grids:
                for i in range(g.height):
                    for j in range(g.width):
                        grid[i, start_j + j] = g[i, j]

                start_j += g.width

            self._grid = grid
        elif "layout" in hints:
            if hints["layout"] == "sequential":
                self._grid = DiagramGrid._sequential_layout(
                    diagram, merged_morphisms)
        else:
            self._grid = DiagramGrid._generic_layout(diagram, merged_morphisms)

        if hints.get("transpose"):
            # Transpose the resulting grid.
            grid = _GrowableGrid(self._grid.height, self._grid.width)
            for i in range(self._grid.height):
                for j in range(self._grid.width):
                    grid[j, i] = self._grid[i, j]
            self._grid = grid

    @property
    def width(self):
        """
        Returns the number of columns in this diagram layout.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import Diagram, DiagramGrid
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> diagram = Diagram([f, g])
        >>> grid = DiagramGrid(diagram)
        >>> grid.width
        2

        """
        return self._grid.width

    @property
    def height(self):
        """
        Returns the number of rows in this diagram layout.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import Diagram, DiagramGrid
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> diagram = Diagram([f, g])
        >>> grid = DiagramGrid(diagram)
        >>> grid.height
        2

        """
        return self._grid.height

    def __getitem__(self, i_j):
        """
        Returns the object placed in the row ``i`` and column ``j``.
        The indices are 0-based.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import Diagram, DiagramGrid
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> diagram = Diagram([f, g])
        >>> grid = DiagramGrid(diagram)
        >>> (grid[0, 0], grid[0, 1])
        (Object("A"), Object("B"))
        >>> (grid[1, 0], grid[1, 1])
        (None, Object("C"))

        """
        i, j = i_j
        return self._grid[i, j]

    @property
    def morphisms(self):
        """
        Returns those morphisms (and their properties) which are
        sufficiently meaningful to be drawn.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import Diagram, DiagramGrid
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> diagram = Diagram([f, g])
        >>> grid = DiagramGrid(diagram)
        >>> grid.morphisms
        {NamedMorphism(Object("A"), Object("B"), "f"): EmptySet,
        NamedMorphism(Object("B"), Object("C"), "g"): EmptySet}

        """
        return self._morphisms

    def __str__(self):
        """
        Produces a string representation of this class.

        This method returns a string representation of the underlying
        list of lists of objects.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import Diagram, DiagramGrid
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> diagram = Diagram([f, g])
        >>> grid = DiagramGrid(diagram)
        >>> print(grid)
        [[Object("A"), Object("B")],
        [None, Object("C")]]

        """
        return repr(self._grid._array)


class ArrowStringDescription:
    r"""
    Stores the information necessary for producing an Xy-pic
    description of an arrow.

    The principal goal of this class is to abstract away the string
    representation of an arrow and to also provide the functionality
    to produce the actual Xy-pic string.

    ``unit`` sets the unit which will be used to specify the amount of
    curving and other distances.  ``horizontal_direction`` should be a
    string of ``"r"`` or ``"l"`` specifying the horizontal offset of the
    target cell of the arrow relatively to the current one.
    ``vertical_direction`` should  specify the vertical offset using a
    series of either ``"d"`` or ``"u"``.  ``label_position`` should be
    either ``"^"``, ``"_"``,  or ``"|"`` to specify that the label should
    be positioned above the arrow, below the arrow or just over the arrow,
    in a break.  Note that the notions "above" and "below" are relative
    to arrow direction.  ``label`` stores the morphism label.

    This works as follows (disregard the yet unexplained arguments):

    >>> from sympy.categories.diagram_drawing import ArrowStringDescription
    >>> astr = ArrowStringDescription(
    ... unit="mm", curving=None, curving_amount=None,
    ... looping_start=None, looping_end=None, horizontal_direction="d",
    ... vertical_direction="r", label_position="_", label="f")
    >>> print(str(astr))
    \ar[dr]_{f}

    ``curving`` should be one of ``"^"``, ``"_"`` to specify in which
    direction the arrow is going to curve. ``curving_amount`` is a number
    describing how many ``unit``'s the morphism is going to curve:

    >>> astr = ArrowStringDescription(
    ... unit="mm", curving="^", curving_amount=12,
    ... looping_start=None, looping_end=None, horizontal_direction="d",
    ... vertical_direction="r", label_position="_", label="f")
    >>> print(str(astr))
    \ar@/^12mm/[dr]_{f}

    ``looping_start`` and ``looping_end`` are currently only used for
    loop morphisms, those which have the same domain and codomain.
    These two attributes should store a valid Xy-pic direction and
    specify, correspondingly, the direction the arrow gets out into
    and the direction the arrow gets back from:

    >>> astr = ArrowStringDescription(
    ... unit="mm", curving=None, curving_amount=None,
    ... looping_start="u", looping_end="l", horizontal_direction="",
    ... vertical_direction="", label_position="_", label="f")
    >>> print(str(astr))
    \ar@(u,l)[]_{f}

    ``label_displacement`` controls how far the arrow label is from
    the ends of the arrow.  For example, to position the arrow label
    near the arrow head, use ">":

    >>> astr = ArrowStringDescription(
    ... unit="mm", curving="^", curving_amount=12,
    ... looping_start=None, looping_end=None, horizontal_direction="d",
    ... vertical_direction="r", label_position="_", label="f")
    >>> astr.label_displacement = ">"
    >>> print(str(astr))
    \ar@/^12mm/[dr]_>{f}

    Finally, ``arrow_style`` is used to specify the arrow style.  To
    get a dashed arrow, for example, use "{-->}" as arrow style:

    >>> astr = ArrowStringDescription(
    ... unit="mm", curving="^", curving_amount=12,
    ... looping_start=None, looping_end=None, horizontal_direction="d",
    ... vertical_direction="r", label_position="_", label="f")
    >>> astr.arrow_style = "{-->}"
    >>> print(str(astr))
    \ar@/^12mm/@{-->}[dr]_{f}

    Notes
    =====

    Instances of :class:`ArrowStringDescription` will be constructed
    by :class:`XypicDiagramDrawer` and provided for further use in
    formatters.  The user is not expected to construct instances of
    :class:`ArrowStringDescription` themselves.

    To be able to properly utilise this class, the reader is encouraged
    to checkout the Xy-pic user guide, available at [Xypic].

    See Also
    ========

    XypicDiagramDrawer

    References
    ==========

    .. [Xypic] https://xy-pic.sourceforge.net/
    """
    def __init__(self, unit, curving, curving_amount, looping_start,
                 looping_end, horizontal_direction, vertical_direction,
                 label_position, label):
        self.unit = unit
        self.curving = curving
        self.curving_amount = curving_amount
        self.looping_start = looping_start
        self.looping_end = looping_end
        self.horizontal_direction = horizontal_direction
        self.vertical_direction = vertical_direction
        self.label_position = label_position
        self.label = label

        self.label_displacement = ""
        self.arrow_style = ""

        # This flag shows that the position of the label of this
        # morphism was set while typesetting a curved morphism and
        # should not be modified later.
        self.forced_label_position = False

    def __str__(self):
        if self.curving:
            curving_str = "@/%s%d%s/" % (self.curving, self.curving_amount,
                                         self.unit)
        else:
            curving_str = ""

        if self.looping_start and self.looping_end:
            looping_str = "@(%s,%s)" % (self.looping_start, self.looping_end)
        else:
            looping_str = ""

        if self.arrow_style:

            style_str = "@" + self.arrow_style
        else:
            style_str = ""

        return "\\ar%s%s%s[%s%s]%s%s{%s}" % \
               (curving_str, looping_str, style_str, self.horizontal_direction,
                self.vertical_direction, self.label_position,
                self.label_displacement, self.label)


class XypicDiagramDrawer:
    r"""
    Given a :class:`~.Diagram` and the corresponding
    :class:`DiagramGrid`, produces the Xy-pic representation of the
    diagram.

    The most important method in this class is ``draw``.  Consider the
    following triangle diagram:

    >>> from sympy.categories import Object, NamedMorphism, Diagram
    >>> from sympy.categories import DiagramGrid, XypicDiagramDrawer
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> diagram = Diagram([f, g], {g * f: "unique"})

    To draw this diagram, its objects need to be laid out with a
    :class:`DiagramGrid`::

    >>> grid = DiagramGrid(diagram)

    Finally, the drawing:

    >>> drawer = XypicDiagramDrawer()
    >>> print(drawer.draw(diagram, grid))
    \xymatrix{
    A \ar[d]_{g\circ f} \ar[r]^{f} & B \ar[ld]^{g} \\
    C &
    }

    For further details see the docstring of this method.

    To control the appearance of the arrows, formatters are used.  The
    dictionary ``arrow_formatters`` maps morphisms to formatter
    functions.  A formatter is accepts an
    :class:`ArrowStringDescription` and is allowed to modify any of
    the arrow properties exposed thereby.  For example, to have all
    morphisms with the property ``unique`` appear as dashed arrows,
    and to have their names prepended with `\exists !`, the following
    should be done:

    >>> def formatter(astr):
    ...   astr.label = r"\exists !" + astr.label
    ...   astr.arrow_style = "{-->}"
    >>> drawer.arrow_formatters["unique"] = formatter
    >>> print(drawer.draw(diagram, grid))
    \xymatrix{
    A \ar@{-->}[d]_{\exists !g\circ f} \ar[r]^{f} & B \ar[ld]^{g} \\
    C &
    }

    To modify the appearance of all arrows in the diagram, set
    ``default_arrow_formatter``.  For example, to place all morphism
    labels a little bit farther from the arrow head so that they look
    more centred, do as follows:

    >>> def default_formatter(astr):
    ...   astr.label_displacement = "(0.45)"
    >>> drawer.default_arrow_formatter = default_formatter
    >>> print(drawer.draw(diagram, grid))
    \xymatrix{
    A \ar@{-->}[d]_(0.45){\exists !g\circ f} \ar[r]^(0.45){f} & B \ar[ld]^(0.45){g} \\
    C &
    }

    In some diagrams some morphisms are drawn as curved arrows.
    Consider the following diagram:

    >>> D = Object("D")
    >>> E = Object("E")
    >>> h = NamedMorphism(D, A, "h")
    >>> k = NamedMorphism(D, B, "k")
    >>> diagram = Diagram([f, g, h, k])
    >>> grid = DiagramGrid(diagram)
    >>> drawer = XypicDiagramDrawer()
    >>> print(drawer.draw(diagram, grid))
    \xymatrix{
    A \ar[r]_{f} & B \ar[d]^{g} & D \ar[l]^{k} \ar@/_3mm/[ll]_{h} \\
    & C &
    }

    To control how far the morphisms are curved by default, one can
    use the ``unit`` and ``default_curving_amount`` attributes:

    >>> drawer.unit = "cm"
    >>> drawer.default_curving_amount = 1
    >>> print(drawer.draw(diagram, grid))
    \xymatrix{
    A \ar[r]_{f} & B \ar[d]^{g} & D \ar[l]^{k} \ar@/_1cm/[ll]_{h} \\
    & C &
    }

    In some diagrams, there are multiple curved morphisms between the
    same two objects.  To control by how much the curving changes
    between two such successive morphisms, use
    ``default_curving_step``:

    >>> drawer.default_curving_step = 1
    >>> h1 = NamedMorphism(A, D, "h1")
    >>> diagram = Diagram([f, g, h, k, h1])
    >>> grid = DiagramGrid(diagram)
    >>> print(drawer.draw(diagram, grid))
    \xymatrix{
    A \ar[r]_{f} \ar@/^1cm/[rr]^{h_{1}} & B \ar[d]^{g} & D \ar[l]^{k} \ar@/_2cm/[ll]_{h} \\
    & C &
    }

    The default value of ``default_curving_step`` is 4 units.

    See Also
    ========

    draw, ArrowStringDescription
    """
    def __init__(self):
        self.unit = "mm"
        self.default_curving_amount = 3
        self.default_curving_step = 4

        # This dictionary maps properties to the corresponding arrow
        # formatters.
        self.arrow_formatters = {}

        # This is the default arrow formatter which will be applied to
        # each arrow independently of its properties.
        self.default_arrow_formatter = None

    @staticmethod
    def _process_loop_morphism(i, j, grid, morphisms_str_info, object_coords):
        """
        Produces the information required for constructing the string
        representation of a loop morphism.  This function is invoked
        from ``_process_morphism``.

        See Also
        ========

        _process_morphism
        """
        curving = ""
        label_pos = "^"
        looping_start = ""
        looping_end = ""

        # This is a loop morphism.  Count how many morphisms stick
        # in each of the four quadrants.  Note that straight
        # vertical and horizontal morphisms count in two quadrants
        # at the same time (i.e., a morphism going up counts both
        # in the first and the second quadrants).

        # The usual numbering (counterclockwise) of quadrants
        # applies.
        quadrant = [0, 0, 0, 0]

        obj = grid[i, j]

        for m, m_str_info in morphisms_str_info.items():
            if (m.domain == obj) and (m.codomain == obj):
                # That's another loop morphism.  Check how it
                # loops and mark the corresponding quadrants as
                # busy.
                (l_s, l_e) = (m_str_info.looping_start, m_str_info.looping_end)

                if (l_s, l_e) == ("r", "u"):
                    quadrant[0] += 1
                elif (l_s, l_e) == ("u", "l"):
                    quadrant[1] += 1
                elif (l_s, l_e) == ("l", "d"):
                    quadrant[2] += 1
                elif (l_s, l_e) == ("d", "r"):
                    quadrant[3] += 1

                continue
            if m.domain == obj:
                (end_i, end_j) = object_coords[m.codomain]
                goes_out = True
            elif m.codomain == obj:
                (end_i, end_j) = object_coords[m.domain]
                goes_out = False
            else:
                continue

            d_i = end_i - i
            d_j = end_j - j
            m_curving = m_str_info.curving

            if (d_i != 0) and (d_j != 0):
                # This is really a diagonal morphism.  Detect the
                # quadrant.
                if (d_i > 0) and (d_j > 0):
                    quadrant[0] += 1
                elif (d_i > 0) and (d_j < 0):
                    quadrant[1] += 1
                elif (d_i < 0) and (d_j < 0):
                    quadrant[2] += 1
                elif (d_i < 0) and (d_j > 0):
                    quadrant[3] += 1
            elif d_i == 0:
                # Knowing where the other end of the morphism is
                # and which way it goes, we now have to decide
                # which quadrant is now the upper one and which is
                # the lower one.
                if d_j > 0:
                    if goes_out:
                        upper_quadrant = 0
                        lower_quadrant = 3
                    else:
                        upper_quadrant = 3
                        lower_quadrant = 0
                else:
                    if goes_out:
                        upper_quadrant = 2
                        lower_quadrant = 1
                    else:
                        upper_quadrant = 1
                        lower_quadrant = 2

                if m_curving:
                    if m_curving == "^":
                        quadrant[upper_quadrant] += 1
                    elif m_curving == "_":
                        quadrant[lower_quadrant] += 1
                else:
                    # This morphism counts in both upper and lower
                    # quadrants.
                    quadrant[upper_quadrant] += 1
                    quadrant[lower_quadrant] += 1
            elif d_j == 0:
                # Knowing where the other end of the morphism is
                # and which way it goes, we now have to decide
                # which quadrant is now the left one and which is
                # the right one.
                if d_i < 0:
                    if goes_out:
                        left_quadrant = 1
                        right_quadrant = 0
                    else:
                        left_quadrant = 0
                        right_quadrant = 1
                else:
                    if goes_out:
                        left_quadrant = 3
                        right_quadrant = 2
                    else:
                        left_quadrant = 2
                        right_quadrant = 3

                if m_curving:
                    if m_curving == "^":
                        quadrant[left_quadrant] += 1
                    elif m_curving == "_":
                        quadrant[right_quadrant] += 1
                else:
                    # This morphism counts in both upper and lower
                    # quadrants.
                    quadrant[left_quadrant] += 1
                    quadrant[right_quadrant] += 1

        # Pick the freest quadrant to curve our morphism into.
        freest_quadrant = 0
        for i in range(4):
            if quadrant[i] < quadrant[freest_quadrant]:
                freest_quadrant = i

        # Now set up proper looping.
        (looping_start, looping_end) = [("r", "u"), ("u", "l"), ("l", "d"),
                                        ("d", "r")][freest_quadrant]

        return (curving, label_pos, looping_start, looping_end)

    @staticmethod
    def _process_horizontal_morphism(i, j, target_j, grid, morphisms_str_info,
                                     object_coords):
        """
        Produces the information required for constructing the string
        representation of a horizontal morphism.  This function is
        invoked from ``_process_morphism``.

        See Also
        ========

        _process_morphism
        """
        # The arrow is horizontal.  Check if it goes from left to
        # right (``backwards == False``) or from right to left
        # (``backwards == True``).
        backwards = False
        start = j
        end = target_j
        if end < start:
            (start, end) = (end, start)
            backwards = True

        # Let's see which objects are there between ``start`` and
        # ``end``, and then count how many morphisms stick out
        # upwards, and how many stick out downwards.
        #
        # For example, consider the situation:
        #
        #    B1 C1
        #    |  |
        # A--B--C--D
        #    |
        #    B2
        #
        # Between the objects `A` and `D` there are two objects:
        # `B` and `C`.  Further, there are two morphisms which
        # stick out upward (the ones between `B1` and `B` and
        # between `C` and `C1`) and one morphism which sticks out
        # downward (the one between `B and `B2`).
        #
        # We need this information to decide how to curve the
        # arrow between `A` and `D`.  First of all, since there
        # are two objects between `A` and `D``, we must curve the
        # arrow.  Then, we will have it curve downward, because
        # there is more space (less morphisms stick out downward
        # than upward).
        up = []
        down = []
        straight_horizontal = []
        for k in range(start + 1, end):
            obj = grid[i, k]
            if not obj:
                continue

            for m in morphisms_str_info:
                if m.domain == obj:
                    (end_i, end_j) = object_coords[m.codomain]
                elif m.codomain == obj:
                    (end_i, end_j) = object_coords[m.domain]
                else:
                    continue

                if end_i > i:
                    down.append(m)
                elif end_i < i:
                    up.append(m)
                elif not morphisms_str_info[m].curving:
                    # This is a straight horizontal morphism,
                    # because it has no curving.
                    straight_horizontal.append(m)

        if len(up) < len(down):
            # More morphisms stick out downward than upward, let's
            # curve the morphism up.
            if backwards:
                curving = "_"
                label_pos = "_"
            else:
                curving = "^"
                label_pos = "^"

            # Assure that the straight horizontal morphisms have
            # their labels on the lower side of the arrow.
            for m in straight_horizontal:
                (i1, j1) = object_coords[m.domain]
                (i2, j2) = object_coords[m.codomain]

                m_str_info = morphisms_str_info[m]
                if j1 < j2:
                    m_str_info.label_position = "_"
                else:
                    m_str_info.label_position = "^"

                # Don't allow any further modifications of the
                # position of this label.
                m_str_info.forced_label_position = True
        else:
            # More morphisms stick out downward than upward, let's
            # curve the morphism up.
            if backwards:
                curving = "^"
                label_pos = "^"
            else:
                curving = "_"
                label_pos = "_"

            # Assure that the straight horizontal morphisms have
            # their labels on the upper side of the arrow.
            for m in straight_horizontal:
                (i1, j1) = object_coords[m.domain]
                (i2, j2) = object_coords[m.codomain]

                m_str_info = morphisms_str_info[m]
                if j1 < j2:
                    m_str_info.label_position = "^"
                else:
                    m_str_info.label_position = "_"

                # Don't allow any further modifications of the
                # position of this label.
                m_str_info.forced_label_position = True

        return (curving, label_pos)

    @staticmethod
    def _process_vertical_morphism(i, j, target_i, grid, morphisms_str_info,
                                   object_coords):
        """
        Produces the information required for constructing the string
        representation of a vertical morphism.  This function is
        invoked from ``_process_morphism``.

        See Also
        ========

        _process_morphism
        """
        # This arrow is vertical.  Check if it goes from top to
        # bottom (``backwards == False``) or from bottom to top
        # (``backwards == True``).
        backwards = False
        start = i
        end = target_i
        if end < start:
            (start, end) = (end, start)
            backwards = True

        # Let's see which objects are there between ``start`` and
        # ``end``, and then count how many morphisms stick out to
        # the left, and how many stick out to the right.
        #
        # See the corresponding comment in the previous branch of
        # this if-statement for more details.
        left = []
        right = []
        straight_vertical = []
        for k in range(start + 1, end):
            obj = grid[k, j]
            if not obj:
                continue

            for m in morphisms_str_info:
                if m.domain == obj:
                    (end_i, end_j) = object_coords[m.codomain]
                elif m.codomain == obj:
                    (end_i, end_j) = object_coords[m.domain]
                else:
                    continue

                if end_j > j:
                    right.append(m)
                elif end_j < j:
                    left.append(m)
                elif not morphisms_str_info[m].curving:
                    # This is a straight vertical morphism,
                    # because it has no curving.
                    straight_vertical.append(m)

        if len(left) < len(right):
            # More morphisms stick out to the left than to the
            # right, let's curve the morphism to the right.
            if backwards:
                curving = "^"
                label_pos = "^"
            else:
                curving = "_"
                label_pos = "_"

            # Assure that the straight vertical morphisms have
            # their labels on the left side of the arrow.
            for m in straight_vertical:
                (i1, j1) = object_coords[m.domain]
                (i2, j2) = object_coords[m.codomain]

                m_str_info = morphisms_str_info[m]
                if i1 < i2:
                    m_str_info.label_position = "^"
                else:
                    m_str_info.label_position = "_"

                # Don't allow any further modifications of the
                # position of this label.
                m_str_info.forced_label_position = True
        else:
            # More morphisms stick out to the right than to the
            # left, let's curve the morphism to the left.
            if backwards:
                curving = "_"
                label_pos = "_"
            else:
                curving = "^"
                label_pos = "^"

            # Assure that the straight vertical morphisms have
            # their labels on the right side of the arrow.
            for m in straight_vertical:
                (i1, j1) = object_coords[m.domain]
                (i2, j2) = object_coords[m.codomain]

                m_str_info = morphisms_str_info[m]
                if i1 < i2:
                    m_str_info.label_position = "_"
                else:
                    m_str_info.label_position = "^"

                # Don't allow any further modifications of the
                # position of this label.
                m_str_info.forced_label_position = True

        return (curving, label_pos)

    def _process_morphism(self, diagram, grid, morphism, object_coords,
                          morphisms, morphisms_str_info):
        """
        Given the required information, produces the string
        representation of ``morphism``.
        """
        def repeat_string_cond(times, str_gt, str_lt):
            """
            If ``times > 0``, repeats ``str_gt`` ``times`` times.
            Otherwise, repeats ``str_lt`` ``-times`` times.
            """
            if times > 0:
                return str_gt * times
            else:
                return str_lt * (-times)

        def count_morphisms_undirected(A, B):
            """
            Counts how many processed morphisms there are between the
            two supplied objects.
            """
            return len([m for m in morphisms_str_info
                        if {m.domain, m.codomain} == {A, B}])

        def count_morphisms_filtered(dom, cod, curving):
            """
            Counts the processed morphisms which go out of ``dom``
            into ``cod`` with curving ``curving``.
            """
            return len([m for m, m_str_info in morphisms_str_info.items()
                        if (m.domain, m.codomain) == (dom, cod) and
                        (m_str_info.curving == curving)])

        (i, j) = object_coords[morphism.domain]
        (target_i, target_j) = object_coords[morphism.codomain]

        # We now need to determine the direction of
        # the arrow.
        delta_i = target_i - i
        delta_j = target_j - j
        vertical_direction = repeat_string_cond(delta_i,
                                                "d", "u")
        horizontal_direction = repeat_string_cond(delta_j,
                                                  "r", "l")

        curving = ""
        label_pos = "^"
        looping_start = ""
        looping_end = ""

        if (delta_i == 0) and (delta_j == 0):
            # This is a loop morphism.
            (curving, label_pos, looping_start,
             looping_end) = XypicDiagramDrawer._process_loop_morphism(
                 i, j, grid, morphisms_str_info, object_coords)
        elif (delta_i == 0) and (abs(j - target_j) > 1):
            # This is a horizontal morphism.
            (curving, label_pos) = XypicDiagramDrawer._process_horizontal_morphism(
                i, j, target_j, grid, morphisms_str_info, object_coords)
        elif (delta_j == 0) and (abs(i - target_i) > 1):
            # This is a vertical morphism.
            (curving, label_pos) = XypicDiagramDrawer._process_vertical_morphism(
                i, j, target_i, grid, morphisms_str_info, object_coords)

        count = count_morphisms_undirected(morphism.domain, morphism.codomain)
        curving_amount = ""
        if curving:
            # This morphisms should be curved anyway.
            curving_amount = self.default_curving_amount + count * \
                self.default_curving_step
        elif count:
            # There are no objects between the domain and codomain of
            # the current morphism, but this is not there already are
            # some morphisms with the same domain and codomain, so we
            # have to curve this one.
            curving = "^"
            filtered_morphisms = count_morphisms_filtered(
                morphism.domain, morphism.codomain, curving)
            curving_amount = self.default_curving_amount + \
                filtered_morphisms * \
                self.default_curving_step

        # Let's now get the name of the morphism.
        morphism_name = ""
        if isinstance(morphism, IdentityMorphism):
            morphism_name = "id_{%s}" + latex(grid[i, j])
        elif isinstance(morphism, CompositeMorphism):
            component_names = [latex(Symbol(component.name)) for
                               component in morphism.components]
            component_names.reverse()
            morphism_name = "\\circ ".join(component_names)
        elif isinstance(morphism, NamedMorphism):
            morphism_name = latex(Symbol(morphism.name))

        return ArrowStringDescription(
            self.unit, curving, curving_amount, looping_start,
            looping_end, horizontal_direction, vertical_direction,
            label_pos, morphism_name)

    @staticmethod
    def _check_free_space_horizontal(dom_i, dom_j, cod_j, grid):
        """
        For a horizontal morphism, checks whether there is free space
        (i.e., space not occupied by any objects) above the morphism
        or below it.
        """
        if dom_j < cod_j:
            (start, end) = (dom_j, cod_j)
            backwards = False
        else:
            (start, end) = (cod_j, dom_j)
            backwards = True

        # Check for free space above.
        if dom_i == 0:
            free_up = True
        else:
            free_up = all(grid[dom_i - 1, j] for j in
                          range(start, end + 1))

        # Check for free space below.
        if dom_i == grid.height - 1:
            free_down = True
        else:
            free_down = not any(grid[dom_i + 1, j] for j in
                                range(start, end + 1))

        return (free_up, free_down, backwards)

    @staticmethod
    def _check_free_space_vertical(dom_i, cod_i, dom_j, grid):
        """
        For a vertical morphism, checks whether there is free space
        (i.e., space not occupied by any objects) to the left of the
        morphism or to the right of it.
        """
        if dom_i < cod_i:
            (start, end) = (dom_i, cod_i)
            backwards = False
        else:
            (start, end) = (cod_i, dom_i)
            backwards = True

        # Check if there's space to the left.
        if dom_j == 0:
            free_left = True
        else:
            free_left = not any(grid[i, dom_j - 1] for i in
                                range(start, end + 1))

        if dom_j == grid.width - 1:
            free_right = True
        else:
            free_right = not any(grid[i, dom_j + 1] for i in
                                 range(start, end + 1))

        return (free_left, free_right, backwards)

    @staticmethod
    def _check_free_space_diagonal(dom_i, cod_i, dom_j, cod_j, grid):
        """
        For a diagonal morphism, checks whether there is free space
        (i.e., space not occupied by any objects) above the morphism
        or below it.
        """
        def abs_xrange(start, end):
            if start < end:
                return range(start, end + 1)
            else:
                return range(end, start + 1)

        if dom_i < cod_i and dom_j < cod_j:
            # This morphism goes from top-left to
            # bottom-right.
            (start_i, start_j) = (dom_i, dom_j)
            (end_i, end_j) = (cod_i, cod_j)
            backwards = False
        elif dom_i > cod_i and dom_j > cod_j:
            # This morphism goes from bottom-right to
            # top-left.
            (start_i, start_j) = (cod_i, cod_j)
            (end_i, end_j) = (dom_i, dom_j)
            backwards = True
        if dom_i < cod_i and dom_j > cod_j:
            # This morphism goes from top-right to
            # bottom-left.
            (start_i, start_j) = (dom_i, dom_j)
            (end_i, end_j) = (cod_i, cod_j)
            backwards = True
        elif dom_i > cod_i and dom_j < cod_j:
            # This morphism goes from bottom-left to
            # top-right.
            (start_i, start_j) = (cod_i, cod_j)
            (end_i, end_j) = (dom_i, dom_j)
            backwards = False

        # This is an attempt at a fast and furious strategy to
        # decide where there is free space on the two sides of
        # a diagonal morphism.  For a diagonal morphism
        # starting at ``(start_i, start_j)`` and ending at
        # ``(end_i, end_j)`` the rectangle defined by these
        # two points is considered.  The slope of the diagonal
        # ``alpha`` is then computed.  Then, for every cell
        # ``(i, j)`` within the rectangle, the slope
        # ``alpha1`` of the line through ``(start_i,
        # start_j)`` and ``(i, j)`` is considered.  If
        # ``alpha1`` is between 0 and ``alpha``, the point
        # ``(i, j)`` is above the diagonal, if ``alpha1`` is
        # between ``alpha`` and infinity, the point is below
        # the diagonal.  Also note that, with some beforehand
        # precautions, this trick works for both the main and
        # the secondary diagonals of the rectangle.

        # I have considered the possibility to only follow the
        # shorter diagonals immediately above and below the
        # main (or secondary) diagonal.  This, however,
        # wouldn't have resulted in much performance gain or
        # better detection of outer edges, because of
        # relatively small sizes of diagram grids, while the
        # code would have become harder to understand.

        alpha = float(end_i - start_i)/(end_j - start_j)
        free_up = True
        free_down = True
        for i in abs_xrange(start_i, end_i):
            if not free_up and not free_down:
                break

            for j in abs_xrange(start_j, end_j):
                if not free_up and not free_down:
                    break

                if (i, j) == (start_i, start_j):
                    continue

                if j == start_j:
                    alpha1 = "inf"
                else:
                    alpha1 = float(i - start_i)/(j - start_j)

                if grid[i, j]:
                    if (alpha1 == "inf") or (abs(alpha1) > abs(alpha)):
                        free_down = False
                    elif abs(alpha1) < abs(alpha):
                        free_up = False

        return (free_up, free_down, backwards)

    def _push_labels_out(self, morphisms_str_info, grid, object_coords):
        """
        For all straight morphisms which form the visual boundary of
        the laid out diagram, puts their labels on their outer sides.
        """
        def set_label_position(free1, free2, pos1, pos2, backwards, m_str_info):
            """
            Given the information about room available to one side and
            to the other side of a morphism (``free1`` and ``free2``),
            sets the position of the morphism label in such a way that
            it is on the freer side.  This latter operations involves
            choice between ``pos1`` and ``pos2``, taking ``backwards``
            in consideration.

            Thus this function will do nothing if either both ``free1
            == True`` and ``free2 == True`` or both ``free1 == False``
            and ``free2 == False``.  In either case, choosing one side
            over the other presents no advantage.
            """
            if backwards:
                (pos1, pos2) = (pos2, pos1)

            if free1 and not free2:
                m_str_info.label_position = pos1
            elif free2 and not free1:
                m_str_info.label_position = pos2

        for m, m_str_info in morphisms_str_info.items():
            if m_str_info.curving or m_str_info.forced_label_position:
                # This is either a curved morphism, and curved
                # morphisms have other magic, or the position of this
                # label has already been fixed.
                continue

            if m.domain == m.codomain:
                # This is a loop morphism, their labels, again have a
                # different magic.
                continue

            (dom_i, dom_j) = object_coords[m.domain]
            (cod_i, cod_j) = object_coords[m.codomain]

            if dom_i == cod_i:
                # Horizontal morphism.
                (free_up, free_down,
                 backwards) = XypicDiagramDrawer._check_free_space_horizontal(
                     dom_i, dom_j, cod_j, grid)

                set_label_position(free_up, free_down, "^", "_",
                                   backwards, m_str_info)
            elif dom_j == cod_j:
                # Vertical morphism.
                (free_left, free_right,
                 backwards) = XypicDiagramDrawer._check_free_space_vertical(
                     dom_i, cod_i, dom_j, grid)

                set_label_position(free_left, free_right, "_", "^",
                                   backwards, m_str_info)
            else:
                # A diagonal morphism.
                (free_up, free_down,
                 backwards) = XypicDiagramDrawer._check_free_space_diagonal(
                     dom_i, cod_i, dom_j, cod_j, grid)

                set_label_position(free_up, free_down, "^", "_",
                                   backwards, m_str_info)

    @staticmethod
    def _morphism_sort_key(morphism, object_coords):
        """
        Provides a morphism sorting key such that horizontal or
        vertical morphisms between neighbouring objects come
        first, then horizontal or vertical morphisms between more
        far away objects, and finally, all other morphisms.
        """
        (i, j) = object_coords[morphism.domain]
        (target_i, target_j) = object_coords[morphism.codomain]

        if morphism.domain == morphism.codomain:
            # Loop morphisms should get after diagonal morphisms
            # so that the proper direction in which to curve the
            # loop can be determined.
            return (3, 0, default_sort_key(morphism))

        if target_i == i:
            return (1, abs(target_j - j), default_sort_key(morphism))

        if target_j == j:
            return (1, abs(target_i - i), default_sort_key(morphism))

        # Diagonal morphism.
        return (2, 0, default_sort_key(morphism))

    @staticmethod
    def _build_xypic_string(diagram, grid, morphisms,
                            morphisms_str_info, diagram_format):
        """
        Given a collection of :class:`ArrowStringDescription`
        describing the morphisms of a diagram and the object layout
        information of a diagram, produces the final Xy-pic picture.
        """
        # Build the mapping between objects and morphisms which have
        # them as domains.
        object_morphisms = {}
        for obj in diagram.objects:
            object_morphisms[obj] = []
        for morphism in morphisms:
            object_morphisms[morphism.domain].append(morphism)

        result = "\\xymatrix%s{\n" % diagram_format

        for i in range(grid.height):
            for j in range(grid.width):
                obj = grid[i, j]
                if obj:
                    result += latex(obj) + " "

                    morphisms_to_draw = object_morphisms[obj]
                    for morphism in morphisms_to_draw:
                        result += str(morphisms_str_info[morphism]) + " "

                # Don't put the & after the last column.
                if j < grid.width - 1:
                    result += "& "

            # Don't put the line break after the last row.
            if i < grid.height - 1:
                result += "\\\\"
            result += "\n"

        result += "}\n"

        return result

    def draw(self, diagram, grid, masked=None, diagram_format=""):
        r"""
        Returns the Xy-pic representation of ``diagram`` laid out in
        ``grid``.

        Consider the following simple triangle diagram.

        >>> from sympy.categories import Object, NamedMorphism, Diagram
        >>> from sympy.categories import DiagramGrid, XypicDiagramDrawer
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> diagram = Diagram([f, g], {g * f: "unique"})

        To draw this diagram, its objects need to be laid out with a
        :class:`DiagramGrid`::

        >>> grid = DiagramGrid(diagram)

        Finally, the drawing:

        >>> drawer = XypicDiagramDrawer()
        >>> print(drawer.draw(diagram, grid))
        \xymatrix{
        A \ar[d]_{g\circ f} \ar[r]^{f} & B \ar[ld]^{g} \\
        C &
        }

        The argument ``masked`` can be used to skip morphisms in the
        presentation of the diagram:

        >>> print(drawer.draw(diagram, grid, masked=[g * f]))
        \xymatrix{
        A \ar[r]^{f} & B \ar[ld]^{g} \\
        C &
        }

        Finally, the ``diagram_format`` argument can be used to
        specify the format string of the diagram.  For example, to
        increase the spacing by 1 cm, proceeding as follows:

        >>> print(drawer.draw(diagram, grid, diagram_format="@+1cm"))
        \xymatrix@+1cm{
        A \ar[d]_{g\circ f} \ar[r]^{f} & B \ar[ld]^{g} \\
        C &
        }

        """
        # This method works in several steps.  It starts by removing
        # the masked morphisms, if necessary, and then maps objects to
        # their positions in the grid (coordinate tuples).  Remember
        # that objects are unique in ``Diagram`` and in the layout
        # produced by ``DiagramGrid``, so every object is mapped to a
        # single coordinate pair.
        #
        # The next step is the central step and is concerned with
        # analysing the morphisms of the diagram and deciding how to
        # draw them.  For example, how to curve the arrows is decided
        # at this step.  The bulk of the analysis is implemented in
        # ``_process_morphism``, to the result of which the
        # appropriate formatters are applied.
        #
        # The result of the previous step is a list of
        # ``ArrowStringDescription``.  After the analysis and
        # application of formatters, some extra logic tries to assure
        # better positioning of morphism labels (for example, an
        # attempt is made to avoid the situations when arrows cross
        # labels).  This functionality constitutes the next step and
        # is implemented in ``_push_labels_out``.  Note that label
        # positions which have been set via a formatter are not
        # affected in this step.
        #
        # Finally, at the closing step, the array of
        # ``ArrowStringDescription`` and the layout information
        # incorporated in ``DiagramGrid`` are combined to produce the
        # resulting Xy-pic picture.  This part of code lies in
        # ``_build_xypic_string``.

        if not masked:
            morphisms_props = grid.morphisms
        else:
            morphisms_props = {}
            for m, props in grid.morphisms.items():
                if m in masked:
                    continue
                morphisms_props[m] = props

        # Build the mapping between objects and their position in the
        # grid.
        object_coords = {}
        for i in range(grid.height):
            for j in range(grid.width):
                if grid[i, j]:
                    object_coords[grid[i, j]] = (i, j)

        morphisms = sorted(morphisms_props,
                           key=lambda m: XypicDiagramDrawer._morphism_sort_key(
                               m, object_coords))

        # Build the tuples defining the string representations of
        # morphisms.
        morphisms_str_info = {}
        for morphism in morphisms:
            string_description = self._process_morphism(
                diagram, grid, morphism, object_coords, morphisms,
                morphisms_str_info)

            if self.default_arrow_formatter:
                self.default_arrow_formatter(string_description)

            for prop in morphisms_props[morphism]:
                # prop is a Symbol.  TODO: Find out why.
                if prop.name in self.arrow_formatters:
                    formatter = self.arrow_formatters[prop.name]
                    formatter(string_description)

            morphisms_str_info[morphism] = string_description

        # Reposition the labels a bit.
        self._push_labels_out(morphisms_str_info, grid, object_coords)

        return XypicDiagramDrawer._build_xypic_string(
            diagram, grid, morphisms, morphisms_str_info, diagram_format)


def xypic_draw_diagram(diagram, masked=None, diagram_format="",
                       groups=None, **hints):
    r"""
    Provides a shortcut combining :class:`DiagramGrid` and
    :class:`XypicDiagramDrawer`.  Returns an Xy-pic presentation of
    ``diagram``.  The argument ``masked`` is a list of morphisms which
    will be not be drawn.  The argument ``diagram_format`` is the
    format string inserted after "\xymatrix".  ``groups`` should be a
    set of logical groups.  The ``hints`` will be passed directly to
    the constructor of :class:`DiagramGrid`.

    For more information about the arguments, see the docstrings of
    :class:`DiagramGrid` and ``XypicDiagramDrawer.draw``.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism, Diagram
    >>> from sympy.categories import xypic_draw_diagram
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> diagram = Diagram([f, g], {g * f: "unique"})
    >>> print(xypic_draw_diagram(diagram))
    \xymatrix{
    A \ar[d]_{g\circ f} \ar[r]^{f} & B \ar[ld]^{g} \\
    C &
    }

    See Also
    ========

    XypicDiagramDrawer, DiagramGrid
    """
    grid = DiagramGrid(diagram, groups, **hints)
    drawer = XypicDiagramDrawer()
    return drawer.draw(diagram, grid, masked, diagram_format)


@doctest_depends_on(exe=('latex', 'dvipng'), modules=('pyglet',))
def preview_diagram(diagram, masked=None, diagram_format="", groups=None,
                    output='png', viewer=None, euler=True, **hints):
    """
    Combines the functionality of ``xypic_draw_diagram`` and
    ``sympy.printing.preview``.  The arguments ``masked``,
    ``diagram_format``, ``groups``, and ``hints`` are passed to
    ``xypic_draw_diagram``, while ``output``, ``viewer, and ``euler``
    are passed to ``preview``.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism, Diagram
    >>> from sympy.categories import preview_diagram
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> d = Diagram([f, g], {g * f: "unique"})
    >>> preview_diagram(d)

    See Also
    ========

    XypicDiagramDrawer
    """
    from sympy.printing import preview
    latex_output = xypic_draw_diagram(diagram, masked, diagram_format,
                                      groups, **hints)
    preview(latex_output, output, viewer, euler, ("xypic",))
