"""
Category Theory module.

Provides some of the fundamental category-theory-related classes,
including categories, morphisms, diagrams.  Functors are not
implemented yet.

The general reference work this module tries to follow is

  [JoyOfCats] J. Adamek, H. Herrlich. G. E. Strecker: Abstract and
              Concrete Categories. The Joy of Cats.

The latest version of this book should be available for free download
from

   katmat.math.uni-bremen.de/acc/acc.pdf

"""

from .baseclasses import (Object, Morphism, IdentityMorphism,
                         NamedMorphism, CompositeMorphism, Category,
                         Diagram)

from .diagram_drawing import (DiagramGrid, XypicDiagramDrawer,
                             xypic_draw_diagram, preview_diagram)

__all__ = [
    'Object', 'Morphism', 'IdentityMorphism', 'NamedMorphism',
    'CompositeMorphism', 'Category', 'Diagram',

    'DiagramGrid', 'XypicDiagramDrawer', 'xypic_draw_diagram',
    'preview_diagram',
]
