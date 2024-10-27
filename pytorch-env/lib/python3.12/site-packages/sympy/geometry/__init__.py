"""
A geometry module for the SymPy library. This module contains all of the
entities and functions needed to construct basic geometrical data and to
perform simple informational queries.

Usage:
======

Examples
========

"""
from sympy.geometry.point import Point, Point2D, Point3D
from sympy.geometry.line import Line, Ray, Segment, Line2D, Segment2D, Ray2D, \
    Line3D, Segment3D, Ray3D
from sympy.geometry.plane import Plane
from sympy.geometry.ellipse import Ellipse, Circle
from sympy.geometry.polygon import Polygon, RegularPolygon, Triangle, rad, deg
from sympy.geometry.util import are_similar, centroid, convex_hull, idiff, \
    intersection, closest_points, farthest_points
from sympy.geometry.exceptions import GeometryError
from sympy.geometry.curve import Curve
from sympy.geometry.parabola import Parabola

__all__ = [
    'Point', 'Point2D', 'Point3D',

    'Line', 'Ray', 'Segment', 'Line2D', 'Segment2D', 'Ray2D', 'Line3D',
    'Segment3D', 'Ray3D',

    'Plane',

    'Ellipse', 'Circle',

    'Polygon', 'RegularPolygon', 'Triangle', 'rad', 'deg',

    'are_similar', 'centroid', 'convex_hull', 'idiff', 'intersection',
    'closest_points', 'farthest_points',

    'GeometryError',

    'Curve',

    'Parabola',
]
