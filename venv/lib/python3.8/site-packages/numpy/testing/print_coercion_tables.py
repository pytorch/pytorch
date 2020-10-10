#!/usr/bin/env python3
"""Prints type-coercion tables for the built-in NumPy types

"""
import numpy as np

# Generic object that can be added, but doesn't do anything else
class GenericObject:
    def __init__(self, v):
        self.v = v

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    dtype = np.dtype('O')

def print_cancast_table(ntypes):
    print('X', end=' ')
    for char in ntypes:
        print(char, end=' ')
    print()
    for row in ntypes:
        print(row, end=' ')
        for col in ntypes:
            print(int(np.can_cast(row, col)), end=' ')
        print()

def print_coercion_table(ntypes, inputfirstvalue, inputsecondvalue, firstarray, use_promote_types=False):
    print('+', end=' ')
    for char in ntypes:
        print(char, end=' ')
    print()
    for row in ntypes:
        if row == 'O':
            rowtype = GenericObject
        else:
            rowtype = np.obj2sctype(row)

        print(row, end=' ')
        for col in ntypes:
            if col == 'O':
                coltype = GenericObject
            else:
                coltype = np.obj2sctype(col)
            try:
                if firstarray:
                    rowvalue = np.array([rowtype(inputfirstvalue)], dtype=rowtype)
                else:
                    rowvalue = rowtype(inputfirstvalue)
                colvalue = coltype(inputsecondvalue)
                if use_promote_types:
                    char = np.promote_types(rowvalue.dtype, colvalue.dtype).char
                else:
                    value = np.add(rowvalue, colvalue)
                    if isinstance(value, np.ndarray):
                        char = value.dtype.char
                    else:
                        char = np.dtype(type(value)).char
            except ValueError:
                char = '!'
            except OverflowError:
                char = '@'
            except TypeError:
                char = '#'
            print(char, end=' ')
        print()


if __name__ == '__main__':
    print("can cast")
    print_cancast_table(np.typecodes['All'])
    print()
    print("In these tables, ValueError is '!', OverflowError is '@', TypeError is '#'")
    print()
    print("scalar + scalar")
    print_coercion_table(np.typecodes['All'], 0, 0, False)
    print()
    print("scalar + neg scalar")
    print_coercion_table(np.typecodes['All'], 0, -1, False)
    print()
    print("array + scalar")
    print_coercion_table(np.typecodes['All'], 0, 0, True)
    print()
    print("array + neg scalar")
    print_coercion_table(np.typecodes['All'], 0, -1, True)
    print()
    print("promote_types")
    print_coercion_table(np.typecodes['All'], 0, 0, False, True)
