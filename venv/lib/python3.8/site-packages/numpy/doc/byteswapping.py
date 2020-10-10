"""

=============================
 Byteswapping and byte order
=============================

Introduction to byte ordering and ndarrays
==========================================

The ``ndarray`` is an object that provide a python array interface to data
in memory.

It often happens that the memory that you want to view with an array is
not of the same byte ordering as the computer on which you are running
Python.

For example, I might be working on a computer with a little-endian CPU -
such as an Intel Pentium, but I have loaded some data from a file
written by a computer that is big-endian.  Let's say I have loaded 4
bytes from a file written by a Sun (big-endian) computer.  I know that
these 4 bytes represent two 16-bit integers.  On a big-endian machine, a
two-byte integer is stored with the Most Significant Byte (MSB) first,
and then the Least Significant Byte (LSB). Thus the bytes are, in memory order:

#. MSB integer 1
#. LSB integer 1
#. MSB integer 2
#. LSB integer 2

Let's say the two integers were in fact 1 and 770.  Because 770 = 256 *
3 + 2, the 4 bytes in memory would contain respectively: 0, 1, 3, 2.
The bytes I have loaded from the file would have these contents:

>>> big_end_buffer = bytearray([0,1,3,2])
>>> big_end_buffer
bytearray(b'\\x00\\x01\\x03\\x02')

We might want to use an ``ndarray`` to access these integers.  In that
case, we can create an array around this memory, and tell numpy that
there are two integers, and that they are 16 bit and big-endian:

>>> import numpy as np
>>> big_end_arr = np.ndarray(shape=(2,),dtype='>i2', buffer=big_end_buffer)
>>> big_end_arr[0]
1
>>> big_end_arr[1]
770

Note the array ``dtype`` above of ``>i2``.  The ``>`` means 'big-endian'
(``<`` is little-endian) and ``i2`` means 'signed 2-byte integer'.  For
example, if our data represented a single unsigned 4-byte little-endian
integer, the dtype string would be ``<u4``.

In fact, why don't we try that?

>>> little_end_u4 = np.ndarray(shape=(1,),dtype='<u4', buffer=big_end_buffer)
>>> little_end_u4[0] == 1 * 256**1 + 3 * 256**2 + 2 * 256**3
True

Returning to our ``big_end_arr`` - in this case our underlying data is
big-endian (data endianness) and we've set the dtype to match (the dtype
is also big-endian).  However, sometimes you need to flip these around.

.. warning::

    Scalars currently do not include byte order information, so extracting
    a scalar from an array will return an integer in native byte order.
    Hence:

    >>> big_end_arr[0].dtype.byteorder == little_end_u4[0].dtype.byteorder
    True

Changing byte ordering
======================

As you can imagine from the introduction, there are two ways you can
affect the relationship between the byte ordering of the array and the
underlying memory it is looking at:

* Change the byte-ordering information in the array dtype so that it
  interprets the underlying data as being in a different byte order.
  This is the role of ``arr.newbyteorder()``
* Change the byte-ordering of the underlying data, leaving the dtype
  interpretation as it was.  This is what ``arr.byteswap()`` does.

The common situations in which you need to change byte ordering are:

#. Your data and dtype endianness don't match, and you want to change
   the dtype so that it matches the data.
#. Your data and dtype endianness don't match, and you want to swap the
   data so that they match the dtype
#. Your data and dtype endianness match, but you want the data swapped
   and the dtype to reflect this

Data and dtype endianness don't match, change dtype to match data
-----------------------------------------------------------------

We make something where they don't match:

>>> wrong_end_dtype_arr = np.ndarray(shape=(2,),dtype='<i2', buffer=big_end_buffer)
>>> wrong_end_dtype_arr[0]
256

The obvious fix for this situation is to change the dtype so it gives
the correct endianness:

>>> fixed_end_dtype_arr = wrong_end_dtype_arr.newbyteorder()
>>> fixed_end_dtype_arr[0]
1

Note the array has not changed in memory:

>>> fixed_end_dtype_arr.tobytes() == big_end_buffer
True

Data and type endianness don't match, change data to match dtype
----------------------------------------------------------------

You might want to do this if you need the data in memory to be a certain
ordering.  For example you might be writing the memory out to a file
that needs a certain byte ordering.

>>> fixed_end_mem_arr = wrong_end_dtype_arr.byteswap()
>>> fixed_end_mem_arr[0]
1

Now the array *has* changed in memory:

>>> fixed_end_mem_arr.tobytes() == big_end_buffer
False

Data and dtype endianness match, swap data and dtype
----------------------------------------------------

You may have a correctly specified array dtype, but you need the array
to have the opposite byte order in memory, and you want the dtype to
match so the array values make sense.  In this case you just do both of
the previous operations:

>>> swapped_end_arr = big_end_arr.byteswap().newbyteorder()
>>> swapped_end_arr[0]
1
>>> swapped_end_arr.tobytes() == big_end_buffer
False

An easier way of casting the data to a specific dtype and byte ordering
can be achieved with the ndarray astype method:

>>> swapped_end_arr = big_end_arr.astype('<i2')
>>> swapped_end_arr[0]
1
>>> swapped_end_arr.tobytes() == big_end_buffer
False

"""
