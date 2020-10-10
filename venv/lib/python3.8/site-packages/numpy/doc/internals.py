"""
===============
Array Internals
===============

Internal organization of numpy arrays
=====================================

It helps to understand a bit about how numpy arrays are handled under the covers to help understand numpy better. This section will not go into great detail. Those wishing to understand the full details are referred to Travis Oliphant's book "Guide to NumPy".

NumPy arrays consist of two major components, the raw array data (from now on,
referred to as the data buffer), and the information about the raw array data.
The data buffer is typically what people think of as arrays in C or Fortran,
a contiguous (and fixed) block of memory containing fixed sized data items.
NumPy also contains a significant set of data that describes how to interpret
the data in the data buffer. This extra information contains (among other things):

 1) The basic data element's size in bytes
 2) The start of the data within the data buffer (an offset relative to the
    beginning of the data buffer).
 3) The number of dimensions and the size of each dimension
 4) The separation between elements for each dimension (the 'stride'). This
    does not have to be a multiple of the element size
 5) The byte order of the data (which may not be the native byte order)
 6) Whether the buffer is read-only
 7) Information (via the dtype object) about the interpretation of the basic
    data element. The basic data element may be as simple as a int or a float,
    or it may be a compound object (e.g., struct-like), a fixed character field,
    or Python object pointers.
 8) Whether the array is to interpreted as C-order or Fortran-order.

This arrangement allow for very flexible use of arrays. One thing that it allows
is simple changes of the metadata to change the interpretation of the array buffer.
Changing the byteorder of the array is a simple change involving no rearrangement
of the data. The shape of the array can be changed very easily without changing
anything in the data buffer or any data copying at all

Among other things that are made possible is one can create a new array metadata
object that uses the same data buffer
to create a new view of that data buffer that has a different interpretation
of the buffer (e.g., different shape, offset, byte order, strides, etc) but
shares the same data bytes. Many operations in numpy do just this such as
slices. Other operations, such as transpose, don't move data elements
around in the array, but rather change the information about the shape and strides so that the indexing of the array changes, but the data in the doesn't move.

Typically these new versions of the array metadata but the same data buffer are
new 'views' into the data buffer. There is a different ndarray object, but it
uses the same data buffer. This is why it is necessary to force copies through
use of the .copy() method if one really wants to make a new and independent
copy of the data buffer.

New views into arrays mean the object reference counts for the data buffer
increase. Simply doing away with the original array object will not remove the
data buffer if other views of it still exist.

Multidimensional Array Indexing Order Issues
============================================

What is the right way to index
multi-dimensional arrays? Before you jump to conclusions about the one and
true way to index multi-dimensional arrays, it pays to understand why this is
a confusing issue. This section will try to explain in detail how numpy
indexing works and why we adopt the convention we do for images, and when it
may be appropriate to adopt other conventions.

The first thing to understand is
that there are two conflicting conventions for indexing 2-dimensional arrays.
Matrix notation uses the first index to indicate which row is being selected and
the second index to indicate which column is selected. This is opposite the
geometrically oriented-convention for images where people generally think the
first index represents x position (i.e., column) and the second represents y
position (i.e., row). This alone is the source of much confusion;
matrix-oriented users and image-oriented users expect two different things with
regard to indexing.

The second issue to understand is how indices correspond
to the order the array is stored in memory. In Fortran the first index is the
most rapidly varying index when moving through the elements of a two
dimensional array as it is stored in memory. If you adopt the matrix
convention for indexing, then this means the matrix is stored one column at a
time (since the first index moves to the next row as it changes). Thus Fortran
is considered a Column-major language. C has just the opposite convention. In
C, the last index changes most rapidly as one moves through the array as
stored in memory. Thus C is a Row-major language. The matrix is stored by
rows. Note that in both cases it presumes that the matrix convention for
indexing is being used, i.e., for both Fortran and C, the first index is the
row. Note this convention implies that the indexing convention is invariant
and that the data order changes to keep that so.

But that's not the only way
to look at it. Suppose one has large two-dimensional arrays (images or
matrices) stored in data files. Suppose the data are stored by rows rather than
by columns. If we are to preserve our index convention (whether matrix or
image) that means that depending on the language we use, we may be forced to
reorder the data if it is read into memory to preserve our indexing
convention. For example if we read row-ordered data into memory without
reordering, it will match the matrix indexing convention for C, but not for
Fortran. Conversely, it will match the image indexing convention for Fortran,
but not for C. For C, if one is using data stored in row order, and one wants
to preserve the image index convention, the data must be reordered when
reading into memory.

In the end, which you do for Fortran or C depends on
which is more important, not reordering data or preserving the indexing
convention. For large images, reordering data is potentially expensive, and
often the indexing convention is inverted to avoid that.

The situation with
numpy makes this issue yet more complicated. The internal machinery of numpy
arrays is flexible enough to accept any ordering of indices. One can simply
reorder indices by manipulating the internal stride information for arrays
without reordering the data at all. NumPy will know how to map the new index
order to the data without moving the data.

So if this is true, why not choose
the index order that matches what you most expect? In particular, why not define
row-ordered images to use the image convention? (This is sometimes referred
to as the Fortran convention vs the C convention, thus the 'C' and 'FORTRAN'
order options for array ordering in numpy.) The drawback of doing this is
potential performance penalties. It's common to access the data sequentially,
either implicitly in array operations or explicitly by looping over rows of an
image. When that is done, then the data will be accessed in non-optimal order.
As the first index is incremented, what is actually happening is that elements
spaced far apart in memory are being sequentially accessed, with usually poor
memory access speeds. For example, for a two dimensional image 'im' defined so
that im[0, 10] represents the value at x=0, y=10. To be consistent with usual
Python behavior then im[0] would represent a column at x=0. Yet that data
would be spread over the whole array since the data are stored in row order.
Despite the flexibility of numpy's indexing, it can't really paper over the fact
basic operations are rendered inefficient because of data order or that getting
contiguous subarrays is still awkward (e.g., im[:,0] for the first row, vs
im[0]), thus one can't use an idiom such as for row in im; for col in im does
work, but doesn't yield contiguous column data.

As it turns out, numpy is
smart enough when dealing with ufuncs to determine which index is the most
rapidly varying one in memory and uses that for the innermost loop. Thus for
ufuncs there is no large intrinsic advantage to either approach in most cases.
On the other hand, use of .flat with an FORTRAN ordered array will lead to
non-optimal memory access as adjacent elements in the flattened array (iterator,
actually) are not contiguous in memory.

Indeed, the fact is that Python
indexing on lists and other sequences naturally leads to an outside-to inside
ordering (the first index gets the largest grouping, the next the next largest,
and the last gets the smallest element). Since image data are normally stored
by rows, this corresponds to position within rows being the last item indexed.

If you do want to use Fortran ordering realize that
there are two approaches to consider: 1) accept that the first index is just not
the most rapidly changing in memory and have all your I/O routines reorder
your data when going from memory to disk or visa versa, or use numpy's
mechanism for mapping the first index to the most rapidly varying data. We
recommend the former if possible. The disadvantage of the latter is that many
of numpy's functions will yield arrays without Fortran ordering unless you are
careful to use the 'order' keyword. Doing this would be highly inconvenient.

Otherwise we recommend simply learning to reverse the usual order of indices
when accessing elements of an array. Granted, it goes against the grain, but
it is more in line with Python semantics and the natural order of the data.

"""
