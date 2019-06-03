Chrono
======

When including the additional header file :file:`pybind11/chrono.h` conversions
from C++11 chrono datatypes to python datetime objects are automatically enabled.
This header also enables conversions of python floats (often from sources such
as ``time.monotonic()``, ``time.perf_counter()`` and ``time.process_time()``)
into durations.

An overview of clocks in C++11
------------------------------

A point of confusion when using these conversions is the differences between
clocks provided in C++11. There are three clock types defined by the C++11
standard and users can define their own if needed. Each of these clocks have
different properties and when converting to and from python will give different
results.

The first clock defined by the standard is ``std::chrono::system_clock``. This
clock measures the current date and time. However, this clock changes with to
updates to the operating system time. For example, if your time is synchronised
with a time server this clock will change. This makes this clock a poor choice
for timing purposes but good for measuring the wall time.

The second clock defined in the standard is ``std::chrono::steady_clock``.
This clock ticks at a steady rate and is never adjusted. This makes it excellent
for timing purposes, however the value in this clock does not correspond to the
current date and time. Often this clock will be the amount of time your system
has been on, although it does not have to be. This clock will never be the same
clock as the system clock as the system clock can change but steady clocks
cannot.

The third clock defined in the standard is ``std::chrono::high_resolution_clock``.
This clock is the clock that has the highest resolution out of the clocks in the
system. It is normally a typedef to either the system clock or the steady clock
but can be its own independent clock. This is important as when using these
conversions as the types you get in python for this clock might be different
depending on the system.
If it is a typedef of the system clock, python will get datetime objects, but if
it is a different clock they will be timedelta objects.

Provided conversions
--------------------

.. rubric:: C++ to Python

- ``std::chrono::system_clock::time_point`` → ``datetime.datetime``
    System clock times are converted to python datetime instances. They are
    in the local timezone, but do not have any timezone information attached
    to them (they are naive datetime objects).

- ``std::chrono::duration`` → ``datetime.timedelta``
    Durations are converted to timedeltas, any precision in the duration
    greater than microseconds is lost by rounding towards zero.

- ``std::chrono::[other_clocks]::time_point`` → ``datetime.timedelta``
    Any clock time that is not the system clock is converted to a time delta.
    This timedelta measures the time from the clocks epoch to now.

.. rubric:: Python to C++

- ``datetime.datetime`` → ``std::chrono::system_clock::time_point``
    Date/time objects are converted into system clock timepoints. Any
    timezone information is ignored and the type is treated as a naive
    object.

- ``datetime.timedelta`` → ``std::chrono::duration``
    Time delta are converted into durations with microsecond precision.

- ``datetime.timedelta`` → ``std::chrono::[other_clocks]::time_point``
    Time deltas that are converted into clock timepoints are treated as
    the amount of time from the start of the clocks epoch.

- ``float`` → ``std::chrono::duration``
    Floats that are passed to C++ as durations be interpreted as a number of
    seconds. These will be converted to the duration using ``duration_cast``
    from the float.

- ``float`` → ``std::chrono::[other_clocks]::time_point``
    Floats that are passed to C++ as time points will be interpreted as the
    number of seconds from the start of the clocks epoch.
