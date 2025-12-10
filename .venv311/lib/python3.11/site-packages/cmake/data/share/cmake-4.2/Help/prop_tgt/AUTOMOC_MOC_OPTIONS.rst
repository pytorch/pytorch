AUTOMOC_MOC_OPTIONS
-------------------

Additional options for ``moc`` when using :prop_tgt:`AUTOMOC`

This property is only used if the :prop_tgt:`AUTOMOC` property is ``ON``
for this target.  In this case, it holds additional command line
options which will be used when ``moc`` is executed during the build, i.e.
it is equivalent to the optional ``OPTIONS`` argument of the
:module:`qt4_wrap_cpp() <FindQt4>` macro.

This property is initialized by the value of the
:variable:`CMAKE_AUTOMOC_MOC_OPTIONS` variable if it is set when a target
is created, or an empty string otherwise.

See the :manual:`cmake-qt(7)` manual for more information on using CMake
with Qt.

EXAMPLE
^^^^^^^

In this example, the ``moc`` tool is invoked with the ``-D_EXTRA_DEFINE``
option when generating the moc file for ``object.cpp``.

``CMakeLists.txt``
  .. code-block:: cmake

    add_executable(mocOptions object.cpp main.cpp)
    set_property(TARGET mocOptions PROPERTY AUTOMOC ON)
    target_compile_options(mocOptions PRIVATE "-D_EXTRA_DEFINE")
    set_property(TARGET mocOptions PROPERTY AUTOMOC_MOC_OPTIONS "-D_EXTRA_DEFINE")
    target_link_libraries(mocOptions Qt6::Core)

``object.hpp``
  .. code-block:: c++

    #ifndef Object_HPP
    #define Object_HPP

    #include <QObject>

    #ifdef _EXTRA_DEFINE
    class Object : public QObject
    {
    Q_OBJECT
    public:

      Object();

    };
    #endif

    #endif
