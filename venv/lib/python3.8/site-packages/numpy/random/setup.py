import os
import platform
import sys
from os.path import join

from numpy.distutils.system_info import platform_bits

is_msvc = (platform.platform().startswith('Windows') and
           platform.python_compiler().startswith('MS'))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_mathlibs
    config = Configuration('random', parent_package, top_path)

    def generate_libraries(ext, build_dir):
        config_cmd = config.get_config_cmd()
        libs = get_mathlibs()
        if sys.platform == 'win32':
            libs.extend(['Advapi32', 'Kernel32'])
        ext.libraries.extend(libs)
        return None

    # enable unix large file support on 32 bit systems
    # (64 bit off_t, lseek -> lseek64 etc.)
    if sys.platform[:3] == "aix":
        defs = [('_LARGE_FILES', None)]
    else:
        defs = [('_FILE_OFFSET_BITS', '64'),
                ('_LARGEFILE_SOURCE', '1'),
                ('_LARGEFILE64_SOURCE', '1')]

    defs.append(('NPY_NO_DEPRECATED_API', 0))
    config.add_subpackage('tests')
    config.add_data_dir('tests/data')
    config.add_data_dir('_examples')

    EXTRA_LINK_ARGS = []
    EXTRA_LIBRARIES = ['npyrandom']
    if os.name != 'nt':
        # Math lib
        EXTRA_LIBRARIES.append('m')
    # Some bit generators exclude GCC inlining
    EXTRA_COMPILE_ARGS = ['-U__GNUC_GNU_INLINE__']

    if is_msvc and platform_bits == 32:
        # 32-bit windows requires explicit sse2 option
        EXTRA_COMPILE_ARGS += ['/arch:SSE2']
    elif not is_msvc:
        # Some bit generators require c99
        EXTRA_COMPILE_ARGS += ['-std=c99']

    # Use legacy integer variable sizes
    LEGACY_DEFS = [('NP_RANDOM_LEGACY', '1')]
    PCG64_DEFS = []
    # One can force emulated 128-bit arithmetic if one wants.
    #PCG64_DEFS += [('PCG_FORCE_EMULATED_128BIT_MATH', '1')]
    depends = ['__init__.pxd', 'c_distributions.pxd', 'bit_generator.pxd']

    # npyrandom - a library like npymath
    npyrandom_sources = [
        'src/distributions/logfactorial.c',
        'src/distributions/distributions.c',
        'src/distributions/random_mvhg_count.c',
        'src/distributions/random_mvhg_marginals.c',
        'src/distributions/random_hypergeometric.c',
    ]
    config.add_installed_library('npyrandom',
        sources=npyrandom_sources,
        install_dir='lib',
        build_info={
            'include_dirs' : [],  # empty list required for creating npyrandom.h
            'extra_compiler_args' : (['/GL-'] if is_msvc else []),
        })

    for gen in ['mt19937']:
        # gen.pyx, src/gen/gen.c, src/gen/gen-jump.c
        config.add_extension(f'_{gen}',
                             sources=[f'_{gen}.c',
                                      f'src/{gen}/{gen}.c',
                                      f'src/{gen}/{gen}-jump.c'],
                             include_dirs=['.', 'src', join('src', gen)],
                             libraries=EXTRA_LIBRARIES,
                             extra_compile_args=EXTRA_COMPILE_ARGS,
                             extra_link_args=EXTRA_LINK_ARGS,
                             depends=depends + [f'_{gen}.pyx'],
                             define_macros=defs,
                             )
    for gen in ['philox', 'pcg64', 'sfc64']:
        # gen.pyx, src/gen/gen.c
        _defs = defs + PCG64_DEFS if gen == 'pcg64' else defs
        config.add_extension(f'_{gen}',
                             sources=[f'_{gen}.c',
                                      f'src/{gen}/{gen}.c'],
                             include_dirs=['.', 'src', join('src', gen)],
                             libraries=EXTRA_LIBRARIES,
                             extra_compile_args=EXTRA_COMPILE_ARGS,
                             extra_link_args=EXTRA_LINK_ARGS,
                             depends=depends + [f'_{gen}.pyx',
                                   'bit_generator.pyx', 'bit_generator.pxd'],
                             define_macros=_defs,
                             )
    for gen in ['_common', 'bit_generator']:
        # gen.pyx
        config.add_extension(gen,
                             sources=[f'{gen}.c'],
                             libraries=EXTRA_LIBRARIES,
                             extra_compile_args=EXTRA_COMPILE_ARGS,
                             extra_link_args=EXTRA_LINK_ARGS,
                             include_dirs=['.', 'src'],
                             depends=depends + [f'{gen}.pyx', f'{gen}.pxd',],
                             define_macros=defs,
                             )
        config.add_data_files(f'{gen}.pxd')
    for gen in ['_generator', '_bounded_integers']:
        # gen.pyx, src/distributions/distributions.c
        config.add_extension(gen,
                             sources=[f'{gen}.c'],
                             libraries=EXTRA_LIBRARIES,
                             extra_compile_args=EXTRA_COMPILE_ARGS,
                             include_dirs=['.', 'src'],
                             extra_link_args=EXTRA_LINK_ARGS,
                             depends=depends + [f'{gen}.pyx'],
                             define_macros=defs,
                             )
    config.add_data_files('_bounded_integers.pxd')
    config.add_extension('mtrand',
                         sources=['mtrand.c',
                                  'src/legacy/legacy-distributions.c',
                                  'src/distributions/distributions.c',
                                 ],
                         include_dirs=['.', 'src', 'src/legacy'],
                         libraries=['m'] if os.name != 'nt' else [],
                         extra_compile_args=EXTRA_COMPILE_ARGS,
                         extra_link_args=EXTRA_LINK_ARGS,
                         depends=depends + ['mtrand.pyx'],
                         define_macros=defs + LEGACY_DEFS,
                         )
    config.add_data_files(*depends)
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(configuration=configuration)
