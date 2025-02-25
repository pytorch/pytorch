@echo off

REM Check for optional components

where /q ninja.exe

IF NOT ERRORLEVEL 1 (
    echo Ninja found, using it to speed up builds
    set CMAKE_GENERATOR=Ninja
)

IF "%USE_SCCACHE%" == "0" goto sccache_end

where /q clcache.exe

IF NOT ERRORLEVEL 1 (
    echo clcache found, using it to speed up builds
    set CC=clcache
    set CXX=clcache
)

where /q sccache-cl.exe

IF NOT ERRORLEVEL 1 (
    echo sccache-cl found, using it to speed up builds
    set CC=sccache-cl
    set CXX=sccache-cl
)

IF "%CC%" == "sccache-cl" IF "%CXX%" == "sccache-cl" goto sccache_end

where /q sccache.exe

IF NOT ERRORLEVEL 1 (
    echo sccache found, using it to speed up builds
    set CC=sccache cl
    set CXX=sccache cl
)

:sccache_end

IF exist "%MKLProductDir%\mkl\lib\intel64_win" (
    echo MKL found, adding it to build
    set "LIB=%MKLProductDir%\mkl\lib\intel64_win;%MKLProductDir%\compiler\lib\intel64_win;%LIB%";
)

exit /b 0
