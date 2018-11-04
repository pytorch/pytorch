
rem Remove to original folder after script is finished
set ORIGINAL_DIR=%cd%

rem Build folders are 
rem %CAFFE2_ROOT%\build\Debug for Debug and
rem %CAFFE2_ROOT%\build\Release for Release
set CAFFE2_ROOT=%~dp0%..

rem Should build folder be deleted and build start from scratch?
if NOT DEFINED CLEAR_BUILD_FOLDER_AND_START_AGAIN (
    set CLEAR_BUILD_FOLDER_AND_START_AGAIN=0
)

rem Debug build enabled by default
if NOT DEFINED BUILD_DEBUG (
    set BUILD_DEBUG=1
)

rem Release build enabled by default
if NOT DEFINED BUILD_RELEASE (
    set BUILD_RELEASE=1
)

rem msbuild /m: option value
if NOT DEFINED NUM_BUILD_PROC (
    set NUM_BUILD_PROC=6
)

if %CLEAR_BUILD_FOLDER_AND_START_AGAIN% EQU 1 (
    if exist %CAFFE2_ROOT%\build (
        rmdir %CAFFE2_ROOT%\build /s /q
    )
)

rem Visual Studio 14 2015 Win64 is supported by this script
rem Define VC_BIN_ROOT where compiler and vcvars64.bat are located
if NOT DEFINED CMAKE_GENERATOR (
    set CMAKE_GENERATOR="Visual Studio 14 2015 Win64"
    set VC_BIN_ROOT="%VS140COMNTOOLS%..\..\VC\bin\amd64"
)

if %VC_BIN_ROOT% EQU "" (
    echo "Error: VC_BIN_ROOT must be specified"
    echo "Exiting..."
    exit
)

rem Explicitly specifiy x64 compiler to have enough heap space
if %CMAKE_GENERATOR% EQU "Visual Studio 14 2015 Win64" (
    set CMAKE_CXX_COMPILER=%VC_BIN_ROOT%\cl.exe
    set CMAKE_C_COMPILER=%VC_BIN_ROOT%\cl.exe
    set CMAKE_LINKER=%VC_BIN_ROOT%\link.exe
)

rem Now checking, that everything is defined for build...
if %CMAKE_CXX_COMPILER% EQU "" (
    echo "Error: CMAKE_CXX_COMPILER must be specified"
    echo "Exiting..."
    exit
)

if %CMAKE_C_COMPILER% EQU "" (
    echo "Error: CMAKE_C_COMPILER must be specified"
    echo "Exiting..."
    exit
)

if %CMAKE_LINKER% EQU "" (
    echo "Error: CMAKE_LINKER must be specified"
    echo "Exiting..."
    exit
)

rem Install pyyaml for Aten codegen
python -m pip install pyyaml

if not exist %CAFFE2_ROOT%\build (
    mkdir %CAFFE2_ROOT%\build
)

rem Building Debug in %CAFFE2_ROOT%\build\Debug
if %BUILD_DEBUG% EQU 1 (

    if not exist %CAFFE2_ROOT%\build\Debug (
        mkdir %CAFFE2_ROOT%\build\Debug
    )   
    cd %CAFFE2_ROOT%\build\Debug
    
    if %CMAKE_GENERATOR% EQU "Visual Studio 14 2015 Win64" (

        cmake %CAFFE2_ROOT% -G%CMAKE_GENERATOR% -DUSE_OPENCV=OFF ^
                                 -DCMAKE_BUILD_TYPE=Debug ^
                                 -DCMAKE_INSTALL_PREFIX=%CAFFE2_ROOT%\build\Debug\install ^
                                 -DCMAKE_CXX_COMPILER=%CMAKE_CXX_COMPILER% ^
                                 -DCMAKE_C_COMPILER=%CMAKE_C_COMPILER% ^
                                 -DCMAKE_LINKER=%CMAKE_C_COMPILER% ^
                                 -DINCLUDE_EXPERIMENTAL_C10_OPS=OFF ^
                                 -DBUILD_BINARY=ON
                                     
        call %VC_BIN_ROOT%\vcvars64.bat
                
    ) else (
        echo "Error: Script supports only Visual Studio 14 2015 Win64 generator"
        echo "Exiting..."
        cd %ORIGINAL_DIR%
        exit
    )
    
    msbuild /p:Configuration=Debug /p:Platform=x64 /m:%NUM_BUILD_PROC% INSTALL.vcxproj /p:PreferredToolArchitecture=x64
)


rem Building Release in %CAFFE2_ROOT%\build\Release
if %BUILD_RELEASE% EQU 1 (

    if not exist %CAFFE2_ROOT%\build\Release (
        mkdir %CAFFE2_ROOT%\build\Release
    )   
    cd %CAFFE2_ROOT%\build\Release
    
    if %CMAKE_GENERATOR% EQU "Visual Studio 14 2015 Win64" (

        cmake %CAFFE2_ROOT% -G%CMAKE_GENERATOR% -DUSE_OPENCV=OFF ^
                                 -DCMAKE_BUILD_TYPE=Release ^
                                 -DCMAKE_INSTALL_PREFIX=%CAFFE2_ROOT%\build\Release\install ^
                                 -DCMAKE_CXX_COMPILER=%CMAKE_CXX_COMPILER% ^
                                 -DCMAKE_C_COMPILER=%CMAKE_C_COMPILER% ^
                                 -DCMAKE_LINKER=%CMAKE_C_COMPILER% ^
                                 -DINCLUDE_EXPERIMENTAL_C10_OPS=OFF ^
                                 -DBUILD_BINARY=ON
                                     
        call %VC_BIN_ROOT%\vcvars64.bat
                
    ) else (
        echo "Error: Script supports only Visual Studio 14 2015 Win64 generator"
        echo "Exiting..."
        cd %ORIGINAL_DIR%
        exit
    )
    
    msbuild /p:Configuration=Release /p:Platform=x64 /m:%NUM_BUILD_PROC% INSTALL.vcxproj /p:PreferredToolArchitecture=x64
)

cd %ORIGINAL_DIR%