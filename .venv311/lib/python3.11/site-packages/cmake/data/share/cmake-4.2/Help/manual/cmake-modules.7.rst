.. cmake-manual-description: CMake Modules Reference

cmake-modules(7)
****************

The modules listed here are part of the CMake distribution.
Projects may provide further modules; their location(s)
can be specified in the :variable:`CMAKE_MODULE_PATH` variable.

Utility Modules
===============

These modules are loaded using the :command:`include` command.

.. toctree::
   :maxdepth: 1

   /module/AndroidTestUtilities
   /module/BundleUtilities
   /module/CheckCCompilerFlag
   /module/CheckCompilerFlag
   /module/CheckCSourceCompiles
   /module/CheckCSourceRuns
   /module/CheckCXXCompilerFlag
   /module/CheckCXXSourceCompiles
   /module/CheckCXXSourceRuns
   /module/CheckCXXSymbolExists
   /module/CheckFortranCompilerFlag
   /module/CheckFortranFunctionExists
   /module/CheckFortranSourceCompiles
   /module/CheckFortranSourceRuns
   /module/CheckFunctionExists
   /module/CheckIncludeFile
   /module/CheckIncludeFileCXX
   /module/CheckIncludeFiles
   /module/CheckIPOSupported
   /module/CheckLanguage
   /module/CheckLibraryExists
   /module/CheckLinkerFlag
   /module/CheckOBJCCompilerFlag
   /module/CheckOBJCSourceCompiles
   /module/CheckOBJCSourceRuns
   /module/CheckOBJCXXCompilerFlag
   /module/CheckOBJCXXSourceCompiles
   /module/CheckOBJCXXSourceRuns
   /module/CheckPIESupported
   /module/CheckPrototypeDefinition
   /module/CheckSourceCompiles
   /module/CheckSourceRuns
   /module/CheckStructHasMember
   /module/CheckSymbolExists
   /module/CheckTypeSize
   /module/CheckVariableExists
   /module/CMakeAddFortranSubdirectory
   /module/CMakeBackwardCompatibilityCXX
   /module/CMakeDependentOption
   /module/CMakeFindDependencyMacro
   /module/CMakePackageConfigHelpers
   /module/CMakePrintHelpers
   /module/CMakePrintSystemInformation
   /module/CMakePushCheckState
   /module/CMakeVerifyManifest
   /module/CPack
   /module/CPackComponent
   /module/CPackIFW
   /module/CPackIFWConfigureFile
   /module/CSharpUtilities
   /module/CTest
   /module/CTestCoverageCollectGCOV
   /module/CTestUseLaunchers
   /module/DeployQt4
   /module/ExternalData
   /module/ExternalProject
   /module/FeatureSummary
   /module/FetchContent
   /module/FindPackageHandleStandardArgs
   /module/FindPackageMessage
   /module/FortranCInterface
   /module/GenerateExportHeader
   /module/GNUInstallDirs
   /module/GoogleTest
   /module/InstallRequiredSystemLibraries
   /module/ProcessorCount
   /module/SelectLibraryConfigurations
   /module/TestForANSIForScope
   /module/TestForANSIStreamHeaders
   /module/TestForSSTREAM
   /module/TestForSTDNamespace
   /module/UseEcos
   /module/UseJava
   /module/UseSWIG
   /module/UsewxWidgets

Find Modules
============

These modules search for third-party software.
They are normally called through the :command:`find_package` command.

.. toctree::
   :maxdepth: 1

   /module/FindALSA
   /module/FindArmadillo
   /module/FindASPELL
   /module/FindAVIFile
   /module/FindBacktrace
   /module/FindBISON
   /module/FindBLAS
   /module/FindBullet
   /module/FindBZip2
   /module/FindCoin3D
   /module/FindCUDAToolkit
   /module/FindCups
   /module/FindCURL
   /module/FindCurses
   /module/FindCVS
   /module/FindCxxTest
   /module/FindCygwin
   /module/FindDCMTK
   /module/FindDevIL
   /module/FindDoxygen
   /module/FindEnvModules
   /module/FindEXPAT
   /module/FindFLEX
   /module/FindFLTK
   /module/FindFLTK2
   /module/FindFontconfig
   /module/FindFreetype
   /module/FindGettext
   /module/FindGIF
   /module/FindGit
   /module/FindGLEW
   /module/FindGLUT
   /module/FindGnuplot
   /module/FindGnuTLS
   /module/FindGSL
   /module/FindGTest
   /module/FindGTK
   /module/FindGTK2
   /module/FindHDF5
   /module/FindHg
   /module/FindHSPELL
   /module/FindHTMLHelp
   /module/FindIce
   /module/FindIconv
   /module/FindIcotool
   /module/FindICU
   /module/FindImageMagick
   /module/FindIntl
   /module/FindJasper
   /module/FindJava
   /module/FindJNI
   /module/FindJPEG
   /module/FindKDE3
   /module/FindKDE4
   /module/FindLAPACK
   /module/FindLATEX
   /module/FindLibArchive
   /module/FindLibinput
   /module/FindLibLZMA
   /module/FindLibXml2
   /module/FindLibXslt
   /module/FindLTTngUST
   /module/FindLua
   /module/FindLua50
   /module/FindLua51
   /module/FindMatlab
   /module/FindMFC
   /module/FindMotif
   /module/FindMPEG
   /module/FindMPEG2
   /module/FindMPI
   /module/FindMsys
   /module/FindODBC
   /module/FindOpenACC
   /module/FindOpenAL
   /module/FindOpenCL
   /module/FindOpenGL
   /module/FindOpenMP
   /module/FindOpenSceneGraph
   /module/FindOpenSP
   /module/FindOpenSSL
   /module/FindOpenThreads
   /module/Findosg
   /module/FindosgAnimation
   /module/FindosgDB
   /module/FindosgFX
   /module/FindosgGA
   /module/FindosgIntrospection
   /module/FindosgManipulator
   /module/FindosgParticle
   /module/FindosgPresentation
   /module/FindosgProducer
   /module/FindosgQt
   /module/FindosgShadow
   /module/FindosgSim
   /module/FindosgTerrain
   /module/FindosgText
   /module/FindosgUtil
   /module/FindosgViewer
   /module/FindosgVolume
   /module/FindosgWidget
   /module/FindPatch
   /module/FindPerl
   /module/FindPerlLibs
   /module/FindPHP4
   /module/FindPhysFS
   /module/FindPike
   /module/FindPkgConfig
   /module/FindPNG
   /module/FindPostgreSQL
   /module/FindProducer
   /module/FindProtobuf
   /module/FindPython
   /module/FindPython2
   /module/FindPython3
   /module/FindQt3
   /module/FindQt4
   /module/FindQuickTime
   /module/FindRTI
   /module/FindRuby
   /module/FindSDL
   /module/FindSDL_gfx
   /module/FindSDL_image
   /module/FindSDL_mixer
   /module/FindSDL_net
   /module/FindSDL_sound
   /module/FindSDL_ttf
   /module/FindSelfPackers
   /module/FindSQLite3
   /module/FindSquish
   /module/FindSubversion
   /module/FindSWIG
   /module/FindTCL
   /module/FindTclsh
   /module/FindTclStub
   /module/FindThreads
   /module/FindTIFF
   /module/FindVulkan
   /module/FindWget
   /module/FindWish
   /module/FindwxWidgets
   /module/FindX11
   /module/FindXalanC
   /module/FindXCTest
   /module/FindXercesC
   /module/FindXMLRPC
   /module/FindZLIB

Deprecated Modules
==================

Deprecated Utility Modules
--------------------------

.. toctree::
   :maxdepth: 1

   /module/AddFileDependencies
   /module/CMakeDetermineVSServicePack
   /module/CMakeExpandImportedTargets
   /module/CMakeFindFrameworks
   /module/CMakeForceCompiler
   /module/CMakeParseArguments
   /module/Dart
   /module/Documentation
   /module/GetPrerequisites
   /module/MacroAddFileDependencies
   /module/TestBigEndian
   /module/TestCXXAcceptsFlag
   /module/Use_wxWindows
   /module/UseJavaClassFilelist
   /module/UseJavaSymlinks
   /module/UsePkgConfig
   /module/WriteBasicConfigVersionFile
   /module/WriteCompilerDetectionHeader

Deprecated Find Modules
-----------------------

.. toctree::
   :maxdepth: 1

   /module/FindBoost
   /module/FindCABLE
   /module/FindCUDA
   /module/FindDart
   /module/FindGCCXML
   /module/FindGDAL
   /module/FindITK
   /module/FindPythonInterp
   /module/FindPythonLibs
   /module/FindQt
   /module/FindUnixCommands
   /module/FindVTK
   /module/FindwxWindows

Legacy CPack Modules
--------------------

These modules used to be mistakenly exposed to the user, and have been moved
out of user visibility. They are for CPack internal use, and should never be
used directly.

.. toctree::
   :maxdepth: 1

   /module/CPackArchive
   /module/CPackBundle
   /module/CPackCygwin
   /module/CPackDeb
   /module/CPackDMG
   /module/CPackFreeBSD
   /module/CPackNSIS
   /module/CPackNuGet
   /module/CPackProductBuild
   /module/CPackRPM
   /module/CPackWIX

Miscellaneous Modules
---------------------

These internal modules are not intended to be included directly in projects:

.. toctree::
   :maxdepth: 1

   /module/CMakeFindPackageMode
   /module/CMakeGraphVizOptions
   /module/CTestScriptMode
   /module/Findosg_functions
   /module/SquishTestScript
