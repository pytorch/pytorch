Slang Compilation Reproduction
==============================

Slang has both API and command line support for reproducing compilations, so called 'repro' functionality.

One use of the feature is if a compilation fails, or produces an unexpected or wrong result, it provides a simple to use mechanism where the compilation can be repeated or 'reproduced', most often on another machine. Instead of having to describe all the options, and make sure all of the files that are used are copied, and in such a way that it repeats the result, all that is required is for the compilation to be run on the host machine with repro capture enabled, and then that 'repro' used for a compilation on the test machine. There are also some mechanisms where the contents of the original compilation can be altered.

The actual data saved is the contents of the SlangCompileRequest. Currently no state is saved from the SlangSession. Saving and loading a SlangCompileRequest into a new SlangCompileRequest should provide two SlangCompileRequests with the same state, and with the second compile request having access to all the files contents the original request had directly in memory.

There are a few command line options

* `-dump-repro [filename]` dumps the compilations state (ie post attempting to compile) to the file specified afterwards
* `-extract-repro [filename]` extracts the contents of the repro file. The contained files are placed in a directory with a name, the same as the repro file minus the extension. Also contains a 'manifest'.
* `-load-repro [filename]` loads the repro and compiles using it's options. Note this must be the last arg on the command line.
* `-dump-repro-on-error` if a compilation fails will attempt to save a repro (using a filename generated from first source filename)
* `-repro-file-system [filename]` makes the repros file contents appear as the file system during a compilation. Does not set any compilation options.
* `-load-repro-directory [directory]` compiles all of the .slang-repro files found in `directory`

The `manifest` made available via `-extract-repro` provides some very useful information

* Provides an approximation of the command line that will produce the same compilation under [compile-line]
* A list of all the unique files held in the repro [files]. It specified their 'unique name' (as used to identify in the repro) and their unique identifier as used by the file system.
* A list of how paths map to unique files. Listed as the path used to access, followed by the unique name used in the repro

First it is worth just describing what is required to reproduce a compilation. Most straightforwardly the options setup for the compilation need to be stored. This would include any flags, and defines, include paths, entry points, input filenames and so forth. Also needed will be the contents of any files that were specified. This might be files on the file system, but could also be 'files' specified as strings through the slang API. Lastly we need any files that were referenced as part of the compilation - this could be include files, or module source files and so forth. All of this information is bundled up together into a file that can then later be loaded and compiled. This is broadly speaking all of the data that is stored within a repro file. 

In order to capture a complete repro file typically a compilation has to be attempted. The state before compilation can be recorded (through the API for example), but it may not be enough to repeat a compilation, as files referenced by the compilation would not yet have been accessed. The repro feature records all of these accesses and contents of such files such that compilation can either be completed or at least to the same point as was reached on the host machine. 

One of the more subtle issues around reproducing a compilation is around filenames. Using the API, a client can specify source files without names, or multiple files with the same name. If files are loaded via `ISlangFileSystem`, they are typically part of a hierarchical file system. This could mean they are referenced relatively. This means there can be distinct files with the same name but differentiated by directory. The files may not easily be reconstructed back into a similar hieararchical file system - as depending on the include paths (or perhaps other mechanisms) the 'files' and their contents could be arranged in a manner very hard to replicate. To work around this the repro feature does not attempt to replicate a hierarchical file system. Instead it gives every file a unique name based on their original name. If there are multiple files with the same name it will 'uniquify' them by appending an index. Doing so means that the contents of the file system can just be held as a flat collection of files. This is not enough to enable repeating the compilation though, as we now need Slang to know which files to reference when they are requested, as they are now no longer part of a hierarchical file system and their names may have been altered. To achieve this the repro functionality stores off a map of all path requests to their contents (or lack there of). Doing so means that the file system still appears to Slang as it did in the original compilation, even with all the files being actually stored using the simpler 'flat' arrangement.

This means that when a repro is 'extracted' it does so to a directory which holds the files with their unique 'flat' names. The name of the directory is the name of the repro file without it's extension, or if it has no extension, with the postfix '-files'. This directory will be referred to from now on as the `repro directory`.

When a repro is loaded, before files are loaded from the repro itself, they will first be looked for via their unique names in the `repro directory`. If they are not there the contents of the repro file will be used. If they are there, their contents will be used instead of the contents in the repro. This provides a simple mechanism to be able to alter the source in a repro. The steps more concretely would be...

1) First extract the repro (say with `-extract-repro`)
2) Go to the `repro directory` and edit files that you wish to change. You can also just delete files that do not need changing, as they will be loaded from the repro.
3) Load the repro - it will now load any files requested from the `repro directory`

Now you might want to change the compilation options. Using -load-repro it will compile with the options as given. It is not possible to change those options as part of -load-repro. If you want to change the compilation options (and files), you can use -extract-repro, and look at the manifest which will list a command line that will typically repeat the compilation. Now you can just attach the repro as a file system, and set the command line options as appropriate, based on the command line listed in the manifest. Note! If there is a fairly complex directory hierarchy, it may be necessary to specify the input sources paths *as if* they are held on the original files system. You can see how these map in the manifest. 

Note that currently it is disabled to access any new source files - they will be determined as `not found`. This behaviour could be changed such that the regular file system was used, or the ISlangFilesystem set on the API is used as a fallback.

There currently isn't a mechanism to alter the options of a repro from the command line (other than altering the contents of the source). The reason for this is because of how command lines are processed currently in Slang. A future update could enable specifying a repro and then altering the command line options used. It can be achieved through the API though. Once the repro is loaded via the `spLoadRepro` function, options can be changed as normal. The two major places where option alteration may have surprising behavior are... 

1) Altering the include paths - unless this may break the mechanism used to map paths to files stored in the repro file
2) Altering the ISlangFileSystem. That to make the contents of the file system appear to be that of the repro, slang uses a ISlangFileSystemExt that uses the contents of the repro file and/or the `repro directory`. If you replace the file system this mechanism will no longer work. 

There are currently several API calls for using the repro functionality 

```
SLANG_API SlangResult spEnableReproCapture(
        SlangCompileRequest* request);
    
SLANG_API SlangResult spLoadRepro(
        SlangCompileRequest* request,
        ISlangFileSystem* fileSystem,
        const void* data,
        size_t size);

SLANG_API SlangResult spSaveRepro(
        SlangCompileRequest* request,
        ISlangBlob** outBlob
    );
    
SLANG_API SlangResult spExtractRepro(
    SlangSession* session, 
    const void* reproData, 
    size_t reproDataSize, 
    ISlangFileSystemExt* fileSystem);    

SLANG_API SlangResult spLoadReproAsFileSystem(
    SlangSession* session,
    const void* reproData,
    size_t reproDataSize,
    ISlangFileSystem* replaceFileSystem,
    ISlangFileSystemExt** outFileSystem);
   
```

The fileSystem parameter passed to `spLoadRepro` provides the mechanism for client code to replace the files that are held within the repro. NOTE! That the files will be loaded from this file system with their `unique names` as if they are part of the flat file system. If an attempt to load a file fails, the file within the repro is used. That `spLoadRepro` is typically performed on a new 'unused' SlangCompileRequest. After a call to `spLoadRepro` normal functions to alter the state of the SlangCompileRequest are available. 

The function `spEnableReproCapture` should be set after any ISlangFileSystem has been set (if any), but before any compilation. It ensures that everything that the ISlangFileSystem accesses will be correctly recorded. Note that if a ISlangFileSystem/ISlangFileSystemExt isn't explicitly set (ie the default is used), then a request will automatically be set up to record everything appropriate and a call to this function isn't strictly required.  
    
The function `spExtractRepro` allows for extracting the files used in a request (along with the associated manifest). They files and manifest are stored under the 'unique names' in the root of the user provided ISlangFileSystemExt.     
    
The function `spLoadReproAsFileSystem` creates a file system that can access the contents of the repro with the same paths that were used on the originating system. The ISlangFileSystemExt produced can be set on a request and used for compilation.    
    
Repro files are currently stored in a binary format. This format is sensitive to changes in the API, as well as internal state within a SlangCompileRequest. This means that the functionality can only be guaranteed to work with exactly the same version of Slang on the same version of compiler. In practice things are typically not so draconian, and future versions will aim to provide a more clear slang repro versioning system, and work will be performed to make more generally usable.

Finally this version of the repo system does not take into account endianness at all. The system the repro is saved from must have the same endianness as the system loaded on.
