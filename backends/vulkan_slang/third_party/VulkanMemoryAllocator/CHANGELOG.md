# 3.4.0 (2026-??-??)

- Added member `VmaAllocationCreateInfo::minAlignment` (#523).
  - Remember to always fully initialize structures with zeros and don't rely on their specific `sizeof` to ensure backward compatibility!
  - Function `vmaCreateBufferWithAlignment` is now deprecated.
- Improvements for external memory export & import (#503):
  - Added functions `vmaCreateDedicatedBuffer`, `vmaCreateDedicatedImage`, `vmaAllocateDedicatedMemory` offering extra parameter `void* pMemoryAllocateNext`.
  - Added function `vmaGetMemoryWin32Handle2` offering extra parameter `VkExternalMemoryHandleTypeFlagBits handleType`.
- Added `VMA_VERSION` macro with library version number (#507).
- Improvements in the algorithm choosing memory type when `VMA_MEMORY_USAGE_AUTO*` is used (#520).
- Fixes for compatibility with C++20 modules on Clang 21 and GCC15 (#513, #514).
- Other fixes and improvements, including compatibility with various platforms and compilers, improvements in documentation, sample application, and tests.

# 3.3.0 (2025-05-12)

Additions to the library API:

- Added function `vmaImportVulkanFunctionsFromVolk`, useful for loading pointers to Vulkan functions with [volk library](https://github.com/zeux/volk).

Other changes:

- Added macro `VMA_DEBUG_DONT_EXCEED_HEAP_SIZE_WITH_ALLOCATION_SIZE` with default value 1.
- Changed macro `VMA_DEBUG_DONT_EXCEED_MAX_MEMORY_ALLOCATION_COUNT` default value from 0 to 1.
- Added documentation chapter "Frequently asked questions".
- Other fixes and improvements, including compatibility with various platforms and compilers.

# 3.2.1 (2025-02-05)

Changes:

- Fixed an assert in `vmaCreateAllocator` function incorrectly failing when Vulkan version 1.4 is used (#457).
- Fix for importing function `vkGetPhysicalDeviceMemoryProperties2` / `vkGetPhysicalDeviceMemoryProperties2KHR` when `VMA_DYNAMIC_VULKAN_FUNCTIONS` macro is enabled (#410).
- Other minor fixes and improvements...

# 3.2.0 (2024-12-30)

Additions to the library API:

- Added support for Vulkan 1.4.
- Added support for VK_KHR_external_memory_win32 extension - `VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT` flag, `vmaGetMemoryWin32Handle` function, and a whole new documentation chapter about it (#442).

Other changes:

- Fixed thread safety issue (#451).
- Many other bug fixes and improvements in the library code, documentation, sample app, Cmake script, mostly to improve compatibility with various compilers and GPUs.

# 3.1.0 (2024-05-27)

This release gathers fixes and improvements made during many months of continuous development on the main branch, mostly based on issues and pull requests on GitHub.

Additions to the library API:

- Added convenience functions `vmaCopyMemoryToAllocation`, `vmaCopyAllocationToMemory`.
- Added functions `vmaCreateAliasingBuffer2`, `vmaCreateAliasingImage2` that offer creating a buffer/image in an existing allocation with additional `allocationLocalOffset`.
- Added function `vmaGetAllocationInfo2`, structure `VmaAllocationInfo2` that return additional information about an allocation, useful for interop with other APIs (#383, #340).
- Added callback `VmaDefragmentationInfo::pfnBreakCallback` that allows breaking long execution of `vmaBeginDefragmentation`.
  Also added `PFN_vmaCheckDefragmentationBreakFunction`, `VmaDefragmentationInfo::pBreakCallbackUserData`.
- Added support for VK_KHR_maintenance4 extension - `VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT` flag (#397).
- Added support for VK_KHR_maintenance5 extension - `VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT` flag (#411).

Other changes:

- Changes in debug and configuration macros:
  - Split macros into separate `VMA_DEBUG_LOG` and `VMA_DEBUG_LOG_FORMAT` (#297).
  - Added macros `VMA_ASSERT_LEAK`, `VMA_LEAK_LOG_FORMAT` separate from normal `VMA_ASSERT`, `VMA_DEBUG_LOG_FORMAT` (#379, #385).
  - Added macro `VMA_EXTENDS_VK_STRUCT` (#347).
- Countless bug fixes and improvements in the code and documentation, mostly to improve compatibility with various compilers and GPUs, including:
  - Fixed missing `#include` that resulted in compilation error about `snprintf` not declared on some compilers (#312).
  - Fixed main memory type selection algorithm for GPUs that have no `HOST_CACHED` memory type, like Raspberry Pi (#362).
- Major changes in Cmake script.
- Fixes in GpuMemDumpVis.py script.

# 3.0.1 (2022-05-26)

- Fixes in defragmentation algorithm.
- Fixes in GpuMemDumpVis.py regarding image height calculation.
- Other bug fixes, optimizations, and improvements in the code and documentation.

# 3.0.0 (2022-03-25)

It has been a long time since the previous official release, so hopefully everyone has been using the latest code from "master" branch, which is always maintained in a good state, not the old version. For completeness, here is the list of changes since v2.3.0. The major version number has changed, so there are some compatibility-breaking changes, but the basic API stays the same and is mostly backward-compatible.

Major features added (some compatibility-breaking):

- Added new API for selecting preferred memory type: flags `VMA_MEMORY_USAGE_AUTO`, `VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE`, `VMA_MEMORY_USAGE_AUTO_PREFER_HOST`, `VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT`, `VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT`, `VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT`. Old values like `VMA_MEMORY_USAGE_GPU_ONLY` still work as before, for backward compatibility, but are not recommended.
- Added new defragmentation API and algorithm, replacing the old one. See structure `VmaDefragmentationInfo`, `VmaDefragmentationMove`, `VmaDefragmentationPassMoveInfo`, `VmaDefragmentationStats`, function `vmaBeginDefragmentation`, `vmaEndDefragmentation`, `vmaBeginDefragmentationPass`, `vmaEndDefragmentationPass`.
- Redesigned API for statistics, replacing the old one. See structures: `VmaStatistics`, `VmaDetailedStatistics`, `VmaTotalStatistics`. `VmaBudget`, functions: `vmaGetHeapBudgets`, `vmaCalculateStatistics`, `vmaGetPoolStatistics`, `vmaCalculatePoolStatistics`, `vmaGetVirtualBlockStatistics`, `vmaCalculateVirtualBlockStatistics`.
- Added "Virtual allocator" feature - possibility to use core allocation algorithms for allocation of custom memory, not necessarily Vulkan device memory. See functions like `vmaCreateVirtualBlock`, `vmaDestroyVirtualBlock` and many more.
- `VmaAllocation` now keeps both `void* pUserData` and `char* pName`. Added function `vmaSetAllocationName`, member `VmaAllocationInfo::pName`. Flag `VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT` is now deprecated.
- Clarified and cleaned up various ways of importing Vulkan functions. See macros `VMA_STATIC_VULKAN_FUNCTIONS`, `VMA_DYNAMIC_VULKAN_FUNCTIONS`, structure `VmaVulkanFunctions`. Added members `VmaVulkanFunctions::vkGetInstanceProcAddr`, `vkGetDeviceProcAddr`, which are now required when using `VMA_DYNAMIC_VULKAN_FUNCTIONS`.

Removed (compatibility-breaking):

- Removed whole "lost allocations" feature. Removed from the interface: `VMA_ALLOCATION_CREATE_CAN_BECOME_LOST_BIT`, `VMA_ALLOCATION_CREATE_CAN_MAKE_OTHER_LOST_BIT`, `vmaCreateLostAllocation`, `vmaMakePoolAllocationsLost`, `vmaTouchAllocation`, `VmaAllocatorCreateInfo::frameInUseCount`, `VmaPoolCreateInfo::frameInUseCount`.
- Removed whole "record & replay" feature. Removed from the API: `VmaAllocatorCreateInfo::pRecordSettings`, `VmaRecordSettings`, `VmaRecordFlagBits`, `VmaRecordFlags`. Removed VmaReplay application.
- Removed "buddy" algorithm - removed flag `VMA_POOL_CREATE_BUDDY_ALGORITHM_BIT`.

Minor but compatibility-breaking changes:

- Changes in `ALLOCATION_CREATE_STRATEGY` flags. Removed flags: `VMA_ALLOCATION_CREATE_STRATEGY_MIN_FRAGMENTATION_BIT`, `VMA_ALLOCATION_CREATE_STRATEGY_WORST_FIT_BIT`, `VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_FRAGMENTATION_BIT`, which were aliases to other existing flags.
- Added a member `void* pUserData` to `VmaDeviceMemoryCallbacks`. Updated `PFN_vmaAllocateDeviceMemoryFunction`, `PFN_vmaFreeDeviceMemoryFunction` to use the new `pUserData` member.
- Removed function `vmaResizeAllocation` that was already deprecated.

Other major changes:

- Added new features to custom pools: support for dedicated allocations, new member `VmaPoolCreateInfo::pMemoryAllocateNext`, `minAllocationAlignment`.
- Added support for Vulkan 1.2, 1.3.
- Added support for VK_KHR_buffer_device_address extension - flag `VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT`.
- Added support for VK_EXT_memory_priority extension - flag `VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT`, members `VmaAllocationCreateInfo::priority`, `VmaPoolCreateInfo::priority`.
- Added support for VK_AMD_device_coherent_memory extension - flag `VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT`.
- Added member `VmaAllocatorCreateInfo::pTypeExternalMemoryHandleTypes`.
- Added function `vmaGetAllocatorInfo`, structure `VmaAllocatorInfo`.
- Added functions `vmaFlushAllocations`, `vmaInvalidateAllocations` for multiple allocations at once.
- Added flag `VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT`.
- Added function `vmaCreateBufferWithAlignment`.
- Added convenience function `vmaGetAllocationMemoryProperties`.
- Added convenience functions: `vmaCreateAliasingBuffer`, `vmaCreateAliasingImage`.

Other minor changes:

- Implemented Two-Level Segregated Fit (TLSF) allocation algorithm, replacing previous default one. It is much faster, especially when freeing many allocations at once or when `bufferImageGranularity` is large.
- Renamed debug macro `VMA_DEBUG_ALIGNMENT` to `VMA_MIN_ALIGNMENT`.
- Added CMake support - CMakeLists.txt files. Removed Premake support.
- Changed `vmaInvalidateAllocation` and `vmaFlushAllocation` to return `VkResult`.
- Added nullability annotations for Clang: `VMA_NULLABLE`, `VMA_NOT_NULL`, `VMA_NULLABLE_NON_DISPATCHABLE`, `VMA_NOT_NULL_NON_DISPATCHABLE`, `VMA_LEN_IF_NOT_NULL`.
- JSON dump format has changed.
- Countless fixes and improvements, including performance optimizations, compatibility with various platforms and compilers, documentation.

# 2.3.0 (2019-12-04)

Major release after a year of development in "master" branch and feature branches. Notable new features: supporting Vulkan 1.1, supporting query for memory budget.

Major changes:

- Added support for Vulkan 1.1.
    - Added member `VmaAllocatorCreateInfo::vulkanApiVersion`.
    - When Vulkan 1.1 is used, there is no need to enable VK_KHR_dedicated_allocation or VK_KHR_bind_memory2 extensions, as they are promoted to Vulkan itself.
- Added support for query for memory budget and staying within the budget.
    - Added function `vmaGetBudget`, structure `VmaBudget`. This can also serve as simple statistics, more efficient than `vmaCalculateStats`.
    - By default the budget it is estimated based on memory heap sizes. It may be queried from the system using VK_EXT_memory_budget extension if you use `VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT` flag and `VmaAllocatorCreateInfo::instance` member.
    - Added flag `VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT` that fails an allocation if it would exceed the budget.
- Added new memory usage options:
    - `VMA_MEMORY_USAGE_CPU_COPY` for memory that is preferably not `DEVICE_LOCAL` but not guaranteed to be `HOST_VISIBLE`.
    - `VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED` for memory that is `LAZILY_ALLOCATED`.
- Added support for VK_KHR_bind_memory2 extension:
    - Added `VMA_ALLOCATION_CREATE_DONT_BIND_BIT` flag that lets you create both buffer/image and allocation, but don't bind them together.
    - Added flag `VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT`, functions `vmaBindBufferMemory2`, `vmaBindImageMemory2` that let you specify additional local offset and `pNext` pointer while binding.
- Added functions `vmaSetPoolName`, `vmaGetPoolName` that let you assign string names to custom pools. JSON dump file format and VmaDumpVis tool is updated to show these names.
- Defragmentation is legal only on buffers and images in `VK_IMAGE_TILING_LINEAR`. This is due to the way it is currently implemented in the library and the restrictions of the Vulkan specification. Clarified documentation in this regard. See discussion in #59.

Minor changes:

- Made `vmaResizeAllocation` function deprecated, always returning failure.
- Made changes in the internal algorithm for the choice of memory type. Be careful! You may now get a type that is not `HOST_VISIBLE` or `HOST_COHERENT` if it's not stated as always ensured by some `VMA_MEMORY_USAGE_*` flag.
- Extended VmaReplay application with more detailed statistics printed at the end.
- Added macros `VMA_CALL_PRE`, `VMA_CALL_POST` that let you decorate declarations of all library functions if you want to e.g. export/import them as dynamically linked library.
- Optimized `VmaAllocation` objects to be allocated out of an internal free-list allocator. This makes allocation and deallocation causing 0 dynamic CPU heap allocations on average.
- Updated recording CSV file format version to 1.8, to support new functions.
- Many additions and fixes in documentation. Many compatibility fixes for various compilers and platforms. Other internal bugfixes, optimizations, updates, refactoring...

# 2.2.0 (2018-12-13)

Major release after many months of development in "master" branch and feature branches. Notable new features: defragmentation of GPU memory, buddy algorithm, convenience functions for sparse binding.

Major changes:

- New, more powerful defragmentation:
  - Added structure `VmaDefragmentationInfo2`, functions `vmaDefragmentationBegin`, `vmaDefragmentationEnd`.
  - Added support for defragmentation of GPU memory.
  - Defragmentation of CPU memory now uses `memmove`, so it can move data to overlapping regions.
  - Defragmentation of CPU memory is now available for memory types that are `HOST_VISIBLE` but not `HOST_COHERENT`.
  - Added structure member `VmaVulkanFunctions::vkCmdCopyBuffer`.
  - Major internal changes in defragmentation algorithm.
  - VmaReplay: added parameters: `--DefragmentAfterLine`, `--DefragmentationFlags`.
  - Old interface (structure `VmaDefragmentationInfo`, function `vmaDefragment`) is now deprecated.
- Added buddy algorithm, available for custom pools - flag `VMA_POOL_CREATE_BUDDY_ALGORITHM_BIT`.
- Added convenience functions for multiple allocations and deallocations at once, intended for sparse binding resources - functions `vmaAllocateMemoryPages`, `vmaFreeMemoryPages`.
- Added function that tries to resize existing allocation in place: `vmaResizeAllocation`.
- Added flags for allocation strategy: `VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT`, `VMA_ALLOCATION_CREATE_STRATEGY_WORST_FIT_BIT`, `VMA_ALLOCATION_CREATE_STRATEGY_FIRST_FIT_BIT`, and their aliases: `VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT`, `VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT`, `VMA_ALLOCATION_CREATE_STRATEGY_MIN_FRAGMENTATION_BIT`.

Minor changes:

- Changed behavior of allocation functions to return `VK_ERROR_VALIDATION_FAILED_EXT` when trying to allocate memory of size 0, create buffer with size 0, or image with one of the dimensions 0.
- VmaReplay: Added support for Windows end of lines.
- Updated recording CSV file format version to 1.5, to support new functions.
- Internal optimization: using read-write mutex on some platforms.
- Many additions and fixes in documentation. Many compatibility fixes for various compilers. Other internal bugfixes, optimizations, refactoring, added more internal validation...

# 2.1.0 (2018-09-10)

Minor bugfixes.

# 2.1.0-beta.1 (2018-08-27)

Major release after many months of development in "development" branch and features branches. Many new features added, some bugs fixed. API stays backward-compatible.

Major changes:

- Added linear allocation algorithm, accessible for custom pools, that can be used as free-at-once, stack, double stack, or ring buffer. See "Linear allocation algorithm" documentation chapter.
  - Added `VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT`, `VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT`.
- Added feature to record sequence of calls to the library to a file and replay it using dedicated application. See documentation chapter "Record and replay".
  - Recording: added `VmaAllocatorCreateInfo::pRecordSettings`.
  - Replaying: added VmaReplay project.
  - Recording file format: added document "docs/Recording file format.md".
- Improved support for non-coherent memory.
  - Added functions: `vmaFlushAllocation`, `vmaInvalidateAllocation`.
  - `nonCoherentAtomSize` is now respected automatically.
  - Added `VmaVulkanFunctions::vkFlushMappedMemoryRanges`, `vkInvalidateMappedMemoryRanges`.
- Improved debug features related to detecting incorrect mapped memory usage. See documentation chapter "Debugging incorrect memory usage".
  - Added debug macro `VMA_DEBUG_DETECT_CORRUPTION`, functions `vmaCheckCorruption`, `vmaCheckPoolCorruption`.
  - Added debug macro `VMA_DEBUG_INITIALIZE_ALLOCATIONS` to initialize contents of allocations with a bit pattern.
  - Changed behavior of `VMA_DEBUG_MARGIN` macro - it now adds margin also before first and after last allocation in a block.
- Changed format of JSON dump returned by `vmaBuildStatsString` (not backward compatible!).
  - Custom pools and memory blocks now have IDs that don't change after sorting.
  - Added properties: "CreationFrameIndex", "LastUseFrameIndex", "Usage".
  - Changed VmaDumpVis tool to use these new properties for better coloring.
  - Changed behavior of `vmaGetAllocationInfo` and `vmaTouchAllocation` to update `allocation.lastUseFrameIndex` even if allocation cannot become lost.

Minor changes:

- Changes in custom pools:
  - Added new structure member `VmaPoolStats::blockCount`.
  - Changed behavior of `VmaPoolCreateInfo::blockSize` = 0 (default) - it now means that pool may use variable block sizes, just like default pools do.
- Improved logic of `vmaFindMemoryTypeIndex` for some cases, especially integrated GPUs.
- VulkanSample application: Removed dependency on external library MathFu. Added own vector and matrix structures.
- Changes that improve compatibility with various platforms, including: Visual Studio 2012, 32-bit code, C compilers.
  - Changed usage of "VK_KHR_dedicated_allocation" extension in the code to be optional, driven by macro `VMA_DEDICATED_ALLOCATION`, for compatibility with Android.
- Many additions and fixes in documentation, including description of new features, as well as "Validation layer warnings".
- Other bugfixes.

# 2.0.0 (2018-03-19)

A major release with many compatibility-breaking changes.

Notable new features:

- Introduction of `VmaAllocation` handle that you must retrieve from allocation functions and pass to deallocation functions next to normal `VkBuffer` and `VkImage`.
- Introduction of `VmaAllocationInfo` structure that you can retrieve from `VmaAllocation` handle to access parameters of the allocation (like `VkDeviceMemory` and offset) instead of retrieving them directly from allocation functions.
- Support for reference-counted mapping and persistently mapped allocations - see `vmaMapMemory`, `VMA_ALLOCATION_CREATE_MAPPED_BIT`.
- Support for custom memory pools - see `VmaPool` handle, `VmaPoolCreateInfo` structure, `vmaCreatePool` function.
- Support for defragmentation (compaction) of allocations - see function `vmaDefragment` and related structures.
- Support for "lost allocations" - see appropriate chapter on documentation Main Page.

# 1.0.1 (2017-07-04)

- Fixes for Linux GCC compilation.
- Changed "CONFIGURATION SECTION" to contain #ifndef so you can define these macros before including this header, not necessarily change them in the file.

# 1.0.0 (2017-06-16)

First public release.
