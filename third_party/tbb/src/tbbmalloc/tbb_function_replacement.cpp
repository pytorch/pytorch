/*
    Copyright (c) 2005-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

#include "tbb/tbb_config.h"

#if !__TBB_WIN8UI_SUPPORT && defined(_WIN32)

#define _CRT_SECURE_NO_DEPRECATE 1
#define __TBB_NO_IMPLICIT_LINKAGE 1

#include <windows.h>
#include <new>
#include <stdio.h>
#include <string.h>
#include "tbb_function_replacement.h"

#include "tbb/tbb_stddef.h"
#include "../tbb/tbb_assert_impl.h"

inline UINT_PTR Ptr2Addrint(LPVOID ptr)
{
    Int2Ptr i2p;
    i2p.lpv = ptr;
    return i2p.uip;
}

inline LPVOID Addrint2Ptr(UINT_PTR ptr)
{
    Int2Ptr i2p;
    i2p.uip = ptr;
    return i2p.lpv;
}

// Is the distance between addr1 and addr2 smaller than dist
inline bool IsInDistance(UINT_PTR addr1, UINT_PTR addr2, __int64 dist)
{
    __int64 diff = addr1>addr2 ? addr1-addr2 : addr2-addr1;
    return diff<dist;
}

/*
 * When inserting a probe in 64 bits process the distance between the insertion
 * point and the target may be bigger than 2^32. In this case we are using
 * indirect jump through memory where the offset to this memory location
 * is smaller than 2^32 and it contains the absolute address (8 bytes).
 *
 * This class is used to hold the pages used for the above trampolines.
 * Since this utility will be used to replace malloc functions this implementation
 * doesn't allocate memory dynamically.
 *
 * The struct MemoryBuffer holds the data about a page in the memory used for
 * replacing functions in 64-bit code where the target is too far to be replaced
 * with a short jump. All the calculations of m_base and m_next are in a multiple
 * of SIZE_OF_ADDRESS (which is 8 in Win64).
 */
class MemoryProvider {
private:
    struct MemoryBuffer {
        UINT_PTR m_base;    // base address of the buffer
        UINT_PTR m_next;    // next free location in the buffer
        DWORD    m_size;    // size of buffer

        // Default constructor
        MemoryBuffer() : m_base(0), m_next(0), m_size(0) {}

        // Constructor
        MemoryBuffer(void *base, DWORD size)
        {
            m_base = Ptr2Addrint(base);
            m_next = m_base;
            m_size = size;
        }
    };

MemoryBuffer *CreateBuffer(UINT_PTR addr)
    {
        // No more room in the pages database
        if (m_lastBuffer - m_pages == MAX_NUM_BUFFERS)
            return 0;

        void *newAddr = Addrint2Ptr(addr);
        // Get information for the region which the given address belongs to
        MEMORY_BASIC_INFORMATION memInfo;
        if (VirtualQuery(newAddr, &memInfo, sizeof(memInfo)) != sizeof(memInfo))
            return 0;

        for(;;) {
            // The new address to check is beyond the current region and aligned to allocation size
            newAddr = Addrint2Ptr( (Ptr2Addrint(memInfo.BaseAddress) + memInfo.RegionSize + m_allocSize) & ~(UINT_PTR)(m_allocSize-1) );

            // Check that the address is in the right distance.
            // VirtualAlloc can only round the address down; so it will remain in the right distance
            if (!IsInDistance(addr, Ptr2Addrint(newAddr), MAX_DISTANCE))
                break;

            if (VirtualQuery(newAddr, &memInfo, sizeof(memInfo)) != sizeof(memInfo))
                break;

            if (memInfo.State == MEM_FREE && memInfo.RegionSize >= m_allocSize)
            {
                // Found a free region, try to allocate a page in this region
                void *newPage = VirtualAlloc(newAddr, m_allocSize, MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE);
                if (!newPage)
                    break;

                // Add the new page to the pages database
                MemoryBuffer *pBuff = new (m_lastBuffer) MemoryBuffer(newPage, m_allocSize);
                ++m_lastBuffer;
                return pBuff;
            }
        }

        // Failed to find a buffer in the distance
        return 0;
    }

public:
    MemoryProvider()
    {
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        m_allocSize = sysInfo.dwAllocationGranularity;
        m_lastBuffer = &m_pages[0];
    }

    // We can't free the pages in the destructor because the trampolines
    // are using these memory locations and a replaced function might be called
    // after the destructor was called.
    ~MemoryProvider()
    {
    }

    // Return a memory location in distance less than 2^31 from input address
    UINT_PTR GetLocation(UINT_PTR addr)
    {
        MemoryBuffer *pBuff = m_pages;
        for (; pBuff<m_lastBuffer && IsInDistance(pBuff->m_next, addr, MAX_DISTANCE); ++pBuff)
        {
            if (pBuff->m_next < pBuff->m_base + pBuff->m_size)
            {
                UINT_PTR loc = pBuff->m_next;
                pBuff->m_next += MAX_PROBE_SIZE;
                return loc;
            }
        }

        pBuff = CreateBuffer(addr);
        if(!pBuff)
            return 0;

        UINT_PTR loc = pBuff->m_next;
        pBuff->m_next += MAX_PROBE_SIZE;
        return loc;
    }

private:
    MemoryBuffer m_pages[MAX_NUM_BUFFERS];
    MemoryBuffer *m_lastBuffer;
    DWORD m_allocSize;
};

static MemoryProvider memProvider;

// Compare opcodes from dictionary (str1) and opcodes from code (str2)
// str1 might contain '*' to mask addresses
// RETURN: 0 if opcodes did not match, 1 on success
size_t compareStrings( const char *str1, const char *str2 )
{
   for (size_t i=0; str1[i]!=0; i++){
       if( str1[i]!='*' && str1[i]!='#' && str1[i]!=str2[i] ) return 0;
   }
   return 1;
}

// Check function prologue with known prologues from the dictionary
// opcodes - dictionary
// inpAddr - pointer to function prologue
// Dictionary contains opcodes for several full asm instructions
// + one opcode byte for the next asm instruction for safe address processing
// RETURN: 1 + the index of the matched pattern, or 0 if no match found.
static UINT CheckOpcodes( const char ** opcodes, void *inpAddr, bool abortOnError )
{
    static size_t opcodesStringsCount = 0;
    static size_t maxOpcodesLength = 0;
    static size_t opcodes_pointer = (size_t)opcodes;
    char opcodeString[2*MAX_PATTERN_SIZE+1];
    size_t i;
    size_t result;

    // Get the values for static variables
    // max length and number of patterns
    if( !opcodesStringsCount || opcodes_pointer != (size_t)opcodes ){
        while( *(opcodes + opcodesStringsCount)!= NULL ){
            if( (i=strlen(*(opcodes + opcodesStringsCount))) > maxOpcodesLength )
                maxOpcodesLength = i;
            opcodesStringsCount++;
        }
        opcodes_pointer = (size_t)opcodes;
        __TBB_ASSERT( maxOpcodesLength/2 <= MAX_PATTERN_SIZE, "Pattern exceeded the limit of 28 opcodes/56 symbols" );
    }

    // Translate prologue opcodes to string format to compare
    for( i=0; i<maxOpcodesLength/2 && i<MAX_PATTERN_SIZE; ++i ){
        sprintf( opcodeString + 2*i, "%.2X", *((unsigned char*)inpAddr+i) );
    }
    opcodeString[2*i] = 0;

    // Compare translated opcodes with patterns
    for( UINT idx=0; idx<opcodesStringsCount; ++idx ){
        result = compareStrings( opcodes[idx],opcodeString );
        if( result )
            return idx+1; // avoid 0 which indicates a failure
    }
    if (abortOnError) {
        // Impossibility to find opcodes in the dictionary is a serious issue,
        // as if we unable to call original function, leak or crash is expected result.
        __TBB_ASSERT_RELEASE( false, "CheckOpcodes failed" );
    }
    return 0;
}

// Modify offsets in original code after moving it to a trampoline.
// We do not have more than one offset to correct in existing opcode patterns.
static void CorrectOffset( UINT_PTR address, const char* pattern, UINT distance )
{
    const char* pos = strstr(pattern, "#*******");
    if( pos ) {
        address += (pos - pattern)/2; // compute the offset position
        UINT value;
        // UINT assignment is not used to avoid potential alignment issues
        memcpy(&value, Addrint2Ptr(address), sizeof(value));
        value += distance;
        memcpy(Addrint2Ptr(address), &value, sizeof(value));
    }
}

// Insert jump relative instruction to the input address
// RETURN: the size of the trampoline or 0 on failure
static DWORD InsertTrampoline32(void *inpAddr, void *targetAddr, const char* pattern, void** storedAddr)
{
    size_t bytesToMove = SIZE_OF_RELJUMP;
    UINT_PTR srcAddr = Ptr2Addrint(inpAddr);
    UINT_PTR tgtAddr = Ptr2Addrint(targetAddr);
    // Check that the target fits in 32 bits
    if (!IsInDistance(srcAddr, tgtAddr, MAX_DISTANCE))
        return 0;

    UINT_PTR offset;
    UINT offset32;
    UCHAR *codePtr = (UCHAR *)inpAddr;

    if ( storedAddr ){ // If requested, store original function code
        bytesToMove = strlen(pattern)/2-1; // The last byte matching the pattern must not be copied
        __TBB_ASSERT_RELEASE( bytesToMove >= SIZE_OF_RELJUMP, "Incorrect bytecode pattern?" );
        UINT_PTR trampAddr = memProvider.GetLocation(srcAddr);
        if (!trampAddr)
            return 0;
        *storedAddr = Addrint2Ptr(trampAddr);
        // Set 'executable' flag for original instructions in the new place
        DWORD pageFlags = PAGE_EXECUTE_READWRITE;
        if (!VirtualProtect(*storedAddr, MAX_PROBE_SIZE, pageFlags, &pageFlags)) return 0;
        // Copy original instructions to the new place
        memcpy(*storedAddr, codePtr, bytesToMove);
        offset = srcAddr - trampAddr;
        offset32 = (UINT)(offset & 0xFFFFFFFF);
        CorrectOffset( trampAddr, pattern, offset32 );
        // Set jump to the code after replacement
        offset32 -= SIZE_OF_RELJUMP;
        *(UCHAR*)(trampAddr+bytesToMove) = 0xE9;
        memcpy((UCHAR*)(trampAddr+bytesToMove+1), &offset32, sizeof(offset32));
    }

    // The following will work correctly even if srcAddr>tgtAddr, as long as
    // address difference is less than 2^31, which is guaranteed by IsInDistance.
    offset = tgtAddr - srcAddr - SIZE_OF_RELJUMP;
    offset32 = (UINT)(offset & 0xFFFFFFFF);
    // Insert the jump to the new code
    *codePtr = 0xE9;
    memcpy(codePtr+1, &offset32, sizeof(offset32));

    // Fill the rest with NOPs to correctly see disassembler of old code in debugger.
    for( unsigned i=SIZE_OF_RELJUMP; i<bytesToMove; i++ ){
        *(codePtr+i) = 0x90;
    }

    return SIZE_OF_RELJUMP;
}

// This function is called when the offset doesn't fit in 32 bits
// 1  Find and allocate a page in the small distance (<2^31) from input address
// 2  Put jump RIP relative indirect through the address in the close page
// 3  Put the absolute address of the target in the allocated location
// RETURN: the size of the trampoline or 0 on failure
static DWORD InsertTrampoline64(void *inpAddr, void *targetAddr, const char* pattern, void** storedAddr)
{
    size_t bytesToMove = SIZE_OF_INDJUMP;

    UINT_PTR srcAddr = Ptr2Addrint(inpAddr);
    UINT_PTR tgtAddr = Ptr2Addrint(targetAddr);

    // Get a location close to the source address
    UINT_PTR location = memProvider.GetLocation(srcAddr);
    if (!location)
        return 0;

    UINT_PTR offset;
    UINT offset32;
    UCHAR *codePtr = (UCHAR *)inpAddr;

    // Fill the location
    UINT_PTR *locPtr = (UINT_PTR *)Addrint2Ptr(location);
    *locPtr = tgtAddr;

    if ( storedAddr ){ // If requested, store original function code
        bytesToMove = strlen(pattern)/2-1; // The last byte matching the pattern must not be copied
        __TBB_ASSERT_RELEASE( bytesToMove >= SIZE_OF_INDJUMP, "Incorrect bytecode pattern?" );
        UINT_PTR trampAddr = memProvider.GetLocation(srcAddr);
        if (!trampAddr)
            return 0;
        *storedAddr = Addrint2Ptr(trampAddr);
        // Set 'executable' flag for original instructions in the new place
        DWORD pageFlags = PAGE_EXECUTE_READWRITE;
        if (!VirtualProtect(*storedAddr, MAX_PROBE_SIZE, pageFlags, &pageFlags)) return 0;
        // Copy original instructions to the new place
        memcpy(*storedAddr, codePtr, bytesToMove);
        offset = srcAddr - trampAddr;
        offset32 = (UINT)(offset & 0xFFFFFFFF);
        CorrectOffset( trampAddr, pattern, offset32 );
        // Set jump to the code after replacement. It is within the distance of relative jump!
        offset32 -= SIZE_OF_RELJUMP;
        *(UCHAR*)(trampAddr+bytesToMove) = 0xE9;
        memcpy((UCHAR*)(trampAddr+bytesToMove+1), &offset32, sizeof(offset32));
    }

    // Fill the buffer
    offset = location - srcAddr - SIZE_OF_INDJUMP;
    offset32 = (UINT)(offset & 0xFFFFFFFF);
    *(codePtr) = 0xFF;
    *(codePtr+1) = 0x25;
    memcpy(codePtr+2, &offset32, sizeof(offset32));

    // Fill the rest with NOPs to correctly see disassembler of old code in debugger.
    for( unsigned i=SIZE_OF_INDJUMP; i<bytesToMove; i++ ){
        *(codePtr+i) = 0x90;
    }

    return SIZE_OF_INDJUMP;
}

// Insert a jump instruction in the inpAddr to the targetAddr
// 1. Get the memory protection of the page containing the input address
// 2. Change the memory protection to writable
// 3. Call InsertTrampoline32 or InsertTrampoline64
// 4. Restore memory protection
// RETURN: FALSE on failure, TRUE on success
static bool InsertTrampoline(void *inpAddr, void *targetAddr, const char ** opcodes, void** origFunc)
{
    DWORD probeSize;
    // Change page protection to EXECUTE+WRITE
    DWORD origProt = 0;
    if (!VirtualProtect(inpAddr, MAX_PROBE_SIZE, PAGE_EXECUTE_WRITECOPY, &origProt))
        return FALSE;

    UINT opcodeIdx = 0;
    if ( origFunc ){ // Need to store original function code
        UCHAR * const codePtr = (UCHAR *)inpAddr;
        if ( *codePtr == 0xE9 ){ // JMP relative instruction
            // For the special case when a system function consists of a single near jump,
            // instead of moving it somewhere we use the target of the jump as the original function.
            unsigned offsetInJmp = *(unsigned*)(codePtr + 1);
            *origFunc = (void*)(Ptr2Addrint(inpAddr) + offsetInJmp + SIZE_OF_RELJUMP);
            origFunc = NULL; // now it must be ignored by InsertTrampoline32/64
        } else {
            // find the right opcode pattern
            opcodeIdx = CheckOpcodes( opcodes, inpAddr, /*abortOnError=*/true );
            __TBB_ASSERT( opcodeIdx > 0, "abortOnError ignored in CheckOpcodes?" );
        }
    }

    const char* pattern = opcodeIdx>0? opcodes[opcodeIdx-1]: NULL; // -1 compensates for +1 in CheckOpcodes
    probeSize = InsertTrampoline32(inpAddr, targetAddr, pattern, origFunc);
    if (!probeSize)
        probeSize = InsertTrampoline64(inpAddr, targetAddr, pattern, origFunc);

    // Restore original protection
    VirtualProtect(inpAddr, MAX_PROBE_SIZE, origProt, &origProt);

    if (!probeSize)
        return FALSE;

    FlushInstructionCache(GetCurrentProcess(), inpAddr, probeSize);
    FlushInstructionCache(GetCurrentProcess(), origFunc, probeSize);

    return TRUE;
}

// Routine to replace the functions
// TODO: replace opcodesNumber with opcodes and opcodes number to check if we replace right code.
FRR_TYPE ReplaceFunctionA(const char *dllName, const char *funcName, FUNCPTR newFunc, const char ** opcodes, FUNCPTR* origFunc)
{
    // Cache the results of the last search for the module
    // Assume that there was no DLL unload between
    static char cachedName[MAX_PATH+1];
    static HMODULE cachedHM = 0;

    if (!dllName || !*dllName)
        return FRR_NODLL;

    if (!cachedHM || strncmp(dllName, cachedName, MAX_PATH) != 0)
    {
        // Find the module handle for the input dll
        HMODULE hModule = GetModuleHandleA(dllName);
        if (hModule == 0)
        {
            // Couldn't find the module with the input name
            cachedHM = 0;
            return FRR_NODLL;
        }

        cachedHM = hModule;
        strncpy(cachedName, dllName, MAX_PATH);
    }

    FARPROC inpFunc = GetProcAddress(cachedHM, funcName);
    if (inpFunc == 0)
    {
        // Function was not found
        return FRR_NOFUNC;
    }

    if (!InsertTrampoline((void*)inpFunc, (void*)newFunc, opcodes, (void**)origFunc)){
        // Failed to insert the trampoline to the target address
        return FRR_FAILED;
    }

    return FRR_OK;
}

FRR_TYPE ReplaceFunctionW(const wchar_t *dllName, const char *funcName, FUNCPTR newFunc, const char ** opcodes, FUNCPTR* origFunc)
{
    // Cache the results of the last search for the module
    // Assume that there was no DLL unload between
    static wchar_t cachedName[MAX_PATH+1];
    static HMODULE cachedHM = 0;

    if (!dllName || !*dllName)
        return FRR_NODLL;

    if (!cachedHM || wcsncmp(dllName, cachedName, MAX_PATH) != 0)
    {
        // Find the module handle for the input dll
        HMODULE hModule = GetModuleHandleW(dllName);
        if (hModule == 0)
        {
            // Couldn't find the module with the input name
            cachedHM = 0;
            return FRR_NODLL;
        }

        cachedHM = hModule;
        wcsncpy(cachedName, dllName, MAX_PATH);
    }

    FARPROC inpFunc = GetProcAddress(cachedHM, funcName);
    if (inpFunc == 0)
    {
        // Function was not found
        return FRR_NOFUNC;
    }

    if (!InsertTrampoline((void*)inpFunc, (void*)newFunc, opcodes, (void**)origFunc)){
        // Failed to insert the trampoline to the target address
        return FRR_FAILED;
    }

    return FRR_OK;
}

bool IsPrologueKnown(HMODULE module, const char *funcName, const char **opcodes)
{
    FARPROC inpFunc = GetProcAddress(module, funcName);
    if (!inpFunc)
        return false;
    return CheckOpcodes( opcodes, (void*)inpFunc, /*abortOnError=*/false ) != 0;
}

#endif /* !__TBB_WIN8UI_SUPPORT && defined(_WIN32) */
