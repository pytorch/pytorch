// unit-test-free-list.cpp

#include "../../source/core/slang-free-list.h"
#include "../../source/core/slang-list.h"
#include "../../source/core/slang-random-generator.h"
#include "unit-test/slang-unit-test.h"

#include <stdio.h>
#include <stdlib.h>

using namespace Slang;

SLANG_UNIT_TEST(freeList)
{
    FreeList freeList;
    freeList.init(sizeof(int), sizeof(void*), 10);

    DefaultRandomGenerator randGen(0x24343);

    List<int*> allocs;

    for (int i = 0; i < 1000; i++)
    {
        const int numAlloc = randGen.nextInt32UpTo(20);

        for (int j = 0; j < numAlloc; j++)
        {
            int* ptr = (int*)freeList.allocate();
            *ptr = i;
            allocs.add(ptr);
        }

        int numDealloc = randGen.nextInt32UpTo(19);
        numDealloc = int(allocs.getCount()) < numDealloc ? int(allocs.getCount()) : numDealloc;

        for (int j = 0; j < numDealloc; j++)
        {
            const int index = randGen.nextInt32UpTo(int(allocs.getCount()));

            int* alloc = allocs[index];

            SLANG_CHECK(*alloc <= i);
            SLANG_CHECK(*alloc >= 0);

            freeList.deallocate(alloc);

            allocs.fastRemoveAt(index);
        }
    }
}
