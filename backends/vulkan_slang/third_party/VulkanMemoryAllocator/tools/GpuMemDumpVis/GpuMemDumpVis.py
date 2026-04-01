#
# Copyright (c) 2018-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

import argparse
import json
from PIL import Image, ImageDraw, ImageFont


PROGRAM_VERSION = 'Vulkan/D3D12 Memory Allocator Dump Visualization 3.0.4'
IMG_WIDTH = 1200
IMG_MARGIN = 8
TEXT_MARGIN = 4
FONT_SIZE = 10
MAP_SIZE = 24
COLOR_TEXT_H1 = (0, 0, 0, 255)
COLOR_TEXT_H2 = (150, 150, 150, 255)
COLOR_OUTLINE = (155, 155, 155, 255)
COLOR_OUTLINE_HARD = (0, 0, 0, 255)
COLOR_GRID_LINE = (224, 224, 224, 255)

currentApi = ""
data = {}


def ParseArgs():
    argParser = argparse.ArgumentParser(description='Visualization of Vulkan/D3D12 Memory Allocator JSON dump.')
    argParser.add_argument('DumpFile', help='Path to source JSON file with memory dump created by Vulkan/D3D12 Memory Allocator library')
    argParser.add_argument('-v', '--version', action='version', version=PROGRAM_VERSION)
    argParser.add_argument('-o', '--output', required=True, help='Path to destination image file (e.g. PNG)')
    return argParser.parse_args()

def GetDataForMemoryPool(poolTypeName):
    global data
    if poolTypeName in data:
        return data[poolTypeName]
    else:
        newPoolData = {'DedicatedAllocations':[], 'Blocks':[], 'CustomPools':{}}
        data[poolTypeName] = newPoolData
        return newPoolData

def ProcessBlock(poolData, block):
    blockInfo = {'ID': block[0], 'Size': int(block[1]['TotalBytes']), 'Suballocations':[]}
    for alloc in block[1]['Suballocations']:
        allocData = {'Type': alloc['Type'], 'Size': int(alloc['Size']), 'Usage': int(alloc['Usage']) if 'Usage' in alloc else 0 }
        blockInfo['Suballocations'].append(allocData)
    poolData['Blocks'].append(blockInfo)
    
def IsDataEmpty():
    global data
    for poolData in data.values():
        if len(poolData['DedicatedAllocations']) > 0:
            return False
        if len(poolData['Blocks']) > 0:
            return False
        for customPool in poolData['CustomPools'].values():
            if len(customPool['Blocks']) > 0:
                return False
            if len(customPool['DedicatedAllocations']) > 0:
                return False
    return True

def RemoveEmptyType():
    global data
    for poolType in list(data.keys()):
        pool = data[poolType]
        if len(pool['DedicatedAllocations']) > 0:
           continue
        if len(pool['Blocks']) > 0:
            continue
        empty = True
        for customPool in pool['CustomPools'].values():
            if len(customPool['Blocks']) > 0:
                empty = False
                break
            if len(customPool['DedicatedAllocations']) > 0:
                empty = False
                break
        if empty:
            del data[poolType]

# Returns tuple:
# [0] image height : integer
# [1] pixels per byte : float
def CalcParams():
    global data
    height = IMG_MARGIN
    height += FONT_SIZE + IMG_MARGIN # Grid lines legend - sizes
    maxBlockSize = 0
    # Get height occupied by every memory pool
    for poolData in data.values():
        height += FONT_SIZE + IMG_MARGIN # Memory pool title
        height += len(poolData['Blocks']) * (FONT_SIZE + MAP_SIZE + IMG_MARGIN * 2)
        height += len(poolData['DedicatedAllocations']) * (FONT_SIZE + MAP_SIZE + IMG_MARGIN * 2)
        # Get longest block size
        for dedicatedAlloc in poolData['DedicatedAllocations']:
            maxBlockSize = max(maxBlockSize, dedicatedAlloc['Size'])
        for block in poolData['Blocks']:
            maxBlockSize = max(maxBlockSize, block['Size'])
        # Same for custom pools
        for customPoolData in poolData['CustomPools'].values():
            height += len(customPoolData['Blocks']) * (FONT_SIZE + MAP_SIZE + IMG_MARGIN * 2)
            height += len(customPoolData['DedicatedAllocations']) * (FONT_SIZE + MAP_SIZE + IMG_MARGIN * 2)
            # Get longest block size
            for dedicatedAlloc in customPoolData['DedicatedAllocations']:
                maxBlockSize = max(maxBlockSize, dedicatedAlloc['Size'])
            for block in customPoolData['Blocks']:
                maxBlockSize = max(maxBlockSize, block['Size'])

    return height, (IMG_WIDTH - IMG_MARGIN * 2) / float(maxBlockSize)

def BytesToStr(bytes):
    if bytes < 1024:
        return "%d B" % bytes
    bytes /= 1024
    if bytes < 1024:
        return "%d KiB" % bytes
    bytes /= 1024
    if bytes < 1024:
        return "%d MiB" % bytes
    bytes /= 1024
    return "%d GiB" % bytes

def TypeToColor(type, usage):
    global currentApi
    if type == 'FREE':
        return 220, 220, 220, 255
    elif type == 'UNKNOWN':
        return 175, 175, 175, 255 # Gray

    if currentApi == 'Vulkan':
        if type == 'BUFFER':
            if (usage & 0x1C0) != 0: # INDIRECT_BUFFER | VERTEX_BUFFER | INDEX_BUFFER
                return 255, 148, 148, 255 # Red
            elif (usage & 0x28) != 0: # STORAGE_BUFFER | STORAGE_TEXEL_BUFFER
                return 255, 187, 121, 255 # Orange
            elif (usage & 0x14) != 0: # UNIFORM_BUFFER | UNIFORM_TEXEL_BUFFER
                return 255, 255, 0, 255 # Yellow
            else:
                return 255, 255, 165, 255 # Light yellow
        elif type == 'IMAGE_OPTIMAL':
            if (usage & 0x20) != 0: # DEPTH_STENCIL_ATTACHMENT
                return 246, 128, 255, 255 # Pink
            elif (usage & 0xD8) != 0: # INPUT_ATTACHMENT | TRANSIENT_ATTACHMENT | COLOR_ATTACHMENT | STORAGE
                return 179, 179, 255, 255 # Blue
            elif (usage & 0x4) != 0: # SAMPLED
                return 0, 255, 255, 255 # Aqua
            else:
                return 183, 255, 255, 255 # Light aqua
        elif type == 'IMAGE_LINEAR' :
            return 0, 255, 0, 255 # Green
        elif type == 'IMAGE_UNKNOWN':
            return 0, 255, 164, 255 # Green/aqua
    elif currentApi == 'Direct3D 12':
        if type == 'BUFFER':
                return 255, 255, 165, 255 # Light yellow
        elif type == 'TEXTURE1D' or type == 'TEXTURE2D' or type == 'TEXTURE3D':
            if (usage & 0x2) != 0: # D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
                return 246, 128, 255, 255 # Pink
            elif (usage & 0x5) != 0: # D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                return 179, 179, 255, 255 # Blue
            elif (usage & 0x8) == 0: # Not having D3D12_RESOURCE_FLAG_DENY_SHARED_RESOURCE
                return 0, 255, 255, 255 # Aqua
            else:
                return 183, 255, 255, 255 # Light aqua
    else:
        print("Unknown graphics API!")
        exit(1)
    assert False
    return 0, 0, 0, 255

def DrawBlock(draw, y, block, pixelsPerByte):
    sizePixels = int(block['Size'] * pixelsPerByte)
    draw.rectangle([IMG_MARGIN, y, IMG_MARGIN + sizePixels, y + MAP_SIZE], fill=TypeToColor('FREE', 0), outline=None)
    byte = 0
    x = 0
    lastHardLineX = -1
    for alloc in block['Suballocations']:
        byteEnd = byte + alloc['Size']
        xEnd = int(byteEnd * pixelsPerByte)
        if alloc['Type'] != 'FREE':
            if xEnd > x + 1:
                draw.rectangle([IMG_MARGIN + x, y, IMG_MARGIN + xEnd, y + MAP_SIZE], fill=TypeToColor(alloc['Type'], alloc['Usage']), outline=COLOR_OUTLINE)
                # Hard line was been overwritten by rectangle outline: redraw it.
                if lastHardLineX == x:
                    draw.line([IMG_MARGIN + x, y, IMG_MARGIN + x, y + MAP_SIZE], fill=COLOR_OUTLINE_HARD)
            else:
                draw.line([IMG_MARGIN + x, y, IMG_MARGIN + x, y + MAP_SIZE], fill=COLOR_OUTLINE_HARD)
                lastHardLineX = x
        byte = byteEnd
        x = xEnd

def DrawDedicatedAllocationBlock(draw, y, dedicatedAlloc, pixelsPerByte): 
    sizePixels = int(dedicatedAlloc['Size'] * pixelsPerByte)
    draw.rectangle([IMG_MARGIN, y, IMG_MARGIN + sizePixels, y + MAP_SIZE], fill=TypeToColor(dedicatedAlloc['Type'], dedicatedAlloc['Usage']), outline=COLOR_OUTLINE)


if __name__ == '__main__':
    args = ParseArgs()
    jsonSrc = json.load(open(args.DumpFile, 'rb'))
 
    if 'General' in jsonSrc:
        currentApi = jsonSrc['General']['API']
    else:
        print("Wrong JSON format, cannot determine graphics API!")
        exit(1)
        
    # Process default pools
    if 'DefaultPools' in jsonSrc:
        for memoryPool in jsonSrc['DefaultPools'].items():
            poolData = GetDataForMemoryPool(memoryPool[0])
            # Get dedicated allocations
            for dedicatedAlloc in memoryPool[1]['DedicatedAllocations']:
                allocData = {'Type': dedicatedAlloc['Type'], 'Size': int(dedicatedAlloc['Size']), 'Usage': int(dedicatedAlloc['Usage'])}
                poolData['DedicatedAllocations'].append(allocData)
            # Get allocations in block vectors
            for block in memoryPool[1]['Blocks'].items():
                ProcessBlock(poolData, block)
    # Process custom pools
    if 'CustomPools' in jsonSrc:
        for memoryPool in jsonSrc['CustomPools'].items():
            poolData = GetDataForMemoryPool(memoryPool[0])
            for pool in memoryPool[1]:
                poolName = pool['Name']
                poolData['CustomPools'][poolName] = {'DedicatedAllocations':[], 'Blocks':[]}
                # Get dedicated allocations
                for dedicatedAlloc in pool['DedicatedAllocations']:
                    allocData = {'Type': dedicatedAlloc['Type'], 'Size': int(dedicatedAlloc['Size']), 'Usage': int(dedicatedAlloc['Usage'])}
                    poolData['CustomPools'][poolName]['DedicatedAllocations'].append(allocData)
                # Get allocations in block vectors
                for block in pool['Blocks'].items():
                    ProcessBlock(poolData['CustomPools'][poolName], block)

    if IsDataEmpty():
        print("There is nothing to put on the image. Please make sure you generated the stats string with detailed map enabled.")
        exit(1)
    RemoveEmptyType()
    # Calculate dimmensions and create data image       
    imgHeight, pixelsPerByte = CalcParams()
    img = Image.new('RGB', (IMG_WIDTH, imgHeight), 'white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('segoeuib.ttf')
    except:
        font = ImageFont.load_default()

    # Draw grid lines
    bytesBetweenGridLines = 32
    while bytesBetweenGridLines * pixelsPerByte < 64:
        bytesBetweenGridLines *= 2
    byte = 0
    y = IMG_MARGIN
    while True:
        x = int(byte * pixelsPerByte)
        if x > IMG_WIDTH - 2 * IMG_MARGIN:
            break
        draw.line([x + IMG_MARGIN, 0, x + IMG_MARGIN, imgHeight], fill=COLOR_GRID_LINE)
        if byte == 0:
            draw.text((x + IMG_MARGIN + TEXT_MARGIN, y), "0", fill=COLOR_TEXT_H2, font=font)
        else:
            text = BytesToStr(byte)
            textLength = draw.textlength(text, font=font)
            draw.text((x + IMG_MARGIN - textLength - TEXT_MARGIN, y), text, fill=COLOR_TEXT_H2, font=font)
        byte += bytesBetweenGridLines
    y += FONT_SIZE + IMG_MARGIN
    
    # Draw main content
    for memType in sorted(data.keys()):
        memPoolData = data[memType]
        draw.text((IMG_MARGIN, y), "Memory pool %s" % memType, fill=COLOR_TEXT_H1, font=font)
        y += FONT_SIZE + IMG_MARGIN
        # Draw block vectors
        for block in memPoolData['Blocks']:
            draw.text((IMG_MARGIN, y), "Default pool block %s" % block['ID'], fill=COLOR_TEXT_H2, font=font)
            y += FONT_SIZE + IMG_MARGIN
            DrawBlock(draw, y, block, pixelsPerByte)
            y += MAP_SIZE + IMG_MARGIN
        index = 0
        # Draw dedicated allocations
        for dedicatedAlloc in memPoolData['DedicatedAllocations']:
            draw.text((IMG_MARGIN, y), "Dedicated allocation %d" % index, fill=COLOR_TEXT_H2, font=font)
            y += FONT_SIZE + IMG_MARGIN
            DrawDedicatedAllocationBlock(draw, y, dedicatedAlloc, pixelsPerByte)
            y += MAP_SIZE + IMG_MARGIN
            index += 1
        for poolName, pool in memPoolData['CustomPools'].items():
            for block in pool['Blocks']:
                draw.text((IMG_MARGIN, y), "Custom pool %s block %s" % (poolName, block['ID']), fill=COLOR_TEXT_H2, font=font)
                y += FONT_SIZE + IMG_MARGIN
                DrawBlock(draw, y, block, pixelsPerByte)
                y += MAP_SIZE + IMG_MARGIN
            index = 0
            for dedicatedAlloc in pool['DedicatedAllocations']:
                draw.text((IMG_MARGIN, y), "Custom pool %s dedicated allocation %d" % (poolName, index), fill=COLOR_TEXT_H2, font=font)
                y += FONT_SIZE + IMG_MARGIN
                DrawDedicatedAllocationBlock(draw, y, dedicatedAlloc, pixelsPerByte)
                y += MAP_SIZE + IMG_MARGIN
                index += 1
    del draw
    img.save(args.output)

"""
Main data structure - variable `data` - is a dictionary. Key is string - memory type name. Value is dictionary of:
- Fixed key 'DedicatedAllocations'. Value is list of objects, each containing dictionary with:
    - Fixed key 'Type'. Value is string.
    - Fixed key 'Size'. Value is int.
    - Fixed key 'Usage'. Value is int.
- Fixed key 'Blocks'. Value is list of objects, each containing dictionary with:
    - Fixed key 'ID'. Value is int.
    - Fixed key 'Size'. Value is int.
    - Fixed key 'Suballocations'. Value is list of objects as above.
- Fixed key 'CustomPools'. Value is dictionary.
  - Key is string with pool ID/name. Value is a dictionary with:
    - Fixed key 'DedicatedAllocations'. Value is list of objects as above.
    - Fixed key 'Blocks'. Value is a list of objects representing memory blocks as above.
"""
