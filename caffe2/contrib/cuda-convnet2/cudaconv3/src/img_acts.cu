/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <vector>

#include "../include/cudaconv2.cuh"

/*
 * Block size: 16x16.
 * blockIdx.x determines case in batches of 16*imgsPerThread.
 * blockIdx.y determines 4x4 image region in target image.
 *
 * threadIdx.x determines case.
 * threadIdx.y determines pixel.
 *
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 * filters:     (numColors, filterPixels, numFilters)                               if conv
 *              (numModulesY, numModulesX, numColors, filterPixels, numFilters)     otherwise
 * targets:     (numColors, imgSizeY, imgSizeX, numImages)
 *
 * Each block reconstructs one 4x4 pixels from 16*imgsPerThread cases.
 *
 * Number of filters must be divisible by 16.
 * Number of images must be divisible by 16*imgsPerThread  if checkCaseBounds is false.
 * 16 * imgsPerThread must be divisible by 32.
 *
 * This version loads 32 cases at a time, so it gets full coalescing on that load.
 * It only loads 16 weights at a time, so those aren't fully coalesced.
 * This version conserves shared memory by loading 16 filters at a time rather than 32.
 */
template <int imgsPerThread, int numColors, bool scale, bool checkCaseBounds, bool conv>
__global__ void img_acts_color(const float* hidActs, const float* filters, float* targets,
                                   const int numModulesY, const int numModulesX, const int numImages, const int numFilters,
                                   const int filterSize, const int imgSizeY, const int imgSizeX,
                                   const int paddingStart, const int moduleStride,
                                   const float scaleTargets, const float scaleOutputs) {
    __shared__ float shFilters[numColors*16][16 + 1];
    __shared__ float shHidActs[16][16*imgsPerThread];

    const int blockCaseIdx = blockIdx.x * 16*imgsPerThread;
    const int numRegionsX = DIVUP(imgSizeX, 4);
    const int blockRegionIdx = blockIdx.y;
    const int blockRegionIdxX = blockRegionIdx % numRegionsX;
    const int blockRegionIdxY = blockRegionIdx / numRegionsX;
    const int blockRegionLeft = blockRegionIdxX * 4;
    const int blockRegionTop = blockRegionIdxY * 4;
    const int pxYInRegion = threadIdx.y / 4, pxXInRegion = threadIdx.y % 4;
    const int pxY = blockRegionTop + pxYInRegion;
    const int pxX = blockRegionLeft + pxXInRegion;
    const int pxIdx = pxY * imgSizeX + pxX;
    const bool isPxInImg = pxY < imgSizeY && pxX < imgSizeX;
    const int numModules = numModulesY * numModulesX;
    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeX * imgSizeY;
    const int tidx = threadIdx.y * 16 + threadIdx.x;
    const int loadY = tidx / 32, loadX = tidx % 32;

    hidActs += blockCaseIdx + loadY * numImages * numModules + loadX;
    filters += threadIdx.x;
    targets += pxIdx * numImages + blockCaseIdx + threadIdx.x;


    float prod[numColors][imgsPerThread];
    #pragma unroll
    for (int c = 0; c < numColors; c++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[c][i] = 0;
        }
    }
    const int startY = blockRegionTop - paddingStart < filterSize ? 0
                        : 1 + (blockRegionTop - paddingStart - filterSize) / moduleStride;
    const int endY = MIN(numModulesY, 1 + (blockRegionTop + 3 - paddingStart) / moduleStride);
    const int startX = blockRegionLeft - paddingStart < filterSize ? 0
                        : 1 + (blockRegionLeft - paddingStart - filterSize) / moduleStride;
    const int endX = MIN(numModulesX, 1 + (blockRegionLeft + 3 - paddingStart) / moduleStride);

    float* shilterLoad = &shFilters[threadIdx.y][threadIdx.x];
    float* shHidActLoad = &shHidActs[loadY][loadX];

    for (int my = startY; my < endY; my++) {
        const int moduleTop = paddingStart + my * moduleStride;
        const int pxInModuleY = pxY - moduleTop;

        for (int mx = startX; mx < endX; mx++) {
            const int moduleIdx = my * numModulesX + mx;
            const int moduleLeft = paddingStart + mx * moduleStride;
            const int pxInModuleX = pxX - moduleLeft;

            const bool isPxInModule = pxInModuleY >= 0 && pxInModuleY < filterSize && pxInModuleX >= 0 && pxInModuleX < filterSize;
            const int pxIdxInModule = pxInModuleY * filterSize + pxInModuleX;

            for (int f = 0; f < numFilters; f += 16) { // multiply with 16 filters at a time
                // Now the threads split up into half-warps, and each half-warp decides if it's interested.
                const float* hLoad = &hidActs[(moduleIdx + f * numModules) * numImages];
                #pragma unroll
                for (int i = 0; i < imgsPerThread * 16; i += 32) {
                    if (!checkCaseBounds || blockCaseIdx + i + loadX < numImages) {
                        #pragma unroll
                        for (int j = 0; j < 16; j += 8) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
                            shHidActLoad[j * 16 * imgsPerThread + i] = hLoad[j * numModules * numImages + i];
                        }
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 16; j += 8) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
                            shHidActLoad[j * 16 * imgsPerThread + i] = 0;
                        }
                    }
                }

                if (isPxInImg && isPxInModule) {
                    // This half-warp is interested, so it's going to load the weights from this module to its pixel.
                    // Not fully coalesced read :(
                    // But taking out this read entirely only reduces the runtime by ~2.8%, so it isn't costing me much.
                    const float* fLoad = conv ? &filters[pxIdxInModule * numFilters + f]
                                              : &filters[(moduleIdx * numColors * filterPixels + pxIdxInModule) * numFilters + f];
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        shilterLoad[c * 16 * (16 + 1)] = fLoad[c * filterPixels * numFilters];
                    }


                }

                __syncthreads();
                // Do some actual computation
                if (isPxInImg && isPxInModule) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        #pragma unroll
                        for (int w = 0; w < 16; w++) {
                            #pragma unroll
                            for (int i = 0; i < imgsPerThread; i++) {
                                prod[c][i] += shFilters[threadIdx.y + c * 16][w] * shHidActs[w][threadIdx.x + i * 16];
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
    // Not fully coalesced write :(... shmem (and fully coalesced) version is actually slightly slower, though
    if (isPxInImg) {
        if (scale) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * 16 < numImages) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        targets[c * imgPixels * numImages + i * 16] = scaleTargets * targets[c * imgPixels * numImages + i * 16] + scaleOutputs * prod[c][i];
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * 16 < numImages) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        targets[c * imgPixels * numImages + i * 16] = scaleOutputs * prod[c][i];
                    }
                }
            }
        }
    }
}

/*
 * Block size: 16x16.
 * blockIdx.x determines case in batches of 16*imgsPerThread, also color in batches of colorsPerThread.
 *  In essence, blockIdx.x.x = 1..numImages/(16*imgsPerThread)
 *              blockIdx.x.y = 1..numImgColors/colorsPerThread
 * blockIdx.y determines 4x4 image region in target image.
 *
 * threadIdx.x determines case.
 * threadIdx.y determines pixel.
 *
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 * filters:     (numFilterColors, filterPixels, numFilters)                             if conv
 *              (numModulesY, numModulesX, numFilterColors, filterPixels, numFilters)   otherwise
 * targets:     (numImageColors, imgSizeY, imgSizeX, numImages)
 *
 * Each block reconstructs one 4x4 pixels from 16*imgsPerThread cases.
 *
 * numImages must be divisible by 16*imgsPerThread if checkCaseBounds is false.
 * 16 * imgsPerThread must be divisible by 32.
 * numImageColors/numGroups must be divisible by colorsPerThread.
 *
 * This version loads 32 cases at a time, so it gets full coalescing on that load.
 * It only loads 16 weights at a time, so those aren't fully coalesced.
 * This version conserves shared memory by loading 16 filters at a time rather than 32.
 *
 * To be used when there are 4-16 color channels.
 */
template <int imgsPerThread, int colorsPerThread,  bool scale, bool checkCaseBounds, bool conv>
__global__ void img_acts_mediumcolor(const float* hidActs, const float* filters, float* targets,
                                       const int numModulesY, const int numModulesX, const int numImages, const int numFilters,
                                       const int filterSize, const int imgSizeY, const int imgSizeX, const int paddingStart,
                                       const int moduleStride, const int numImgColors, const int numGroups,
                                       const float scaleTargets, const float scaleOutputs) {
    __shared__ float shFilters[colorsPerThread*16][16 + 1];
    __shared__ float shHidActs[16][16*imgsPerThread];

    const int numImgBlocks = DIVUP(numImages,16*imgsPerThread);
    const int blockCaseIdx = (blockIdx.x % numImgBlocks) * 16*imgsPerThread;

    const int imgColorIdx = (blockIdx.x / numImgBlocks) * colorsPerThread; // color idx globally
    const int numFilterColors = numImgColors / numGroups;
    const int blockGroupIdx = imgColorIdx / numFilterColors;
    const int filterColorIdx = imgColorIdx % numFilterColors; // color idx within group
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockFilterIdx = blockGroupIdx * numFiltersPerGroup;

    const int numRegionsX = DIVUP(imgSizeX, 4);
    const int blockRegionIdx = blockIdx.y;
    const int blockRegionIdxX = blockRegionIdx % numRegionsX;
    const int blockRegionIdxY = blockRegionIdx / numRegionsX;
    const int blockRegionLeft = blockRegionIdxX * 4;
    const int blockRegionTop = blockRegionIdxY * 4;
    const int pxYInRegion = threadIdx.y / 4, pxXInRegion = threadIdx.y % 4;
    const int pxY = blockRegionTop + pxYInRegion;
    const int pxX = blockRegionLeft + pxXInRegion;
    const int pxIdx = pxY * imgSizeX + pxX;
    const bool isPxInImg = pxY < imgSizeY && pxX < imgSizeX;
    const uint numModules = numModulesY * numModulesX;
    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;
    const int tidx = threadIdx.y * 16 + threadIdx.x;
    const int loadY = tidx / 32, loadX = tidx % 32;

    hidActs += blockCaseIdx + (blockFilterIdx + loadY) * numImages * numModules + loadX;
    filters += blockFilterIdx + filterColorIdx * filterPixels * numFilters + threadIdx.x;
    targets += imgColorIdx * imgPixels * numImages + pxIdx * numImages + blockCaseIdx + threadIdx.x;

    float prod[colorsPerThread][imgsPerThread];
    #pragma unroll
    for (int c = 0; c < colorsPerThread; c++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[c][i] = 0;
        }
    }
    const int startY = blockRegionTop - paddingStart < filterSize ? 0
                        : 1 + (blockRegionTop - paddingStart - filterSize) / moduleStride;
    const int endY = MIN(numModulesY, 1 + (blockRegionTop + 3 - paddingStart) / moduleStride);
    const int startX = blockRegionLeft - paddingStart < filterSize ? 0
                        : 1 + (blockRegionLeft - paddingStart - filterSize) / moduleStride;
    const int endX = MIN(numModulesX, 1 + (blockRegionLeft + 3 - paddingStart) / moduleStride);

    float* shFilterLoad = &shFilters[threadIdx.y][threadIdx.x];
    float* shHidActLoad = &shHidActs[loadY][loadX];

    for (int my = startY; my < endY; my++) {
        const int moduleTop = paddingStart + my * moduleStride;
        const int pxInModuleY = pxY - moduleTop;

        for (int mx = startX; mx < endX; mx++) {
            const int moduleIdx = my * numModulesX + mx;
            const int moduleLeft = paddingStart + mx * moduleStride;
            const int pxInModuleX = pxX - moduleLeft;

            const bool isPxInModule = pxInModuleY >= 0 && pxInModuleY < filterSize && pxInModuleX >= 0 && pxInModuleX < filterSize;
            const int pxIdxInModule = pxInModuleY * filterSize + pxInModuleX;

            for (int f = 0; f < numFiltersPerGroup; f += 16) { // multipply with 16 filters at a time
                // Now the threads split up into half-warps, and each half-warp decides if it's interested.
                const float* hLoad = &hidActs[(moduleIdx + f * numModules) * numImages];
                #pragma unroll
                for (int i = 0; i < imgsPerThread * 16; i += 32) {
                    if (!checkCaseBounds || blockCaseIdx + loadX + i < numImages) {
                        #pragma unroll
                        for (int j = 0; j < 16; j += 8) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
                            shHidActLoad[j * 16 * imgsPerThread + i] = hLoad[j * numModules * numImages + i];
                        }
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 16; j += 8) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
                            shHidActLoad[j * 16 * imgsPerThread + i] = 0;
                        }
                    }
                }

                if (isPxInImg && isPxInModule) {
                    // This half-warp is interested, so it's going to load the weights from this module to its pixel.

                    // Not fully coalesced read :(
                    // But taking out this read entirely only reduces the runtime by ~2.8%, so it isn't costing me much.
                    const float* fLoad = conv ? &filters[pxIdxInModule * numFilters + f]
                                              : &filters[moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInModule * numFilters + f];
                    #pragma unroll
                    for (int c = 0; c < colorsPerThread; c++) {
                        shFilterLoad[c * 16 * (16 + 1)] = fLoad[c * filterPixels * numFilters];
                    }
                }

                __syncthreads();
                // Do some actual computation
                if (isPxInImg && isPxInModule) {
                    #pragma unroll
                    for (int c = 0; c < colorsPerThread; c++) {
                        #pragma unroll
                        for (int w = 0; w < 16; w++) {
                            #pragma unroll
                            for (int i = 0; i < imgsPerThread; i++) {
                                prod[c][i] += shFilters[threadIdx.y + c * 16][w] * shHidActs[w][threadIdx.x + i * 16];
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
    // Not fully coalesced write :(... shmem (and fully coalesced) version is actually slightly slower, though
    if (isPxInImg) {
        if (scale) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * 16 < numImages) {
                    #pragma unroll
                    for (int c = 0; c < colorsPerThread; c++) {
                        targets[c * imgPixels * numImages + i * 16] = scaleTargets * targets[c * imgPixels * numImages + i * 16] + scaleOutputs * prod[c][i];
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * 16 < numImages) {
                    #pragma unroll
                    for (int c = 0; c < colorsPerThread; c++) {
                        targets[c * imgPixels * numImages + i * 16] = scaleOutputs * prod[c][i];
                    }
                }
            }
        }
    }
}

/*
 * Block size: B_YxB_X.
 * blockIdx.x determines case in batches of B_X*imgsPerThread, also color in batches of B_Y*colorsPerThread.
 *  In essence, blockIdx.x.x = 1..numImages/(B_X*imgsPerThread)
 *              blockIdx.x.y = 1..numImgColors/(B_Y*colorsPerThread)
 * blockIdx.y determines image pixel in target image.
 *
 * threadIdx.x determines case.
 * threadIdx.y determines color.
 *
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 * filters:     (numFilterColors, filterPixels, numFilters)                             if conv
 *              (numModulesY, numModulesX, numFilterColors, filterPixels, numFilters)   otherwise
 * targets:     (numImageColors, imgSizeY, imgSizeX, numImages)
 *
 * Each block reconstructs one B_Y*colorsPerThread colors from 1 pixel from B_X*imgsPerThread cases.
 *
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false.
 * numFiltersPerGroup must be divisible by filterCache.
 *
 * B_X * imgsPerThread must be divisible by 32.
 * numFilterColors must be divisible by B_Y*colorsPerThread.
 * B_X*B_Y must be divisible by 32.
 * filterCache must be divisible by B_X*B_Y/32
 * B_X*B_Y must be divisible by filterCache

 * This version loads 32 cases at a time, so it gets full coalescing on that load.
 * It only loads filterCache weights at a time, so those aren't fully coalesced (depending on size of filterCache).
 *
 * To be used when there are >= 16 color channels.
 */
template <int B_Y, int B_X, int imgsPerThread, int colorsPerThread, int filterCache, bool scale, bool checkCaseBounds, bool conv>
__global__ void conv_img_acts_manycolor(const float* hidActs, const float* filters, float* targets,
                                          const int numModulesY, const int numModulesX, const int numImages, const int numFilters,
                                          const int filterSize, const int imgSizeY, const int imgSizeX, const int paddingStart, const int moduleStride,
                                          const int numImgColors, const int numGroups,
                                          const float scaleTargets, const float scaleOutputs) {
    __shared__ float shFilters[colorsPerThread*B_Y][filterCache + 1];
    __shared__ float shHidActs[filterCache][B_X*imgsPerThread];

    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int blockCaseIdx = (blockIdx.x % numImgBlocks) * B_X*imgsPerThread;

    const int imgColorIdx = (blockIdx.x / numImgBlocks) * B_Y*colorsPerThread; // color idx globally
    const int numFilterColors = numImgColors / numGroups;
    const int blockGroupIdx = imgColorIdx / numFilterColors;
    const int filterColorIdx = imgColorIdx % numFilterColors; // color idx within group
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockFilterIdx = blockGroupIdx * numFiltersPerGroup;

    const int blockPixelIdx = blockIdx.y;
    const int blockPixelIdxX = blockPixelIdx % imgSizeX;
    const int blockPixelIdxY = blockPixelIdx / imgSizeX;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;
    const int tidx = threadIdx.y * B_X + threadIdx.x;
    const int hidActLoadY = tidx / 32, hidActLoadX = tidx % 32;
    const int filtersLoadY = tidx / filterCache, filtersLoadX = tidx % filterCache;
    const int numModules = numModulesY * numModulesX;

    hidActs += blockCaseIdx + (blockFilterIdx + hidActLoadY) * numImages * numModules + hidActLoadX;
    filters += blockFilterIdx + (filterColorIdx + filtersLoadY) * filterPixels * numFilters + filtersLoadX;
    targets += (imgColorIdx + threadIdx.y) * imgPixels * numImages + blockPixelIdx * numImages + blockCaseIdx + threadIdx.x;

    float prod[colorsPerThread][imgsPerThread];
    #pragma unroll
    for (int c = 0; c < colorsPerThread; c++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[c][i] = 0;
        }
    }

    const int startY = blockPixelIdxY - paddingStart < filterSize ? 0
                        : 1 + (blockPixelIdxY - paddingStart - filterSize) / moduleStride;
    const int endY = MIN(numModulesY, 1 + (blockPixelIdxY - paddingStart) / moduleStride);
    const int startX = blockPixelIdxX - paddingStart < filterSize ? 0
                        : 1 + (blockPixelIdxX - paddingStart - filterSize) / moduleStride;
    const int endX = MIN(numModulesX, 1 + (blockPixelIdxX - paddingStart) / moduleStride);

    float* shFilterLoad = &shFilters[filtersLoadY][filtersLoadX];
    float* shHidActLoad = &shHidActs[hidActLoadY][hidActLoadX];

    for (int my = startY; my < endY; my++) {
        const int moduleTop = paddingStart + my * moduleStride;
        const int pxInFilterY = blockPixelIdxY - moduleTop;

        for (int mx = startX; mx < endX; mx++) {
            const int moduleIdx = my * numModulesX + mx;
            const int moduleLeft = paddingStart + mx * moduleStride;
            const int pxInFilterX = blockPixelIdxX - moduleLeft;

            const int pxIdxInFilter = pxInFilterY * filterSize + pxInFilterX;

            for (int f = 0; f < numFiltersPerGroup; f += filterCache) { // multiply with filterCache filters at a time
                const float* hLoad = &hidActs[(moduleIdx + f * numModules) * numImages];
                #pragma unroll
                for (int i = 0; i < imgsPerThread * B_X; i += 32) {
                    if (!checkCaseBounds || blockCaseIdx + hidActLoadX + i < numImages) {
                        #pragma unroll
                        for (int j = 0; j < filterCache; j += B_X*B_Y/32) { // load filterCache rows of imgsPerThread*B_X cols, 8 * 32 elements at a time.
                            shHidActLoad[j * B_X * imgsPerThread + i] = hLoad[j * numModules * numImages + i];
                        }
                    } else {
                        #pragma unroll
                        for (int j = 0; j < filterCache; j += B_X*B_Y/32) { // load filterCache rows of imgsPerThread*B_X cols, 8 * 32 elements at a time.
                            shHidActLoad[j * B_X * imgsPerThread + i] = 0;
                        }
                    }
                }
                const float* fLoad = conv ? &filters[pxIdxInFilter * numFilters + f]
                                          : &filters[moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInFilter * numFilters + f];
                #pragma unroll
                for (int i = 0; i < colorsPerThread*B_Y; i+= B_X*B_Y/filterCache) {
                    if ((colorsPerThread*B_Y) % (B_X*B_Y/filterCache) == 0 || i + filtersLoadY < colorsPerThread*B_Y) {
                        shFilterLoad[i * (filterCache + 1)] = fLoad[i * filterPixels * numFilters];
                    }
                }

                __syncthreads();
                // Do some actual computation
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for (int w = 0; w < filterCache; w++) {
                        #pragma unroll
                        for (int c = 0; c < colorsPerThread; c++) {
                            prod[c][i] += shFilters[c * B_Y + threadIdx.y][w] * shHidActs[w][threadIdx.x + i * B_X];
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
    if (scale) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * B_X < numImages) {
                #pragma unroll
                for (int c = 0; c < colorsPerThread; c++) {
                    targets[c * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * targets[c * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[c][i];
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * B_X < numImages) {
                #pragma unroll
                for (int c = 0; c < colorsPerThread; c++) {
                    targets[c * B_Y * imgPixels * numImages + i * B_X] = scaleOutputs * prod[c][i];
                }
            }
        }
    }
}


/*
 * Block size: B_YxB_X.
 * blockIdx.x determines case in batches of B_X*imgsPerThread, also color in batches of B_Y*colorsPerThread.
 *  In essence, blockIdx.x.x = 1..numImages/(B_X*imgsPerThread)
 *              blockIdx.x.y = 1..numImgColors/(B_Y*colorsPerThread)
 * blockIdx.y determines image pixel in target image.
 *
 * threadIdx.x determines case.
 * threadIdx.y determines color.
 *
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 * filters:     (numFilterColors, filterPixels, numFilters)                             if conv
 *              (numModulesY, numModulesX, numFilterColors, filterPixels, numFilters)   otherwise
 * targets:     (numImageColors, imgSizeY, imgSizeX, numImages)
 *
 * Each block reconstructs one B_Y*colorsPerThread colors from 1 pixel from B_X*imgsPerThread cases.
 *
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false.
 * numFiltersPerGroup must be divisible by filterCacheF.
 *
 * numFilterColors must be divisible by B_Y*colorsPerThread.
 * B_X*B_Y must be divisible by filterCacheF
 * filterCacheF must be divisible by filterCacheH
 *
 * This version loads 32 cases at a time, so it gets full coalescing on that load.
 * It only loads filterCacheF weights at a time, so those aren't fully coalesced (depending on size of filterCacheF).
 *
 * To be used when there are >= 16 color channels.
 */
template <int B_Y, int B_X, int imgsPerThread, int colorsPerThread, int filterCacheF, int filterCacheH, bool scale, bool checkCaseBounds, bool conv>
__global__ void conv_img_acts_manycolor_kepler(const float* hidActs, const float* filters, float* targets,
                                          const int numModulesY, const int numModulesX, const int numImages, const int numFilters,
                                          const int filterSize, const int imgSizeY, const int imgSizeX, const int paddingStart, const int moduleStride,
                                          const int numImgColors, const int numGroups,
                                          const float scaleTargets, const float scaleOutputs) {
    __shared__ float shFilters[colorsPerThread*B_Y][filterCacheF];
    __shared__ float shHidActs[filterCacheH][B_X*imgsPerThread];

    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int blockCaseIdx = (blockIdx.x % numImgBlocks) * B_X*imgsPerThread;

    const int imgColorIdx = (blockIdx.x / numImgBlocks) * B_Y*colorsPerThread; // color idx globally
    const int numFilterColors = numImgColors / numGroups;
    const int blockGroupIdx = imgColorIdx / numFilterColors;
    const int filterColorIdx = imgColorIdx % numFilterColors; // color idx within group
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockFilterIdx = blockGroupIdx * numFiltersPerGroup;

    const int blockPixelIdx = blockIdx.y;
    const int blockPixelIdxX = blockPixelIdx % imgSizeX;
    const int blockPixelIdxY = blockPixelIdx / imgSizeX;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;
    const int tidx = threadIdx.y * B_X + threadIdx.x;
    const int hidActLoadY = threadIdx.y, hidActLoadX = threadIdx.x;
    //const int hidActLoadY = tidx / (B_X*imgsPerThread), hidActLoadX = tidx % (B_X*imgsPerThread);
    const int filtersLoadY = tidx / filterCacheF, filtersLoadX = tidx % filterCacheF;
    // nvcc is behaving idiotically again, these useless declarations save registers
    //const int outputY = threadIdx.y, outputX = threadIdx.x;
    //const int ty = threadIdx.y, tx = threadIdx.x;
    const int numModules = numModulesY * numModulesX;

    hidActs += blockCaseIdx + (blockFilterIdx + hidActLoadY) * numImages * numModules + hidActLoadX;
    filters += blockFilterIdx + (filterColorIdx + filtersLoadY) * filterPixels * numFilters + filtersLoadX;
    targets += (imgColorIdx + threadIdx.y) * imgPixels * numImages + blockPixelIdx * numImages + blockCaseIdx + threadIdx.x;

    float prod[colorsPerThread][imgsPerThread];
    #pragma unroll
    for (int c = 0; c < colorsPerThread; c++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[c][i] = 0;
        }
    }

    const int startY = blockPixelIdxY - paddingStart < filterSize ? 0
                        : 1 + (blockPixelIdxY - paddingStart - filterSize) / moduleStride;
    const int endY = min(numModulesY, 1 + (blockPixelIdxY - paddingStart) / moduleStride);
    const int startX = blockPixelIdxX - paddingStart < filterSize ? 0
                        : 1 + (blockPixelIdxX - paddingStart - filterSize) / moduleStride;
    const int endX = min(numModulesX, 1 + (blockPixelIdxX - paddingStart) / moduleStride);

    float* shFilterLoad = &shFilters[filtersLoadY][filtersLoadX];
    float* shHidActLoad = &shHidActs[hidActLoadY][hidActLoadX];
    //const bool noFLoop = filterCacheF == filterCacheH;
    for (int my = startY; my < endY; my++) {
        const int moduleTop = paddingStart + my * moduleStride;
        const int pxInFilterY = blockPixelIdxY - moduleTop;

        for (int mx = startX; mx < endX; mx++) {
            const int moduleIdx = my * numModulesX + mx;
            const int moduleLeft = paddingStart + mx * moduleStride;
            const int pxInFilterX = blockPixelIdxX - moduleLeft;

            const int pxIdxInFilter = pxInFilterY * filterSize + pxInFilterX;

            for (int f = 0; f < numFiltersPerGroup; f += filterCacheF) { // multiply with filterCacheF filters at a time
                const float* fLoad = conv ? &filters[pxIdxInFilter * numFilters + f]
                                          : &filters[moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInFilter * numFilters + f];
                #pragma unroll
                for (int i = 0; i < colorsPerThread*B_Y; i+= B_X*B_Y/filterCacheF) {
                    if ((colorsPerThread*B_Y) % (B_X*B_Y/filterCacheF) == 0 || i + filtersLoadY < colorsPerThread*B_Y) {
                        shFilterLoad[i * filterCacheF] = fLoad[i * filterPixels * numFilters];
                    }
                }
                //#pragma unroll

                for (int fh = f; fh < f + filterCacheF; fh += filterCacheH) {
                    //conv_img_acts_manycolor_dummy_fhLoop<B_Y, B_X, imgsPerThread, colorsPerThread, filterCacheF, filterCacheH, checkCaseBounds>(hidActs, shHidActLoad, shHidActs, shFilters, moduleIdx, numImages, hidActLoadY, hidActLoadX, blockCaseIdx, numModules, f, fh, prod);

                    const float* hLoad = &hidActs[(moduleIdx + fh * numModules) * numImages];

                    #pragma unroll
                    for (int j = 0; j < filterCacheH; j += B_Y) {
                        if (filterCacheH % B_Y == 0 || hidActLoadY + j < filterCacheH) {
                            #pragma unroll
                            for (int i = 0; i < imgsPerThread*B_X; i += B_X) {
                                if (!checkCaseBounds || blockCaseIdx + hidActLoadX + i < numImages) {
                                    shHidActLoad[j * B_X * imgsPerThread + i] = hLoad[j * numModules * numImages + i];
                                } else {
                                    shHidActLoad[j * B_X * imgsPerThread + i] = 0;
                                }
                            }
                        }
                    }

                    __syncthreads();

                    // Do some actual computation
                    // Using these variables causes register usage to go from 161 --> 123.
                    // But nonetheless, the high-register version is faster.
                    //const float* shF = &shFilters[threadIdx.y][fh-f];
                    //const float* const shF2 = &shFilters[threadIdx.y][fh];
                    //const float*  shH = &shHidActs[0][threadIdx.x];
                    #pragma unroll
                    for (int w = 0; w < filterCacheH; w++) {
                        #pragma unroll
                        for (int c = 0; c < colorsPerThread; c++) {
                            #pragma unroll
                            for (int i = 0; i < imgsPerThread; i++) {
                                prod[c][i] += shFilters[c * B_Y + threadIdx.y][fh-f + w] * shHidActs[w][threadIdx.x + i * B_X];

                            }
                        }
                    }
                    __syncthreads();

                }
            }
        }
    }
    if (scale) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * B_X < numImages) {
                #pragma unroll
                for (int c = 0; c < colorsPerThread; c++) {
                    targets[c * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * targets[c * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[c][i];
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * B_X < numImages) {
                #pragma unroll
                for (int c = 0; c < colorsPerThread; c++) {
                    targets[c * B_Y * imgPixels * numImages + i * B_X] = scaleOutputs * prod[c][i];
                }
            }
        }
    }
}

/*
 * New Titan-optimized stuff.
 */

__device__ __forceinline__ void conv_img_acts_manycolor_preload_ty_8_tx_32_c_8_ff_32_fh_16_setCoords(const int my, const int mx, const int numModulesX,
        const int paddingStart, const int moduleStride, const int blockPixelIdxY, const int blockPixelIdxX, const int filterSize, int &moduleIdx, int &pxIdxInFilter) {
    const int moduleTop = paddingStart + my * moduleStride;
    const int pxInFilterY = blockPixelIdxY - moduleTop;

    moduleIdx = my * numModulesX + mx; // out
    const int moduleLeft = paddingStart + mx * moduleStride;
    const int pxInFilterX = blockPixelIdxX - moduleLeft;

    pxIdxInFilter = pxInFilterY * filterSize + pxInFilterX; // out
}

#define IA_PRELOAD_LOOP(w,offset) _Pragma("unroll") \
for (int i = 0; i < imgsPerThread; i++) { \
    _Pragma("unroll") \
    for (int c = 0; c < colorsPerThread; c++) { \
        prod[c][i] += shFilters[c * B_Y + threadIdx.y][(w)+(offset)] * shHidActs[w][threadIdx.x * imgsPerThread + i]; \
    } \
} \

/*
 * Same loop as above but inverted.
 */
#define IA_PRELOAD_LOOP2(w,offset) _Pragma("unroll") \
for (int c = 0; c < colorsPerThread; c++) { \
    _Pragma("unroll") \
    for (int i = 0; i < imgsPerThread; i++) { \
        prod[c][i] += shFilters[c * B_Y + threadIdx.y][(w)+(offset)] * shHidActs[w][threadIdx.x * imgsPerThread + i]; \
    } \
} \

#define IA_PRELOAD_LOOP3(i,offset) _Pragma("unroll") \
for (int w = 0; w < filterCacheH; w++) { \
    _Pragma("unroll") \
    for (int c = 0; c < colorsPerThread; c++) { \
        prod[c][i] += shFilters[c * B_Y + threadIdx.y][(w)+(offset)] * shHidActs[w][threadIdx.x * imgsPerThread + i]; \
    } \
} \

#define IA_PRELOAD_W(z) wPreload[z] = fLoad[(z) * B_X*B_Y/filterCacheF * filterPixels * numFilters];
#define IA_PRELOAD_W_TX(z) wPreload[z] = tex1Dfetch<float>(filters, filtersLoadOffset + (z) * B_X*B_Y/filterCacheF * filterPixels * numFilters);
#define IA_PRELOAD_H(y,x) if (!checkCaseBounds || myCaseIdx + (x) * B_X < numImages) { \
    hPreload[y][x] =  hLoad[(y) * B_Y * numModules * numImages + (x) * B_X]; \
}
#define IA_PRELOAD_H_TX(y,x) if (!checkCaseBounds || myCaseIdx + (x) * B_X < numImages) { \
    hPreload[y][x] =  tex1Dfetch<float>(hidActs, hidActsLoadOffset + (y) * B_Y * numModules * numImages + (x) * B_X); \
}

template <int B_Y, int B_X, int imgsPerThread, int colorsPerThread, int filterCacheF, int filterCacheH, bool scale, bool checkCaseBounds, bool conv>
__global__ void
__launch_bounds__(256, 2)   // 256 threads per block, 2 blocks per multiprocessor
                            // These launch bounds ensure 25% occupancy (128 registers used)
                            // as oppposed to 13% (130 registers) achieved by defaults.
conv_img_acts_manycolor_preloadfh_ty_8_tx_32_c_8_ff_32_fh_16_tex(cudaTextureObject_t hidActs, cudaTextureObject_t filters, float* targets,
                                          const int numModulesY, const int numModulesX, const int numImages, const int numFilters,
                                          const int filterSize, const int imgSizeY, const int imgSizeX, const int paddingStart, const int moduleStride,
                                          const int numImgColors, const int numGroups,
                                          const float scaleTargets, const float scaleOutputs) {
    __shared__ float shFilters[colorsPerThread*B_Y][filterCacheF];
    __shared__ float shHidActs[filterCacheH][B_X*imgsPerThread];

    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int blockCaseIdx = (blockIdx.x % numImgBlocks) * B_X*imgsPerThread;
    const int myCaseIdx = blockCaseIdx + threadIdx.x;

    const int imgColorIdx = (blockIdx.x / numImgBlocks) * B_Y*colorsPerThread; // color idx globally
    const int numFilterColors = numImgColors / numGroups;
    const int blockGroupIdx = imgColorIdx / numFilterColors;
    const int filterColorIdx = imgColorIdx % numFilterColors; // color idx within group
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockFilterIdx = blockGroupIdx * numFiltersPerGroup;

    const int blockPixelIdx = blockIdx.y;
    const int blockPixelIdxX = blockPixelIdx % imgSizeX;
    const int blockPixelIdxY = blockPixelIdx / imgSizeX;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;
    const int tidx = threadIdx.y * B_X + threadIdx.x;
//    const int hidActLoadY = threadIdx.y % B_Y, hidActLoadX = threadIdx.x % B_X;
    //const int hidActLoadY = tidx / (B_X*imgsPerThread), hidActLoadX = tidx % (B_X*imgsPerThread);
    const int filtersLoadY = tidx / filterCacheF, filtersLoadX = tidx % filterCacheF;
    // nvcc is behaving idiotically again, these useless declarations save registers
    //const int outputY = threadIdx.y, outputX = threadIdx.x;
    //const int ty = threadIdx.y, tx = threadIdx.x;
    const int numModules = numModulesY * numModulesX;
    const int hidActsOffset = (blockFilterIdx + threadIdx.y) * numImages * numModules + myCaseIdx;
    const int filtersOffset = blockFilterIdx + (filterColorIdx + filtersLoadY) * filterPixels * numFilters + filtersLoadX;
//    hidActs += (blockFilterIdx + threadIdx.y) * numImages * numModules + myCaseIdx;
//    filters += blockFilterIdx + (filterColorIdx + filtersLoadY) * filterPixels * numFilters + filtersLoadX;
    targets += (imgColorIdx + threadIdx.y) * imgPixels * numImages + blockPixelIdx * numImages + myCaseIdx;

    float prod[colorsPerThread][imgsPerThread];
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            prod[c][i] = 0;
        }
    }

    const int startY = blockPixelIdxY - paddingStart < filterSize ? 0
                        : 1 + (blockPixelIdxY - paddingStart - filterSize) / moduleStride;
    const int endY = min(numModulesY, 1 + (blockPixelIdxY - paddingStart) / moduleStride);
    const int startX = blockPixelIdxX - paddingStart < filterSize ? 0
                        : 1 + (blockPixelIdxX - paddingStart - filterSize) / moduleStride;
    const int endX = min(numModulesX, 1 + (blockPixelIdxX - paddingStart) / moduleStride);

    float* shFilterLoad = &shFilters[filtersLoadY][filtersLoadX];
    float* shHidActLoad = &shHidActs[threadIdx.y][threadIdx.x * imgsPerThread];
    //const bool noFLoop = filterCacheF == filterCacheH;

    /*
     * Initial preload
     */
    float hPreload[filterCacheH/B_Y][imgsPerThread]; // [2][4]
    float wPreload[filterCacheF*colorsPerThread/B_X]; // [8]

    int moduleIdx, pxIdxInFilter;
    conv_img_acts_manycolor_preload_ty_8_tx_32_c_8_ff_32_fh_16_setCoords(startY, startX, numModulesX, paddingStart, moduleStride, blockPixelIdxY,
                                                                         blockPixelIdxX, filterSize, moduleIdx, pxIdxInFilter);
//    const float* fLoad = conv ? &filters[pxIdxInFilter * numFilters + 0]
//                              : &filters[moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInFilter * numFilters + 0];
    int filtersLoadOffset = filtersOffset + (conv ? pxIdxInFilter * numFilters + 0
                                                  : moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInFilter * numFilters);
    #pragma unroll
    for (int i = 0; i < colorsPerThread*B_Y; i+= B_X*B_Y/filterCacheF) {
        if ((colorsPerThread*B_Y) % (B_X*B_Y/filterCacheF) == 0 || i + filtersLoadY < colorsPerThread*B_Y) {
            wPreload[i * filterCacheF/(B_X*B_Y)] = tex1Dfetch<float>(filters, filtersLoadOffset + i * filterPixels * numFilters);
        }
    }

//    const float* hLoad = &hidActs[(moduleIdx + 0 * numModules) * numImages];
    int hidActsLoadOffset = hidActsOffset + (moduleIdx + 0 * numModules) * numImages;
    #pragma unroll
    for (int j = 0; j < filterCacheH; j += B_Y) {
        if (filterCacheH % B_Y == 0 || threadIdx.y + j < filterCacheH) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || myCaseIdx + i * B_X < numImages) {
                    hPreload[j/B_Y][i] = tex1Dfetch<float>(hidActs, hidActsLoadOffset + j * numModules * numImages + i * B_X);
                }
            }
        }
    }

    for (int my = startY; my < endY; my++) {
        const int moduleTop = paddingStart + my * moduleStride;
        const int pxInFilterY = blockPixelIdxY - moduleTop;

        for (int mx = startX; mx < endX; mx++) {
            moduleIdx = my * numModulesX + mx;
            const int moduleLeft = paddingStart + mx * moduleStride;
            const int pxInFilterX = blockPixelIdxX - moduleLeft;

            pxIdxInFilter = pxInFilterY * filterSize + pxInFilterX;
            int myNext = my, mxNext = mx, moduleIdxNext, pxIdxInFilterNext;
            const bool lastModule = my == endY - 1 && mx == endX - 1;
            if (!lastModule) {
                mxNext = mx + 1 == endX ? startX : mx + 1;
                myNext = my + (mx + 1 == endX);
            }
            conv_img_acts_manycolor_preload_ty_8_tx_32_c_8_ff_32_fh_16_setCoords(myNext, mxNext, numModulesX, paddingStart, moduleStride, blockPixelIdxY,
                                                                                 blockPixelIdxX, filterSize, moduleIdxNext, pxIdxInFilterNext);
            for (int f = 0; f < numFiltersPerGroup; f += filterCacheF) { // multiply with filterCacheF filters at a time
                #pragma unroll
                for (int i = 0; i < colorsPerThread*B_Y; i+= B_X*B_Y/filterCacheF) {
                    if ((colorsPerThread*B_Y) % (B_X*B_Y/filterCacheF) == 0 || i + filtersLoadY < colorsPerThread*B_Y) {
                        shFilterLoad[i * filterCacheF] = wPreload[i * filterCacheF/(B_X*B_Y)];
                    }
                }

                filtersLoadOffset = filtersOffset + (conv ? pxIdxInFilter * numFilters + f + filterCacheF
                                                          : moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInFilter * numFilters + f + filterCacheF);
                if (f == numFiltersPerGroup - filterCacheF) {
                    filtersLoadOffset = filtersOffset + (conv ? pxIdxInFilterNext * numFilters
                                                              : moduleIdxNext * numFilterColors * filterPixels * numFilters + pxIdxInFilterNext * numFilters);
                }

                #pragma unroll
                for (int j = 0; j < filterCacheH; j += B_Y) {
                    if (filterCacheH % B_Y == 0 || threadIdx.y + j < filterCacheH) {
                        #pragma unroll
                        for (int i = 0; i < imgsPerThread; i++) {
                            // NOTE: bank conflicts here!
                            if (!checkCaseBounds || myCaseIdx + i * B_X < numImages) {
                                shHidActLoad[j * B_X * imgsPerThread + i] = hPreload[j/B_Y][i];
                            }
                        }
                    }
                }

                __syncthreads();

                hidActsLoadOffset = hidActsOffset + (moduleIdx + (f + filterCacheH) * numModules) * numImages;

                #pragma unroll
                for (int z = 0; z < 4; ++z) {
                    IA_PRELOAD_LOOP(z,0);
                    IA_PRELOAD_W_TX(z);
                }

                #pragma unroll
                for (int z = 4; z < 12; ++z) {
                    IA_PRELOAD_LOOP(z,0);
                    IA_PRELOAD_H_TX((z-4)/4,z%4);
                }

                #pragma unroll
                for (int z = 12; z < 16; ++z) {
                    IA_PRELOAD_LOOP(z,0);
                }

                __syncthreads();

                #pragma unroll
                for (int j = 0; j < filterCacheH; j += B_Y) {
                    if (filterCacheH % B_Y == 0 || threadIdx.y + j < filterCacheH) {
                        #pragma unroll
                        for (int i = 0; i < imgsPerThread; i++) {
                            if (!checkCaseBounds || myCaseIdx + i * B_X < numImages) {
                                shHidActLoad[j * B_X * imgsPerThread + i] = hPreload[j/B_Y][i];
                            }
                        }
                    }
                }

                __syncthreads();

                hidActsLoadOffset = hidActsOffset + (moduleIdx + (f + filterCacheF) * numModules) * numImages;
                if (f == numFiltersPerGroup - filterCacheF) {
                    hidActsLoadOffset = hidActsOffset + moduleIdxNext * numImages;
                }

                #pragma unroll
                for (int z = 0; z < 4; ++z) {
                    IA_PRELOAD_LOOP(z,filterCacheH);
                    IA_PRELOAD_W_TX(z+4);
                }

                #pragma unroll
                for (int z = 4; z < 12; ++z) {
                    IA_PRELOAD_LOOP(z,filterCacheH);
                    IA_PRELOAD_H_TX((z-4)/4, z%4);
                }

                #pragma unroll
                for (int z = 12; z < 16; ++z) {
                    IA_PRELOAD_LOOP(z,filterCacheH);
                }

                __syncthreads();
            }
        }
    }
    if (scale) {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || myCaseIdx + i * B_X < numImages) {
                    targets[c * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * targets[c * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[c][i];
                }
            }
        }
    } else {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || myCaseIdx + i * B_X < numImages) {
                    targets[c * B_Y * imgPixels * numImages + i * B_X] = scaleOutputs * prod[c][i];
                }
            }
        }
    }
}


template <int B_Y, int B_X, int imgsPerThread, int colorsPerThread, int filterCacheF, int filterCacheH, bool scale, bool checkCaseBounds, bool conv>
__global__ void
//__launch_bounds__(128, 3)   // 128 threads per block, 3 blocks per multiprocessor
conv_img_acts_manycolor_preloadfh_ty_4_tx_32_c_12_ff_16_fh_16(cudaTextureObject_t hidActs, cudaTextureObject_t filters, float* targets,
                                          const int numModulesY, const int numModulesX, const int numImages, const int numFilters,
                                          const int filterSize, const int imgSizeY, const int imgSizeX, const int paddingStart, const int moduleStride,
                                          const int numImgColors, const int numGroups,
                                          const float scaleTargets, const float scaleOutputs) {
    __shared__ float shFilters[colorsPerThread*B_Y][filterCacheF];
    __shared__ float shHidActs[filterCacheH][B_X*imgsPerThread];

    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int blockCaseIdx = (blockIdx.x % numImgBlocks) * B_X*imgsPerThread;
    const int myCaseIdx = blockCaseIdx + threadIdx.x;

    const int imgColorIdx = (blockIdx.x / numImgBlocks) * B_Y*colorsPerThread; // color idx globally
    const int numFilterColors = numImgColors / numGroups;
    const int blockGroupIdx = imgColorIdx / numFilterColors;
    const int filterColorIdx = imgColorIdx % numFilterColors; // color idx within group
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockFilterIdx = blockGroupIdx * numFiltersPerGroup;

    const int blockPixelIdx = blockIdx.y;
    const int blockPixelIdxX = blockPixelIdx % imgSizeX;
    const int blockPixelIdxY = blockPixelIdx / imgSizeX;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;
    const int tidx = threadIdx.y * B_X + threadIdx.x;
//    const int hidActLoadY = threadIdx.y % B_Y, hidActLoadX = threadIdx.x % B_X;
    //const int hidActLoadY = tidx / (B_X*imgsPerThread), hidActLoadX = tidx % (B_X*imgsPerThread);
    const int filtersLoadY = tidx / filterCacheF, filtersLoadX = tidx % filterCacheF;
    // nvcc is behaving idiotically again, these useless declarations save registers
    //const int outputY = threadIdx.y, outputX = threadIdx.x;
    //const int ty = threadIdx.y, tx = threadIdx.x;
    const int numModules = numModulesY * numModulesX;

    const int hidActsOffset = (blockFilterIdx + threadIdx.y) * numImages * numModules + myCaseIdx;
    const int filtersOffset = blockFilterIdx + (filterColorIdx + filtersLoadY) * filterPixels * numFilters + filtersLoadX;

//    hidActs += (blockFilterIdx + threadIdx.y) * numImages * numModules + myCaseIdx;
//    filters += blockFilterIdx + (filterColorIdx + filtersLoadY) * filterPixels * numFilters + filtersLoadX;
    targets += (imgColorIdx + threadIdx.y) * imgPixels * numImages + blockPixelIdx * numImages + myCaseIdx;

    float prod[colorsPerThread][imgsPerThread];
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            prod[c][i] = 0;
        }
    }

    const int startY = blockPixelIdxY - paddingStart < filterSize ? 0
                        : 1 + (blockPixelIdxY - paddingStart - filterSize) / moduleStride;
    const int endY = min(numModulesY, 1 + (blockPixelIdxY - paddingStart) / moduleStride);
    const int startX = blockPixelIdxX - paddingStart < filterSize ? 0
                        : 1 + (blockPixelIdxX - paddingStart - filterSize) / moduleStride;
    const int endX = min(numModulesX, 1 + (blockPixelIdxX - paddingStart) / moduleStride);

    float* shFilterLoad = &shFilters[filtersLoadY][filtersLoadX];
    float* shHidActLoad = &shHidActs[threadIdx.y][threadIdx.x * imgsPerThread];
    //const bool noFLoop = filterCacheF == filterCacheH;

    /*
     * Initial preload
     */
    float hPreload[filterCacheH/B_Y][imgsPerThread]; // [4][4]
    float wPreload[filterCacheF*colorsPerThread/B_X]; // [6]

    int moduleIdx, pxIdxInFilter;
    conv_img_acts_manycolor_preload_ty_8_tx_32_c_8_ff_32_fh_16_setCoords(startY, startX, numModulesX, paddingStart, moduleStride, blockPixelIdxY,
                                                                         blockPixelIdxX, filterSize, moduleIdx, pxIdxInFilter);
//    const float* fLoad = conv ? &filters[pxIdxInFilter * numFilters + 0]
//                              : &filters[moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInFilter * numFilters + 0];
    int filtersLoadOffset = filtersOffset + (conv ? pxIdxInFilter * numFilters
                                                : moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInFilter * numFilters);
    #pragma unroll
    for (int i = 0; i < colorsPerThread*B_Y; i+= B_X*B_Y/filterCacheF) {
        if ((colorsPerThread*B_Y) % (B_X*B_Y/filterCacheF) == 0 || i + filtersLoadY < colorsPerThread*B_Y) {
            wPreload[i * filterCacheF/(B_X*B_Y)] = tex1Dfetch<float>(filters, filtersLoadOffset + i * filterPixels * numFilters);
        }
    }

//    const float* hLoad = &hidActs[moduleIdx * numImages];
    int hidActsLoadOffset = hidActsOffset + moduleIdx * numImages;
    #pragma unroll
    for (int j = 0; j < filterCacheH; j += B_Y) {
        if (filterCacheH % B_Y == 0 || threadIdx.y + j < filterCacheH) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || myCaseIdx + i * B_X < numImages) {
                    hPreload[j/B_Y][i] = tex1Dfetch<float>(hidActs, hidActsLoadOffset + j * numModules * numImages + i * B_X);
                }
            }
        }
    }

    for (int my = startY; my < endY; my++) {
        const int moduleTop = paddingStart + my * moduleStride;
        const int pxInFilterY = blockPixelIdxY - moduleTop;

        for (int mx = startX; mx < endX; mx++) {
            moduleIdx = my * numModulesX + mx;
            const int moduleLeft = paddingStart + mx * moduleStride;
            const int pxInFilterX = blockPixelIdxX - moduleLeft;

            pxIdxInFilter = pxInFilterY * filterSize + pxInFilterX;
            int myNext = my, mxNext = mx, moduleIdxNext, pxIdxInFilterNext;
            const bool lastModule = my == endY - 1 && mx == endX - 1;
            if (!lastModule) {
                mxNext = mx + 1 == endX ? startX : mx + 1;
                myNext = my + (mx + 1 == endX);
            }
            conv_img_acts_manycolor_preload_ty_8_tx_32_c_8_ff_32_fh_16_setCoords(myNext, mxNext, numModulesX, paddingStart, moduleStride, blockPixelIdxY,
                                                                                 blockPixelIdxX, filterSize, moduleIdxNext, pxIdxInFilterNext);
            for (int f = 0; f < numFiltersPerGroup; f += filterCacheF) { // multiply with filterCacheF filters at a time
                #pragma unroll
                for (int i = 0; i < colorsPerThread*B_Y; i+= B_X*B_Y/filterCacheF) {
                    if ((colorsPerThread*B_Y) % (B_X*B_Y/filterCacheF) == 0 || i + filtersLoadY < colorsPerThread*B_Y) {
                        shFilterLoad[i * filterCacheF] = wPreload[i * filterCacheF/(B_X*B_Y)];
                    }
                }

                filtersLoadOffset = filtersOffset + (conv ? pxIdxInFilter * numFilters + f + filterCacheF
                                                          : moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInFilter * numFilters + f + filterCacheF);
                if (f == numFiltersPerGroup - filterCacheF) {
                    filtersLoadOffset = filtersOffset + (conv ? pxIdxInFilterNext * numFilters
                                                              : moduleIdxNext * numFilterColors * filterPixels * numFilters + pxIdxInFilterNext * numFilters);
                }

                #pragma unroll
                for (int j = 0; j < filterCacheH; j += B_Y) {
                    if (filterCacheH % B_Y == 0 || threadIdx.y + j < filterCacheH) {
                        #pragma unroll
                        for (int i = 0; i < imgsPerThread; i++) {
                            // NOTE: bank conflicts here!
                            if (!checkCaseBounds || myCaseIdx + i * B_X < numImages) {
                                shHidActLoad[j * B_X * imgsPerThread + i] = hPreload[j/B_Y][i];
                            }
                        }
                    }
                }
                hidActsLoadOffset = hidActsOffset + (moduleIdx + (f + filterCacheF) * numModules) * numImages;
                if (f == numFiltersPerGroup - filterCacheF) {
                    hidActsLoadOffset = hidActsOffset + moduleIdxNext * numImages;
                }

                __syncthreads();

                // It seems that there is no point explicitly interleaving loads
                // and computations because the scheduler does that anyway.

                IA_PRELOAD_LOOP2(0,0);
                IA_PRELOAD_LOOP2(1,0);
                IA_PRELOAD_LOOP2(2,0);
                IA_PRELOAD_LOOP2(3,0);
                IA_PRELOAD_LOOP2(4,0);
                IA_PRELOAD_LOOP2(5,0);
                IA_PRELOAD_LOOP2(6,0);
                IA_PRELOAD_LOOP2(7,0);
                IA_PRELOAD_LOOP2(8,0);
                IA_PRELOAD_LOOP2(9,0);
                IA_PRELOAD_LOOP2(10,0);
                IA_PRELOAD_LOOP2(11,0);
                IA_PRELOAD_LOOP2(12,0);
                IA_PRELOAD_LOOP2(13,0);
                IA_PRELOAD_LOOP2(14,0);
                IA_PRELOAD_LOOP2(15,0);

                IA_PRELOAD_W_TX(0);
                IA_PRELOAD_W_TX(1);
                IA_PRELOAD_W_TX(2);
                IA_PRELOAD_W_TX(3);
                IA_PRELOAD_W_TX(4);
                IA_PRELOAD_W_TX(5);

                IA_PRELOAD_H_TX(0,0);
                IA_PRELOAD_H_TX(0,1);
                IA_PRELOAD_H_TX(0,2);
                IA_PRELOAD_H_TX(0,3);
                IA_PRELOAD_H_TX(1,0);
                IA_PRELOAD_H_TX(1,1);
                IA_PRELOAD_H_TX(1,2);
                IA_PRELOAD_H_TX(1,3);
                IA_PRELOAD_H_TX(2,0);
                IA_PRELOAD_H_TX(2,1);
                IA_PRELOAD_H_TX(2,2);
                IA_PRELOAD_H_TX(2,3);
                IA_PRELOAD_H_TX(3,0);
                IA_PRELOAD_H_TX(3,1);
                IA_PRELOAD_H_TX(3,2);
                IA_PRELOAD_H_TX(3,3);

                __syncthreads();
            }
        }
    }
    if (scale) {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || myCaseIdx + i * B_X < numImages) {
                    targets[c * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * targets[c * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[c][i];
                }
            }
        }
    } else {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || myCaseIdx + i * B_X < numImages) {
                    targets[c * B_Y * imgPixels * numImages + i * B_X] = scaleOutputs * prod[c][i];
                }
            }
        }
    }
}

/*
 * hidActs:         (numFilters, numModules, numImages)
 * filters:         (numFilterColors, filterPixels, numFilters)               if conv
 *                  (numModules, numFilterColors, filterPixels, numFilters)   otherwise
 * targets:         (overSample, numImgColors, imgPixels, numImages)
 *
 * Note: all of these convolution routines are optimized for the case when
 * the number of images (i.e. the minibatch size) is a multiple of 128.
 * Other batch sizes will work, but but I made no attempt whatsoever
 * to make them work fast.
 */
void _imgActs(caffe2::CUDAContext* context, caffe2::TensorCUDA* hidActs, caffe2::TensorCUDA* filters, caffe2::TensorCUDA* targets,
              int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
              float scaleTargets, float scaleOutput, bool conv) {
    CAFFE_ENFORCE(hidActs->ndim() == 2);
    CAFFE_ENFORCE(filters->ndim() == 2);
    CAFFE_ENFORCE(targets->ndim() == 2);

    int numFilterColors = numImgColors / numGroups;
    int numImages = hidActs->dim32(1);
    int numFilters = filters->dim32(1);
    int numModules = hidActs->dim32(0) / numFilters;
    int filterModuleMult = conv ? 1 : numModules;
    int filterPixels = filters->dim32(0) / (filterModuleMult * numFilterColors);
    int filterSize = sqrt(filterPixels);
    int imgPixels = imgSizeY * imgSizeX;
    int numModulesX = numModules / numModulesY;

    CAFFE_ENFORCE(numImgColors % numGroups == 0);
    CAFFE_ENFORCE(numFilters % (16*numGroups) == 0); // TODO: insisting on 32 filters due to bug in calling code below. fix that.
    CAFFE_ENFORCE(numGroups > 1 || (numImgColors > 0 && (numImgColors <= 3 || numImgColors % 2 == 0)));
    CAFFE_ENFORCE(numGroups == 1 || numFilterColors % 4 == 0);

    CAFFE_ENFORCE(filterPixels == filterSize * filterSize);
    CAFFE_ENFORCE(hidActs->dim32(0) == numModules * numFilters);
    CAFFE_ENFORCE(filters->dim32(0) == filterModuleMult * numFilterColors * filterPixels);
    CAFFE_ENFORCE(numModules == numModulesY * numModulesX);

    // These routines don't handle the case when only part of the image is visited in the convolution
    CAFFE_ENFORCE(paddingStart <= 0);
    CAFFE_ENFORCE(paddingStart + (numModulesX-1)*moduleStride + filterSize >= imgSizeX);
    CAFFE_ENFORCE(paddingStart + (numModulesY-1)*moduleStride + filterSize >= imgSizeY);
    CAFFE_ENFORCE(moduleStride <= filterSize);

    dim3 blocks;
    dim3 threads;
    int colorsPerThread, imgsPerThread;
    if (numFilterColors % 8 == 0) {
        threads = dim3(32, numFilterColors % 64 == 0 ? 8 : 4);
        colorsPerThread = numFilterColors % 64 == 0 ? 8
                        : numFilterColors % 48 == 0 ? 12
                        : numFilterColors % 32 == 0 ? 8
                        : numFilterColors % 16 == 0 ? 4
                        : 2;
        imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
        CAFFE_ENFORCE(numFilterColors % (threads.y * colorsPerThread) == 0);

        blocks = dim3(DIVUP(numImages, threads.x*imgsPerThread) * (numImgColors/(threads.y*colorsPerThread)), imgPixels);
        // NOTE: the case when channels % 32 == 0 but channels % 48 != 0 and channels % 64 != 0 has not been optimized!!
    } else if (numFilterColors > 3) {
        // NOTE: THIS CASE HAS NOT BEEN OPTIMIZED FOR KEPLER!!
        imgsPerThread = numImages % 128 == 0 ? 8 : numImages % 64 == 0 ? 4 : 2;
        threads = dim3(16, 16);
        colorsPerThread = numFilterColors % 4 == 0 ? 4 : 2;
        blocks = dim3(DIVUP(numImages,threads.x*imgsPerThread) * (numImgColors / colorsPerThread), DIVUP(imgSizeY,4) * DIVUP(imgSizeX,4));
    } else {
        // NOTE: THIS CASE HAS NOT BEEN OPTIMIZED FOR KEPLER!!
        imgsPerThread = numImages % 128 == 0 ? 8 : numImages % 64 == 0 ? 4 : 2;
        threads = dim3(16, 16);
        blocks = dim3(DIVUP(numImages,threads.x*imgsPerThread), DIVUP(imgSizeY,4) * DIVUP(imgSizeX,4));
    }
    bool checkCaseBounds = numImages % (threads.x * imgsPerThread) != 0;

    if (scaleTargets == 0) { // do not scale or use targets matrix
        targets->Resize(std::vector<int>{numImgColors*imgPixels, numImages});
    } else {
        CAFFE_ENFORCE(targets->dim32(0) == numImgColors * imgPixels);
        CAFFE_ENFORCE(targets->dim32(1) == numImages);
    }
    const bool scale = scaleTargets != 0;

    cudaTextureObject_t tex_hidacts = GetTensorTextureObject(hidActs);
    cudaTextureObject_t tex_filters = GetTensorTextureObject(filters);
    float* hidacts_data = hidActs->mutable_data<float>();
    float* filters_data = filters->mutable_data<float>();
    float* targets_data = targets->mutable_data<float>();

    cudaStream_t stream = context->cuda_stream();
//    cudaFuncSetCacheConfig(conv_img_acts_manycolor_preloadfh_ty_4_tx_32_c_12_ff_16_fh_16< 4, 32, 4, 12, 16, 16, false, false, true >, cudaFuncCachePreferShared);
//    conv_img_acts_manycolor_preloadfh_ty_4_tx_32_c_12_ff_16_fh_16< 4, 32, 4, 12, 16, 16, false, false, true ><<<blocks, threads, 0, stream>>>(
//            tex_hidacts, tex_filters, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize,
//            imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);

    //return;
//    printf("conv: %d\n", conv);
//    printf("scale: %d\n", scale);
//    printf("checkCaseBounds: %d\n", checkCaseBounds);
//    printf("numFilterColors: %d\n", numFilterColors);
//    printf("numImages: %d\n", numImages);
//    cudaStream_t stream = NVMatrix::getDefaultStream();

    if (conv == true) {
        if (scale == false) {
            if (checkCaseBounds == false) {
                if (numFilterColors % 8 == 0) {
                    if (numFilterColors % 64 == 0) {
                        if (numFilters % 32 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_preloadfh_ty_8_tx_32_c_8_ff_32_fh_16_tex< 8, 32, 4, 8, 32, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_preloadfh_ty_8_tx_32_c_8_ff_32_fh_16_tex< 8, 32, 4, 8, 32, 16, false, false, true ><<<blocks, threads, 0, stream>>>(tex_hidacts, tex_filters, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 2, 8, 32, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 2, 8, 32, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                        else if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 4, 8, 16, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 4, 8, 16, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 2, 8, 16, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 2, 8, 16, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 48 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_preloadfh_ty_4_tx_32_c_12_ff_16_fh_16< 4, 32, 4, 12, 16, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_preloadfh_ty_4_tx_32_c_12_ff_16_fh_16< 4, 32, 4, 12, 16, 16, false, false, true ><<<blocks, threads, 0, stream>>>(tex_hidacts, tex_filters, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 12, 16, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 12, 16, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 32 == 0) {
                        if (numFilters % 32 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 4, 8, 32, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 4, 8, 32, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 8, 32, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 8, 32, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                        else if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 4, 8, 16, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 4, 8, 16, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 8, 16, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 8, 16, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 16 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 4, 4, 16, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 4, 4, 16, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 4, 16, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 4, 16, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 8 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 4, 2, 16, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 4, 2, 16, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 2, 16, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 2, 16, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, false, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
                else if (numFilterColors > 3) {
                    if (numFilterColors == 4) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 8, 4, false, false, true >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 8, 4, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 4, 4, false, false, true >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 4, 4, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 2, 4, false, false, true >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 2, 4, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 2, 4, false, false, true >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 2, 4, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 2) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 8, 2, false, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 8, 2, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 4, 2, false, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 4, 2, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, false, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, false, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
                else if (numFilterColors <= 3) {
                    if (numFilterColors == 3) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 8, 3, false, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 8, 3, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 4, 3, false, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 4, 3, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 3, false, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 3, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 3, false, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 3, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 2) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 8, 2, false, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 8, 2, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 4, 2, false, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 4, 2, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, false, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, false, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 1) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 8, 1, false, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 8, 1, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 4, 1, false, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 4, 1, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 1, false, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 1, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 1, false, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 1, false, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
            }
            else if (checkCaseBounds == true) {
                if (numFilterColors % 8 == 0) {
                    if (numFilterColors % 64 == 0) {
                        if (numFilters % 32 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, false, true, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, false, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                        else if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, false, true, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, false, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 48 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, false, true, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, false, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 32 == 0) {
                        if (numFilters % 32 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, false, true, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, false, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                        else if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, false, true, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, false, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 16 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, false, true, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, false, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 8 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, false, true, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, false, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
                else if (numFilterColors > 3) {
                    if (numFilterColors == 4) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 2, 4, false, true, true >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 2, 4, false, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 2) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, false, true, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, false, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
                else if (numFilterColors <= 3) {
                    if (numFilterColors == 3) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 3, false, true, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 3, false, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 2) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, false, true, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, false, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 1) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 1, false, true, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 1, false, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
            }
        }
        else if (scale == true) {
            if (checkCaseBounds == false) {
                if (numFilterColors % 8 == 0) {
                    if (numFilterColors % 64 == 0) {
                        if (numFilters % 32 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_preloadfh_ty_8_tx_32_c_8_ff_32_fh_16_tex< 8, 32, 4, 8, 32, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_preloadfh_ty_8_tx_32_c_8_ff_32_fh_16_tex< 8, 32, 4, 8, 32, 16, true, false, true ><<<blocks, threads, 0, stream>>>(tex_hidacts, tex_filters, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 2, 8, 32, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 2, 8, 32, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                        else if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 4, 8, 16, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 4, 8, 16, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 2, 8, 16, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 2, 8, 16, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 48 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_preloadfh_ty_4_tx_32_c_12_ff_16_fh_16< 4, 32, 4, 12, 16, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_preloadfh_ty_4_tx_32_c_12_ff_16_fh_16< 4, 32, 4, 12, 16, 16, true, false, true ><<<blocks, threads, 0, stream>>>(tex_hidacts, tex_filters, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 12, 16, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 12, 16, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 32 == 0) {
                        if (numFilters % 32 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 4, 8, 32, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 4, 8, 32, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 8, 32, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 8, 32, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                        else if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 4, 8, 16, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 4, 8, 16, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 8, 16, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 8, 16, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 16 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 4, 4, 16, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 4, 4, 16, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 4, 16, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 4, 16, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 8 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 4, 2, 16, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 4, 2, 16, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 2, 16, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 2, 16, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, true, false, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
                else if (numFilterColors > 3) {
                    if (numFilterColors == 4) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 8, 4, true, false, true >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 8, 4, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 4, 4, true, false, true >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 4, 4, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 2, 4, true, false, true >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 2, 4, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 2, 4, true, false, true >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 2, 4, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 2) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 8, 2, true, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 8, 2, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 4, 2, true, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 4, 2, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, true, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, true, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
                else if (numFilterColors <= 3) {
                    if (numFilterColors == 3) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 8, 3, true, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 8, 3, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 4, 3, true, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 4, 3, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 3, true, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 3, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 3, true, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 3, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 2) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 8, 2, true, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 8, 2, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 4, 2, true, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 4, 2, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, true, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, true, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 1) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 8, 1, true, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 8, 1, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 4, 1, true, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 4, 1, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 1, true, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 1, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 1, true, false, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 1, true, false, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
            }
            else if (checkCaseBounds == true) {
                if (numFilterColors % 8 == 0) {
                    if (numFilterColors % 64 == 0) {
                        if (numFilters % 32 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, true, true, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, true, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                        else if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, true, true, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, true, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 48 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, true, true, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, true, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 32 == 0) {
                        if (numFilters % 32 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, true, true, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, true, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                        else if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, true, true, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, true, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 16 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, true, true, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, true, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 8 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, true, true, true >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, true, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
                else if (numFilterColors > 3) {
                    if (numFilterColors == 4) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 2, 4, true, true, true >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 2, 4, true, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 2) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, true, true, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, true, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
                else if (numFilterColors <= 3) {
                    if (numFilterColors == 3) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 3, true, true, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 3, true, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 2) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, true, true, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, true, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 1) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 1, true, true, true >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 1, true, true, true ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
            }
        }
    }
    else if (conv == false) {
        if (scale == false) {
            if (checkCaseBounds == false) {
                if (numFilterColors % 8 == 0) {
                    if (numFilterColors % 64 == 0) {
                        if (numFilters % 32 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_preloadfh_ty_8_tx_32_c_8_ff_32_fh_16_tex< 8, 32, 4, 8, 32, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_preloadfh_ty_8_tx_32_c_8_ff_32_fh_16_tex< 8, 32, 4, 8, 32, 16, false, false, false ><<<blocks, threads, 0, stream>>>(tex_hidacts, tex_filters, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 2, 8, 32, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 2, 8, 32, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                        else if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 4, 8, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 4, 8, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 2, 8, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 2, 8, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 48 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_preloadfh_ty_4_tx_32_c_12_ff_16_fh_16< 4, 32, 4, 12, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_preloadfh_ty_4_tx_32_c_12_ff_16_fh_16< 4, 32, 4, 12, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(tex_hidacts, tex_filters, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 12, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 12, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 32 == 0) {
                        if (numFilters % 32 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 4, 8, 32, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 4, 8, 32, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 8, 32, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 8, 32, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                        else if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 4, 8, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 4, 8, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 8, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 8, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 16 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 4, 4, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 4, 4, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 4, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 4, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 8 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 4, 2, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 4, 2, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 2, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 2, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
                else if (numFilterColors > 3) {
                    if (numFilterColors == 4) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 8, 4, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 8, 4, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 4, 4, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 4, 4, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 2, 4, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 2, 4, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 2, 4, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 2, 4, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 2) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 8, 2, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 8, 2, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 4, 2, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 4, 2, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
                else if (numFilterColors <= 3) {
                    if (numFilterColors == 3) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 8, 3, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 8, 3, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 4, 3, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 4, 3, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 3, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 3, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 3, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 3, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 2) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 8, 2, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 8, 2, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 4, 2, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 4, 2, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 1) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 8, 1, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 8, 1, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 4, 1, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 4, 1, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 1, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 1, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 1, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 1, false, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
            }
            else if (checkCaseBounds == true) {
                if (numFilterColors % 8 == 0) {
                    if (numFilterColors % 64 == 0) {
                        if (numFilters % 32 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, false, true, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, false, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                        else if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, false, true, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, false, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 48 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, false, true, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, false, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 32 == 0) {
                        if (numFilters % 32 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, false, true, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, false, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                        else if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, false, true, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, false, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 16 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, false, true, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, false, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 8 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, false, true, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, false, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
                else if (numFilterColors > 3) {
                    if (numFilterColors == 4) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 2, 4, false, true, false >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 2, 4, false, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 2) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, false, true, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, false, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
                else if (numFilterColors <= 3) {
                    if (numFilterColors == 3) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 3, false, true, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 3, false, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 2) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, false, true, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, false, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 1) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 1, false, true, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 1, false, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
            }
        }
        else if (scale == true) {
            if (checkCaseBounds == false) {
                if (numFilterColors % 8 == 0) {
                    if (numFilterColors % 64 == 0) {
                        if (numFilters % 32 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_preloadfh_ty_8_tx_32_c_8_ff_32_fh_16_tex< 8, 32, 4, 8, 32, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_preloadfh_ty_8_tx_32_c_8_ff_32_fh_16_tex< 8, 32, 4, 8, 32, 16, true, false, false ><<<blocks, threads, 0, stream>>>(tex_hidacts, tex_filters, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 2, 8, 32, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 2, 8, 32, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                        else if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 4, 8, 16, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 4, 8, 16, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 2, 8, 16, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 2, 8, 16, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 48 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_preloadfh_ty_4_tx_32_c_12_ff_16_fh_16< 4, 32, 4, 12, 16, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_preloadfh_ty_4_tx_32_c_12_ff_16_fh_16< 4, 32, 4, 12, 16, 16, true, false, false ><<<blocks, threads, 0, stream>>>(tex_hidacts, tex_filters, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 12, 16, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 12, 16, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 32 == 0) {
                        if (numFilters % 32 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 4, 8, 32, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 4, 8, 32, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 8, 32, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 8, 32, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                        else if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 4, 8, 16, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 4, 8, 16, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 8, 16, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 8, 16, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 16 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 4, 4, 16, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 4, 4, 16, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 4, 16, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 4, 16, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 8 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 4, 2, 16, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 4, 2, 16, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 2, 16, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 2, 16, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, true, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
                else if (numFilterColors > 3) {
                    if (numFilterColors == 4) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 8, 4, true, false, false >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 8, 4, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 4, 4, true, false, false >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 4, 4, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 2, 4, true, false, false >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 2, 4, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 2, 4, true, false, false >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 2, 4, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 2) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 8, 2, true, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 8, 2, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 4, 2, true, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 4, 2, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, true, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, true, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
                else if (numFilterColors <= 3) {
                    if (numFilterColors == 3) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 8, 3, true, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 8, 3, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 4, 3, true, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 4, 3, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 3, true, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 3, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 3, true, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 3, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 2) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 8, 2, true, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 8, 2, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 4, 2, true, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 4, 2, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, true, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, true, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 1) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 8, 1, true, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 8, 1, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 4, 1, true, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 4, 1, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 1, true, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 1, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 1, true, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 1, true, false, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
            }
            else if (checkCaseBounds == true) {
                if (numFilterColors % 8 == 0) {
                    if (numFilterColors % 64 == 0) {
                        if (numFilters % 32 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, true, true, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, true, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                        else if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, true, true, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, true, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 48 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, true, true, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, true, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 32 == 0) {
                        if (numFilters % 32 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, true, true, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, true, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                        else if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, true, true, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, true, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 16 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, true, true, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, true, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 8 == 0) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, true, true, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, true, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
                else if (numFilterColors > 3) {
                    if (numFilterColors == 4) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 2, 4, true, true, false >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 2, 4, true, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 2) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, true, true, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, true, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
                else if (numFilterColors <= 3) {
                    if (numFilterColors == 3) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 3, true, true, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 3, true, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 2) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, true, true, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, true, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 1) {
                        if (numFilters % 16 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 1, true, true, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 1, true, true, false ><<<blocks, threads, 0, stream>>>(hidacts_data, filters_data, targets_data, numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
            }
        }
    }

    checkCudaErrors(cudaDestroyTextureObject(tex_hidacts));
    checkCudaErrors(cudaDestroyTextureObject(tex_filters));

    getLastCudaError("imgActs: kernel execution failed");
}


void convImgActs(caffe2::CUDAContext* context, caffe2::TensorCUDA* hidActs, caffe2::TensorCUDA* filters, caffe2::TensorCUDA* targets,
                 int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups) {
    _imgActs(context, hidActs, filters, targets, imgSizeY, imgSizeX, numModulesY, paddingStart, moduleStride, numImgColors, numGroups, 0, 1, true);
}

void convImgActs(caffe2::CUDAContext* context, caffe2::TensorCUDA* hidActs, caffe2::TensorCUDA* filters, caffe2::TensorCUDA* targets,
                 int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
                 float scaleTargets, float scaleOutput) {
    _imgActs(context, hidActs, filters, targets, imgSizeY, imgSizeX, numModulesY, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, true);
}

void localImgActs(caffe2::CUDAContext* context, caffe2::TensorCUDA* hidActs, caffe2::TensorCUDA* filters, caffe2::TensorCUDA* targets,
                  int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups) {
    _imgActs(context, hidActs, filters, targets, imgSizeY, imgSizeX, numModulesY, paddingStart, moduleStride, numImgColors, numGroups, 0, 1, false);
}

void localImgActs(caffe2::CUDAContext* context, caffe2::TensorCUDA* hidActs, caffe2::TensorCUDA* filters, caffe2::TensorCUDA* targets,
                  int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
                  float scaleTargets, float scaleOutput) {
    _imgActs(context, hidActs, filters, targets, imgSizeY, imgSizeX, numModulesY, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, false);
}

