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

#define LO16(x)     ((x) & 0x0000FFFF)
#define HI16(x)     ((x) >> 16)

#define WA_LOOP(r) _Pragma("unroll") \
for (int c = 0; c < colorsPerThread; c++) { \
    _Pragma("unroll") \
    for (int f = 0; f < filtersPerThread; f++) { \
        prod[f][c] += shImages[threadIdx.y + c * B_Y][(r)] * shHidActs[threadIdx.x + f * B_X][(r)]; \
    } \
}

#define WA_LOOP2(r) _Pragma("unroll") \
for (int f = 0; f < filtersPerThread; f++) { \
    _Pragma("unroll") \
    for (int c = 0; c < colorsPerThread; c++) { \
        prod[f][c] += shImages[threadIdx.y + c * B_Y][(r)] * shHidActs[threadIdx.x + f * B_X][(r)]; \
    } \
}

#define WA_IMLOAD(r) imPreload[r] = im[(r) * B_X * B_Y / preloadCases * imgPixels * imgStride];
#define WA_IMLOAD_TX(r) imPreload[r] = tex1Dfetch<float>(images, imgOffset2 + (r) * B_X * B_Y / preloadCases * imgPixels * imgStride);
#define WA_HALOAD(r) haPreload[r] = ha[(r) * B_X * B_Y / preloadCases * numImages * numModules];
#define WA_HALOAD_TX(r) haPreload[r] = tex1Dfetch<float>(hidActs, hidActsOffset2 + (r) * B_X * B_Y / preloadCases * numImages * numModules);

__device__ __forceinline__ void conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_8_r_16_setCoords(
        const int my, const int mx, const int paddingStart, const int numModulesX, const int moduleStride,
        const int blockPixelY, const int blockPixelX, const int imgSizeX,
        const int imgStride, int& pixIdx, int& m) {
    const int imgLoadModPosY = paddingStart + my * moduleStride;
    const int imgLoadModPosX = paddingStart + mx * moduleStride;
    const int pxY = imgLoadModPosY + blockPixelY; // pixel x,y coords in image
    const int pxX = imgLoadModPosX + blockPixelX;
    pixIdx = (pxY * imgSizeX + pxX) * imgStride; // pixel idx in image
    m = my * numModulesX + mx;
}

/*
 * Each block computes weight gradients for B_Y * pixelsPerThread pixels and B_X filters
 * threadIdx.x determines filter
 * threadIdx.y determines pixel in filter
 *
 * blockIdx.x determines filter batch of B_X * filtersPerThread, module batch of partialSum
 * blockIdx.y determines pixel batch of B_Y * pixelsPerThread
 *
 * Number of filters must be divisible by B_X * filtersPerThread
 * Number of images (cases) should be divisible by preloadCases if checkCaseBounds is false.
 *
 * images:      (numColors, imgSizeY, imgSizeX, numImages), with stride given
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * targets:     (numModulesY*numModulesX/partialSum, numColors, filterPixels, numFilters)
 *
 * B_Y * B_X should be divisible by preloadCases.
 * preloadCases one of 16, 32.
 * B_X one of 4, 8, 16, 32
 * B_Y arbitrary (satisfying divisibility constraints)
 * numModules must be divisible by partialSum
 * pixelsPerThread must be divisible by pixelCache
 *
 * After adding pixelsPerThread, register usage went from 20 to 23 (when pixelsPerThread = 1)...
 * so the compiler is messing up here somehow. It's unable to optimize that case away.
 */
template <int B_Y, int B_X, int pixelCache, int pixelsPerThread, int filtersPerThread, int preloadCases, int numColors, bool scale, bool checkCaseBounds>
__global__ void conv_weight_acts_c_kepler(float* images, float* hidActs, float* targets,
                                   const int numImages, const int numFilters,
                                   const int numModulesY, const int numModulesX,
                                   const int imgSizeY, const int imgSizeX, const int filterSize,
                                   const int paddingStart, const int moduleStride, const int imgStride,
                                   const int partialSum,
                                   const float scaleTargets, const float scaleOutputs) {
    __shared__ float shImages[pixelCache * B_Y * numColors][preloadCases]; // preload preloadCases cases of B_Y * pixelsPerThread pixels
    __shared__ float shHidActs[B_X * filtersPerThread][preloadCases + 1]; // preload preloadCases cases of B_X hidActs

    const int tidx = B_X * threadIdx.y + threadIdx.x;
    const int loadY = tidx / preloadCases, loadX = tidx % preloadCases;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;

    const int filterBlocksPerModule = numFilters / (B_X*filtersPerThread);
    const int outputModuleIdx = blockIdx.x / filterBlocksPerModule;
    const int moduleIdx = partialSum * outputModuleIdx;
    const int blockFilterIdx = B_X * filtersPerThread* (blockIdx.x % filterBlocksPerModule);

//    const int moduleStride = (imgSize - filterSize + 1) / numModulesX;
    const int numModules = numModulesY * numModulesX;

    const int blockPixelOffset = blockIdx.y * B_Y * pixelsPerThread;

    images += loadX;
    hidActs += blockFilterIdx * numImages * numModules
            + loadY * numImages * numModules
            + loadX;

    targets += (outputModuleIdx * numFilters) * filterPixels * numColors
            + blockPixelOffset * numFilters
            + blockFilterIdx
            + threadIdx.y * numFilters + threadIdx.x;

    float prod[numColors][pixelsPerThread][filtersPerThread];
    #pragma unroll
    for (int c = 0; c < numColors; c++) {
        #pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                prod[c][p][f] = 0;
            }
        }
    }

    __shared__ int pxIdxes[B_Y*pixelsPerThread];
    //__shared__ bool isPxInImage[B_Y*pixelsPerThread];
    for (int m = moduleIdx; m < moduleIdx + partialSum; m++) {

        __syncthreads();
        if (tidx < B_Y * pixelsPerThread) {
            const int imgLoadModPosY = paddingStart + (m / numModulesX) * moduleStride;
            const int imgLoadModPosX = paddingStart + (m % numModulesX) * moduleStride;
            int pxY = (imgLoadModPosY + (blockPixelOffset + tidx) / filterSize);
            int pxX = (imgLoadModPosX + (blockPixelOffset + tidx) % filterSize);
            int pixIdx = (pxY * imgSizeX + pxX) * imgStride;
            pxIdxes[tidx] = pxY >= 0 && pxY < imgSizeY && pxX >= 0 && pxX < imgSizeX ? pixIdx : -1;
            //isPxInImage[tidx] = ;
        }
        __syncthreads();
        for (int caseIdx = 0; caseIdx < numImages; caseIdx += preloadCases) {
            if (/*loadY < B_X*filtersPerThread &&*/ (!checkCaseBounds || caseIdx + loadX < numImages)) {
                #pragma unroll
                for (int y = 0; y < B_X*filtersPerThread; y += (B_X * B_Y) / preloadCases) {
                    // Make sure number of rows in the array is divisible by number of rows filled per iteration
                    if ((B_X*filtersPerThread) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_X*filtersPerThread) {
                        shHidActs[loadY+y][loadX]= hidActs[caseIdx + y * numImages * numModules + m * numImages];
                    }
                }
            }
            #pragma unroll
            for (int pp = 0; pp < pixelsPerThread; pp += pixelCache) {
                //if (loadY < B_Y * pixelCache) { // This condition is not necessary for correctness, but it speeds things a bit
                /*
                 * As long as B_Y * B_X is divisible by preloadCases this will loop the right
                 * number of times.
                 *
                 * This will load some imgGrads from filter pixels that don't exit (it'll set those to 0),
                 * but the code does not produce any output for those pixels (see last lines).
                 */
                #pragma unroll
                for (int y = 0; y < B_Y * pixelCache; y += (B_X * B_Y) / preloadCases) {
                    // Make sure number of rows in the array is divisible by number of rows filled per iteration
                    if ((B_Y * pixelCache) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_Y * pixelCache) {
                        const int pxIdx = pp * B_Y + loadY + y; // pixel idx in filter

                        if (pxIdx + blockPixelOffset < filterPixels && (!checkCaseBounds || caseIdx + loadX < numImages)) {
                            const int pixIdx = pxIdxes[pxIdx];//(pxY * imgSizeX + pxX) * imgStride;

                            if (pixIdx >= 0) {
                                #pragma unroll
                                for (int c = 0; c < numColors; c++) {
                                    shImages[loadY+y + c * pixelCache * B_Y][loadX] = images[caseIdx + c * imgPixels * imgStride + pixIdx];
                                }
                            } else {
                                #pragma unroll
                                for (int c = 0; c < numColors; c++) {
                                    shImages[loadY+y + c * pixelCache * B_Y][loadX] = 0;
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int c = 0; c < numColors; c++) {
                                shImages[loadY+y + c * pixelCache * B_Y][loadX]= 0;
                            }
                        }
                    }
                }
                //}


                __syncthreads();

                #pragma unroll
                for (int i = 0; i < preloadCases; i++) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        #pragma unroll
                        for (int p = 0; p < pixelCache; p++) {
                            #pragma unroll
                            for (int c = 0; c < numColors; c++) {
                                prod[c][pp + p][f] += shImages[threadIdx.y + p * B_Y + c * pixelCache * B_Y][i] * shHidActs[threadIdx.x + f * B_X][i];
                            }
                        }
                    }
                }

                __syncthreads();
            }
        }
    }

    if (scale) {
        #pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
            if (blockPixelOffset + p * B_Y + threadIdx.y < filterPixels) {
                #pragma unroll
                for (int c = 0; c < numColors; c++) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        targets[p * B_Y * numFilters + c * filterPixels * numFilters + f * B_X] = scaleTargets * targets[p * B_Y * numFilters + c * filterPixels * numFilters + f * B_X] + scaleOutputs * prod[c][p][f];
                    }
                }
            }
        }
    } else {
        #pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
            if (blockPixelOffset + p * B_Y + threadIdx.y < filterPixels) {
                #pragma unroll
                for (int c = 0; c < numColors; c++) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        targets[p * B_Y * numFilters + c * filterPixels * numFilters + f * B_X] = scaleOutputs * prod[c][p][f];
                    }
                }
            }
        }
    }
}

/*
 * Each block computes weight gradients for 1 pixel, B_Y * colorsPerThread colors and B_X * filtersPerThread filters
 * threadIdx.x determines filter
 * threadIdx.y determines color
 *
 * blockIdx.x determines filter batch of B_X * filtersPerThread, module batch of partialSum
 * blockIdx.y determines color batch of B_Y * colorsPerThread
 * blockIdx.z determines pixel in filter
 *            NOTE: blockIdx.z is limited to values < 2^16. This means that this routine will
 *                  fail for filters >= 256*256. I'm assuming I won't ever use such large filters.

 * images:      (numImgColors, imgSizeY, imgSizeX, numImages), with stride given
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * targets:     (numModulesY*numModulesX/partialSum, numFilterColors, filterPixels, numFilters)

 * B_X * B_Y must be divisible by preloadCases
 */
template <int B_Y, int B_X, int filtersPerThread, int colorsPerThread, int preloadCases, bool scale>
__global__ void conv_weight_acts_mc_mf_kepler(float* images, float* hidActs, float* targets,
                                       const int numImages, const int numFilters,
                                       const int numModulesY, const int numModulesX,
                                       const int imgSizeY, const int imgSizeX, const int filterSize,
                                       const int paddingStart, const int moduleStride, const int imgStride,
                                       const int numImgColors, const int numGroups, const int partialSum,
                                       const float scaleTargets, const float scaleOutputs) {
    __shared__ float shImages[colorsPerThread * B_Y][preloadCases]; // preload preloadCases cases
    __shared__ float shHidActs[filtersPerThread * B_X][preloadCases + 1]; // preload preloadCases cases of B_X hidacts

    const int tidx = B_X * threadIdx.y + threadIdx.x;
    const int loadY = tidx / preloadCases, loadX = tidx % preloadCases;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;

    const int numFilterBlocks = numFilters / (B_X * filtersPerThread);
    const int outputModuleIdx = blockIdx.x / numFilterBlocks;
    const int moduleIdx = partialSum * outputModuleIdx;
    const int blockFilterIdx = filtersPerThread * B_X * (blockIdx.x % numFilterBlocks);
    const int numModules = numModulesY * numModulesX;

    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;
    const int numFilterColors = numImgColors / numGroups;

    const int blockPixelOffset = blockIdx.z; // pixel idx in filter
    const int blockPixelY = blockPixelOffset / filterSize, blockPixelX = blockPixelOffset % filterSize;
    const int blockFilterColorIdx = blockIdx.y  * B_Y * colorsPerThread;
    const int imgColorIdx = blockFilterColorIdx + blockGroupIdx * numFilterColors;

    images += (imgColorIdx + loadY) * imgPixels * imgStride + loadX;

    hidActs +=
             blockFilterIdx * numImages * numModules
            + loadY * numImages * numModules
            + loadX;

    targets += outputModuleIdx * numFilters * filterPixels * numFilterColors
            + (blockFilterColorIdx + threadIdx.y) * filterPixels * numFilters
            + blockPixelOffset * numFilters
            + blockFilterIdx
            + threadIdx.x;
    //if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) return;
    float* shHidActLoad = &shHidActs[loadY][loadX];
    float* shImgLoad = &shImages[loadY][loadX];
    float prod[colorsPerThread][filtersPerThread];
    #pragma unroll
    for (int c = 0; c < colorsPerThread; c++) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
            prod[c][f] = 0;
        }
    }

    for (int m = moduleIdx; m < moduleIdx + partialSum; m++) {
        const int imgLoadModPosY = paddingStart + (m / numModulesX) * moduleStride;
        const int imgLoadModPosX = paddingStart + (m % numModulesX) * moduleStride;
        const int pxY = imgLoadModPosY + blockPixelY; // pixel x,y coords in image
        const int pxX = imgLoadModPosX + blockPixelX;
        const int pixIdx = (pxY * imgSizeX + pxX) * imgStride; // pixel idx in image
        if (pxY >= 0 && pxY < imgSizeY && pxX >= 0 && pxX < imgSizeX) {
            for (int caseIdx = 0; caseIdx < numImages; caseIdx += preloadCases) {
                // Checking this condition actually makes things faster ... :/
                // So I've removed the !checkCaseBounds flag and just check it all the time.
                if (caseIdx + loadX < numImages) {
                    /*
                     * As long as B_Y * B_X is divisible by preloadCases this will loop the right
                     * number of times.
                     *
                     * This will load some images from filter pixels that don't exist (it'll set those to 0),
                     * but the code does not produce any output for those pixels (see last lines).
                     */
                    if (loadY < B_Y * colorsPerThread) {
                        #pragma unroll
                        for (int y = 0; y < B_Y * colorsPerThread; y += (B_X * B_Y) / preloadCases) {
                            // Make sure number of rows in the array is divisible by number of rows filled per iteration
                            if ((B_Y*colorsPerThread) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_Y*colorsPerThread) {
                                shImgLoad[(y) * preloadCases] = images[caseIdx + y * imgPixels * imgStride + pixIdx];
                            }
                        }
                    }

                    if (loadY < B_X * filtersPerThread) {
                        #pragma unroll
                        for (int y = 0; y < B_X * filtersPerThread; y += (B_X * B_Y) / preloadCases) {
                            // Make sure number of rows in the array is divisible by number of rows filled per iteration
                            if ((B_X * filtersPerThread) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_X * filtersPerThread) {
                                shHidActLoad[y * (preloadCases + 1)] = hidActs[caseIdx + y * numImages * numModules + m * numImages];
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for (int y = 0; y < B_Y * colorsPerThread; y += (B_X * B_Y) / preloadCases) {
                        // Make sure number of rows in the array is divisible by number of rows filled per iteration
                        if ((B_Y*colorsPerThread) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_Y*colorsPerThread) {
                            shImgLoad[(y) * preloadCases] = 0;
                        }
                    }
                    #pragma unroll
                    for (int y = 0; y < B_X * filtersPerThread; y += (B_X * B_Y) / preloadCases) {
                        // Make sure number of rows in the array is divisible by number of rows filled per iteration
                        if ((B_X * filtersPerThread) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_X * filtersPerThread) {
                            shHidActLoad[y * (preloadCases + 1)] = 0;
                        }
                    }
                }

                __syncthreads();
                #pragma unroll
                for (int i = 0; i < preloadCases; i++) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        #pragma unroll
                        for (int c = 0; c < colorsPerThread; c++) {
                            prod[c][f] += shImages[threadIdx.y + c * B_Y][i] * shHidActs[threadIdx.x + f * B_X][i];
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
    if (scale) {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                targets[c * B_Y * filterPixels * numFilters + f * B_X] = scaleTargets * targets[c * B_Y * filterPixels * numFilters + f * B_X] + scaleOutputs * prod[c][f];
            }
        }
    } else {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                targets[c * B_Y * filterPixels * numFilters + f * B_X] = scaleOutputs * prod[c][f];
            }
        }
    }
}


/*
 * Each block computes weight gradients for 1 pixel, B_Y * colorsPerThread colors and B_X * filtersPerThread filters
 * threadIdx.x determines filter
 * threadIdx.y determines color
 *
 * blockIdx.x determines filter batch of B_X * filtersPerThread, module batch of partialSum
 * blockIdx.y determines color batch of B_Y * colorsPerThread
 * blockIdx.z determines pixel in filter
 *            NOTE: blockIdx.z is limited to values < 2^16. This means that this routine will
 *                  fail for filters >= 256*256. I'm assuming I won't ever use such large filters.

 * images:      (numImgColors, imgSizeY, imgSizeX, numImages), with stride given
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * targets:     (numModulesY*numModulesX/partialSum, numFilterColors, filterPixels, numFilters)

 * B_X * B_Y must be divisible by preloadCases
 */
template <int B_Y, int B_X, int filtersPerThread, int colorsPerThread, int preloadCases, bool scale>
__global__ void conv_weight_acts_mc_mf_kepler_sw(float* images, float* hidActs, float* targets,
                                       const int numImages, const int numFilters,
                                       const int numModulesY, const int numModulesX,
                                       const int imgSizeY, const int imgSizeX, const int filterSize,
                                       const int paddingStart, const int moduleStride, const int imgStride,
                                       const int numImgColors, const int numGroups, const int sumWidth,
                                       const float scaleTargets, const float scaleOutputs) {
    __shared__ float shImages[colorsPerThread * B_Y][preloadCases]; // preload preloadCases cases
    __shared__ float shHidActs[filtersPerThread * B_X][preloadCases + 1]; // preload preloadCases cases of B_X hidacts

    const int tidx = B_X * threadIdx.y + threadIdx.x;
    const int loadY = tidx / preloadCases, loadX = tidx % preloadCases;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;

    const int numFilterBlocks = numFilters / (B_X * filtersPerThread);
    const int blockModuleChunkIdx = blockIdx.x / numFilterBlocks;

    const int numModuleChunksX = DIVUP(numModulesX, sumWidth);
//    const int numModuleChunksY = DIVUP(numModulesY, sumWidth);

    const int blockModuleChunkX = blockModuleChunkIdx % numModuleChunksX;
    const int blockModuleChunkY = blockModuleChunkIdx / numModuleChunksX;

    const int blockModuleStartX = blockModuleChunkX * sumWidth;
    const int blockModuleStartY = blockModuleChunkY * sumWidth;

    const int blockFilterIdx = filtersPerThread * B_X * (blockIdx.x % numFilterBlocks);
    const int numModules = numModulesY * numModulesX;

    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;
    const int numFilterColors = numImgColors / numGroups;

    const int blockPixelOffset = blockIdx.z; // pixel idx in filter
    const int blockPixelY = blockPixelOffset / filterSize, blockPixelX = blockPixelOffset % filterSize;
    const int blockFilterColorIdx = blockIdx.y  * B_Y * colorsPerThread;
    const int imgColorIdx = blockFilterColorIdx + blockGroupIdx * numFilterColors;

    images += (imgColorIdx + loadY) * imgPixels * imgStride + loadX;

    hidActs +=
             blockFilterIdx * numImages * numModules
            + loadY * numImages * numModules
            + loadX;

    targets += blockModuleChunkIdx * numFilters * filterPixels * numFilterColors
            + (blockFilterColorIdx + threadIdx.y) * filterPixels * numFilters
            + blockPixelOffset * numFilters
            + blockFilterIdx
            + threadIdx.x;
    //if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) return;

    const int mStartX = max(blockModuleStartX, DIVUP(-blockPixelX - paddingStart, moduleStride));
    const int mStartY = max(blockModuleStartY, DIVUP(-blockPixelY - paddingStart, moduleStride));
    const int mEndX = min(numModulesX, min(blockModuleStartX + sumWidth, DIVUP(imgSizeX - blockPixelX - paddingStart, moduleStride)));
    const int mEndY = min(numModulesY, min(blockModuleStartY + sumWidth, DIVUP(imgSizeY - blockPixelY - paddingStart, moduleStride)));

//    if (mStartY == mEndY || mStartX == mEndX) {
//        return;
//    }

    float* shHidActLoad = &shHidActs[loadY][loadX];
    float* shImgLoad = &shImages[loadY][loadX];
    float prod[colorsPerThread][filtersPerThread];
    #pragma unroll
    for (int c = 0; c < colorsPerThread; c++) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
            prod[c][f] = 0;
        }
    }

    /*
     * Note; iterating this way is about 1% slower and uses a few more registers than iterating
     * over the modules linearly. But it's consistent with the preload routines,
     * so I'm using it.
     */
    for (int my = mStartY; my < mEndY; my++) {
        const int imgLoadModPosY = paddingStart + my * moduleStride;
        const int pxY = imgLoadModPosY + blockPixelY; // pixel x,y coords in image
        for (int mx = mStartX; mx < mEndX; mx++) {
            const int m = my * numModulesX + mx;
            const int imgLoadModPosX = paddingStart + mx * moduleStride;
            const int pxX = imgLoadModPosX + blockPixelX;
            const int pixIdx = (pxY * imgSizeX + pxX) * imgStride; // pixel idx in image
            for (int caseIdx = 0; caseIdx < numImages; caseIdx += preloadCases) {
                // Checking this condition actually makes things faster ... :/
                // So I've removed the !checkCaseBounds flag and just check it all the time.
                if (caseIdx + loadX < numImages) {
                    /*
                     * As long as B_Y * B_X is divisible by preloadCases this will loop the right
                     * number of times.
                     *
                     * This will load some images from filter pixels that don't exist (it'll set those to 0),
                     * but the code does not produce any output for those pixels (see last lines).
                     */
                    if (loadY < B_Y * colorsPerThread) {
                        #pragma unroll
                        for (int y = 0; y < B_Y * colorsPerThread; y += (B_X * B_Y) / preloadCases) {
                            // Make sure number of rows in the array is divisible by number of rows filled per iteration
                            if ((B_Y*colorsPerThread) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_Y*colorsPerThread) {
                                shImgLoad[(y) * preloadCases] = images[caseIdx + y * imgPixels * imgStride + pixIdx];
                            }
                        }
                    }

                    if (loadY < B_X * filtersPerThread) {
                        #pragma unroll
                        for (int y = 0; y < B_X * filtersPerThread; y += (B_X * B_Y) / preloadCases) {
                            // Make sure number of rows in the array is divisible by number of rows filled per iteration
                            if ((B_X * filtersPerThread) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_X * filtersPerThread) {
                                shHidActLoad[y * (preloadCases + 1)] = hidActs[caseIdx + y * numImages * numModules + m * numImages];
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for (int y = 0; y < B_Y * colorsPerThread; y += (B_X * B_Y) / preloadCases) {
                        // Make sure number of rows in the array is divisible by number of rows filled per iteration
                        if ((B_Y*colorsPerThread) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_Y*colorsPerThread) {
                            shImgLoad[(y) * preloadCases] = 0;
                        }
                    }
                    #pragma unroll
                    for (int y = 0; y < B_X * filtersPerThread; y += (B_X * B_Y) / preloadCases) {
                        // Make sure number of rows in the array is divisible by number of rows filled per iteration
                        if ((B_X * filtersPerThread) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_X * filtersPerThread) {
                            shHidActLoad[y * (preloadCases + 1)] = 0;
                        }
                    }
                }

                __syncthreads();
                #pragma unroll
                for (int i = 0; i < preloadCases; i++) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        #pragma unroll
                        for (int c = 0; c < colorsPerThread; c++) {
                            prod[c][f] += shImages[threadIdx.y + c * B_Y][i] * shHidActs[threadIdx.x + f * B_X][i];
                        }
                    }
                }
                __syncthreads();
            }

        }
    }
    if (scale) {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                targets[c * B_Y * filterPixels * numFilters + f * B_X] = scaleTargets * targets[c * B_Y * filterPixels * numFilters + f * B_X] + scaleOutputs * prod[c][f];
            }
        }
    } else {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                targets[c * B_Y * filterPixels * numFilters + f * B_X] = scaleOutputs * prod[c][f];
            }
        }
    }
}


/*
 * Each block computes weight gradients for B_Y * pixelsPerThread pixels and B_X filters
 * threadIdx.x determines filter
 * threadIdx.y determines pixel in filter
 *
 * blockIdx.x determines filter batch of B_X * filtersPerThread, module batch of partialSum
 * blockIdx.y determines pixel batch of B_Y * pixelsPerThread
 *
 * Number of filters must be divisible by B_X * filtersPerThread
 * Number of images (cases) should be divisible by preloadCases if checkCaseBounds is false.
 *
 * images:      (numColors, imgSizeY, imgSizeX, numImages), with stride given
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * targets:     (numModulesY*numModulesX/partialSum, numColors, filterPixels, numFilters)
 *
 * B_Y * B_X should be divisible by preloadCases.
 * preloadCases one of 16, 32.
 * B_X one of 4, 8, 16, 32
 * B_Y arbitrary (satisfying divisibility constraints)
 * numModules must be divisible by partialSum
 * pixelsPerThread must be divisible by pixelCache
 *
 * After adding pixelsPerThread, register usage went from 20 to 23 (when pixelsPerThread = 1)...
 * so the compiler is messing up here somehow. It's unable to optimize that case away.
 */
template <int B_Y, int B_X, int pixelCache, int pixelsPerThread, int filtersPerThread, int preloadCases, int numColors, bool scale, bool checkCaseBounds>
__global__ void conv_weight_acts_c_kepler_sw(float* images, float* hidActs, float* targets,
                                   const int numImages, const int numFilters,
                                   const int numModulesY, const int numModulesX,
                                   const int imgSizeY, const int imgSizeX, const int filterSize,
                                   const int paddingStart, const int moduleStride, const int imgStride,
                                   const int sumWidth,
                                   const float scaleTargets, const float scaleOutputs) {
    __shared__ float shImages[pixelCache * B_Y * numColors][preloadCases]; // preload preloadCases cases of B_Y * pixelsPerThread pixels
    __shared__ float shHidActs[B_X * filtersPerThread][preloadCases + 1]; // preload preloadCases cases of B_X hidActs

    const int tidx = B_X * threadIdx.y + threadIdx.x;
    const int loadY = tidx / preloadCases, loadX = tidx % preloadCases;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;

    const int numFilterBlocks = numFilters / (B_X*filtersPerThread);

    const int blockModuleChunkIdx = blockIdx.x / numFilterBlocks;

    const int numModuleChunksX = DIVUP(numModulesX, sumWidth);
//    const int numModuleChunksY = DIVUP(numModulesY, sumWidth);

    const int blockModuleChunkX = blockModuleChunkIdx % numModuleChunksX;
    const int blockModuleChunkY = blockModuleChunkIdx / numModuleChunksX;

    const int blockModuleStartX = blockModuleChunkX * sumWidth;
    const int blockModuleStartY = blockModuleChunkY * sumWidth;

    const int blockFilterIdx = B_X * filtersPerThread* (blockIdx.x % numFilterBlocks);

//    const int moduleStride = (imgSize - filterSize + 1) / numModulesX;
    const int numModules = numModulesY * numModulesX;

    const int blockPixelOffset = blockIdx.y * B_Y * pixelsPerThread;

    images += loadX;
    hidActs += blockFilterIdx * numImages * numModules
//            + loadY * numImages * numModules
            + loadX;

    targets += (blockModuleChunkIdx * numFilters) * filterPixels * numColors
            + blockPixelOffset * numFilters
            + blockFilterIdx
            + threadIdx.y * numFilters + threadIdx.x;

    //float* shImgLoad = &shImages[loadY][loadX];
    //float* shHidActLoad = &shHidActs[loadY][loadX];

    float prod[numColors][pixelsPerThread][filtersPerThread];
    #pragma unroll
    for (int c = 0; c < numColors; c++) {
        #pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                prod[c][p][f] = 0;
            }
        }
    }
    const int mStartX = blockModuleStartX;
    const int mStartY = blockModuleStartY;
    const int mEndX = min(numModulesX, blockModuleStartX + sumWidth);
    const int mEndY = min(numModulesY, blockModuleStartY + sumWidth);

//    if (mStartY == mEndY || mStartX == mEndX) {
//        return;
//    }

    const int fYOff = (blockPixelOffset + tidx) / filterSize;
    const int fXOff = (blockPixelOffset + tidx) % filterSize;
    __shared__ int pxIdxes[B_Y*pixelsPerThread];
    for (int my = mStartY; my < mEndY; my++) {
        const int imgLoadModPosY = paddingStart + my * moduleStride;
        for (int mx = mStartX; mx < mEndX; mx++) {
            const int m = my * numModulesX + mx;

            __syncthreads();
            const int imgLoadModPosX = paddingStart + mx * moduleStride;
            if (tidx < B_Y * pixelsPerThread) {
//                const int imgLoadModPosY = paddingStart + my * moduleStride;
//                const int imgLoadModPosX = paddingStart + mx * moduleStride;
                int pxY = (imgLoadModPosY + fYOff);
                int pxX = (imgLoadModPosX + fXOff);
                int pixIdx = (pxY * imgSizeX + pxX) * imgStride;
                pxIdxes[tidx] = pxY >= 0 && pxY < imgSizeY && pxX >= 0 && pxX < imgSizeX ? pixIdx : -1;
            }
            __syncthreads();
            for (int caseIdx = 0; caseIdx < numImages; caseIdx += preloadCases) {
                if (/*loadY < B_X*filtersPerThread &&*/ (!checkCaseBounds || caseIdx + loadX < numImages)) {
                    #pragma unroll
                    for (int y = 0; y < B_X*filtersPerThread; y += (B_X * B_Y) / preloadCases) {
                        const int fIdx = ((loadY + y) % filtersPerThread) * B_X + (loadY + y) / filtersPerThread;
                        // Make sure number of rows in the array is divisible by number of rows filled per iteration
                        if ((B_X*filtersPerThread) % (B_X * B_Y / preloadCases) == 0 || loadY+y < B_X*filtersPerThread) {
                            shHidActs[loadY+y][loadX]= hidActs[caseIdx + fIdx * numImages * numModules + m * numImages];
                        }
                    }
                } else {
                    #pragma unroll
                    for (int y = 0; y < B_X*filtersPerThread; y += (B_X * B_Y) / preloadCases) {
                    //                        const int fIdx = ((loadY + y) % filtersPerThread) * B_X + (loadY + y) / filtersPerThread;
                        // Make sure number of rows in the array is divisible by number of rows filled per iteration
                        if ((B_X*filtersPerThread) % (B_X * B_Y / preloadCases) == 0 || loadY+y < B_X*filtersPerThread) {
                            shHidActs[loadY+y][loadX] = 0;
                        }
                    }
                }
                #pragma unroll
                for (int pp = 0; pp < pixelsPerThread; pp += pixelCache) {
                    //if (loadY < B_Y * pixelCache) { // This condition is not necessary for correctness, but it speeds things a bit
                    /*
                     * As long as B_Y * B_X is divisible by preloadCases this will loop the right
                     * number of times.
                     *
                     * This will load some imgGrads from filter pixels that don't exit (it'll set those to 0),
                     * but the code does not produce any output for those pixels (see last lines).
                     */
                    #pragma unroll
                    for (int y = 0; y < B_Y * pixelCache; y += (B_X * B_Y) / preloadCases) {
                        // Make sure number of rows in the array is divisible by number of rows filled per iteration
                        if ((B_Y * pixelCache) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_Y * pixelCache) {
                            const int pxIdx = pp * B_Y + loadY + y; // pixel idx in filter

                            if (pxIdx + blockPixelOffset < filterPixels && (!checkCaseBounds || caseIdx + loadX < numImages)) {
                                const int pixIdx = pxIdxes[pxIdx];//(pxY * imgSizeX + pxX) * imgStride;

                                if (pixIdx >= 0) {
                                    #pragma unroll
                                    for (int c = 0; c < numColors; c++) {
                                        shImages[loadY+y + c * pixelCache * B_Y][loadX] = images[caseIdx + c * imgPixels * imgStride + pixIdx];
                                    }
                                } else {
                                    #pragma unroll
                                    for (int c = 0; c < numColors; c++) {
                                        shImages[loadY+y + c * pixelCache * B_Y][loadX] = 0;
                                    }
                                }
                            } else {
                                #pragma unroll
                                for (int c = 0; c < numColors; c++) {
                                    shImages[loadY+y + c * pixelCache * B_Y][loadX]= 0;
                                }
                            }
                        }
                    }
                    //}

                    __syncthreads();

                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        #pragma unroll
                        for (int i = 0; i < preloadCases; i++) {
                            #pragma unroll
                            for (int p = 0; p < pixelCache; p++) {
                                #pragma unroll
                                for (int f = 0; f < filtersPerThread; f++) {
                                    prod[c][pp + p][f] += shImages[threadIdx.y + p * B_Y + c * pixelCache * B_Y][i] * shHidActs[threadIdx.x * filtersPerThread + f][i];
                                }
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
        for (int p = 0; p < pixelsPerThread; p++) {
            if (blockPixelOffset + p * B_Y + threadIdx.y < filterPixels) {
                #pragma unroll
                for (int c = 0; c < numColors; c++) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        targets[p * B_Y * numFilters + c * filterPixels * numFilters + f * B_X] = scaleTargets * targets[p * B_Y * numFilters + c * filterPixels * numFilters + f * B_X] + scaleOutputs * prod[c][p][f];
                    }
                }
            }
        }
    } else {
        #pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
            if (blockPixelOffset + p * B_Y + threadIdx.y < filterPixels) {
                #pragma unroll
                for (int c = 0; c < numColors; c++) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        targets[p * B_Y * numFilters + c * filterPixels * numFilters + f * B_X] = scaleOutputs * prod[c][p][f];
                    }
                }
            }
        }
    }
}


#define WA_C3_LOOP(pp, c) _Pragma("unroll") \
for (int i = 0; i < preloadCases; i++) { \
    _Pragma("unroll") \
    for (int p = 0; p < pixelCache; p++) { \
        _Pragma("unroll") \
        for (int f = 0; f < filtersPerThread; f++) { \
            prod[c][(pp) + p][f] += shImages[threadIdx.y + p * B_Y + (c) * pixelCache * B_Y][i] * shHidActs[threadIdx.x * filtersPerThread + f][i]; \
        } \
    } \
}

#define WA_C3_LOOP2(pp) _Pragma("unroll") \
for (int p = 0; p < pixelCache; p++) { \
    _Pragma("unroll") \
    for (int i = 0; i < preloadCases; i++) { \
        _Pragma("unroll") \
        for (int f = 0; f < filtersPerThread; f++) { \
            _Pragma("unroll") \
            for (int c = 0; c < 3; ++c) { \
                prod[c][(pp) + p][f] += shImages[threadIdx.y + p * B_Y + (c) * pixelCache * B_Y][i] * shHidActs[threadIdx.x * filtersPerThread + f][i]; \
            } \
        } \
    } \
}

#define WA_3_FIDX(y) (((loadY + (y)*B_X*B_Y/preloadCases) % filtersPerThread) * B_X + (loadY + (y)*B_X*B_Y/preloadCases) / filtersPerThread)


/*
 * Each block computes weight gradients for B_Y * pixelsPerThread pixels and B_X filters
 * threadIdx.x determines filter
 * threadIdx.y determines pixel in filter
 *
 * blockIdx.x determines filter batch of B_X * filtersPerThread, module batch of partialSum
 * blockIdx.y determines pixel batch of B_Y * pixelsPerThread
 *
 * Number of filters must be divisible by B_X * filtersPerThread
 * Number of images (cases) should be divisible by preloadCases if checkCaseBounds is false.
 *
 * images:      (numColors, imgSizeY, imgSizeX, numImages), with stride given
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * targets:     (numModulesY*numModulesX/partialSum, numColors, filterPixels, numFilters)
 *
 * B_Y * B_X should be divisible by preloadCases.
 * preloadCases one of 16, 32.
 * B_X one of 4, 8, 16, 32
 * B_Y arbitrary (satisfying divisibility constraints)
 * numModules must be divisible by partialSum
 * pixelsPerThread must be divisible by pixelCache
 *
 * After adding pixelsPerThread, register usage went from 20 to 23 (when pixelsPerThread = 1)...
 * so the compiler is messing up here somehow. It's unable to optimize that case away.
 */
template <int B_Y, int B_X, int pixelCache, int pixelsPerThread, int filtersPerThread, int preloadCases, int numColors, bool scale, bool checkCaseBounds>
//__launch_bounds__(256,2)
__global__ void conv_weight_acts_c_preload_pc_2_pt_2_f_4_r_32_c_3(cudaTextureObject_t images, cudaTextureObject_t hidActs, float* targets,
                                   const int numImages, const int numFilters,
                                   const int numModulesY, const int numModulesX,
                                   const int imgSizeY, const int imgSizeX, const int filterSize,
                                   const int paddingStart, const int moduleStride, const int imgStride,
                                   const int sumWidth,
                                   const float scaleTargets, const float scaleOutputs) {
    __shared__ float shImages[pixelCache * B_Y * numColors][preloadCases]; // preload preloadCases cases of B_Y * pixelsPerThread pixels
    __shared__ float shHidActs[B_X * filtersPerThread][preloadCases + 1]; // preload preloadCases cases of B_X hidActs

    const int tidx = B_X * threadIdx.y + threadIdx.x;
    const int loadY = tidx / preloadCases, loadX = tidx % preloadCases;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;

    const int numFilterBlocks = numFilters / (B_X*filtersPerThread);

    const int blockModuleChunkIdx = blockIdx.x / numFilterBlocks;

    const int numModuleChunksX = DIVUP(numModulesX, sumWidth);
//    const int numModuleChunksY = DIVUP(numModulesY, sumWidth);

    const int blockModuleChunkX = blockModuleChunkIdx % numModuleChunksX;
    const int blockModuleChunkY = blockModuleChunkIdx / numModuleChunksX;

    const int blockModuleStartX = blockModuleChunkX * sumWidth;
    const int blockModuleStartY = blockModuleChunkY * sumWidth;

    const int blockFilterIdx = B_X * filtersPerThread* (blockIdx.x % numFilterBlocks);

//    const int moduleStride = (imgSize - filterSize + 1) / numModulesX;
    const int numModules = numModulesY * numModulesX;

    const int blockPixelOffset = blockIdx.y * B_Y * pixelsPerThread;
    const int imgOffset = loadX;
    const int hidActsOffset = blockFilterIdx * numImages * numModules + loadX;
//    images += loadX;
//    hidActs += blockFilterIdx * numImages * numModules
//            + loadX;

    targets += (blockModuleChunkIdx * numFilters) * filterPixels * numColors
            + blockPixelOffset * numFilters
            + blockFilterIdx
            + threadIdx.y * numFilters + threadIdx.x;

    //float* shImgLoad = &shImages[loadY][loadX];
    //float* shHidActLoad = &shHidActs[loadY][loadX];

    float prod[numColors][pixelsPerThread][filtersPerThread];
    #pragma unroll
    for (int c = 0; c < numColors; c++) {
        #pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                prod[c][p][f] = 0;
            }
        }
    }
    const int mStartX = blockModuleStartX;
    const int mStartY = blockModuleStartY;
    const int mEndX = min(numModulesX, blockModuleStartX + sumWidth);
    const int mEndY = min(numModulesY, blockModuleStartY + sumWidth);

    const bool doWork = mStartY < mEndY && mStartX < mEndX;
//    if (!doWork) {
//        hidActs -=
//    }
//    if (mStartY == mEndY || mStartX == mEndX) {
//        return;
//    }

//    float imPreload[pixelCache * numColors * preloadCases / B_X]; // [12]
    float haPreload[filtersPerThread * preloadCases / B_Y]; // [8]
//    if (blockIdx.x != 0 || blockIdx.y !=0) {
//        return;
//    }
//    printf("mStartX: %d, mStartX: %d, mStartX: %d, mStartX: %d\n", mStartX, mStartY, mEndX, mEndY);
    const int fYOff = (blockPixelOffset + tidx) / filterSize;
    const int fXOff = (blockPixelOffset + tidx) % filterSize;
    __shared__ int pxIdxes[B_Y*pixelsPerThread];
//    __shared__ int fidx[filtersPerThread * preloadCases / B_Y]; // [8]

    int m = mStartY * numModulesX + mStartX;

    int fidx[filtersPerThread * preloadCases / B_Y];
    if (doWork) {
        #pragma unroll
        for (int y = 0; y < filtersPerThread * preloadCases / B_Y; ++y) {
            const int fIdx = WA_3_FIDX(y);
//            if (doWork) {
            haPreload[y] =  tex1Dfetch<float>(hidActs, hidActsOffset + fIdx * numImages * numModules + m * numImages);
//            }
            fidx[y] = fIdx * numImages * numModules;
        }
    }

    for (int my = mStartY; my < mEndY; my++) {
        const int imgLoadModPosY = paddingStart + my * moduleStride;
        for (int mx = mStartX; mx < mEndX; mx++) {
            m = my * numModulesX + mx;

//            __syncthreads();
            const int imgLoadModPosX = paddingStart + mx * moduleStride;
            if (tidx < B_Y * pixelsPerThread) {
//                const int imgLoadModPosY = paddingStart + my * moduleStride;
//                const int imgLoadModPosX = paddingStart + mx * moduleStride;
                const int pxY = (imgLoadModPosY + fYOff);
                const int pxX = (imgLoadModPosX + fXOff);
                const int pixIdx = (pxY * imgSizeX + pxX) * imgStride;
                pxIdxes[tidx] = pxY >= 0 && pxY < imgSizeY && pxX >= 0 && pxX < imgSizeX ? pixIdx : -1;
            }
            __syncthreads();

            int myNext = my, mxNext = mx, mNext = m;
            const bool lastModule = my == mEndY - 1 && mx == mEndX - 1;

            if (!lastModule) {
                mxNext = mx + 1 == mEndX ? mStartX : mx + 1;
                myNext = my + (mx + 1 == mEndX);
                mNext = myNext * numModulesX + mxNext;
            }

            for (int caseIdx = 0; caseIdx < numImages; caseIdx += preloadCases) {
                const bool lastBatch = caseIdx + preloadCases == numImages;
//                const float* im = &images[caseIdx + preloadCases + pixIdx];
//                const float* ha = &hidActs[caseIdx + preloadCases + m * numImages];
                int hidActsOffset2 = hidActsOffset + caseIdx + preloadCases + m * numImages;

                if (lastBatch) {
//                    ha = &hidActs[mNext * numImages];
                    hidActsOffset2 = hidActsOffset + mNext * numImages;
                }

                #pragma unroll
                for (int y = 0; y < B_X*filtersPerThread; y += (B_X * B_Y) / preloadCases) {
                    shHidActs[loadY+y][loadX] = haPreload[y*preloadCases/(B_X*B_Y)];
                }

                /* ==================================================================================
                 * Iteration 0
                 * ==================================================================================
                 */
                #pragma unroll
                for (int y = 0; y < B_Y * pixelCache; y += (B_X * B_Y) / preloadCases) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        shImages[loadY+y + c * pixelCache * B_Y][loadX] = 0;
                    }
                }
                #pragma unroll
                for (int y = 0; y < B_Y * pixelCache; y += (B_X * B_Y) / preloadCases) {
                    const int pxIdx = 0 * B_Y + loadY + y; // pixel idx in filter
                    if (pxIdx + blockPixelOffset < filterPixels) {
                        const int pixIdx = pxIdxes[pxIdx];//(pxY * imgSizeX + pxX) * imgStride;
                        if (pixIdx >= 0) {
                            #pragma unroll
                            for (int c = 0; c < numColors; c++) {
                                shImages[loadY+y + c * pixelCache * B_Y][loadX] = tex1Dfetch<float>(images, imgOffset + caseIdx + c * imgPixels * imgStride + pixIdx);
                            }
                        }
                    }
                }

                __syncthreads();

                haPreload[0] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[0]);
                haPreload[1] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[1]);
                WA_C3_LOOP(0,0);
                haPreload[2] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[2]);
                haPreload[3] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[3]);
                WA_C3_LOOP(0,1);
                haPreload[4] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[4]);
                haPreload[5] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[5]);
                WA_C3_LOOP(0,2);
                haPreload[6] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[6]);
                haPreload[7] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[7]);

                __syncthreads();
            }
        }
    }

    if (scale) {
        #pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
            if (blockPixelOffset + p * B_Y + threadIdx.y < filterPixels) {
                #pragma unroll
                for (int c = 0; c < numColors; c++) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        targets[p * B_Y * numFilters + c * filterPixels * numFilters + f * B_X] = scaleTargets * targets[p * B_Y * numFilters + c * filterPixels * numFilters + f * B_X] + scaleOutputs * prod[c][p][f];
                    }
                }
            }
        }
    } else {
        #pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
            if (blockPixelOffset + p * B_Y + threadIdx.y < filterPixels) {
                #pragma unroll
                for (int c = 0; c < numColors; c++) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
//                        if (threadIdx.x == 3)
                        targets[p * B_Y * numFilters + c * filterPixels * numFilters + f * B_X] = scaleOutputs * prod[c][p][f];
                    }
                }
            }
        }
    }
}


/*
 * Each block computes weight gradients for B_Y * pixelsPerThread pixels and B_X filters
 * threadIdx.x determines filter
 * threadIdx.y determines pixel in filter
 *
 * blockIdx.x determines filter batch of B_X * filtersPerThread, module batch of partialSum
 * blockIdx.y determines pixel batch of B_Y * pixelsPerThread
 *
 * Number of filters must be divisible by B_X * filtersPerThread
 * Number of images (cases) should be divisible by preloadCases if checkCaseBounds is false.
 *
 * images:      (numColors, imgSizeY, imgSizeX, numImages), with stride given
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * targets:     (numModulesY*numModulesX/partialSum, numColors, filterPixels, numFilters)
 *
 * B_Y * B_X should be divisible by preloadCases.
 * preloadCases one of 16, 32.
 * B_X one of 4, 8, 16, 32
 * B_Y arbitrary (satisfying divisibility constraints)
 * numModules must be divisible by partialSum
 * pixelsPerThread must be divisible by pixelCache
 *
 * After adding pixelsPerThread, register usage went from 20 to 23 (when pixelsPerThread = 1)...
 * so the compiler is messing up here somehow. It's unable to optimize that case away.
 */
template <int B_Y, int B_X, int pixelCache, int pixelsPerThread, int filtersPerThread, int preloadCases, int numColors, bool scale, bool checkCaseBounds>
__launch_bounds__(256,2)
__global__ void conv_weight_acts_c_preload_pc_2_pt_4_f_3_r_32_c_3(cudaTextureObject_t images, cudaTextureObject_t hidActs, float* targets,
                                   const int numImages, const int numFilters,
                                   const int numModulesY, const int numModulesX,
                                   const int imgSizeY, const int imgSizeX, const int filterSize,
                                   const int paddingStart, const int moduleStride, const int imgStride,
                                   const int sumWidth,
                                   const float scaleTargets, const float scaleOutputs) {
    __shared__ float shImages[pixelCache * B_Y * numColors][preloadCases]; // preload preloadCases cases of B_Y * pixelsPerThread pixels
    __shared__ float shHidActs[B_X * filtersPerThread][preloadCases + 1]; // preload preloadCases cases of B_X hidActs

    const int tidx = B_X * threadIdx.y + threadIdx.x;
    const int loadY = tidx / preloadCases, loadX = tidx % preloadCases;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;

    const int numFilterBlocks = numFilters / (B_X*filtersPerThread);

    const int blockModuleChunkIdx = blockIdx.x / numFilterBlocks;

    const int numModuleChunksX = DIVUP(numModulesX, sumWidth);
//    const int numModuleChunksY = DIVUP(numModulesY, sumWidth);

    const int blockModuleChunkX = blockModuleChunkIdx % numModuleChunksX;
    const int blockModuleChunkY = blockModuleChunkIdx / numModuleChunksX;

    const int blockModuleStartX = blockModuleChunkX * sumWidth;
    const int blockModuleStartY = blockModuleChunkY * sumWidth;

    const int blockFilterIdx = B_X * filtersPerThread* (blockIdx.x % numFilterBlocks);

//    const int moduleStride = (imgSize - filterSize + 1) / numModulesX;
    const int numModules = numModulesY * numModulesX;

    const int blockPixelOffset = blockIdx.y * B_Y * pixelsPerThread;
    const int imgOffset = loadX;
    const int hidActsOffset = blockFilterIdx * numImages * numModules
                        + loadX;
//    images += loadX;
//    hidActs += blockFilterIdx * numImages * numModules
//            + loadX;

    targets += (blockModuleChunkIdx * numFilters) * filterPixels * numColors
            + blockPixelOffset * numFilters
            + blockFilterIdx
            + threadIdx.y * numFilters + threadIdx.x;

    //float* shImgLoad = &shImages[loadY][loadX];
    //float* shHidActLoad = &shHidActs[loadY][loadX];

    float prod[numColors][pixelsPerThread][filtersPerThread];
    #pragma unroll
    for (int c = 0; c < numColors; c++) {
        #pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                prod[c][p][f] = 0;
            }
        }
    }
    const int mStartX = blockModuleStartX;
    const int mStartY = blockModuleStartY;
    const int mEndX = min(numModulesX, blockModuleStartX + sumWidth);
    const int mEndY = min(numModulesY, blockModuleStartY + sumWidth);

    const bool doWork = mStartY < mEndY && mStartX < mEndX;
//    if (mStartY == mEndY || mStartX == mEndX) {
//        return;
//    }

//    float imPreload[pixelCache * numColors * preloadCases / B_X]; // [12]
    float haPreload[filtersPerThread * preloadCases / B_Y]; // [6]
//    if (blockIdx.x != 0 || blockIdx.y !=0) {
//        return;
//    }
//    printf("mStartX: %d, mStartX: %d, mStartX: %d, mStartX: %d\n", mStartX, mStartY, mEndX, mEndY);
    const int fYOff = (blockPixelOffset + tidx) / filterSize;
    const int fXOff = (blockPixelOffset + tidx) % filterSize;
    __shared__ int pxIdxes[B_Y*pixelsPerThread];
//    __shared__ int fidx[filtersPerThread * preloadCases / B_Y]; // [6]

    int m = mStartY * numModulesX + mStartX;
    int fidx[filtersPerThread * preloadCases / B_Y];
//    if (doWork) {
    #pragma unroll
    for (int y = 0; y < filtersPerThread * preloadCases / B_Y; ++y) {
        fidx[y] = WA_3_FIDX(y) * numImages * numModules;
        if (doWork) { // Not actually necessary, I think
            haPreload[y] = tex1Dfetch<float>(hidActs, hidActsOffset + fidx[y] + m * numImages);
        }
    }
//    }
    int mNext = mStartY * numModulesX + mStartX;
    for (int my = mStartY; my < mEndY; my++) {
//        const int imgLoadModPosY = paddingStart + my * moduleStride;
        for (int mx = mStartX; mx < mEndX; mx++) {
            m = mNext;//my * numModulesX + mx;

//            __syncthreads();
//            const int imgLoadModPosX = paddingStart + mx * moduleStride;
            if (tidx < B_Y * pixelsPerThread) {
                const int imgLoadModPosY = paddingStart + my * moduleStride;
                const int imgLoadModPosX = paddingStart + mx * moduleStride;
                const int pxY = (imgLoadModPosY + fYOff);
                const int pxX = (imgLoadModPosX + fXOff);
                const int pixIdx = (pxY * imgSizeX + pxX) * imgStride;
                pxIdxes[tidx] = pxY >= 0 && pxY < imgSizeY && pxX >= 0 && pxX < imgSizeX ? pixIdx : -1;
            }
            __syncthreads();


            const bool lastModule = my == mEndY - 1 && mx == mEndX - 1;
            mNext = lastModule * m + !lastModule * ((my + (mx + 1 == mEndX)) * numModulesX + (mx + 1 == mEndX ? mStartX : mx + 1));
//            if (!lastModule) {
//                const int mxNext = mx + 1 == mEndX ? mStartX : mx + 1;
//                const int myNext = my + (mx + 1 == mEndX);
//                mNext = myNext * numModulesX + mxNext;
//            }

            for (int caseIdx = 0; caseIdx < numImages; caseIdx += preloadCases) {
                const bool lastBatch = caseIdx + preloadCases == numImages;
//                const float* im = &images[caseIdx + preloadCases + pixIdx];
//                const float* ha = hidActs + !lastBatch * (caseIdx + preloadCases + m * numImages) + lastBatch * mNext * numImages;
                const int hidActsOffset2 = hidActsOffset + !lastBatch * (caseIdx + preloadCases + m * numImages) + lastBatch * mNext * numImages;
//                if (lastBatch) {
//                    ha = &hidActs[mNext * numImages];
//                }

                #pragma unroll
                for (int y = 0; y < B_X*filtersPerThread; y += (B_X * B_Y) / preloadCases) {
                    shHidActs[loadY+y][loadX] = haPreload[y*preloadCases/(B_X*B_Y)];
                }

                /* ==================================================================================
                 * Iteration 0
                 * ==================================================================================
                 */
                #pragma unroll
                for (int y = 0; y < B_Y * pixelCache; y += (B_X * B_Y) / preloadCases) {
                    // Make sure number of rows in the array is divisible by number of rows filled per iteration
                    if ((B_Y * pixelCache) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_Y * pixelCache) {
                        #pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shImages[loadY+y + c * pixelCache * B_Y][loadX] = 0;
                        }
                    }
                }
                #pragma unroll
                for (int y = 0; y < B_Y * pixelCache; y += (B_X * B_Y) / preloadCases) {
                    // Make sure number of rows in the array is divisible by number of rows filled per iteration
                    if ((B_Y * pixelCache) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_Y * pixelCache) {
                        const int pxIdx = 0 * B_Y + loadY + y; // pixel idx in filter
                        const int pixIdx = pxIdxes[pxIdx];//(pxY * imgSizeX + pxX) * imgStride;
                        if (pixIdx >= 0 && pxIdx + blockPixelOffset < filterPixels && (!checkCaseBounds || caseIdx + loadX < numImages)) {
                            #pragma unroll
                            for (int c = 0; c < numColors; c++) {
                                shImages[loadY+y + c * pixelCache * B_Y][loadX] = tex1Dfetch<float>(images, imgOffset + caseIdx + c * imgPixels * imgStride + pixIdx);
                            }
                        }
                    }
                }

                __syncthreads();

                haPreload[0] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[0]);
                haPreload[1] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[1]);
                haPreload[2] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[2]);
                haPreload[3] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[3]);
                haPreload[4] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[4]);
                haPreload[5] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[5]);

                WA_C3_LOOP2(0);

                __syncthreads();

                /* ==================================================================================
                 * Iteration 1
                 * ==================================================================================
                 */
                #pragma unroll
                for (int y = 0; y < B_Y * pixelCache; y += (B_X * B_Y) / preloadCases) {
                    // Make sure number of rows in the array is divisible by number of rows filled per iteration
                    if ((B_Y * pixelCache) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_Y * pixelCache) {
//                        const int pxIdx = 2 * B_Y + loadY + y; // pixel idx in filter
                        #pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shImages[loadY+y + c * pixelCache * B_Y][loadX] = 0;
                        }
                    }
                }

                #pragma unroll
                for (int y = 0; y < B_Y * pixelCache; y += (B_X * B_Y) / preloadCases) {
                    // Make sure number of rows in the array is divisible by number of rows filled per iteration
                    if ((B_Y * pixelCache) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_Y * pixelCache) {
                        const int pxIdx = 2 * B_Y + loadY + y; // pixel idx in filter
                        const int pixIdx = pxIdxes[pxIdx];//(pxY * imgSizeX + pxX) * imgStride;
                        if (pixIdx >= 0 && pxIdx + blockPixelOffset < filterPixels && (!checkCaseBounds || caseIdx + loadX < numImages)) {
                            #pragma unroll
                            for (int c = 0; c < numColors; c++) {
                                shImages[loadY+y + c * pixelCache * B_Y][loadX] = tex1Dfetch<float>(images, imgOffset + caseIdx + c * imgPixels * imgStride + pixIdx);
                            }
                        }
                    }
                }

                __syncthreads();

                WA_C3_LOOP2(2);

                __syncthreads();

            }
        }
    }

    if (scale) {
        #pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
            if (blockPixelOffset + p * B_Y + threadIdx.y < filterPixels) {
                #pragma unroll
                for (int c = 0; c < numColors; c++) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        targets[p * B_Y * numFilters + c * filterPixels * numFilters + f * B_X] = scaleTargets * targets[p * B_Y * numFilters + c * filterPixels * numFilters + f * B_X] + scaleOutputs * prod[c][p][f];
                    }
                }
            }
        }
    } else {
        #pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
            if (blockPixelOffset + p * B_Y + threadIdx.y < filterPixels) {
                #pragma unroll
                for (int c = 0; c < numColors; c++) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        targets[p * B_Y * numFilters + c * filterPixels * numFilters + f * B_X] = scaleOutputs * prod[c][p][f];
                    }
                }
            }
        }
    }
}

/*****************************Function Revision Record*****************************
 * Author: Tencent BestImage Team(ankerguo@tencent.com)                           *
 * Date:   2015-05-18                                                             *
 * Reason: Optimizing kernel to get faster speed according to GPU features        *
 * Method:                                                                        *
 *         1. reorganizing data structure to avoid bank conflict;                 *
 *         2. using vectorized data type;                                         *
 *         3. improving instruction-level parallelism;                            *
 *         4. removing redundant 'if' branches;                                   *
 *         5. removing local variables to save registers.                         *
 *********************************************************************************/

/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages), with stride given
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * targets:     (numModulesY*numModulesX/partialSum, numFilterColors, filterPixels, numFilters)
 */
template <int B_Y, int B_X, int filtersPerThread, int colorsPerThread, int preloadCases, bool scale>
__launch_bounds__(128, 4)
__global__ void conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_16_f_4_c_8_r_16(cudaTextureObject_t images, cudaTextureObject_t hidActs, float* targets,
                                       const int numImages, const int numFilters,
                                       const int numModulesY, const int numModulesX,
                                       const int imgSizeY, const int imgSizeX, const int filterSize,
                                       const int paddingStart, const int moduleStride, const int imgStride,
                                       const int numImgColors, const int numGroups, const int sumWidth,
                                       const float scaleTargets, const float scaleOutputs) {
    // avoid bank conflict by reorganizing the data structure, and improve the band width by using 'float2'  instead of 'float'
    __shared__ float2 shImages[preloadCases][colorsPerThread * B_Y / 2 + 2]; // preload preloadCases cases
    __shared__ float2 shHidActs[preloadCases][filtersPerThread * B_X / 2 + 2]; // preload preloadCases cases of B_X hidacts

    const int tx = threadIdx.x % B_X, ty = threadIdx.y % B_Y;
    const int tidx = B_X * ty + tx;
    const int loadY = tidx / preloadCases, loadX = tidx % preloadCases;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;

    const int numFilterBlocks = numFilters / (B_X * filtersPerThread);
    const int blockModuleChunkIdx = blockIdx.x / numFilterBlocks;

    const int numModuleChunksX = DIVUP(numModulesX, sumWidth);
//    const int numModuleChunksY = DIVUP(numModulesY, sumWidth);

    const int blockModuleChunkX = blockModuleChunkIdx % numModuleChunksX;
    const int blockModuleChunkY = blockModuleChunkIdx / numModuleChunksX;

    const int blockModuleStartX = blockModuleChunkX * sumWidth;
    const int blockModuleStartY = blockModuleChunkY * sumWidth;

//    const int moduleIdx = partialSum * outputModuleIdx;
    const int blockFilterIdx = filtersPerThread * B_X * (blockIdx.x % numFilterBlocks);
    const int numModules = numModulesY * numModulesX;

    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;
    const int numFilterColors = numImgColors / numGroups;

    const int blockPixelOffset = blockIdx.z; // pixel idx in filter
    const int blockPixelY = blockPixelOffset / filterSize, blockPixelX = blockPixelOffset % filterSize;
    const int blockFilterColorIdx = blockIdx.y  * B_Y * colorsPerThread;
    const int imgColorIdx = blockFilterColorIdx + blockGroupIdx * numFilterColors;
    const int imgOffset = (imgColorIdx + loadY) * imgPixels * imgStride + loadX;
//    images += (imgColorIdx + loadY) * imgPixels * imgStride + loadX;
    const int hidActsOffset = blockFilterIdx * numImages * numModules
            + loadY * numImages * numModules
            + loadX;
//
//    hidActs +=
//             blockFilterIdx * numImages * numModules
//            + loadY * numImages * numModules
//            + loadX;

    targets += blockModuleChunkIdx * numFilters * filterPixels * numFilterColors
            + (blockFilterColorIdx + ty) * filterPixels * numFilters
            + blockPixelOffset * numFilters
            + blockFilterIdx
            + tx;
    // if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) return;

    const int mStartX = max(blockModuleStartX, DIVUP(-blockPixelX - paddingStart, moduleStride));
    const int mStartY = max(blockModuleStartY, DIVUP(-blockPixelY - paddingStart, moduleStride));
    const int mEndX = min(numModulesX, min(blockModuleStartX + sumWidth, DIVUP(imgSizeX - blockPixelX - paddingStart, moduleStride)));
    const int mEndY = min(numModulesY, min(blockModuleStartY + sumWidth, DIVUP(imgSizeY - blockPixelY - paddingStart, moduleStride)));

    // if (mStartY == mEndY || mStartX == mEndX) {
    //     return;
    // }
    const bool doWork = mStartY < mEndY && mStartX < mEndX;

    // reduce 2 registers
    //float* shHidActLoad = &shHidActs[loadY][loadX];
    //float* shImgLoad = &shImages[loadY][loadX];

    float imPreload[preloadCases*colorsPerThread/B_X]; // [8]
    float haPreload[preloadCases*filtersPerThread/B_Y]; // [8]

    float prod[filtersPerThread][colorsPerThread];

    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            prod[f][c] = 0;
        }
    }
    int pixIdx, pixIdxNext, m, mNext;

    conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_8_r_16_setCoords(
            mStartY, mStartX, paddingStart, numModulesX, moduleStride,
            blockPixelY, blockPixelX, imgSizeX, imgStride,
            pixIdx, m);

    if (doWork) {
    #pragma unroll
        for (int y = 0; y < B_Y * colorsPerThread; y += (B_X * B_Y) / preloadCases) {
            // It's bizarre, but this is the fastest way I've found to get it not to load nonexistent pixels.
            // All other ways cause crazy excessive register usage.
            const int idx = (mStartY < mEndY && mStartX < mEndX) * (0 + y * imgPixels * imgStride + pixIdx);
            imPreload[y * preloadCases/(B_X * B_Y)] = tex1Dfetch<float>(images, imgOffset + idx);
        }
    }
    
    if (doWork) {
        #pragma unroll
        for (int y = 0; y < B_X * filtersPerThread; y += (B_X * B_Y) / preloadCases) {
            // Almost certainly not necessary here.
            const int idx = (mStartY < mEndY && mStartX < mEndX) * (0 + y * numImages * numModules + m * numImages);
            haPreload[y * preloadCases / (B_X * B_Y)] = tex1Dfetch<float>(hidActs, hidActsOffset + idx);
        }
    }


    for (int my = mStartY; my < mEndY; my++) {
        for (int mx = mStartX; mx < mEndX; mx++) {
            int myNext = my, mxNext = mx;
            const bool lastModule = my == mEndY - 1 && mx == mEndX - 1;

            if (!lastModule) {
                mxNext = mx + 1 == mEndX ? mStartX : mx + 1;
                myNext = my + (mx + 1 == mEndX);
            }

            conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_8_r_16_setCoords(
                    myNext, mxNext, paddingStart, numModulesX, moduleStride,
                    blockPixelY, blockPixelX, imgSizeX, imgStride,
                    pixIdxNext, mNext);

            for (int caseIdx = 0; caseIdx < numImages; caseIdx += preloadCases) {
                // store the preloaded image's pixel into shared memory
                #pragma unroll
                for (int y = 0; y < 4; y++) {
                    shImages[loadX][loadY+y*8].x = imPreload[y];
                    shImages[loadX][loadY+y*8].y = imPreload[y+4];
                }
                //const float* im = &images[caseIdx + preloadCases + pixIdx];
                //const float* ha = &hidActs[caseIdx + preloadCases + m * numImages];
                int imgOffset2 = imgOffset + caseIdx + preloadCases + pixIdx;
                int hidActsOffset2 = hidActsOffset + caseIdx + preloadCases + m * numImages;
                if (caseIdx + preloadCases == numImages) {
                    pixIdx = pixIdxNext;
                    m = mNext;
                    imgOffset2 = imgOffset + pixIdxNext;
                    hidActsOffset2 = hidActsOffset + mNext * numImages;
                }
        
                // store the images and hidActs 
                shHidActs[loadX][loadY].x = haPreload[0];
                shHidActs[loadX][loadY].y = haPreload[2];
                shHidActs[loadX][loadY+16].x = haPreload[4];
                shHidActs[loadX][loadY+16].y = haPreload[6];
                shHidActs[loadX][loadY+8].x = haPreload[1];
                shHidActs[loadX][loadY+8].y = haPreload[3];
                shHidActs[loadX][loadY+24].x = haPreload[5];
                shHidActs[loadX][loadY+24].y = haPreload[7];

                // preloade the image's and hidAct's pixel
                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    imPreload[r] = tex1Dfetch<float>(images, imgOffset2 + (r) * 8 * imgPixels * imgStride);
                    haPreload[r] = tex1Dfetch<float>(hidActs, hidActsOffset2 + (r) * 8 * numImages * numModules);
                }

                __syncthreads();
                // put together the instructions of same type to improve instruction-level parallelism
                #pragma unroll
                for (int r = 0; r < 16; r++) {
                    for (int c = 0; c < 4; c++) { 
                        prod[0][c] += shImages[r][ty + c * B_Y].x * shHidActs[(r)][tx].x; 
                        prod[1][c] += shImages[r][ty + c * B_Y].x * shHidActs[(r)][tx].y; 
                        prod[2][c] += shImages[r][ty + c * B_Y].x * shHidActs[(r)][tx + B_X].x; 
                        prod[3][c] += shImages[r][ty + c * B_Y].x * shHidActs[(r)][tx + B_X].y; 
                        prod[0][c+4] += shImages[r][ty + c * B_Y].y * shHidActs[(r)][tx].x; 
                        prod[1][c+4] += shImages[r][ty + c * B_Y].y * shHidActs[(r)][tx].y; 
                        prod[2][c+4] += shImages[r][ty + c * B_Y].y * shHidActs[(r)][tx + B_X].x; 
                        prod[3][c+4] += shImages[r][ty + c * B_Y].y * shHidActs[(r)][tx + B_X].y; 
                    }
                }

                __syncthreads();
            }
        }
    }

    if (scale) {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                targets[c * B_Y * filterPixels * numFilters + f * B_X] = scaleTargets * targets[c * B_Y * filterPixels * numFilters + f * B_X] + scaleOutputs * prod[f][c];
            }
        }
    } else {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                targets[c * B_Y * filterPixels * numFilters + f * B_X] = scaleOutputs * prod[f][c];
            }
        }
    }
}

/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages), with stride given
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * targets:     (numModulesY*numModulesX/partialSum, numFilterColors, filterPixels, numFilters)
 */
template <int B_Y, int B_X, int filtersPerThread, int colorsPerThread, int preloadCases, bool scale>
__launch_bounds__(256, 2)
__global__ void conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_6_r_32(cudaTextureObject_t images, cudaTextureObject_t hidActs, float* targets,
                                       const int numImages, const int numFilters,
                                       const int numModulesY, const int numModulesX,
                                       const int imgSizeY, const int imgSizeX, const int filterSize,
                                       const int paddingStart, const int moduleStride, const int imgStride,
                                       const int numImgColors, const int numGroups, const int sumWidth,
                                       const float scaleTargets, const float scaleOutputs) {
    __shared__ float shImages[colorsPerThread * B_Y][preloadCases]; // preload preloadCases cases
    __shared__ float shHidActs[filtersPerThread * B_X][preloadCases + 1]; // preload preloadCases cases of B_X hidacts

    const int tidx = B_X * threadIdx.y + threadIdx.x;
    const int loadY = tidx / preloadCases, loadX = tidx % preloadCases;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;

    const int numFilterBlocks = numFilters / (B_X * filtersPerThread);
    const int blockModuleChunkIdx = blockIdx.x / numFilterBlocks;

    const int numModuleChunksX = DIVUP(numModulesX, sumWidth);
//    const int numModuleChunksY = DIVUP(numModulesY, sumWidth);

    const int blockModuleChunkX = blockModuleChunkIdx % numModuleChunksX;
    const int blockModuleChunkY = blockModuleChunkIdx / numModuleChunksX;

    const int blockModuleStartX = blockModuleChunkX * sumWidth;
    const int blockModuleStartY = blockModuleChunkY * sumWidth;

//    const int moduleIdx = partialSum * outputModuleIdx;
    const int blockFilterIdx = filtersPerThread * B_X * (blockIdx.x % numFilterBlocks);
    const int numModules = numModulesY * numModulesX;

    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;
    const int numFilterColors = numImgColors / numGroups;

    const int blockPixelOffset = blockIdx.z; // pixel idx in filter
    const int blockPixelY = blockPixelOffset / filterSize, blockPixelX = blockPixelOffset % filterSize;
    const int blockFilterColorIdx = blockIdx.y  * B_Y * colorsPerThread;
    const int imgColorIdx = blockFilterColorIdx + blockGroupIdx * numFilterColors;

    const int imgOffset = (imgColorIdx + loadY) * imgPixels * imgStride + loadX;
    const int hidActsOffset = blockFilterIdx * numImages * numModules
            + loadY * numImages * numModules
            + loadX;
//    images += (imgColorIdx + loadY) * imgPixels * imgStride + loadX;
//
//    hidActs +=
//             blockFilterIdx * numImages * numModules
//            + loadY * numImages * numModules
//            + loadX;

    targets += blockModuleChunkIdx * numFilters * filterPixels * numFilterColors
            + (blockFilterColorIdx + threadIdx.y) * filterPixels * numFilters
            + blockPixelOffset * numFilters
            + blockFilterIdx
            + threadIdx.x;
//    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) return;

    const int mStartX = max(blockModuleStartX, DIVUP(-blockPixelX - paddingStart, moduleStride));
    const int mStartY = max(blockModuleStartY, DIVUP(-blockPixelY - paddingStart, moduleStride));
    const int mEndX = min(numModulesX, min(blockModuleStartX + sumWidth, DIVUP(imgSizeX - blockPixelX - paddingStart, moduleStride)));
    const int mEndY = min(numModulesY, min(blockModuleStartY + sumWidth, DIVUP(imgSizeY - blockPixelY - paddingStart, moduleStride)));

//    if (mStartY == mEndY || mStartX == mEndX) {
//        return;
//    }
    const bool doWork = mStartY < mEndY && mStartX < mEndX;

    float* shHidActLoad = &shHidActs[loadY][loadX];
    float* shImgLoad = &shImages[loadY][loadX];

    float imPreload[preloadCases*colorsPerThread/B_X]; // [6]
    float haPreload[preloadCases*filtersPerThread/B_Y]; // [16]

    float prod[filtersPerThread][colorsPerThread];

    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            prod[f][c] = 0;
        }
    }
    int pixIdx, pixIdxNext, m, mNext;

    conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_8_r_16_setCoords(
            mStartY, mStartX, paddingStart, numModulesX, moduleStride,
            blockPixelY, blockPixelX, imgSizeX, imgStride,
            pixIdx, m);

    if (doWork) {
        #pragma unroll
        for (int y = 0; y < B_Y * colorsPerThread; y += (B_X * B_Y) / preloadCases) {
            imPreload[y * preloadCases/(B_X * B_Y)] = tex1Dfetch<float>(images, imgOffset + y * imgPixels * imgStride + pixIdx);
        }

        #pragma unroll
        for (int y = 0; y < B_X * filtersPerThread; y += (B_X * B_Y) / preloadCases) {
            haPreload[y * preloadCases / (B_X * B_Y)] = tex1Dfetch<float>(hidActs, hidActsOffset + y * numImages * numModules + m * numImages);
        }
    }
//    if (mStartY > mEndY || mStartX > mEndX) {
//        printf("crzy!!\n");
//    }

    for (int my = mStartY; my < mEndY; my++) {
        for (int mx = mStartX; mx < mEndX; mx++) {
            int myNext = my, mxNext = mx;
            const bool lastModule = my == mEndY - 1 && mx == mEndX - 1;

            if (!lastModule) {
                mxNext = mx + 1 == mEndX ? mStartX : mx + 1;
                myNext = my + (mx + 1 == mEndX);
            }

            conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_8_r_16_setCoords(
                    myNext, mxNext, paddingStart, numModulesX, moduleStride,
                    blockPixelY, blockPixelX, imgSizeX, imgStride,
                    pixIdxNext, mNext);

            for (int caseIdx = 0; caseIdx < numImages; caseIdx += preloadCases) {
                #pragma unroll
                for (int y = 0; y < B_Y * colorsPerThread; y += (B_X * B_Y) / preloadCases) {
                    shImgLoad[(y) * preloadCases] = imPreload[y * preloadCases / (B_X * B_Y)];
                }

                #pragma unroll
                for (int y = 0; y < B_X * filtersPerThread; y += (B_X * B_Y) / preloadCases) {
                    shHidActLoad[y * (preloadCases + 1)] = haPreload[y * preloadCases / (B_X * B_Y)];
                }

                __syncthreads();

//                const float* im = &images[caseIdx + preloadCases + pixIdx];
//                const float* ha = &hidActs[caseIdx + preloadCases + m * numImages];
                int imgOffset2 = imgOffset + caseIdx + preloadCases + pixIdx;
                int hidActsOffset2 = hidActsOffset + caseIdx + preloadCases + m * numImages;
                if (caseIdx + preloadCases == numImages) {
                    pixIdx = pixIdxNext;
                    m = mNext;
                    imgOffset2 = imgOffset + pixIdxNext;
                    hidActsOffset2 = hidActsOffset + mNext * numImages;
                }

                WA_LOOP(0);
                WA_LOOP(1);
                WA_LOOP(2);
                WA_LOOP(3);
                WA_LOOP(4);

                WA_LOOP(5);
                WA_IMLOAD_TX(0);
                WA_LOOP(6);
                WA_IMLOAD_TX(1);
                WA_LOOP(7);
                WA_IMLOAD_TX(2);
                WA_LOOP(8);
                WA_IMLOAD_TX(3);
                WA_LOOP(9);
                WA_IMLOAD_TX(4);
                WA_LOOP(10);
                WA_IMLOAD_TX(5);

                WA_LOOP(11);
                WA_HALOAD_TX(0);
                WA_LOOP(12);
                WA_HALOAD_TX(1);
                WA_LOOP(13);
                WA_HALOAD_TX(2);
                WA_LOOP(14);
                WA_HALOAD_TX(3);
                WA_LOOP(15);
                WA_HALOAD_TX(4);
                WA_LOOP(16);
                WA_HALOAD_TX(5);
                WA_LOOP(17);
                WA_HALOAD_TX(6);
                WA_LOOP(18);
                WA_HALOAD_TX(7);
                WA_LOOP(19);
                WA_HALOAD_TX(8);
                WA_LOOP(20);
                WA_HALOAD_TX(9);
                WA_LOOP(21);
                WA_HALOAD_TX(10);
                WA_LOOP(22);
                WA_HALOAD_TX(11);
                WA_LOOP(23);
                WA_HALOAD_TX(12);
                WA_LOOP(24);
                WA_HALOAD_TX(13);
                WA_LOOP(25);
                WA_HALOAD_TX(14);
                WA_LOOP(26);
                WA_HALOAD_TX(15);

                WA_LOOP(27);
                WA_LOOP(28);
                WA_LOOP(29);
                WA_LOOP(30);
                WA_LOOP(31);

                __syncthreads();
            }
        }
    }

    if (scale) {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                targets[c * B_Y * filterPixels * numFilters + f * B_X] = scaleTargets * targets[c * B_Y * filterPixels * numFilters + f * B_X] + scaleOutputs * prod[f][c];
            }
        }
    } else {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                targets[c * B_Y * filterPixels * numFilters + f * B_X] = scaleOutputs * prod[f][c];
            }
        }
    }
}

/*****************************Function Revision Record*****************************
 * Author: Tencent BestImage Team(ankerguo@tencent.com)                           *
 * Date:   2015-05-18                                                             *
 * Reason: Optimizing kernel to get faster speed according to GPU features        *
 * Method:                                                                        *
 *         1. reorganizing data structure to avoid bank conflict;                 *
 *         2. using vectorized data type;                                         *
 *         3. improving instruction-level parallelism;                            *
 *         4. removing redundant 'if' branches;                                   *
 *         5. removing local variables to save registers.                         *
 *********************************************************************************/

/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages), with stride given
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * targets:     (numModulesY*numModulesX/partialSum, numFilterColors, filterPixels, numFilters)
 */
template <int B_Y, int B_X, int filtersPerThread, int colorsPerThread, int preloadCases, bool scale>
__launch_bounds__(256, 2)
__global__ void conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_8_r_16(cudaTextureObject_t images, cudaTextureObject_t hidActs, float* targets,
                                       const int numImages, const int numFilters,
                                       const int numModulesY, const int numModulesX,
                                       const int imgSizeY, const int imgSizeX, const int filterSize,
                                       const int paddingStart, const int moduleStride, const int imgStride,
                                       const int numImgColors, const int numGroups, const int sumWidth,
                                       const float scaleTargets, const float scaleOutputs) {
    // avoid bank conflict by re-organizing the data structure, and improve band width by using 'float2' instead of 'float'
    __shared__ float2 shImages[preloadCases][colorsPerThread * B_Y / 2 + 2]; // preload preloadCases cases
    __shared__ float2 shHidActs[preloadCases][filtersPerThread * B_X / 2 + 2]; // preload preloadCases cases of B_X hidacts
    const int tx = threadIdx.x % B_X, ty = threadIdx.y % B_Y;
    //const int tidx = B_X * threadIdx.y + threadIdx.x;
    // reduce two registers
    //const int loadY = tidx / preloadCases, loadX = tidx % preloadCases;

    //const int filterPixels = filterSize * filterSize;
    // reduce one register
    const int filterPixelsAll = numFilters * filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;

    const int numFilterBlocks = numFilters / (B_X * filtersPerThread);
    const int blockModuleChunkIdx = blockIdx.x / numFilterBlocks;

    const int numModuleChunksX = DIVUP(numModulesX, sumWidth);
    // const int numModuleChunksY = DIVUP(numModulesY, sumWidth);

    const int blockModuleChunkX = blockModuleChunkIdx % numModuleChunksX;
    const int blockModuleChunkY = blockModuleChunkIdx / numModuleChunksX;

    const int blockModuleStartX = blockModuleChunkX * sumWidth;
    const int blockModuleStartY = blockModuleChunkY * sumWidth;

    // const int moduleIdx = partialSum * outputModuleIdx;
    const int blockFilterIdx = filtersPerThread * B_X * (blockIdx.x % numFilterBlocks);
    const int numModules = numModulesY * numModulesX;

    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;
    const int numFilterColors = numImgColors / numGroups;

    const int blockPixelOffset = blockIdx.z; // pixel idx in filter
    const int blockPixelY = blockPixelOffset / filterSize, blockPixelX = blockPixelOffset % filterSize;
    const int blockFilterColorIdx = blockIdx.y  * B_Y * colorsPerThread;
    const int imgColorIdx = blockFilterColorIdx + blockGroupIdx * numFilterColors;
    const int imgOffset = (imgColorIdx + (ty * B_X + tx) / preloadCases) * imgPixels * imgStride + (ty * B_X + tx) % preloadCases;
    // images += (imgColorIdx + loadY) * imgPixels * imgStride + loadX;
    const int hidActsOffset = blockFilterIdx * numImages * numModules
            + ((ty * B_X + tx) / preloadCases) * numImages * numModules
            + ((ty * B_X + tx) % preloadCases);
    //
    // hidActs +=
    //             blockFilterIdx * numImages * numModules
    //            + loadY * numImages * numModules
    //            + loadX;

    // usie one temporary register instead of multiple registers
    const int pIdxBase = imgStride * ((paddingStart + blockPixelY) * imgSizeX + paddingStart + blockPixelX);

    targets += blockModuleChunkIdx * numFilters * filterSize * filterSize * numFilterColors
            + (blockFilterColorIdx + ty) * filterSize * filterSize * numFilters
            + blockPixelOffset * numFilters
            + blockFilterIdx
            + tx;
    // if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) return;

    const int mStartX = max(blockModuleStartX, DIVUP(-blockPixelX - paddingStart, moduleStride));
    const int mStartY = max(blockModuleStartY, DIVUP(-blockPixelY - paddingStart, moduleStride));
    const int mEndX = min(numModulesX, min(blockModuleStartX + sumWidth, DIVUP(imgSizeX - blockPixelX - paddingStart, moduleStride)));
    const int mEndY = min(numModulesY, min(blockModuleStartY + sumWidth, DIVUP(imgSizeY - blockPixelY - paddingStart, moduleStride)));

    // reduce 3 registers
    const bool doWork = mStartY < mEndY && mStartX < mEndX;

    //float* shHidActLoad = &shHidActs[loadY][loadX];
    //float* shImgLoad = &shImages[loadY][loadX];

    float imPreload[preloadCases*colorsPerThread/B_X]; // [4]
    float haPreload[preloadCases*filtersPerThread/B_Y]; // [8]

    float prod[filtersPerThread][colorsPerThread];

    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            prod[f][c] = 0;
        }
    }
    //int pixIdx, pixIdxNext, m, mNext;

    //conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_8_r_16_setCoords(
    //        mStartY, mStartX, paddingStart, numModulesX, moduleStride,
    //        blockPixelY, blockPixelX, imgSizeX, imgStride,
    //        pixIdx, m);
    
    const int pixIdx = pIdxBase + (mStartY * imgSizeX + mStartX) * moduleStride * imgStride;
    const int m = (mStartY * numModulesX + mStartX);

    // preload the image's pixel 
    if (doWork && (ty * B_X + tx) / preloadCases < (B_Y * colorsPerThread / 4)) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            imPreload[i] = tex1Dfetch<float>(images, imgOffset + 16 * i * imgPixels * imgStride + pixIdx);
        }
    }

    // preload the hidAct's pixel
    if (doWork && (ty * B_X + tx) / preloadCases < (B_X * filtersPerThread) / 8) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            haPreload[i] = tex1Dfetch<float>(hidActs, hidActsOffset + 16 * i * numImages * numModules + m * numImages);
        }
    }

    for (int my = mStartY; my < mEndY; my++) {
        for (int mx = mStartX; mx < mEndX; mx++) {

            for (int caseIdx = 0; caseIdx < numImages; caseIdx += preloadCases) {
                int imgOffset2 = imgOffset + caseIdx + preloadCases + pIdxBase + (my * imgSizeX + mx) * moduleStride * imgStride;
                int hidActsOffset2 = hidActsOffset + caseIdx + preloadCases + (my * numModulesX + mx) * numImages;

                if (caseIdx + preloadCases == numImages) {
                    const int mxNext = mx + 1 == mEndX ? mStartX : mx + 1;
                    const int myNext = my + (mx + 1 == mEndX);

                    imgOffset2 = imgOffset + + pIdxBase + (myNext * imgSizeX + mxNext) * moduleStride * imgStride;
                    hidActsOffset2 = hidActsOffset + (myNext * numModulesX + mxNext) * numImages;
                }

                if ((ty * B_X + tx) / preloadCases < (B_Y * colorsPerThread / 4)) {
                    // store the previousely preloaded pixel into shared memory
                    shImages[(ty * B_X + tx) % preloadCases][(ty * B_X + tx) / preloadCases].x = imPreload[0];
                    shImages[(ty * B_X + tx) % preloadCases][(ty * B_X + tx) / preloadCases].y = imPreload[2];
                    shImages[(ty * B_X + tx) % preloadCases][(ty * B_X + tx) / preloadCases + 16].x = imPreload[1];
                    shImages[(ty * B_X + tx) % preloadCases][(ty * B_X + tx) / preloadCases + 16].y = imPreload[3];
		}

                if ((ty * B_X + tx) / preloadCases < (B_X * filtersPerThread / 8)) {
                    shHidActs[(ty * B_X + tx) % preloadCases][(ty * B_X + tx) / preloadCases].x = haPreload[0];
                    shHidActs[(ty * B_X + tx) % preloadCases][(ty * B_X + tx) / preloadCases].y = haPreload[2];
                    shHidActs[(ty * B_X + tx) % preloadCases][(ty * B_X + tx) / preloadCases + 32].x = haPreload[4];
                    shHidActs[(ty * B_X + tx) % preloadCases][(ty * B_X + tx) / preloadCases + 32].y = haPreload[6];
                    shHidActs[(ty * B_X + tx) % preloadCases][(ty * B_X + tx) / preloadCases + 16].x = haPreload[1];
                    shHidActs[(ty * B_X + tx) % preloadCases][(ty * B_X + tx) / preloadCases + 16].y = haPreload[3];
                    shHidActs[(ty * B_X + tx) % preloadCases][(ty * B_X + tx) / preloadCases + 48].x = haPreload[5];
                    shHidActs[(ty * B_X + tx) % preloadCases][(ty * B_X + tx) / preloadCases + 48].y = haPreload[7];
		}

                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    haPreload[r] = tex1Dfetch<float>(hidActs, hidActsOffset2 + r * 16 * numImages * numModules);
                }

                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    imPreload[r] = tex1Dfetch<float>(images, imgOffset2 + r * 16 * imgPixels * imgStride);
                }
                __syncthreads();

                // put together the instructions of same type to improve instruction-level parallelism
                // calculate the derivative of the hidAct with respect to weight
                #pragma unroll
                for (int r = 0; r < 16; r++) {
                    #pragma unroll
                    for (int c = 0; c < 4; c++) { 
                        prod[0][c] += shImages[r][ty + c * B_Y].x * shHidActs[r][tx].x; 
                        prod[1][c] += shImages[r][ty + c * B_Y].x * shHidActs[r][tx].y; 
                        prod[2][c] += shImages[r][ty + c * B_Y].x * shHidActs[r][tx + B_X].x; 
                        prod[3][c] += shImages[r][ty + c * B_Y].x * shHidActs[r][tx + B_X].y; 
                        prod[0][c+4] += shImages[r][ty + c * B_Y].y * shHidActs[r][tx].x; 
                        prod[1][c+4] += shImages[r][ty + c * B_Y].y * shHidActs[r][tx].y; 
                        prod[2][c+4] += shImages[r][ty + c * B_Y].y * shHidActs[r][tx + B_X].x; 
                        prod[3][c+4] += shImages[r][ty + c * B_Y].y * shHidActs[r][tx + B_X].y; 
                    }
                }    

                __syncthreads();
            } 
        } 
    } 

    if (scale) {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                targets[c * B_Y * filterPixelsAll + f * B_X] = scaleTargets * targets[c * B_Y * filterPixelsAll + f * B_X] + scaleOutputs * prod[f][c];
            }
        }
    } else {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                targets[c * B_Y * filterPixelsAll + f * B_X] = scaleOutputs * prod[f][c];
            }
        }
    }
}

std::pair<int,int> getWeightActsOutputSize(int numModulesY, int numModulesX, int numFilterColors,
                                                  int filterSize, int numFilters, int sumWidth) {
    const int outputModuleChunksX = DIVUP(numModulesX, sumWidth);
    const int outputModuleChunksY = DIVUP(numModulesY, sumWidth);
    const int outputModuleChunks = outputModuleChunksX * outputModuleChunksY;
    return std::pair<int,int>(outputModuleChunks * numFilterColors * filterSize * filterSize, numFilters);
}

/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages), with stride given
 * hidActs:     (numFilters, numModules, numImages)
 *
 * targets:     (numModuleY*numModulesX/partialSum, numFilterColors, filterPixels, numFilters)
 *
 * TODO: you can get a slight speed boost for local non-convolutional units by writing special
 * routines for partialSum = 1. But I dunno if the code duplication is worth it...
 *
 * Note: all of these convolution routines are optimized for the case when
 * the number of images (i.e. the minibatch size) is a multiple of 128.
 * Other batch sizes will work, but but I made no attempt whatsoever
 * to make them work fast.
 */
void _weightActs(caffe2::CUDAContext* context, caffe2::TensorCUDA* images, caffe2::TensorCUDA* hidActs, caffe2::TensorCUDA* targets,
                 int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride, int numImgColors,
                 int numGroups, int sumWidth, float scaleTargets, float scaleOutput) {
    CAFFE_ENFORCE(images->ndim() == 2);
    CAFFE_ENFORCE(hidActs->ndim() == 2);
    CAFFE_ENFORCE(targets->ndim() == 2);

    int numFilterColors = numImgColors / numGroups;
    int imgStride = images->dim32(1);
    int numImages = images->dim32(1);
    int imgPixels = images->dim32(0) / numImgColors;
    int imgSizeX = imgPixels / imgSizeY;
    int numModules = numModulesY * numModulesX;
    int numFilters = hidActs->dim32(0) / numModules;
    int numFiltersPerGroup = numFilters / numGroups;

    CAFFE_ENFORCE(numImgColors % numGroups == 0);
    CAFFE_ENFORCE(numFilters % (16*numGroups) == 0);
    CAFFE_ENFORCE(numGroups > 1 || (numImgColors > 0 && (numImgColors <= 3 || numImgColors % 16 == 0)));
    CAFFE_ENFORCE(numGroups == 1 || numFilterColors % 16 == 0);
    CAFFE_ENFORCE(imgSizeY * imgSizeX == imgPixels);
    CAFFE_ENFORCE(images->dim32(0) == imgPixels * numImgColors);

    int filterPixels = filterSize * filterSize;
    int outputModuleChunksX = DIVUP(numModulesX, sumWidth);
    int outputModuleChunksY = DIVUP(numModulesY, sumWidth);
    int outputModuleChunks = outputModuleChunksX * outputModuleChunksY;
//    partialSum = partialSum == 0 ? numModules : partialSum;

//    CAFFE_ENFORCE(numModules % partialSum == 0);
    CAFFE_ENFORCE(hidActs->dim32(1) == numImages);

    // These routines don't handle the case when only part of the image is visited in the convolution
    CAFFE_ENFORCE(paddingStart <= 0);
    CAFFE_ENFORCE(paddingStart + (numModulesX-1)*moduleStride + filterSize >= imgSizeX);
    CAFFE_ENFORCE(paddingStart + (numModulesY-1)*moduleStride + filterSize >= imgSizeY);
    CAFFE_ENFORCE(moduleStride <= filterSize);

    CAFFE_ENFORCE(numModules * numFilters == hidActs->dim32(0));

    int preloadCases = 32;

    dim3 blocks, threads;
    int bx, by;
    int pixelsPerThread, filtersPerThread, colorsPerThread;
    // Worth playing with these parameters to find best values for your problem.
    // These values work relatively well, but not optimal for all problems.
    if (numFilterColors > 3) {
        filtersPerThread = numFiltersPerGroup % 64 == 0 ? 4
                        : numFiltersPerGroup % 32 == 0 ? 2
                        : 1;
        colorsPerThread = numFilterColors % 64 == 0 ? 8
                        : numFilterColors % 48 == 0 ? 6
                        : numFilterColors % 32 == 0 ? 8
                        : 4;
        by = (numFilterColors / colorsPerThread) % 8 == 0 ? 8 : 4;
        bx = numFiltersPerGroup % 128 == 0 ? 32 : 16;
        preloadCases = filtersPerThread * colorsPerThread < 32 ? 32 : 16;
        blocks = dim3(outputModuleChunks*(numFilters/(bx*filtersPerThread)), numFilterColors / (by*colorsPerThread), filterPixels);
        CAFFE_ENFORCE(numFilterColors % (by*colorsPerThread) == 0);
    } else { // This is ugly but it's nice to spell it out clearly
        CAFFE_ENFORCE(numGroups == 1); // Just for sanity
        // NOTE: these things are only optimized for colors = 3. I didn't really test other cases.
        if (numFilters % 64 == 0) { // TODO: having a separate case for 128 would make things faster, but I probably don't care about 128
            filtersPerThread = 4;
            pixelsPerThread = 2;
            by = 16;
            bx = 16;
            preloadCases = 32;
        } else if (numFilters % 48 == 0) {
            filtersPerThread = 3;
            pixelsPerThread = 4;
            by = 16;
            bx = 16;
            preloadCases = 32;
        } else if (numFilters % 32 == 0) {
            filtersPerThread = 2;
            pixelsPerThread = 2;
            by = 8;
            bx = 16;
            preloadCases = 16;
        } else { // This case is completely untested. It might be really slow. But no time now.
            filtersPerThread = 1;
            pixelsPerThread = 16;
            by = 16;
            bx = 16;
            preloadCases = 32;
        }
        blocks = dim3(outputModuleChunks*(numFilters/(bx*filtersPerThread)), DIVUP(filterPixels, by*pixelsPerThread));
    }
    CAFFE_ENFORCE((by * bx) % preloadCases == 0);
    CAFFE_ENFORCE(numFilters % (bx * filtersPerThread) == 0);
    threads = dim3(bx, by);
    bool checkCaseBounds = numImages % preloadCases != 0;
    bool scale = scaleTargets != 0;
    std::pair<int,int> targetSize = getWeightActsOutputSize(numModulesY, numModulesX, numFilterColors, filterSize, numFilters, sumWidth);
    if (!scale) {
        targets->Resize(std::vector<int>{targetSize.first, targetSize.second});
    } else {
        CAFFE_ENFORCE(targets->dim32(0) == targetSize.first);
        CAFFE_ENFORCE(targets->dim32(1) == targetSize.second);
    }

    cudaTextureObject_t tex_images = GetTensorTextureObject(images);
    cudaTextureObject_t tex_hidacts = GetTensorTextureObject(hidActs);
    float* images_data = images->mutable_data<float>();
    float* hidacts_data = hidActs->mutable_data<float>();
    float* targets_data = targets->mutable_data<float>();
    const std::size_t images_bytes = images->nbytes();

    cudaStream_t stream = context->cuda_stream();
    
    checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

    if (scale == false) {
        if (checkCaseBounds == false) {
            if (numFilterColors > 3)  {
                if (numFilterColors % 64 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_8_r_16< 8, 32, 4, 8, 16, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_8_r_16< 8, 32, 4, 8, 16, false ><<<blocks, threads, 0, stream>>>(tex_images, tex_hidacts, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_16_f_4_c_8_r_16< 8, 16, 4, 8, 16, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_16_f_4_c_8_r_16< 8, 16, 4, 8, 16, false ><<<blocks, threads, 0, stream>>>(tex_images, tex_hidacts, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 16, 2, 8, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 16, 2, 8, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 16, 1, 8, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 16, 1, 8, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                }
                else if (numFilterColors % 48 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_6_r_32< 8, 32, 4, 6, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_6_r_32< 8, 32, 4, 6, 32, false ><<<blocks, threads, 0, stream>>>(tex_images, tex_hidacts, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 16, 4, 6, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 16, 4, 6, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 16, 2, 6, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 16, 2, 6, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 16, 1, 6, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 16, 1, 6, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                }
                else if (numFilterColors % 32 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 32, 4, 8, 16, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 32, 4, 8, 16, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 4, 8, 16, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 4, 8, 16, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 2, 8, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 2, 8, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 1, 8, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 1, 8, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                }
                else if (numFilterColors % 16 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 32, 4, 4, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 32, 4, 4, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 4, 4, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 4, 4, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {

                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 2, 4, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 2, 4, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 1, 4, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 1, 4, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                }
            }
            else if (numFilterColors <= 3) {
                if (numFilterColors == 3) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_preload_pc_2_pt_2_f_4_r_32_c_3 < 16, 16, 2, 2, 4, 32, 3, false, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_preload_pc_2_pt_2_f_4_r_32_c_3 < 16, 16, 2, 2, 4, 32, 3, false, false ><<<blocks, threads, 0, stream>>>(tex_images, tex_hidacts, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_preload_pc_2_pt_4_f_3_r_32_c_3 < 16, 16, 2, 4, 3, 32, 3, false, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_preload_pc_2_pt_4_f_3_r_32_c_3 < 16, 16, 2, 4, 3, 32, 3, false, false ><<<blocks, threads, 0, stream>>>(tex_images, tex_hidacts, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 3, false, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 3, false, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 3, false, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 3, false, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                }
                else if (numFilterColors == 2) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 2, 4, 32, 2, false, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 2, 4, 32, 2, false, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 4, 3, 32, 2, false, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 4, 3, 32, 2, false, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 2, false, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 2, false, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 2, false, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 2, false, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                }
                else if (numFilterColors == 1) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 2, 4, 32, 1, false, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 2, 4, 32, 1, false, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 4, 3, 32, 1, false, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 4, 3, 32, 1, false, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 1, false, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 1, false, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 1, false, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 1, false, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                }
            }
        }
        else if (checkCaseBounds == true) {
            if (numFilterColors > 3) {
                if (numFilterColors % 64 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 32, 4, 8, 16, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 32, 4, 8, 16, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 16, 4, 8, 16, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 16, 4, 8, 16, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 16, 2, 8, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 16, 2, 8, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 16, 1, 8, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 16, 1, 8, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                }
                else if (numFilterColors % 48 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 32, 4, 6, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 32, 4, 6, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 16, 4, 6, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 16, 4, 6, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 16, 2, 6, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 16, 2, 6, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 16, 1, 6, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 16, 1, 6, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                }
                else if (numFilterColors % 32 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 32, 4, 8, 16, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 32, 4, 8, 16, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 4, 8, 16, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 4, 8, 16, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 2, 8, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 2, 8, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 1, 8, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 1, 8, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                }
                else if (numFilterColors % 16 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 32, 4, 4, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 32, 4, 4, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 4, 4, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 4, 4, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 2, 4, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 2, 4, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 1, 4, 32, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 1, 4, 32, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                }
            }
            else if (numFilterColors <= 3) {
                if (numFilterColors == 3) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 2, 4, 32, 3, false, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 2, 4, 32, 3, false, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 4, 3, 32, 3, false, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 4, 3, 32, 3, false, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 3, false, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 3, false, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 3, false, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 3, false, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                }
                else if (numFilterColors == 2) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 2, 4, 32, 2, false, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 2, 4, 32, 2, false, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 4, 3, 32, 2, false, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 4, 3, 32, 2, false, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 2, false, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 2, false, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 2, false, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 2, false, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                }
                else if (numFilterColors == 1) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 2, 4, 32, 1, false, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 2, 4, 32, 1, false, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 4, 3, 32, 1, false, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 4, 3, 32, 1, false, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 1, false, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 1, false, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 1, false, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 1, false, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                }
            }
        }
    }
    else if (scale == true) {
        if (checkCaseBounds == false) {
            if (numFilterColors > 3) {
                if (numFilterColors % 64 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_8_r_16< 8, 32, 4, 8, 16, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_8_r_16< 8, 32, 4, 8, 16, true ><<<blocks, threads, 0, stream>>>(tex_images, tex_hidacts, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_16_f_4_c_8_r_16< 8, 16, 4, 8, 16, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_16_f_4_c_8_r_16< 8, 16, 4, 8, 16, true ><<<blocks, threads, 0, stream>>>(tex_images, tex_hidacts, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 16, 2, 8, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 16, 2, 8, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 16, 1, 8, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 16, 1, 8, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                }
                else if (numFilterColors % 48 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_6_r_32< 8, 32, 4, 6, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_6_r_32< 8, 32, 4, 6, 32, true ><<<blocks, threads, 0, stream>>>(tex_images, tex_hidacts, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 16, 4, 6, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 16, 4, 6, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 16, 2, 6, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 16, 2, 6, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 16, 1, 6, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 16, 1, 6, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                }
                else if (numFilterColors % 32 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 32, 4, 8, 16, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 32, 4, 8, 16, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 4, 8, 16, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 4, 8, 16, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 2, 8, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 2, 8, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 1, 8, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 1, 8, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                }
                else if (numFilterColors % 16 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 32, 4, 4, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 32, 4, 4, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 4, 4, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 4, 4, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 2, 4, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 2, 4, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 1, 4, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 1, 4, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                }
            }
            else if (numFilterColors <= 3) {
                if (numFilterColors == 3) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_preload_pc_2_pt_2_f_4_r_32_c_3 < 16, 16, 2, 2, 4, 32, 3, true, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_preload_pc_2_pt_2_f_4_r_32_c_3 < 16, 16, 2, 2, 4, 32, 3, true, false ><<<blocks, threads, 0, stream>>>(tex_images, tex_hidacts, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_preload_pc_2_pt_4_f_3_r_32_c_3 < 16, 16, 2, 4, 3, 32, 3, true, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_preload_pc_2_pt_4_f_3_r_32_c_3 < 16, 16, 2, 4, 3, 32, 3, true, false ><<<blocks, threads, 0, stream>>>(tex_images, tex_hidacts, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 3, true, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 3, true, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 3, true, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 3, true, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                }
                else if (numFilterColors == 2) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 2, 4, 32, 2, true, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 2, 4, 32, 2, true, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 4, 3, 32, 2, true, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 4, 3, 32, 2, true, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 2, true, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 2, true, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 2, true, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 2, true, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                }
                else if (numFilterColors == 1) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 2, 4, 32, 1, true, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 2, 4, 32, 1, true, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 4, 3, 32, 1, true, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 4, 3, 32, 1, true, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 1, true, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 1, true, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 1, true, false >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 1, true, false ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                }
            }
        }
        else if (checkCaseBounds == true) {
            if (numFilterColors > 3) {
                if (numFilterColors % 64 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 32, 4, 8, 16, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 32, 4, 8, 16, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 16, 4, 8, 16, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 16, 4, 8, 16, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 16, 2, 8, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 16, 2, 8, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 16, 1, 8, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 16, 1, 8, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                }
                else if (numFilterColors % 48 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 32, 4, 6, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 32, 4, 6, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 16, 4, 6, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 16, 4, 6, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 16, 2, 6, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 16, 2, 6, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 8, 16, 1, 6, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 8, 16, 1, 6, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                }
                else if (numFilterColors % 32 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 32, 4, 8, 16, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 32, 4, 8, 16, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 4, 8, 16, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 4, 8, 16, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 2, 8, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 2, 8, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 1, 8, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 1, 8, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                }
                else if (numFilterColors % 16 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 32, 4, 4, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 32, 4, 4, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 4, 4, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 4, 4, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 2, 4, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 2, 4, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_mc_mf_kepler_sw < 4, 16, 1, 4, 32, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw < 4, 16, 1, 4, 32, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, numImgColors, numGroups, sumWidth, scaleTargets, scaleOutput);
                    }
                }
            }
            else if (numFilterColors <= 3) {
                if (numFilterColors == 3) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 2, 4, 32, 3, true, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 2, 4, 32, 3, true, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 4, 3, 32, 3, true, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 4, 3, 32, 3, true, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 3, true, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 3, true, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 3, true, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 3, true, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                }
                else if (numFilterColors == 2) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 2, 4, 32, 2, true, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 2, 4, 32, 2, true, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 4, 3, 32, 2, true, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 4, 3, 32, 2, true, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 2, true, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 2, true, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 2, true, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 2, true, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                }
                else if (numFilterColors == 1) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 2, 4, 32, 1, true, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 2, 4, 32, 1, true, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 4, 3, 32, 1, true, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 4, 3, 32, 1, true, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 1, true, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 8, 16, 2, 2, 2, 16, 1, true, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                    else if (numFiltersPerGroup % 16 == 0) {
                        cudaFuncSetCacheConfig(conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 1, true, true >, cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw < 16, 16, 2, 16, 1, 32, 1, true, true ><<<blocks, threads, 0, stream>>>(images_data, hidacts_data, targets_data, numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                }
            }
        }
    }
    checkCudaErrors(cudaDestroyTextureObject(tex_images));
    checkCudaErrors(cudaDestroyTextureObject(tex_hidacts));
    checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));
    getLastCudaError("weightActs: kernel execution failed");
}

void convWeightActs(caffe2::CUDAContext* context, caffe2::TensorCUDA* images, caffe2::TensorCUDA* hidActs, caffe2::TensorCUDA* targets,
                    int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride, int numImgColors, int numGroups, int partialSum) {
    _weightActs(context, images, hidActs, targets, imgSizeY, numModulesY, numModulesX, filterSize, paddingStart, moduleStride, numImgColors, numGroups, partialSum, 0, 1);
}

void convWeightActs(caffe2::CUDAContext* context, caffe2::TensorCUDA* images, caffe2::TensorCUDA* hidActs, caffe2::TensorCUDA* targets,
                    int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride, int numImgColors, int numGroups, int partialSum,
                    float scaleTargets, float scaleOutput) {
    _weightActs(context, images, hidActs, targets, imgSizeY, numModulesY, numModulesX, filterSize, paddingStart, moduleStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
}

void localWeightActs(caffe2::CUDAContext* context, caffe2::TensorCUDA* images, caffe2::TensorCUDA* hidActs, caffe2::TensorCUDA* targets,
                     int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride, int numImgColors, int numGroups) {
    _weightActs(context, images, hidActs, targets, imgSizeY, numModulesY, numModulesX, filterSize, paddingStart, moduleStride, numImgColors, numGroups, 1, 0, 1);
}

void localWeightActs(caffe2::CUDAContext* context, caffe2::TensorCUDA* images, caffe2::TensorCUDA* hidActs, caffe2::TensorCUDA* targets,
                    int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride,
                    int numImgColors, int numGroups, float scaleTargets, float scaleOutput) {
    _weightActs(context, images, hidActs, targets, imgSizeY, numModulesY, numModulesX, filterSize, paddingStart, moduleStride, numImgColors, numGroups, 1, scaleTargets, scaleOutput);
}
