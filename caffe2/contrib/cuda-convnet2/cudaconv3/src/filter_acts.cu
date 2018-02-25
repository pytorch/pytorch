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

#include "../../nvmatrix/include/nvmatrix.cuh"
#include "../include/cudaconv2.cuh"

__device__ __forceinline__ void filterActs_YxX_color_preload_ty_4_tx_32_f_16_cc_3_setImgCoords(int fPidx, int imgLoadModPosY, int imgLoadModPosX,
                                                                                        int imgSizeX, int filterSize, int& iPidx) {
    int x = imgLoadModPosX + (fPidx) % filterSize;
    int y = imgLoadModPosY + (fPidx) / filterSize;
    iPidx = y >= 0 && y < imgSizeX && x >= 0 && x < imgSizeX ? y * imgSizeX + x : -1;
}

#define FA_COLOR3_IMPRELOAD(c,i) imPreload[c][i] = iPidxNext < 0 || (checkImgBounds && myImgIdx + i * B_X >= numImages) ? 0 : mm[c * imgPixels * imgStride + i * B_X];
#define FA_COLOR3_IMPRELOAD_TX(c,i) imPreload[c][i] = iPidxNext < 0 || (checkImgBounds && myImgIdx + i * B_X >= numImages) ? 0 : tex1Dfetch<float>(images, imagesOffset2 + c * imgPixels * imgStride + i * B_X);


/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 */
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int numColors, int pixelCache,
          bool scale, bool checkImgBounds>
//__launch_bounds__(128,3)
__global__ void filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_16_px_4_cc_3_tex(cudaTextureObject_t images, cudaTextureObject_t filters, float* targets,
                                       const int numImages, const int numFilters,
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesY, const int numModulesX, const int imgStride,
                                       const float scaleTargets, const float scaleOutputs,
                                       const bool conv/*, const bool noloads*/) {
    __shared__ float shFilters[numColors][pixelCache][B_Y * filtersPerThread]; // pre-load 1 pixel from B_Y*filtersPerThread filters
    __shared__ float shImages[numColors][pixelCache][B_X * imgsPerThread]; // pre-load 1 pixel from B_X*imgsPerThread images
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;
    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);

    const int numModules = numModulesX * numModulesY;
    // Another fun insanity: the % B_X makes things faster, even thought threadIdx.x is
    // in the range 0..31. It appears that this allows the compiler to optimize?
    const int tx = threadIdx.x % B_X;
    const int ty = threadIdx.y % B_Y;
    const int tidx = ty * B_X + threadIdx.x;

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;

//    images += myImgIdx;
//    filters += blockFilterIdx
//            + shFilterLoadY * numFilters + shFilterLoadX;
//    if (!conv) { // NOTE: UNTESTED!
//        filters += moduleIdx * numColors * filterPixels * numFilters;
//    }

    const int imagesOffset = myImgIdx;
    const int filtersOffset = blockFilterIdx + shFilterLoadY * numFilters + shFilterLoadX
                            + (conv ? 0 : moduleIdx * numColors * filterPixels * numFilters);

    targets += moduleIdx * numImages
            + (blockFilterIdx + threadIdx.y * filtersPerThread) * numImages * numModules
            + myImgIdx;

    float prod[imgsPerThread][filtersPerThread];
    #pragma unroll
    for(int i = 0; i < imgsPerThread; i++) {
        #pragma unroll
        for(int f = 0; f < filtersPerThread; f++) {
            prod[i][f] = 0;
        }
    }

    int iPidxNext;
    float imPreload[numColors][imgsPerThread];
    float fPreload[numColors][pixelCache*filtersPerThread/B_X];

    #pragma unroll
    for (int c = 0; c < numColors; ++c) {
        #pragma unroll
        for (int p = 0; p < pixelCache; p += B_X/filtersPerThread) {
            if (p + shFilterLoadY < filterPixels) {
                fPreload[c][p*filtersPerThread/B_X] = tex1Dfetch<float>(filters, filtersOffset + p * numFilters + c * numFilters * filterPixels);
            } else{
                fPreload[c][p*filtersPerThread/B_X] = 0;
            }
        }
    }

    filterActs_YxX_color_preload_ty_4_tx_32_f_16_cc_3_setImgCoords(ty, imgLoadModPosY, imgLoadModPosX, imgSizeX, filterSize, iPidxNext);

    #pragma unroll
    for (int c = 0; c < numColors; ++c) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (iPidxNext >= 0 && (!checkImgBounds || myImgIdx + i * B_X < numImages)) {
                imPreload[c][i] = tex1Dfetch<float>(images, imagesOffset + (c * imgPixels + iPidxNext) * imgStride + i * B_X);
            } else {
                imPreload[c][i] =  0;
            }
        }
    }

    for (int p = 0; p < filterPixels; p += pixelCache) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            #pragma unroll
            for (int c = 0; c < numColors; ++c) {
                // NOTE: bank conflicts here!
                shImages[c][ty][tx * imgsPerThread + i] = imPreload[c][i];
            }
        }

        const int fPidxNext = p + pixelCache >= filterPixels ? 0 : p + pixelCache;
        filterActs_YxX_color_preload_ty_4_tx_32_f_16_cc_3_setImgCoords(fPidxNext + ty, imgLoadModPosY, imgLoadModPosX, imgSizeX, filterSize, iPidxNext);

//        const float* ff = &filters[numFilters * fPidxNext];
//        const float* mm = &images[imgStride * iPidxNext];
        const int filtersOffset2 = filtersOffset + numFilters * fPidxNext;
        const int imagesOffset2 = imagesOffset + imgStride * iPidxNext;

        FA_COLOR3_IMPRELOAD_TX(0,0);
        FA_COLOR3_IMPRELOAD_TX(0,1);
        FA_COLOR3_IMPRELOAD_TX(0,2);
        FA_COLOR3_IMPRELOAD_TX(0,3);

        #pragma unroll
        for (int c = 0; c < numColors; ++c) {
            #pragma unroll
            for (int pp = 0; pp < pixelCache; pp += B_X/filtersPerThread) {
                shFilters[c][pp + shFilterLoadY][shFilterLoadX] = fPreload[c][pp*filtersPerThread/B_X];
            }
        }

        __syncthreads();
        FA_COLOR3_IMPRELOAD_TX(1,0);
        FA_COLOR3_IMPRELOAD_TX(1,1);
        FA_COLOR3_IMPRELOAD_TX(1,2);
        FA_COLOR3_IMPRELOAD_TX(1,3);
        FA_COLOR3_IMPRELOAD_TX(2,0);
        FA_COLOR3_IMPRELOAD_TX(2,1);
        FA_COLOR3_IMPRELOAD_TX(2,2);
        FA_COLOR3_IMPRELOAD_TX(2,3);
        #pragma unroll
        for (int c = 0; c < numColors; c++) {
            #pragma unroll
            for (int pp = 0; pp < pixelCache*filtersPerThread/B_X; pp++) {
                fPreload[c][pp] = fPidxNext + pp*(B_X/filtersPerThread) + shFilterLoadY >= filterPixels ? 0 : tex1Dfetch<float>(filters, filtersOffset2 + c * numFilters* filterPixels + pp*(B_X/filtersPerThread) * numFilters);
            }
        }
        #pragma unroll
        for (int pp = 0; pp < pixelCache; pp++) {
            #pragma unroll
            for (int c = 0; c < numColors; c++) {
                #pragma unroll
                for(int f = 0; f < filtersPerThread; f++) {
                    #pragma unroll
                    for(int i = 0; i < imgsPerThread; i++) {
                        prod[i][f] += shImages[c][pp][tx * imgsPerThread + i] * shFilters[c][pp][ty * filtersPerThread + f];
                    }
                }
            }
        }

        __syncthreads();
    }

    if (scale) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                    targets[i * B_X + f * numImages * numModules] = scaleTargets * targets[i * B_X + f * numImages * numModules] + scaleOutputs * prod[i][f];
                }
            }
        }
    } else {
        // Note: reversing order of these loops saves 2 registers, but costs time
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                    targets[i * B_X + f * numImages * numModules] = scaleOutputs * prod[i][f];
                }
            }
        }
    }
}

/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * This won't be pretty.
 */
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int numColors, int pixelCache,
          bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_12_px_4_cc_3_tex(cudaTextureObject_t images, cudaTextureObject_t filters, float* targets,
                                       const int numImages, const int numFilters,
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesY, const int numModulesX, const int imgStride,
                                       const float scaleTargets, const float scaleOutputs,
                                       const bool conv/*, const bool noloads*/) {
    __shared__ float shFilters[numColors][pixelCache][B_Y * filtersPerThread]; // pre-load 1 pixel from B_Y*filtersPerThread filters
    __shared__ float shImages[numColors][pixelCache][B_X * imgsPerThread]; // pre-load 1 pixel from B_X*imgsPerThread images
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;
    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);

    const int numModules = numModulesX * numModulesY;
    // Another fun insanity: the % B_X makes things faster, even though threadIdx.x is
    // in the range 0..31. It appears that this allows the compiler to optimize?
    const int tx = threadIdx.x % B_X;
    const int ty = threadIdx.y % B_Y;
    const int tidx = ty * B_X + threadIdx.x;
    const int warp = tidx / 32;

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;

//    images += myImgIdx;
//    filters += blockFilterIdx
//            + shFilterLoadY * numFilters + shFilterLoadX;
//    if (!conv) { // NOTE: UNTESTED!
//        filters += moduleIdx * numColors * filterPixels * numFilters;
//    }

    const int imagesOffset = myImgIdx;
    const int filtersOffset = blockFilterIdx + shFilterLoadY * numFilters + shFilterLoadX
                            + (conv ? 0 : moduleIdx * numColors * filterPixels * numFilters);

    targets += moduleIdx * numImages
            + (blockFilterIdx + threadIdx.y * filtersPerThread) * numImages * numModules
            + myImgIdx;

    float prod[imgsPerThread][filtersPerThread];
    #pragma unroll
    for(int i = 0; i < imgsPerThread; i++) {
        #pragma unroll
        for(int f = 0; f < filtersPerThread; f++) {
            prod[i][f] = 0;
        }
    }

    int iPidxNext;
    float imPreload[numColors][imgsPerThread];
    float fPreload[numColors][DIVUP(pixelCache*filtersPerThread,B_X)];

    if (warp < 3) {
        #pragma unroll
        for (int c = 0; c < numColors; ++c) {
            #pragma unroll
            for (int p = 0; p < pixelCache; p += 2) {
                if (p + shFilterLoadY < filterPixels) {
                    fPreload[c][p/2] = tex1Dfetch<float>(filters, filtersOffset + p * numFilters + c * numFilters * filterPixels);
                } else {
                    fPreload[c][p/2] = 0;
                }
            }
        }
    }

    filterActs_YxX_color_preload_ty_4_tx_32_f_16_cc_3_setImgCoords(ty, imgLoadModPosY, imgLoadModPosX, imgSizeX, filterSize, iPidxNext);

    #pragma unroll
    for (int c = 0; c < numColors; ++c) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (iPidxNext >= 0 && (!checkImgBounds || myImgIdx + i * B_X < numImages)) {
                imPreload[c][i] = tex1Dfetch<float>(images, imagesOffset + (c * imgPixels + iPidxNext) * imgStride + i * B_X);
            } else {
                imPreload[c][i] =  0;
            }
        }
    }

    for (int p = 0; p < filterPixels; p += pixelCache) {
        const int fPidxNext = p + pixelCache >= filterPixels ? 0 : p + pixelCache;
        filterActs_YxX_color_preload_ty_4_tx_32_f_16_cc_3_setImgCoords(fPidxNext + ty, imgLoadModPosY, imgLoadModPosX, imgSizeX, filterSize, iPidxNext);

        #pragma unroll
        for (int c = 0; c < numColors; ++c) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                // NOTE: bank conflicts here!
                shImages[c][ty][tx * imgsPerThread + i] = imPreload[c][i];
            }
        }

        if (warp < 3) {
            #pragma unroll
            for (int c = 0; c < numColors; ++c) {
                #pragma unroll
                for (int pp = 0; pp < pixelCache; pp += 2) {
                    shFilters[c][pp + shFilterLoadY][shFilterLoadX] = fPreload[c][pp/2];
                }
            }
        }

        __syncthreads();
//        const float* ff = &filters[numFilters * fPidxNext];
//        const float* mm = &images[imgStride * iPidxNext];
        const int filtersOffset2 = filtersOffset + numFilters * fPidxNext;
        const int imagesOffset2 = imagesOffset + imgStride * iPidxNext;

        #pragma unroll
        for (int i = 0; i < imgsPerThread; ++i) {
            #pragma unroll
            for (int c = 0; c < numColors; c++) {
                FA_COLOR3_IMPRELOAD_TX(c,i);
            }
        }

        #pragma unroll
        for (int c = 0; c < numColors; c++) {
            #pragma unroll
            for (int pp = 0; pp < 2; pp++) {
                fPreload[c][pp] = warp >= 3 || fPidxNext + pp*2 + shFilterLoadY >= filterPixels ? 0 : tex1Dfetch<float>(filters, filtersOffset2 +  c * numFilters* filterPixels + pp*2 * numFilters);
            }
            #pragma unroll
            for (int pp = 0; pp < pixelCache; pp++) {
                #pragma unroll
                for(int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for(int f = 0; f < filtersPerThread; f++) {
                        prod[i][f] += shImages[c][pp][tx * imgsPerThread + i] * shFilters[c][pp][ty * filtersPerThread + f];
                    }
                }
            }

        }
        __syncthreads();
    }

    if (scale) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                    targets[i * B_X + f * numImages * numModules] = scaleTargets * targets[i * B_X + f * numImages * numModules] + scaleOutputs * prod[i][f];
                }
            }
        }
    } else {
        // Note: reversing order of these loops costs 2 registers, but saves time
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                    targets[i * B_X + f * numImages * numModules] = scaleOutputs * prod[i][f];
                }
            }
        }
    }
}

__device__ inline void filterActs_YxX_sparse2_preload_ty_4_tx_32_f_16_c_4_setPixelCoords(int filterSize, int imgSizeX, int imgLoadModPosY, int imgLoadModPosX, int imgY, int imgX, int& fPidx, int& iPidx) {
    int filterPxY = imgY - imgLoadModPosY;
    int filterPxX = imgX - imgLoadModPosX;
    fPidx = filterPxY * filterSize + filterPxX;
    iPidx = imgY * imgSizeX + imgX; // Pixel index in img
}

/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * Note: in git there's a 1.5% faster version of this which sues 167 registers instead of 154...
 * it's basically the same thing, but it doesn't do the next-pixel computation. It just avoids
 * pre-loading when it rolls over to the next pixel.
 */
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int colorCache,
          bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4(float* images, float* filters, float* targets,
                                       const int numImages, const int numFilters,
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesY, const int numModulesX, const int imgStride, const int numImgColors,
                                       const int numGroups,
                                       const float scaleTargets, const float scaleOutputs,
                                       const bool conv/*, const bool noloads*/) {
    __shared__ float shFilters[colorCache][B_Y * filtersPerThread]; // pre-load 1 pixel from B_Y*filtersPerThread filters
    __shared__ float shImages[colorCache][B_X * imgsPerThread]; // pre-load 1 pixel from B_X*imgsPerThread images
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;
    const int numFilterColors = numImgColors / numGroups;
    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;

    const int numModules = numModulesX * numModulesY;
    const int blockColorIdx = numFilterColors * blockGroupIdx;
    // Another fun insanity: the % B_X makes things faster, even thought threadIdx.x is
    // in the range 0..31. It appears that this allows the compiler to optimize?
    const int tx = threadIdx.x % B_X;
    const int ty = threadIdx.y % B_Y;
    const int tidx = ty * B_X + threadIdx.x;

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;

    images += (blockColorIdx + threadIdx.y) * imgPixels * imgStride + myImgIdx;
    filters +=blockFilterIdx
            + shFilterLoadY * numFilters * filterPixels + shFilterLoadX;
    if (!conv) {
        filters += moduleIdx * numFilterColors * filterPixels * numFilters;
    }

    targets += moduleIdx * numImages
            + (blockFilterIdx + threadIdx.y * filtersPerThread) * numImages * numModules
            + myImgIdx;

    float prod[imgsPerThread][filtersPerThread];
//    float fCache[filtersPerThread];
    #pragma unroll
    for(int i = 0; i < imgsPerThread; i++) {
        #pragma unroll
        for(int f = 0; f < filtersPerThread; f++) {
            prod[i][f] = 0;
        }
    }
    // NOTE: these max/min functions increase register usage as compared to my macros
    const int imgStartX = max(0, imgLoadModPosX);
    const int imgStartY = max(0, imgLoadModPosY);
    const int imgEndX = min(imgLoadModPosX + filterSize, imgSizeX);
    const int imgEndY = min(imgLoadModPosY + filterSize, imgSizeY);
//    __shared__ int imgPos[]

    int fPidx, iPidx;
    float imPreload[imgsPerThread];
    float fPreload[colorCache*filtersPerThread/B_X];
//    float fCache[filtersPerThread];

    filterActs_YxX_sparse2_preload_ty_4_tx_32_f_16_c_4_setPixelCoords(filterSize, imgSizeX, imgLoadModPosY, imgLoadModPosX, imgStartY, imgStartX, fPidx, iPidx);

    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
            imPreload[i] = images[imgStride * iPidx + i * B_X];
        } else {
            imPreload[i] = 0;
        }
    }
    if (/*B_X % filtersPerThread == 0 ||*/ shFilterLoadY < B_X/filtersPerThread) { // This if statement reduces reg usage..
        #pragma unroll
        for (int c = 0; c < colorCache; c += B_X/filtersPerThread) {
            fPreload[c*filtersPerThread/B_X] = filters[(c * filterPixels + fPidx) * numFilters];
        }
    }
    for (int imgY = imgStartY; imgY < imgEndY; ++imgY) {
//        const int filterPxY = imgY - imgLoadModPosY;
        for (int imgX = imgStartX; imgX < imgEndX; ++imgX) {
//            const int filterPxX = imgX - imgLoadModPosX;
//            const int p = filterPxY * filterSize + filterPxX;
//            const int pixIdx = imgY * imgSizeX + imgX;// Pixel index in img
//            setPixelCoords(filterSize, imgSizeX, imgLoadModPosY, imgLoadModPosX, imgY, imgX, &p, &pixIdx);
//            float* m = &images[imgStride * pixIdx];
            const bool lastPixel = imgY == imgEndY - 1 && imgX == imgEndX - 1;
            int imgYNext = imgY;
            int imgXNext = imgX;
            int fPidxNext, iPidxNext;
            if (!lastPixel) {
                imgYNext = imgY + (imgX + 1 == imgEndX);
                imgXNext = imgX + 1 == imgEndX ? imgStartX : imgX + 1;
            }
            filterActs_YxX_sparse2_preload_ty_4_tx_32_f_16_c_4_setPixelCoords(filterSize, imgSizeX, imgLoadModPosY, imgLoadModPosX, imgYNext, imgXNext, fPidxNext, iPidxNext);
            for (int oc = 0; oc < numFilterColors; oc += colorCache) { // oc stands for outer color (loop)
                const float* ff = &filters[numFilters * ((oc + colorCache) * filterPixels + fPidx)];
                const float* mm = &images[imgStride * ((oc + colorCache) * imgPixels + iPidx)];
                if (oc == numFilterColors - colorCache) {
                    ff = &filters[fPidxNext * numFilters];
                    mm = &images[iPidxNext * imgStride];
                    fPidx = fPidxNext;
                    iPidx = iPidxNext;
                }

                #pragma unroll
                for (int c = 0; c < colorCache; c += B_X/filtersPerThread) {
                    shFilters[c + shFilterLoadY][shFilterLoadX] = fPreload[c*filtersPerThread/B_X];
                }

                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    // NOTE: bank conflicts here!
                    shImages[ty][tx * imgsPerThread + i] = imPreload[i];
                }
                imPreload[0] = (checkImgBounds && myImgIdx + 0 * B_X >= numImages) ? 0 : mm[0 * B_X];
                imPreload[1] = (checkImgBounds && myImgIdx + 1 * B_X >= numImages) ? 0 : mm[1 * B_X];
                imPreload[2] = (checkImgBounds && myImgIdx + 2 * B_X >= numImages) ? 0 : mm[2 * B_X];

                __syncthreads();

                #pragma unroll
                for(int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for(int f = 0; f < filtersPerThread; f++) {
                        prod[i][f] += shImages[0][threadIdx.x * imgsPerThread + i] * shFilters[0][threadIdx.y * filtersPerThread + f];
                    }
                }

                fPreload[0] = ff[0];

                #pragma unroll
                for(int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for(int f = 0; f < filtersPerThread; f++) {
                        prod[i][f] += shImages[1][threadIdx.x * imgsPerThread + i] * shFilters[1][threadIdx.y * filtersPerThread + f];
                    }
                }

                fPreload[1] = ff[(B_X/filtersPerThread * filterPixels) * numFilters];

                #pragma unroll
                for(int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for(int f = 0; f < filtersPerThread; f++) {
                        prod[i][f] += shImages[2][threadIdx.x * imgsPerThread + i] * shFilters[2][threadIdx.y * filtersPerThread + f];
                    }
                }

                imPreload[3] = (checkImgBounds && myImgIdx + 3 * B_X >= numImages) ? 0 : mm[3 * B_X];

                #pragma unroll
                for(int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for(int f = 0; f < filtersPerThread; f++) {
                        prod[i][f] += shImages[3][threadIdx.x * imgsPerThread + i] * shFilters[3][threadIdx.y * filtersPerThread + f];
                    }
                }
                __syncthreads();
            }
        }
    }

    if (scale) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                    targets[i * B_X + f * numImages * numModules] = scaleTargets * targets[i * B_X + f * numImages * numModules] + scaleOutputs * prod[i][f];
                }
            }
        }
    } else {
        // Note: reversing order of these loops saves 2 registers, but costs time
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                    targets[i * B_X + f * numImages * numModules] = scaleOutputs * prod[i][f];
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
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 */
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int colorCache,
          bool scale, bool checkImgBounds>
__global__ void 
__launch_bounds__(128, 4)
filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex(cudaTextureObject_t images, cudaTextureObject_t filters, float* targets,
                                       const int numImages, const int numFilters,
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesY, const int numModulesX, const int imgStride, const int numImgColors,
                                       const int numGroups,
                                       const float scaleTargets, const float scaleOutputs,
                                       const bool conv/*, const bool noloads*/) {
    // avoid bank conflict by reorganizing the data structure and improve the band width by using 'float2' instead of 'float'
    __shared__ float2 shFilters[colorCache / 2][B_Y * filtersPerThread]; // pre-load 1 pixel from B_Y*filtersPerThread filters
    __shared__ float2 shImages[colorCache][B_X * imgsPerThread / 2]; // pre-load 1 pixel from B_X*imgsPerThread images
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;
    const int numFilterColors = numImgColors / numGroups;
    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;

    const int numModules = numModulesX * numModulesY;
    const int blockColorIdx = numFilterColors * blockGroupIdx;
    // Another fun insanity: the % B_X makes things faster, even thought threadIdx.x is
    // in the range 0..31. It appears that this allows the compiler to optimize?
    const int tx = threadIdx.x % B_X;
    const int ty = threadIdx.y % B_Y;
    //const int tidx = ty * B_X + threadIdx.x; // reduce one register

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

    // reduce two registers
    //const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    //const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + tx;
    const int imgOffset = (blockColorIdx + ty) * imgPixels * imgStride + myImgIdx;

//    images += (blockColorIdx + threadIdx.y) * imgPixels * imgStride + myImgIdx;
    const int filterOffset = blockFilterIdx
            + ((ty * B_X + tx) / (B_Y * filtersPerThread)) * numFilters * filterPixels + ((ty * B_X + tx) % (B_Y * filtersPerThread)) + (conv ? 0 : moduleIdx * numFilterColors * filterPixels * numFilters);
//    filters +=blockFilterIdx
//            + shFilterLoadY * numFilters * filterPixels + shFilterLoadX;
//    if (!conv) {
//        filters += moduleIdx * numFilterColors * filterPixels * numFilters;
//    }

    targets += moduleIdx * numImages
            + (blockFilterIdx + threadIdx.y * filtersPerThread) * numImages * numModules
            + myImgIdx;

    // combine two registers into one
    const int numModImages = numModules * numImages;
    float prod[imgsPerThread][filtersPerThread];
//    float fCache[filtersPerThread];
    #pragma unroll
    for(int i = 0; i < imgsPerThread; i++) {
        #pragma unroll
        for(int f = 0; f < filtersPerThread; f++) {
            prod[i][f] = 0;
        }
    }
    // NOTE: these max/min functions increase register usage as compared to my macros
    const int imgStartX = max(0, imgLoadModPosX);
    const int imgStartY = max(0, imgLoadModPosY);
    const int imgEndX = min(imgLoadModPosX + filterSize, imgSizeX);
    const int imgEndY = min(imgLoadModPosY + filterSize, imgSizeY);
//    __shared__ int imgPos[]

    int fPidx, iPidx;
    float imPreload[imgsPerThread]; // [4]
    float fPreload[colorCache*filtersPerThread/B_X]; // [2]
//    float fCache[filtersPerThread];

    filterActs_YxX_sparse2_preload_ty_4_tx_32_f_16_c_4_setPixelCoords(filterSize, imgSizeX, imgLoadModPosY, imgLoadModPosX, imgStartY, imgStartX, fPidx, iPidx);

    // remove redundant conditions
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        imPreload[i]  = tex1Dfetch<float>(images, imgOffset + imgStride * iPidx + i * B_X);
    }

    #pragma unroll
    for (int c = 0; c < colorCache; c += B_X/filtersPerThread) {
        fPreload[c*filtersPerThread/B_X] = tex1Dfetch<float>(filters, filterOffset + (c * filterPixels + fPidx) * numFilters);
    }
    for (int imgY = imgStartY; imgY < imgEndY; ++imgY) {
//        const int filterPxY = imgY - imgLoadModPosY;
        for (int imgX = imgStartX; imgX < imgEndX; ++imgX) {
//            const int filterPxX = imgX - imgLoadModPosX;
//            const int p = filterPxY * filterSize + filterPxX;
//            const int pixIdx = imgY * imgSizeX + imgX;// Pixel index in img
//            setPixelCoords(filterSize, imgSizeX, imgLoadModPosY, imgLoadModPosX, imgY, imgX, &p, &pixIdx);
//            float* m = &images[imgStride * pixIdx];
            const bool lastPixel = imgY == imgEndY - 1 && imgX == imgEndX - 1;
            int imgYNext = imgY;
            int imgXNext = imgX;
            int fPidxNext, iPidxNext;
            if (!lastPixel) {
                imgYNext = imgY + (imgX + 1 == imgEndX);
                imgXNext = imgX + 1 == imgEndX ? imgStartX : imgX + 1;
            }
            filterActs_YxX_sparse2_preload_ty_4_tx_32_f_16_c_4_setPixelCoords(filterSize, imgSizeX, imgLoadModPosY, imgLoadModPosX, imgYNext, imgXNext, fPidxNext, iPidxNext);
            for (int oc = 0; oc < numFilterColors; oc += colorCache) { // oc stands for outer color (loop)
                // store the preloaded pixel of filter and image into shared memory
                shFilters[(ty * B_X + tx) / (B_Y * filtersPerThread)][(ty * B_X + tx) % (B_Y * filtersPerThread)].x = fPreload[0];
                shFilters[(ty * B_X + tx) / (B_Y * filtersPerThread)][(ty * B_X + tx) % (B_Y * filtersPerThread)].y = fPreload[1];
                shImages[ty][tx].x = imPreload[0];
                shImages[ty][tx].y = imPreload[1];                
                shImages[ty][tx+B_X].x = imPreload[2]; 
                shImages[ty][tx+B_X].y = imPreload[3];

                int imgOffset2 = imgOffset + imgStride * ((oc + colorCache) * imgPixels + iPidx);
                int filterOffset2 = filterOffset + numFilters * ((oc + colorCache) * filterPixels + fPidx);
                if (oc == numFilterColors - colorCache) {
                    filterOffset2 = filterOffset + fPidxNext * numFilters;
                    imgOffset2 = imgOffset + iPidxNext * imgStride;
                    fPidx = fPidxNext;
                    iPidx = iPidxNext;
                }

                // preload one pixel of filter and image from texture, and no need to check 'checkImgBounds' with all callers setting it as false
                imPreload[0] = tex1Dfetch<float>(images, imgOffset2);
                imPreload[1] = tex1Dfetch<float>(images, imgOffset2 + B_X);
                imPreload[2] = tex1Dfetch<float>(images, imgOffset2 + 2 * B_X);
                imPreload[3] = tex1Dfetch<float>(images, imgOffset2 + 3 * B_X);
                fPreload[0] = tex1Dfetch<float>(filters, filterOffset2);
                fPreload[1] = tex1Dfetch<float>(filters, filterOffset2 + 2 * filterPixels * numFilters);

                __syncthreads();

                // put together the instructions with same type to improve instruction-level parallelism 
                // calculate the convolution between images and filters
                #pragma unroll 
                for (int f = 0; f < filtersPerThread; f++) {
                    #pragma unroll
                    for (int r = 0; r < colorCache / 2; r++) {
                        prod[0][f] += shImages[r][tx].x      * shFilters[r][ty*filtersPerThread+f].x;
                        prod[1][f] += shImages[r][tx].y      * shFilters[r][ty*filtersPerThread+f].x;
                        prod[2][f] += shImages[r][tx+B_X].x   * shFilters[r][ty*filtersPerThread+f].x;
                        prod[3][f] += shImages[r][tx+B_X].y   * shFilters[r][ty*filtersPerThread+f].x;
                        prod[0][f] += shImages[r+2][tx].x    * shFilters[r][ty*filtersPerThread+f].y;
                        prod[1][f] += shImages[r+2][tx].y    * shFilters[r][ty*filtersPerThread+f].y;
                        prod[2][f] += shImages[r+2][tx+B_X].x * shFilters[r][ty*filtersPerThread+f].y;
                        prod[3][f] += shImages[r+2][tx+B_X].y * shFilters[r][ty*filtersPerThread+f].y;
                    }
                }
                __syncthreads();
            }
        }
    }

    if (scale) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                // remove the redundant condition for less registers
                targets[i * B_X + f * numModImages] = scaleTargets * targets[i * B_X + f * numModImages] + scaleOutputs * prod[i][f];
            }
        }
    } else {
        // Note: reversing order of these loops saves 2 registers, but costs time
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                // remove the redundant condition for less registers
                targets[i * B_X + f * numModImages] = scaleOutputs * prod[i][f];
            }
        }
    }
}

/*
 * Block size B_YxB_X. Each block applies B_Y * filtersPerThread filters to B_X * imgsPerThread images.
 * threadIdx.x determines image
 * threadIdx.y determines filter
 *
 * blockIdx.x determines image batch of B_X * imgsPerThread
 * blockIdx.y determines filter batch of module and B_Y * filtersPerThread
 *
 * images:      (numColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numColors, filterPixels, numFilters) if conv
 *              (numModules, numColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 *
 * Number of filters per module should be divisible by B_Y * filtersPerThread
 * checkImgBounds indicates whether number of images is divisible by B_X * imgsPerThread
 *
 * The imgSize here is the size of the actual image without the padding.
 *
 */
 template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int numColors, int pixelCache,
          bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_color(float* images, float* filters, float* targets,
                                   const int numImages, const int numFilters,
                                   const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                   const int moduleStride,
                                   const int numModulesY, const int numModulesX, const int imgStride,
                                   const float scaleTargets, const float scaleOutputs,
                                   const bool conv) {
    __shared__ float shFilters[pixelCache*numColors][B_Y * filtersPerThread]; // pre-load pixelCache pixels from B_Y*filtersPerThread filters
    __shared__ float shImages[pixelCache*numColors][B_X * imgsPerThread]; // pre-load pixelCache pixels from B_X*imgsPerThread images
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;

    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = blockIdx.y % blocksPerModule;

    const int tidx = threadIdx.y * B_X + threadIdx.x;

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;
    const int numModules = numModulesY * numModulesX;
    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;
    images += myImgIdx;
    filters += filtersPerThread * B_Y * blockFilterIdx
             + shFilterLoadY * numFilters + shFilterLoadX;
    if (!conv) {
        filters += moduleIdx * numColors * filterPixels * numFilters;
    }

    targets += moduleIdx * numImages
            + (blockFilterIdx * B_Y * filtersPerThread + threadIdx.y*filtersPerThread) * numImages * numModulesY * numModulesX
            + myImgIdx;


    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for(int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for(int g = 0; g < imgsPerThread; g++) {
            prod[f][g] = 0;
        }
    }
    //float* shImgLoad = &shImages[0][threadIdx.x];
    for (int p = 0; p < filterPixels; p += pixelCache) {
        /*
         * Load pixelCache pixels from B_Y*filtersPerThread filters
         * This condition covers the case when B_X is not divisible by filtersPerThread.
         * In this case, not all of the threads will participate in the loading operation.
         * This ensures that in each loop iteration, an integer number of rows of shFilters
         * are filled, which makes indexing simple.
         */
        if (B_X % filtersPerThread == 0 || shFilterLoadY < B_X/filtersPerThread) {
            #pragma unroll
            for (int p2 = 0; p2 < pixelCache; p2 += B_X/filtersPerThread) {
                const bool omit = pixelCache % (B_X / filtersPerThread) == 0;
                const int preloadPx = shFilterLoadY + p2;
                if (omit || preloadPx < pixelCache) {
                    if (p + preloadPx < filterPixels) {
                        #pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shFilters[shFilterLoadY + p2 + c * pixelCache][shFilterLoadX] = filters[(c * filterPixels + p + p2) * numFilters];
                        }
                    } else {
                        #pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shFilters[shFilterLoadY + p2 + c * pixelCache][shFilterLoadX] = 0;
                        }
                    }
                }
            }
        }

        /*
         * Load pixelCache pixels from B_X*imgsPerThread images.
         */
        #pragma unroll
        for (int ly = 0; ly < pixelCache; ly += B_Y) {
            const int preloadPx = ly + threadIdx.y;
            const int pixIdx = p + preloadPx;
            const bool omit = pixelCache % B_Y == 0; // Compile-time condition
            /*
             * Don't load any image pixels corresponding to filter pixels that don't exist.
             */
            if (pixIdx < filterPixels && (omit || preloadPx < pixelCache)) {
                const int x = imgLoadModPosX + pixIdx % filterSize;
                const int y = imgLoadModPosY + pixIdx / filterSize;

                if (y >= 0 && y < imgSizeY && x >= 0 && x < imgSizeX) {
                    float* m = &images[imgStride * (y * imgSizeX + x)];

                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        #pragma unroll
                        for (int i = 0; i < imgsPerThread; i++) {
                            if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                                shImages[preloadPx + c * pixelCache][threadIdx.x * imgsPerThread + i] = m[c * imgStride * imgPixels + i * B_X];
                            } else {
                                shImages[preloadPx + c * pixelCache][threadIdx.x * imgsPerThread + i] = 0;
                            }
                        }
                    }
                } else { // Padding
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        #pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shImages[preloadPx + c * pixelCache][threadIdx.x * imgsPerThread + i] = 0;
                        }
                    }
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < pixelCache*numColors; i++) {
            #pragma unroll
            for(int f = 0; f < filtersPerThread; f++) {
                #pragma unroll
                for(int g = 0; g < imgsPerThread; g++) {
                    prod[f][g] += shImages[i][g + threadIdx.x * imgsPerThread] * shFilters[i][threadIdx.y * filtersPerThread + f];
                }
            }
        }
        __syncthreads();
    }

    if (scale) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
            #pragma unroll
            for (int g = 0; g < imgsPerThread; g++) {
                if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                    targets[g * B_X + f * numImages * numModules] = scaleTargets * targets[g * B_X + f * numImages * numModules] + scaleOutputs * prod[f][g];
                }
            }
        }
    } else {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[g * B_X + f * numImages * numModules] = scaleOutputs * prod[f][g];
                }
            }
        }
    }
}

/*
 * Block size B_YxB_X. Each block applies B_Y * filtersPerThread filters to B_X * imgsPerThread images.
 * threadIdx.x determines image
 * threadIdx.y determines filter
 *
 * blockIdx.x determines image batch of B_X * imgsPerThread
 * blockIdx.y determines filter batch of B_Y * filtersPerThread
 *
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * B_Y one of 4, 8, 16
 * B_X one of 16, 32
 * imgsPerThread one of 1, 2, 4
 * filtersPerThread one of 1, 2, 4, 8
 * colorCache: how many colors to put into shmem
 *
 * numFilters should be divisible by B_Y * filtersPerThread
 * numImages be divisible by B_X * imgsPerThread
 * numFilterColors should be divisible by colorCache.
 * numImgColors must be even.
 * numFilters must be divisible by numGroups.
 * no restrictions on pixelCache
 * The imgSize here is the size of the actual image without the padding.
 * As always, try to make B_X * imgsPerThread == B_Y * filtersPerThread for maximum efficiency.
 *
 */
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int colorCache,
          bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_sparse2(float* images, float* filters, float* targets,
                                       const int numImages, const int numFilters,
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesY, const int numModulesX, const int imgStride, const int numImgColors,
                                       const int numGroups, 
                                       const float scaleTargets, const float scaleOutputs,
                                       const bool conv) {
    __shared__ float shFilters[colorCache][B_Y * filtersPerThread]; // pre-load 1 pixel from B_Y*filtersPerThread filters
    __shared__ float shImages[colorCache][B_X * imgsPerThread]; // pre-load 1 pixel from B_X*imgsPerThread images
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;
    const int numFilterColors = numImgColors / numGroups;
    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;

    const int numModules = numModulesX * numModulesY;
    const int blockColorIdx = numFilterColors * blockGroupIdx;

    const int tidx = threadIdx.y * B_X + threadIdx.x;

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;

    images += (blockColorIdx + threadIdx.y) * imgPixels * imgStride + myImgIdx;
    filters +=blockFilterIdx
            + shFilterLoadY * numFilters * filterPixels + shFilterLoadX;
    if (!conv) {
        filters += moduleIdx * numFilterColors * filterPixels * numFilters;
    }

    targets += moduleIdx * numImages
            + (blockFilterIdx + threadIdx.y) * numImages * numModules
            + myImgIdx;

    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for(int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for(int g = 0; g < imgsPerThread; g++) {
            prod[f][g] = 0;
        }
    }
    const int imgStartX = MAX(0, imgLoadModPosX);
    const int imgStartY = MAX(0, imgLoadModPosY);
    const int imgEndX = MIN(imgLoadModPosX + filterSize, imgSizeX);
    const int imgEndY = MIN(imgLoadModPosY + filterSize, imgSizeY);
//    __shared__ int imgPos[]

    for (int imgY = imgStartY; imgY < imgEndY; ++imgY) {
        const int filterPxY = imgY - imgLoadModPosY;
        for (int imgX = imgStartX; imgX < imgEndX; ++imgX) {
            const int filterPxX = imgX - imgLoadModPosX;
            const int p = filterPxY * filterSize + filterPxX;
            for (int oc = 0; oc < numFilterColors; oc += colorCache) { // oc stands for outer color (loop)

                /*
                 * Load a pixel from B_Y*filtersPerThread filters
                 * This condition covers the case when B_X is not divisible by filtersPerThread.
                 * In this case, not all of the threads will participate in the loading operation.
                 * This ensures that in each loop iteration, an integer number of rows of shFilters
                 * are filled, which makes indexing simple.

                 * nvcc is behaving in a completely insane way: removing this condition under
                 * template parameters that guarantee it to be true actually slows down
                 * the computation.
                 *
                 */
                if (/*B_X % filtersPerThread == 0 ||*/ shFilterLoadY < B_X/filtersPerThread) {
                    #pragma unroll
                    for (int c = 0; c < colorCache; c += B_X/filtersPerThread) {
                        if (colorCache % (B_X/filtersPerThread) == 0 || c + shFilterLoadY < colorCache) {
                            shFilters[c + shFilterLoadY][shFilterLoadX] = filters[((oc+c) * filterPixels + p) * numFilters];
                        }
                    }
                }

                /*
                 * Load a pixel from B_X*imgsPerThread images.
                 */
                const int pixIdx = imgY * imgSizeX + imgX;// Pixel index in img

                float* m = &images[imgStride * (oc * imgPixels + pixIdx)];
                #pragma unroll
                for (int c = 0; c < colorCache; c += B_Y) {
                    if (colorCache % B_Y == 0 || threadIdx.y + c < colorCache) {
                        #pragma unroll
                        for (int i = 0; i < imgsPerThread; i++) {
                            if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                                shImages[c + threadIdx.y][threadIdx.x + i * B_X] = m[c * imgStride * imgPixels + i * B_X];
                            } else {
                                shImages[c + threadIdx.y][threadIdx.x + i * B_X] = 0;
                            }
                        }
                    }
                }

                __syncthreads();

                for (int c = 0; c < colorCache; c++) {
                    #pragma unroll
                    for(int g = 0; g < imgsPerThread; g++) {
                        #pragma unroll
                        for(int f = 0; f < filtersPerThread; f++) {
                            prod[f][g] += shImages[c][g * B_X + threadIdx.x] * shFilters[c][threadIdx.y + f * B_Y];
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    if (scale) {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[g * B_X + f * B_Y * numImages * numModules] = scaleTargets * targets[g * B_X + f * B_Y * numImages * numModules] + scaleOutputs * prod[f][g];
                }
            }
        }
    } else {
        // Note: reversing order of these loops saves 2 registers, but costs time
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
            #pragma unroll
            for (int g = 0; g < imgsPerThread; g++) {
                if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                    targets[g * B_X + f * B_Y * numImages * numModules] = scaleOutputs * prod[f][g];
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
 * Note:   This function can be used when each thread loads even number of filter *
 *         pixels(filtersPerThread * colorCache / B_X is even), and this can be   *
 *         optimized more when the number of loaded image's pixel is even.        *
 *********************************************************************************/
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int colorCache,
          bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_sparse2_f_vec(float* images, float* filters, float* targets,
                                       const int numImages, const int numFilters,
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesY, const int numModulesX, const int imgStride, const int numImgColors,
                                       const int numGroups, 
                                       const float scaleTargets, const float scaleOutputs,
                                       const bool conv) {
    // improve shared memory's band width by using 'float2' instead of 'float'
    __shared__ float2 shFilters[colorCache/2][B_Y * filtersPerThread]; // pre-load 1 pixel from B_Y*filtersPerThread filters
    __shared__ float shImages[colorCache][B_X * imgsPerThread]; // pre-load 1 pixel from B_X*imgsPerThread images

    const int tx = threadIdx.x % B_X, ty = threadIdx.y % B_Y;
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;
    const int numFilterColors = numImgColors / numGroups;
    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;

    const int numModules = numModulesX * numModulesY;
    const int blockColorIdx = numFilterColors * blockGroupIdx;

    const int tidx = ty * B_X + tx;

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

    // load position of filters' pixels for current thread
    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    // load position of images' pixels for current thread
    const int shImgLoadY = tidx / (B_X * imgsPerThread);
    const int shImgLoadX = tidx % (B_X * imgsPerThread);

    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + shImgLoadX;
    images += (blockColorIdx + shImgLoadY) * imgPixels * imgStride + myImgIdx;

    filters +=blockFilterIdx
            + shFilterLoadY * numFilters * filterPixels + shFilterLoadX;
    if (!conv) {
        filters += moduleIdx * numFilterColors * filterPixels * numFilters;
    }

    targets += moduleIdx * numImages
            + (blockFilterIdx + ty) * numImages * numModules
            + blockIdx.x * B_X * imgsPerThread + tx;

    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for(int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for(int g = 0; g < imgsPerThread; g++) {
            prod[f][g] = 0;
        }
    }

    const int imgStartX = MAX(0, imgLoadModPosX);
    const int imgStartY = MAX(0, imgLoadModPosY);
    const int imgEndX = MIN(imgLoadModPosX + filterSize, imgSizeX);
    const int imgEndY = MIN(imgLoadModPosY + filterSize, imgSizeY);

    // temporary buffer to store the filter's loaded pixels during each loop
    float fPreload[colorCache * filtersPerThread / B_X];
    // temporary buffer to store the image's loaded pixels during each loop
    float iPreload[colorCache * imgsPerThread / B_Y];

    // preload filter's pixels
    #pragma unroll
    for (int c = 0; c < colorCache; c += B_X/filtersPerThread) {
        fPreload[c * filtersPerThread / B_X] = filters[(c * filterPixels + (imgStartY - imgLoadModPosY) * filterSize + (imgStartX - imgLoadModPosX)) * numFilters];
    }

    // preload image's pixels
    if (!checkImgBounds || myImgIdx < numImages) {
        #pragma unroll
        for (int c = 0; c < colorCache; c += B_Y / imgsPerThread) {
            iPreload[c * imgsPerThread / B_Y] = images[(c * imgPixels + imgStartY * imgSizeX + imgStartX) * imgStride];
        }
    } else {
        #pragma unroll
        for (int c = 0; c < colorCache; c += B_Y / imgsPerThread) {
            iPreload[c * imgsPerThread / B_Y] = 0;
        }
    }

    for (int imgY = imgStartY; imgY < imgEndY; ++imgY) {
        //const int filterPxY = imgY - imgLoadModPosY;
        for (int imgX = imgStartX; imgX < imgEndX; ++imgX) {
            for (int oc = 0; oc < numFilterColors; oc += colorCache) { // oc stands for outer color (loop)
                // store the preloaded filter's pixels into shared memory
                #pragma unroll
                for (int c = 0; c < colorCache / 2; c += B_X / filtersPerThread) {
                    shFilters[c + shFilterLoadY][shFilterLoadX].x = fPreload[c * filtersPerThread / B_X];
                    shFilters[c + shFilterLoadY][shFilterLoadX].y = fPreload[(c + colorCache / 2) * filtersPerThread / B_X];
                }

                // store the preloaded image's pixels into shared memory
                #pragma unroll
                for (int c = 0; c < colorCache; c += B_Y / imgsPerThread) {
                    shImages[c + shImgLoadY][shImgLoadX] = iPreload[c * imgsPerThread / B_Y];
                }
                /*
                 * Load a pixel from B_Y*filtersPerThread filters
                 * This condition covers the case when B_X is not divisible by filtersPerThread.
                 * In this case, not all of the threads will participate in the loading operation.
                 * This ensures that in each loop iteration, an integer number of rows of shFilters
                 * are filled, which makes indexing simple.

                 * nvcc is behaving in a completely insane way: removing this condition under
                 * template parameters that guarantee it to be true actually slows down
                 * the computation.
                 *
                 */

                /* preload image and filter pixels' data */
                if ((oc + colorCache) == numFilterColors) { // move to next pixel when all colors of current pixel have been finished
                   int imgXn = (imgX < (imgEndX - 1)) ? (imgX + 1) : imgStartX; 
                   int imgYn = imgY + (imgXn != (imgX + 1));

                    #pragma unroll
                    for (int c = 0; c < colorCache; c += B_X/filtersPerThread) {
                        fPreload[c * filtersPerThread / B_X] = filters[(c * filterPixels + (imgYn - imgLoadModPosY) * filterSize + (imgXn - imgLoadModPosX)) * numFilters];
                    }

                    if (!checkImgBounds || myImgIdx < numImages) {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c += B_Y / imgsPerThread) {
                            iPreload[c * imgsPerThread / B_Y] = images[(c * imgPixels + imgYn * imgSizeX + imgXn) * imgStride];
                        }
                    } else {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c += B_Y / imgsPerThread) {
                            iPreload[c * imgsPerThread / B_Y] = 0;
                        }
                    }
                } else { // move next colorCache
                    #pragma unroll
                    for (int c = 0; c < colorCache; c += B_X/filtersPerThread) {
                        fPreload[c * filtersPerThread / B_X] = filters[((c + oc + colorCache) * filterPixels + (imgY - imgLoadModPosY) * filterSize + (imgX - imgLoadModPosX)) * numFilters];
                    }

                    if (!checkImgBounds || myImgIdx < numImages) {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c += B_Y / imgsPerThread) {
                            iPreload[c * imgsPerThread / B_Y] = images[((c + oc + colorCache) * imgPixels + imgY * imgSizeX + imgX) * imgStride];
                        }
                    } else {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c += B_Y / imgsPerThread) {
                            iPreload[c * imgsPerThread / B_Y] = 0;
                        }
                    }
                }

                __syncthreads();

                // convolution
                for (int c = 0; c < colorCache / 2; c++) {
                    #pragma unroll
                    for(int g = 0; g < imgsPerThread; g++) {
                        #pragma unroll
                        for(int f = 0; f < filtersPerThread; f++) {
                            prod[f][g] += shImages[c][g * B_X + tx]      * shFilters[c][ty + f * B_Y].x;
                            prod[f][g] += shImages[c + colorCache / 2][g * B_X + tx] * shFilters[c][ty + f * B_Y].y;
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    // write convolution result into global memory
    if (scale) {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[g * B_X + f * B_Y * numImages * numModules] = scaleTargets * targets[g * B_X + f * B_Y * numImages * numModules] + scaleOutputs * prod[f][g];
                }
            }
        }
    } else {
        // Note: reversing order of these loops saves 2 registers, but costs time
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
            #pragma unroll
            for (int g = 0; g < imgsPerThread; g++) {
                if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                    targets[g * B_X + f * B_Y * numImages * numModules] = scaleOutputs * prod[f][g];
                }
            }
        }
    }
}
/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters)             if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModules, numImages)
 *
 * Note: all of these convolution routines are optimized for the case when
 * the number of images (i.e. the minibatch size) is a multiple of 128.
 * Other batch sizes will work, but but I made no attempt whatsoever
 * to make them work fast.
 */
 void _filterActs(caffe2::CUDAContext* context, caffe2::TensorCUDA* images, caffe2::TensorCUDA* filters, caffe2::TensorCUDA* targets,
                  int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                  int numImgColors, int numGroups,
                  float scaleTargets, float scaleOutput, bool conv) {

    CAFFE_ENFORCE(images->ndim() == 2);
    CAFFE_ENFORCE(filters->ndim() == 2);
    CAFFE_ENFORCE(targets->ndim() == 2);

    int numFilterColors = numImgColors / numGroups;
    int numFilters = filters->dim32(1);
    int numModules = numModulesY * numModulesX;
    int numImages = images->dim32(1);
    int imgPixels = images->dim32(0) / numImgColors;
    int imgSizeX = imgPixels / imgSizeY;
    int filterModuleMult = conv ? 1 : numModules;

    CAFFE_ENFORCE(numGroups > 1 || (numImgColors > 0 && (numImgColors <= 3 || numImgColors % 4 == 0)));
    CAFFE_ENFORCE(numGroups == 1 || numFilterColors % 4 == 0);
    CAFFE_ENFORCE(numFilters % (16 * numGroups) == 0);
    CAFFE_ENFORCE(numImgColors % numGroups == 0);
    CAFFE_ENFORCE(images->dim32(0) == imgPixels * numImgColors);
    CAFFE_ENFORCE(imgSizeY * imgSizeX == imgPixels);
    int numFiltersPerGroup = numFilters / numGroups;

    int imgStride = images->dim32(1);

    int filterPixels = filters->dim32(0) / (filterModuleMult * numFilterColors);
    int filterSize = int(sqrt(filterPixels));
    CAFFE_ENFORCE(filterSize * filterSize == filterPixels);
    CAFFE_ENFORCE(filters->dim32(0) == filterModuleMult * numFilterColors * filterPixels);

    // These routines don't handle the case when only part of the image is visited in the convolution
    CAFFE_ENFORCE(paddingStart <= 0);
    CAFFE_ENFORCE(paddingStart + (numModulesX-1)*moduleStride + filterSize >= imgSizeX);
    CAFFE_ENFORCE(paddingStart + (numModulesY-1)*moduleStride + filterSize >= imgSizeY);
    CAFFE_ENFORCE(moduleStride <= filterSize);

    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    int filtersPerThread, threadsY = 4;
    if (numImgColors <= 3) {
        // Special kernels written for colors = 3, filters = 64 and colors = 3, filters = 48 cases.
        // The remaining cases use the old routines.
        // TODO: Modernize the remaining cases if you care about them.
        filtersPerThread = numFiltersPerGroup % 64 == 0 ? 16 : numFiltersPerGroup % 48 == 0 ? 12 : numFiltersPerGroup % 32 == 0 ? 8 : 4;
    } else {
        filtersPerThread = numFiltersPerGroup % 64 == 0 ? 16 : numFiltersPerGroup % 32 == 0 ? 8 : 4;
        threadsY = numFiltersPerGroup % 128 == 0 && numFilterColors % 8 == 0  && imgsPerThread != 4 ?  8 : 4;
    }
    int threadsX = 32;
    dim3 threads(threadsX, threadsY);
    dim3 blocks = dim3(DIVUP(numImages, threads.x * imgsPerThread), (numModules * numFilters) / (threads.y * filtersPerThread));

    bool checkImgBounds = numImages % (threads.x*imgsPerThread) != 0;
    bool scale = scaleTargets != 0;
    if (scaleTargets == 0) {
        targets->Resize(std::vector<int>{numFilters * numModules, numImages});
    } else {
        CAFFE_ENFORCE(targets->dim32(0) == numFilters * numModules);
        CAFFE_ENFORCE(targets->dim32(1) == numImages);
    }

    cudaTextureObject_t tex_images = GetTensorTextureObject(images);
    cudaTextureObject_t tex_filters = GetTensorTextureObject(filters);
    float* images_data = images->mutable_data<float>();
    float* filters_data = filters->mutable_data<float>();
    float* targets_data = targets->mutable_data<float>();
    const std::size_t images_bytes = images->nbytes();

    cudaStream_t stream = context->cuda_stream();

    checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte)); // using wider band width
    
    // Auto-generated calling code...
    // NOTE: The calling code is set up such that if checkImgBounds is true, then imgsPerThread = 1.
    // In principle it doesn't have to be this way, and you may want to optimize for that case.

    if (scale == false) {
        if (checkImgBounds == false) {
            if (numFilterColors % 8 == 0) {
                if (numImages % 128 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        if (images_bytes < TEXTURE_SIZE_MAX) {
                            cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, false, false >, cudaFuncCachePreferL1);
                            filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, false, false > <<<blocks, threads, 0, stream>>>(tex_images, tex_filters, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, false, false >, cudaFuncCachePreferL1);
                            filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                        }
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        if (images_bytes < TEXTURE_SIZE_MAX) {
                            cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, false, false >, cudaFuncCachePreferL1);
                            filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, false, false > <<<blocks, threads, 0, stream>>>(tex_images, tex_filters, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, false, false >, cudaFuncCachePreferL1);
                            filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                        }
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 4, 32, 4, 8, 8, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2_f_vec < 4, 32, 4, 8, 8, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 4, 8, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 4, 4, 8, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 64 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 8, 32, 2, 16, 8, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2_f_vec < 8, 32, 2, 16, 8, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 4, 32, 2, 16, 8, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2_f_vec < 4, 32, 2, 16, 8, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 4, 32, 2, 8, 8, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2_f_vec < 4, 32, 2, 8, 8, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 4, 8, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 4, 8, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 32 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 8, 32, 1, 16, 8, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2_f_vec < 8, 32, 1, 16, 8, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 4, 32, 1, 16, 8, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2_f_vec < 4, 32, 1, 16, 8, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 4, 32, 1, 8, 8, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2_f_vec < 4, 32, 1, 8, 8, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors % 4 == 0) {
                if (numImages % 128 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 8, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 4, 8, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 4, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 4, 4, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 64 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 8, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 8, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 4, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 4, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 32 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 3) {
                if (numImages % 128 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_16_px_4_cc_3_tex < 4, 32, 4, 16, 3, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_16_px_4_cc_3_tex < 4, 32, 4, 16, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(tex_images, tex_filters, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_12_px_4_cc_3_tex < 4, 32, 4, 12, 3, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_12_px_4_cc_3_tex < 4, 32, 4, 12, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(tex_images, tex_filters, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 3, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 8, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 3, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 4, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 64 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 3, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 16, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 3, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 12, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 3, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 8, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 3, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 4, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 32 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 3, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 3, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 3, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 3, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 2) {
                if (numImages % 128 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 16, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 16, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 12, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 12, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 8, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 4, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 64 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 16, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 12, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 8, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 4, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 32 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 1) {
                if (numImages % 128 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 16, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 16, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 12, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 12, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 8, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 4, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 64 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 16, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 12, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 8, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 4, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 32 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
        }
        else if (checkImgBounds == true) {
            if (numFilterColors % 8 == 0) {
                if (numImages % 1 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 8, 32, 1, 16, 8, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 8, 32, 1, 16, 8, false, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 8, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 16, 8, false, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 8, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 8, 8, false, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, false, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors % 4 == 0) {
                if (numImages % 1 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, false, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, false, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 3) {
                if (numImages % 1 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 3, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 3, 4, false, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 3, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 3, 4, false, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 3, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 3, 4, false, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 3, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 3, 4, false, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 2) {
                if (numImages % 1 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 2, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 2, 4, false, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 2, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 2, 4, false, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 2, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 2, 4, false, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 2, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 2, 4, false, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 1) {
                if (numImages % 1 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 1, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 1, 4, false, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 1, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 1, 4, false, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 1, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 1, 4, false, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 1, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 1, 4, false, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
        }
    }
    else if (scale == true) {
        if (checkImgBounds == false) {
            if (numFilterColors % 8 == 0) {
                if (numImages % 128 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        if (images_bytes < TEXTURE_SIZE_MAX) {
                            cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, true, false >, cudaFuncCachePreferL1);
                            filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, true, false > <<<blocks, threads, 0, stream>>>(tex_images, tex_filters, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, true, false >, cudaFuncCachePreferL1);
                            filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                        }
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        if (images_bytes < TEXTURE_SIZE_MAX) {
                            cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, true, false >, cudaFuncCachePreferL1);
                            filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, true, false > <<<blocks, threads, 0, stream>>>(tex_images, tex_filters, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, true, false >, cudaFuncCachePreferL1);
                            filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                        }
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 4, 32, 4, 8, 8, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2_f_vec < 4, 32, 4, 8, 8, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 4, 8, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 4, 4, 8, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 64 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 8, 32, 2, 16, 8, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2_f_vec < 8, 32, 2, 16, 8, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 4, 32, 2, 16, 8, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2_f_vec < 4, 32, 2, 16, 8, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 4, 32, 2, 8, 8, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2_f_vec < 4, 32, 2, 8, 8, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 4, 8, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 4, 8, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 32 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 8, 32, 1, 16, 8, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2_f_vec < 8, 32, 1, 16, 8, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 4, 32, 1, 16, 8, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2_f_vec < 4, 32, 1, 16, 8, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 4, 32, 1, 8, 8, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2_f_vec < 4, 32, 1, 8, 8, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors % 4 == 0) {
                if (numImages % 128 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 8, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 4, 8, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 4, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 4, 4, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 64 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 8, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 8, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 4, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 4, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 32 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 3) {
                if (numImages % 128 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_16_px_4_cc_3_tex < 4, 32, 4, 16, 3, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_16_px_4_cc_3_tex < 4, 32, 4, 16, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(tex_images, tex_filters, targets_data,numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_12_px_4_cc_3_tex < 4, 32, 4, 12, 3, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_12_px_4_cc_3_tex < 4, 32, 4, 12, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(tex_images, tex_filters, targets_data,numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 3, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 8, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 3, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 4, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 64 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 3, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 16, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 3, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 12, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 3, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 8, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 3, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 4, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 32 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 3, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 3, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 3, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 3, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 2) {
                if (numImages % 128 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 16, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 16, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 12, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 12, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 8, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 4, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 64 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 16, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 12, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 8, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 4, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 32 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 1) {
                if (numImages % 128 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 16, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 16, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 12, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 12, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 8, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 4, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 64 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 16, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 12, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 8, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 4, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 32 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
        }
        else if (checkImgBounds == true) {
            if (numFilterColors % 8 == 0) {
                if (numImages % 1 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 8, 32, 1, 16, 8, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 8, 32, 1, 16, 8, true, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 8, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 16, 8, true, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 8, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 8, 8, true, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, true, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors % 4 == 0) {
                if (numImages % 1 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, true, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, true, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 3) {
                if (numImages % 1 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 3, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 3, 4, true, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 3, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 3, 4, true, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 3, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 3, 4, true, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 3, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 3, 4, true, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 2) {
                if (numImages % 1 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 2, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 2, 4, true, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 2, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 2, 4, true, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 2, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 2, 4, true, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 2, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 2, 4, true, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 1) {
                if (numImages % 1 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 1, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 1, 4, true, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 1, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 1, 4, true, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 1, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 1, 4, true, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 1, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 1, 4, true, true > <<<blocks, threads, 0, stream>>>(images_data, filters_data, targets_data, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
        }
    }

    checkCudaErrors(cudaDestroyTextureObject(tex_images));
    checkCudaErrors(cudaDestroyTextureObject(tex_filters));
    checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));

    getLastCudaError("filterActs: kernel execution failed");
}

void convFilterActs(caffe2::CUDAContext* context, caffe2::TensorCUDA* images, caffe2::TensorCUDA* filters, caffe2::TensorCUDA* targets,
                    int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                    int numImgColors, int numGroups) {
    convFilterActs(context, images, filters, targets, imgSizeY, numModulesY, numModulesX, paddingStart, moduleStride, numImgColors, numGroups, 0, 1);
}

void convFilterActs(caffe2::CUDAContext* context, caffe2::TensorCUDA* images, caffe2::TensorCUDA* filters, caffe2::TensorCUDA* targets,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors, int numGroups,
                   float scaleTargets, float scaleOutput) {
     _filterActs(context, images, filters, targets, imgSizeY, numModulesY, numModulesX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, true);
}

void localFilterActs(caffe2::CUDAContext* context, caffe2::TensorCUDA* images, caffe2::TensorCUDA* filters, caffe2::TensorCUDA* targets,
                     int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                     int numImgColors, int numGroups) {
    localFilterActs(context, images, filters, targets, imgSizeY, numModulesY, numModulesX, paddingStart, moduleStride, numImgColors, numGroups, 0, 1);
}

void localFilterActs(caffe2::CUDAContext* context, caffe2::TensorCUDA* images, caffe2::TensorCUDA* filters, caffe2::TensorCUDA* targets,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors, int numGroups,
                   float scaleTargets, float scaleOutput) {
     _filterActs(context, images, filters, targets, imgSizeY, numModulesY, numModulesX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, false);
}

