
#include "IOSGLImageAllocator.h"

#include "../core/GLImage.h"
#include "../core/GLImageAllocator.h"
#include "../core/GLPlainTexture.h"

#include "IOSGLContext.h"
#include "IOSGLTexture.h"

#include "../core/arm_neon_support.h"

template <class T>
GLImageVector<T>* IOSGLImageAllocator<T>::newImage(int num_images,
                                                   int width,
                                                   int height,
                                                   int channels,
                                                   int tile_x,
                                                   int tile_y,
                                                   bool useCVPixelBuffer) {
  GLImageVector<T>* output_images =
      new GLImageVector<T>(num_images, width, height, channels, tile_x, tile_y);
  if (useCVPixelBuffer) {
    IOSGLContext* gl_context = (IOSGLContext*)GLContext::getGLContext();
    for (int i = 0; i < num_images; i++) {
      GLImage<T>* output_image = new GLImage<T>(
          width, height, channels, tile_x, tile_y, [&](int slice) -> const GLTexture* {
            gl_log(GL_VERBOSE,
                   "%s pixelbuffers.size(): %ld\n",
                   __PRETTY_FUNCTION__,
                   pixelbuffers.size());

            CVPixelBufferRef buffer = NULL;
            int slices = (channels + 3) / 4;
            int slice_index = i * slices + slice;
            if (pixelbuffers.size() < slice_index + 1) {
              const int texture_width = width * tile_x;
              const int texture_height = height * tile_y;
              buffer =
                  IOSGLTexture::createCVPixelBuffer(pixelFormat, texture_width, texture_height);
              gl_log(GL_VERBOSE,
                     "created a new buffer %p for image %d slice %d of dimensions %dx%d\n",
                     buffer,
                     i,
                     slice,
                     texture_width,
                     texture_height);
              pixelbuffers.push_back(buffer);
            } else {
              buffer = pixelbuffers[slice_index];

              gl_log(GL_VERBOSE, "reused buffer %p for image %d slice %d\n", buffer, i, slice);
            }

            return gl_context->createNewTexture(buffer, GLImageAllocator<T>::type);
          });
      output_images->push_back(output_image);
    }
  } else {
    for (int i = 0; i < num_images; i++) {
      GLImage<T>* image = new GLImage<T>(
          width, height, channels, tile_x, tile_y, [&](int slice) -> const GLTexture* {
            return new GLPlainTexture(
                GLImageAllocator<T>::type, nullptr, width * tile_x, height * tile_y);
          });
      output_images->push_back(image);
    }
  }
  return output_images;
}

template <>
const FourCharCode IOSGLImageAllocator<float16_t>::pixelFormat = kCVPixelFormatType_64RGBAHalf;
template <>
const FourCharCode IOSGLImageAllocator<uint8_t>::pixelFormat = kCVPixelFormatType_32BGRA;

template class IOSGLImageAllocator<float16_t>;
template class IOSGLImageAllocator<uint8_t>;
