
#pragma once

#include "GLImage.h"
#include "GLPlainTexture.h"

template <class T>
class GLImageAllocator {
 public:
  static const GLTexture::Type& type;

  GLImageAllocator() { gl_log(GL_VERBOSE, "%s\n", __PRETTY_FUNCTION__); }

  virtual ~GLImageAllocator() { gl_log(GL_VERBOSE, "%s\n", __PRETTY_FUNCTION__); }

  virtual GLImageVector<T>* newImage(
      int num_images, int width, int height, int channels, int tile_x, int tile_y, bool is_output);

  virtual GLImageVector<T>* newImage(
      int num_images,
      int width,
      int height,
      int channels,
      int tile_x,
      int tile_y,
      std::function<const GLTexture*(const int width, const int height)> textureAllocator);

  virtual GLImageVector<T>* ShareTexture(const GLuint textureID,
                                         int num_images,
                                         int width,
                                         int height,
                                         int channels,
                                         int tile_x = 1,
                                         int tile_y = 1);

  static GLImageAllocator<T>* newGLImageAllocator();
};
