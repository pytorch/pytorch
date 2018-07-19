
#include "IOSGLTexture.h"
#include "../core/DataTransfer.h"

IOSGLTexture::IOSGLTexture(const Type& type,
                           CVOpenGLESTextureCacheRef textureCache,
                           CVPixelBufferRef _sourceImage,
                           GLint filter,
                           GLint wrap)
    : GLTexture(type,
                CVPixelBufferGetWidth(_sourceImage),
                CVPixelBufferGetHeight(_sourceImage),
                CVPixelBufferGetBytesPerRow(_sourceImage) / (type.channels() * type.dataSize()),
                false,
                filter,
                wrap),
      sourceImage(_sourceImage) {
  CVReturn err = CVOpenGLESTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
                                                              textureCache,
                                                              _sourceImage,
                                                              NULL,
                                                              GL_TEXTURE_2D,
                                                              _type.internalFormat,
                                                              _width,
                                                              _height,
                                                              _type.format,
                                                              _type.type,
                                                              0,
                                                              &textureRef);

  if (!textureRef || err) {
    gl_log(GL_ERR,
           "something went wrong, sourceImage: %p, width: %d, height: %d, filter: %d, wrap: %d\n",
           _sourceImage,
           _width,
           _height,
           filter,
           wrap);
  }
  _textureId = name();
  gl_log(
      GL_VERBOSE,
      "IOSGLTexture() - allocated textureId %d, internalFormat: 0x%X, format: 0x%X, type: 0x%X\n",
      _textureId,
      _type.internalFormat,
      _type.format,
      _type.type);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, _textureId);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);

#if GL_EXT_texture_border_clamp
  GLfloat borderColor[] = {0.0f, 0.0f, 0.0f, 0.0f};
  glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR_EXT, borderColor);
  // Set the texture to use the border clamp wrapping mode.
  wrap = GL_CLAMP_TO_BORDER_EXT;
#endif

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);

  glBindTexture(GL_TEXTURE_2D, 0);
}

CVPixelBufferRef IOSGLTexture::createCVPixelBuffer(OSType pixelFormat,
                                                   int32_t width,
                                                   int32_t height) {
  NSDictionary* pixelBufferAttributes = @{
    (id)kCVPixelBufferPixelFormatTypeKey : @(pixelFormat),
    (id)kCVPixelFormatOpenGLESCompatibility : @YES,
    (id)kCVPixelBufferIOSurfacePropertiesKey : @{/*empty dictionary*/}
  };

  CVPixelBufferRef buffer = NULL;
  CVPixelBufferCreate(kCFAllocatorDefault,
                      width,
                      height,
                      pixelFormat,
                      (__bridge CFDictionaryRef)(pixelBufferAttributes),
                      &buffer);
  return buffer;
}

void IOSGLTexture::map_buffer(std::function<void(void* buffer,
                                                 size_t width,
                                                 size_t height,
                                                 size_t stride,
                                                 size_t channels,
                                                 const Type& type)> process) const {
  if (CVPixelBufferLockBaseAddress(sourceImage, 0) == kCVReturnSuccess) {
    void* buffer = CVPixelBufferGetBaseAddress(sourceImage);
    int buffer_stride = CVPixelBufferGetBytesPerRow(sourceImage) / (_channels * _type.dataSize());
    process(buffer, _width, _height, buffer_stride, _channels, _type);

    CVPixelBufferUnlockBaseAddress(sourceImage, 0);
  }
}

void IOSGLTexture::map_load(std::function<void(void* buffer,
                                               size_t width,
                                               size_t height,
                                               size_t stride,
                                               size_t channels,
                                               const Type& type)> process) const {
  map_buffer(process);
}

void IOSGLTexture::map_read(std::function<void(const void* buffer,
                                               size_t width,
                                               size_t height,
                                               size_t stride,
                                               size_t channels,
                                               const Type& type)> process) const {
  // TODO: why is glFlush() only necessary when running tests
  glFlush();

  map_buffer(process);
}
