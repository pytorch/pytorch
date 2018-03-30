
#include "IOSGLContext.h"
#include "IOSGLTexture.h"
#import <sstream>

#import <OpenGLES/EAGL.h>

IOSGLContext::IOSGLContext() {
  auto const currentContext = [EAGLContext currentContext];
  oldContext = (void*)CFBridgingRetain(currentContext);

  if (currentContext != nil && [currentContext API] == kEAGLRenderingAPIOpenGLES3) {
    oglContext = (void*)CFBridgingRetain(currentContext);

    gl_log(GL_LOG, "Reusing current context %p\n", oglContext);
  } else {
    oglContext =
        (void*)CFBridgingRetain([[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES3]);

    gl_log(GL_LOG, "Created a new context %p\n", oglContext);
  }

  if (!oglContext) {
    throw std::runtime_error("Problem with OpenGL context");
  }

  set_context();
  textureCache = NULL;
  CVReturn err = CVOpenGLESTextureCacheCreate(
      kCFAllocatorDefault, NULL, (__bridge EAGLContext*)oglContext, NULL, &textureCache);

  if (err) {
    std::stringstream errmsg;
    errmsg << "Error at CVOpenGLESTextureCacheCreate " << err;
    throw std::runtime_error(errmsg.str());
  }
}

IOSGLContext::~IOSGLContext() {
  gl_log(GL_VERBOSE, "~IOSGLContext()");

  set_context();
  if (textureCache) {
    CFRelease(textureCache);
    textureCache = 0;
  }
  reset_context();

  // Explicitly release only after we `reset_context` since otherwise we are going to read from a
  // dangling pointer.
  if (oglContext) {
    CFBridgingRelease(oglContext);
  }
  if (oldContext) {
    CFBridgingRelease(oldContext);
  }
}

const GLTexture* IOSGLContext::createNewTexture(CVPixelBufferRef pixelBuffer,
                                                const GLTexture::Type& type) {
  return new IOSGLTexture(type, textureCache, pixelBuffer);
}

void IOSGLContext::set_context() {
  auto const currentContext = [EAGLContext currentContext];

  if ((__bridge void*)currentContext != oglContext) {
    if (![EAGLContext setCurrentContext:(__bridge EAGLContext*)oglContext]) {
      throw std::runtime_error("Problem setting OpenGL context");
    }
    GLenum glError = glGetError();
    if (glError != GL_NO_ERROR) {
      gl_log(GL_ERR, "There is an error: 0x%X\n", glError);
    }
    gl_log(GL_VERBOSE, "Set context to %p\n", oglContext);
  }
}

void IOSGLContext::reset_context() {
  EAGLContext* currentContext = [EAGLContext currentContext];

  if ((__bridge void*)currentContext != oldContext) {
    GLenum glError = glGetError();
    if (glError != GL_NO_ERROR) {
      gl_log(GL_ERR, "There is an error before: 0x%X\n", glError);
    }
    if (![EAGLContext setCurrentContext:(__bridge EAGLContext*)oldContext]) {
      throw std::runtime_error("Problem setting OpenGL context");
    }
    glError = glGetError();
    if (glError != GL_NO_ERROR) {
      gl_log(GL_ERR, "There is an error after: 0x%X\n", glError);
    }
    gl_log(GL_VERBOSE, "Reset context to %p\n", oldContext);
  }
}

void IOSGLContext::flush_context() { CVOpenGLESTextureCacheFlush(textureCache, 0); }
