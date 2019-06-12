
// clang-format off

#include <EGL/egl.h>
#include "gl3stub.h"

GLboolean gl3stubInit() {
    #define FIND_PROC(s) s = (void*)eglGetProcAddress(#s)
    FIND_PROC(glReadBuffer);
    FIND_PROC(glDrawRangeElements);
    FIND_PROC(glTexImage3D);
    FIND_PROC(glTexSubImage3D);
    FIND_PROC(glCopyTexSubImage3D);
    FIND_PROC(glCompressedTexImage3D);
    FIND_PROC(glCompressedTexSubImage3D);
    FIND_PROC(glGenQueries);
    FIND_PROC(glDeleteQueries);
    FIND_PROC(glIsQuery);
    FIND_PROC(glBeginQuery);
    FIND_PROC(glEndQuery);
    FIND_PROC(glGetQueryiv);
    FIND_PROC(glGetQueryObjectuiv);
    FIND_PROC(glUnmapBuffer);
    FIND_PROC(glGetBufferPointerv);
    FIND_PROC(glDrawBuffers);
    FIND_PROC(glUniformMatrix2x3fv);
    FIND_PROC(glUniformMatrix3x2fv);
    FIND_PROC(glUniformMatrix2x4fv);
    FIND_PROC(glUniformMatrix4x2fv);
    FIND_PROC(glUniformMatrix3x4fv);
    FIND_PROC(glUniformMatrix4x3fv);
    FIND_PROC(glBlitFramebuffer);
    FIND_PROC(glRenderbufferStorageMultisample);
    FIND_PROC(glFramebufferTextureLayer);
    FIND_PROC(glMapBufferRange);
    FIND_PROC(glFlushMappedBufferRange);
    FIND_PROC(glBindVertexArray);
    FIND_PROC(glDeleteVertexArrays);
    FIND_PROC(glGenVertexArrays);
    FIND_PROC(glIsVertexArray);
    FIND_PROC(glGetIntegeri_v);
    FIND_PROC(glBeginTransformFeedback);
    FIND_PROC(glEndTransformFeedback);
    FIND_PROC(glBindBufferRange);
    FIND_PROC(glBindBufferBase);
    FIND_PROC(glTransformFeedbackVaryings);
    FIND_PROC(glGetTransformFeedbackVarying);
    FIND_PROC(glVertexAttribIPointer);
    FIND_PROC(glGetVertexAttribIiv);
    FIND_PROC(glGetVertexAttribIuiv);
    FIND_PROC(glVertexAttribI4i);
    FIND_PROC(glVertexAttribI4ui);
    FIND_PROC(glVertexAttribI4iv);
    FIND_PROC(glVertexAttribI4uiv);
    FIND_PROC(glGetUniformuiv);
    FIND_PROC(glGetFragDataLocation);
    FIND_PROC(glUniform1ui);
    FIND_PROC(glUniform2ui);
    FIND_PROC(glUniform3ui);
    FIND_PROC(glUniform4ui);
    FIND_PROC(glUniform1uiv);
    FIND_PROC(glUniform2uiv);
    FIND_PROC(glUniform3uiv);
    FIND_PROC(glUniform4uiv);
    FIND_PROC(glClearBufferiv);
    FIND_PROC(glClearBufferuiv);
    FIND_PROC(glClearBufferfv);
    FIND_PROC(glClearBufferfi);
    FIND_PROC(glGetStringi);
    FIND_PROC(glCopyBufferSubData);
    FIND_PROC(glGetUniformIndices);
    FIND_PROC(glGetActiveUniformsiv);
    FIND_PROC(glGetUniformBlockIndex);
    FIND_PROC(glGetActiveUniformBlockiv);
    FIND_PROC(glGetActiveUniformBlockName);
    FIND_PROC(glUniformBlockBinding);
    FIND_PROC(glDrawArraysInstanced);
    FIND_PROC(glDrawElementsInstanced);
    FIND_PROC(glFenceSync);
    FIND_PROC(glIsSync);
    FIND_PROC(glDeleteSync);
    FIND_PROC(glClientWaitSync);
    FIND_PROC(glWaitSync);
    FIND_PROC(glGetInteger64v);
    FIND_PROC(glGetSynciv);
    FIND_PROC(glGetInteger64i_v);
    FIND_PROC(glGetBufferParameteri64v);
    FIND_PROC(glGenSamplers);
    FIND_PROC(glDeleteSamplers);
    FIND_PROC(glIsSampler);
    FIND_PROC(glBindSampler);
    FIND_PROC(glSamplerParameteri);
    FIND_PROC(glSamplerParameteriv);
    FIND_PROC(glSamplerParameterf);
    FIND_PROC(glSamplerParameterfv);
    FIND_PROC(glGetSamplerParameteriv);
    FIND_PROC(glGetSamplerParameterfv);
    FIND_PROC(glVertexAttribDivisor);
    FIND_PROC(glBindTransformFeedback);
    FIND_PROC(glDeleteTransformFeedbacks);
    FIND_PROC(glGenTransformFeedbacks);
    FIND_PROC(glIsTransformFeedback);
    FIND_PROC(glPauseTransformFeedback);
    FIND_PROC(glResumeTransformFeedback);
    FIND_PROC(glGetProgramBinary);
    FIND_PROC(glProgramBinary);
    FIND_PROC(glProgramParameteri);
    FIND_PROC(glInvalidateFramebuffer);
    FIND_PROC(glInvalidateSubFramebuffer);
    FIND_PROC(glTexStorage2D);
    FIND_PROC(glTexStorage3D);
    FIND_PROC(glGetInternalformativ);

    // Bind GL_EXT_texture_border_clamp

    FIND_PROC(glTexParameterIivEXT);
    FIND_PROC(glTexParameterIuivEXT);
    FIND_PROC(glGetTexParameterIivEXT);
    FIND_PROC(glGetTexParameterIuivEXT);
    FIND_PROC(glSamplerParameterIivEXT);
    FIND_PROC(glSamplerParameterIuivEXT);
    FIND_PROC(glGetSamplerParameterIivEXT);
    FIND_PROC(glGetSamplerParameterIuivEXT);

    #undef FIND_PROC

    if (!glReadBuffer ||
        !glDrawRangeElements ||
        !glTexImage3D ||
        !glTexSubImage3D ||
        !glCopyTexSubImage3D ||
        !glCompressedTexImage3D ||
        !glCompressedTexSubImage3D ||
        !glGenQueries ||
        !glDeleteQueries ||
        !glIsQuery ||
        !glBeginQuery ||
        !glEndQuery ||
        !glGetQueryiv ||
        !glGetQueryObjectuiv ||
        !glUnmapBuffer ||
        !glGetBufferPointerv ||
        !glDrawBuffers ||
        !glUniformMatrix2x3fv ||
        !glUniformMatrix3x2fv ||
        !glUniformMatrix2x4fv ||
        !glUniformMatrix4x2fv ||
        !glUniformMatrix3x4fv ||
        !glUniformMatrix4x3fv ||
        !glBlitFramebuffer ||
        !glRenderbufferStorageMultisample ||
        !glFramebufferTextureLayer ||
        !glMapBufferRange ||
        !glFlushMappedBufferRange ||
        !glBindVertexArray ||
        !glDeleteVertexArrays ||
        !glGenVertexArrays ||
        !glIsVertexArray ||
        !glGetIntegeri_v ||
        !glBeginTransformFeedback ||
        !glEndTransformFeedback ||
        !glBindBufferRange ||
        !glBindBufferBase ||
        !glTransformFeedbackVaryings ||
        !glGetTransformFeedbackVarying ||
        !glVertexAttribIPointer ||
        !glGetVertexAttribIiv ||
        !glGetVertexAttribIuiv ||
        !glVertexAttribI4i ||
        !glVertexAttribI4ui ||
        !glVertexAttribI4iv ||
        !glVertexAttribI4uiv ||
        !glGetUniformuiv ||
        !glGetFragDataLocation ||
        !glUniform1ui ||
        !glUniform2ui ||
        !glUniform3ui ||
        !glUniform4ui ||
        !glUniform1uiv ||
        !glUniform2uiv ||
        !glUniform3uiv ||
        !glUniform4uiv ||
        !glClearBufferiv ||
        !glClearBufferuiv ||
        !glClearBufferfv ||
        !glClearBufferfi ||
        !glGetStringi ||
        !glCopyBufferSubData ||
        !glGetUniformIndices ||
        !glGetActiveUniformsiv ||
        !glGetUniformBlockIndex ||
        !glGetActiveUniformBlockiv ||
        !glGetActiveUniformBlockName ||
        !glUniformBlockBinding ||
        !glDrawArraysInstanced ||
        !glDrawElementsInstanced ||
        !glFenceSync ||
        !glIsSync ||
        !glDeleteSync ||
        !glClientWaitSync ||
        !glWaitSync ||
        !glGetInteger64v ||
        !glGetSynciv ||
        !glGetInteger64i_v ||
        !glGetBufferParameteri64v ||
        !glGenSamplers ||
        !glDeleteSamplers ||
        !glIsSampler ||
        !glBindSampler ||
        !glSamplerParameteri ||
        !glSamplerParameteriv ||
        !glSamplerParameterf ||
        !glSamplerParameterfv ||
        !glGetSamplerParameteriv ||
        !glGetSamplerParameterfv ||
        !glVertexAttribDivisor ||
        !glBindTransformFeedback ||
        !glDeleteTransformFeedbacks ||
        !glGenTransformFeedbacks ||
        !glIsTransformFeedback ||
        !glPauseTransformFeedback ||
        !glResumeTransformFeedback ||
        !glGetProgramBinary ||
        !glProgramBinary ||
        !glProgramParameteri ||
        !glInvalidateFramebuffer ||
        !glInvalidateSubFramebuffer ||
        !glTexStorage2D ||
        !glTexStorage3D ||
        !glGetInternalformativ)
    {
        return GL_FALSE;
    }

    return GL_TRUE;
}

/* Function pointer definitions */
GL_APICALL void           (* GL_APIENTRY glReadBuffer) (GLenum mode);
GL_APICALL void           (* GL_APIENTRY glDrawRangeElements) (GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const GLvoid* indices);
GL_APICALL void           (* GL_APIENTRY glTexImage3D) (GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const GLvoid* pixels);
GL_APICALL void           (* GL_APIENTRY glTexSubImage3D) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const GLvoid* pixels);
GL_APICALL void           (* GL_APIENTRY glCopyTexSubImage3D) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);
GL_APICALL void           (* GL_APIENTRY glCompressedTexImage3D) (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const GLvoid* data);
GL_APICALL void           (* GL_APIENTRY glCompressedTexSubImage3D) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const GLvoid* data);
GL_APICALL void           (* GL_APIENTRY glGenQueries) (GLsizei n, GLuint* ids);
GL_APICALL void           (* GL_APIENTRY glDeleteQueries) (GLsizei n, const GLuint* ids);
GL_APICALL GLboolean      (* GL_APIENTRY glIsQuery) (GLuint id);
GL_APICALL void           (* GL_APIENTRY glBeginQuery) (GLenum target, GLuint id);
GL_APICALL void           (* GL_APIENTRY glEndQuery) (GLenum target);
GL_APICALL void           (* GL_APIENTRY glGetQueryiv) (GLenum target, GLenum pname, GLint* params);
GL_APICALL void           (* GL_APIENTRY glGetQueryObjectuiv) (GLuint id, GLenum pname, GLuint* params);
GL_APICALL GLboolean      (* GL_APIENTRY glUnmapBuffer) (GLenum target);
GL_APICALL void           (* GL_APIENTRY glGetBufferPointerv) (GLenum target, GLenum pname, GLvoid** params);
GL_APICALL void           (* GL_APIENTRY glDrawBuffers) (GLsizei n, const GLenum* bufs);
GL_APICALL void           (* GL_APIENTRY glUniformMatrix2x3fv) (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
GL_APICALL void           (* GL_APIENTRY glUniformMatrix3x2fv) (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
GL_APICALL void           (* GL_APIENTRY glUniformMatrix2x4fv) (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
GL_APICALL void           (* GL_APIENTRY glUniformMatrix4x2fv) (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
GL_APICALL void           (* GL_APIENTRY glUniformMatrix3x4fv) (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
GL_APICALL void           (* GL_APIENTRY glUniformMatrix4x3fv) (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
GL_APICALL void           (* GL_APIENTRY glBlitFramebuffer) (GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);
GL_APICALL void           (* GL_APIENTRY glRenderbufferStorageMultisample) (GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height);
GL_APICALL void           (* GL_APIENTRY glFramebufferTextureLayer) (GLenum target, GLenum attachment, GLuint texture, GLint level, GLint layer);
GL_APICALL GLvoid*        (* GL_APIENTRY glMapBufferRange) (GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access);
GL_APICALL void           (* GL_APIENTRY glFlushMappedBufferRange) (GLenum target, GLintptr offset, GLsizeiptr length);
GL_APICALL void           (* GL_APIENTRY glBindVertexArray) (GLuint array);
GL_APICALL void           (* GL_APIENTRY glDeleteVertexArrays) (GLsizei n, const GLuint* arrays);
GL_APICALL void           (* GL_APIENTRY glGenVertexArrays) (GLsizei n, GLuint* arrays);
GL_APICALL GLboolean      (* GL_APIENTRY glIsVertexArray) (GLuint array);
GL_APICALL void           (* GL_APIENTRY glGetIntegeri_v) (GLenum target, GLuint index, GLint* data);
GL_APICALL void           (* GL_APIENTRY glBeginTransformFeedback) (GLenum primitiveMode);
GL_APICALL void           (* GL_APIENTRY glEndTransformFeedback) (void);
GL_APICALL void           (* GL_APIENTRY glBindBufferRange) (GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size);
GL_APICALL void           (* GL_APIENTRY glBindBufferBase) (GLenum target, GLuint index, GLuint buffer);
GL_APICALL void           (* GL_APIENTRY glTransformFeedbackVaryings) (GLuint program, GLsizei count, const GLchar* const* varyings, GLenum bufferMode);
GL_APICALL void           (* GL_APIENTRY glGetTransformFeedbackVarying) (GLuint program, GLuint index, GLsizei bufSize, GLsizei* length, GLsizei* size, GLenum* type, GLchar* name);
GL_APICALL void           (* GL_APIENTRY glVertexAttribIPointer) (GLuint index, GLint size, GLenum type, GLsizei stride, const GLvoid* pointer);
GL_APICALL void           (* GL_APIENTRY glGetVertexAttribIiv) (GLuint index, GLenum pname, GLint* params);
GL_APICALL void           (* GL_APIENTRY glGetVertexAttribIuiv) (GLuint index, GLenum pname, GLuint* params);
GL_APICALL void           (* GL_APIENTRY glVertexAttribI4i) (GLuint index, GLint x, GLint y, GLint z, GLint w);
GL_APICALL void           (* GL_APIENTRY glVertexAttribI4ui) (GLuint index, GLuint x, GLuint y, GLuint z, GLuint w);
GL_APICALL void           (* GL_APIENTRY glVertexAttribI4iv) (GLuint index, const GLint* v);
GL_APICALL void           (* GL_APIENTRY glVertexAttribI4uiv) (GLuint index, const GLuint* v);
GL_APICALL void           (* GL_APIENTRY glGetUniformuiv) (GLuint program, GLint location, GLuint* params);
GL_APICALL GLint          (* GL_APIENTRY glGetFragDataLocation) (GLuint program, const GLchar *name);
GL_APICALL void           (* GL_APIENTRY glUniform1ui) (GLint location, GLuint v0);
GL_APICALL void           (* GL_APIENTRY glUniform2ui) (GLint location, GLuint v0, GLuint v1);
GL_APICALL void           (* GL_APIENTRY glUniform3ui) (GLint location, GLuint v0, GLuint v1, GLuint v2);
GL_APICALL void           (* GL_APIENTRY glUniform4ui) (GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3);
GL_APICALL void           (* GL_APIENTRY glUniform1uiv) (GLint location, GLsizei count, const GLuint* value);
GL_APICALL void           (* GL_APIENTRY glUniform2uiv) (GLint location, GLsizei count, const GLuint* value);
GL_APICALL void           (* GL_APIENTRY glUniform3uiv) (GLint location, GLsizei count, const GLuint* value);
GL_APICALL void           (* GL_APIENTRY glUniform4uiv) (GLint location, GLsizei count, const GLuint* value);
GL_APICALL void           (* GL_APIENTRY glClearBufferiv) (GLenum buffer, GLint drawbuffer, const GLint* value);
GL_APICALL void           (* GL_APIENTRY glClearBufferuiv) (GLenum buffer, GLint drawbuffer, const GLuint* value);
GL_APICALL void           (* GL_APIENTRY glClearBufferfv) (GLenum buffer, GLint drawbuffer, const GLfloat* value);
GL_APICALL void           (* GL_APIENTRY glClearBufferfi) (GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil);
GL_APICALL const GLubyte* (* GL_APIENTRY glGetStringi) (GLenum name, GLuint index);
GL_APICALL void           (* GL_APIENTRY glCopyBufferSubData) (GLenum readTarget, GLenum writeTarget, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size);
GL_APICALL void           (* GL_APIENTRY glGetUniformIndices) (GLuint program, GLsizei uniformCount, const GLchar* const* uniformNames, GLuint* uniformIndices);
GL_APICALL void           (* GL_APIENTRY glGetActiveUniformsiv) (GLuint program, GLsizei uniformCount, const GLuint* uniformIndices, GLenum pname, GLint* params);
GL_APICALL GLuint         (* GL_APIENTRY glGetUniformBlockIndex) (GLuint program, const GLchar* uniformBlockName);
GL_APICALL void           (* GL_APIENTRY glGetActiveUniformBlockiv) (GLuint program, GLuint uniformBlockIndex, GLenum pname, GLint* params);
GL_APICALL void           (* GL_APIENTRY glGetActiveUniformBlockName) (GLuint program, GLuint uniformBlockIndex, GLsizei bufSize, GLsizei* length, GLchar* uniformBlockName);
GL_APICALL void           (* GL_APIENTRY glUniformBlockBinding) (GLuint program, GLuint uniformBlockIndex, GLuint uniformBlockBinding);
GL_APICALL void           (* GL_APIENTRY glDrawArraysInstanced) (GLenum mode, GLint first, GLsizei count, GLsizei instanceCount);
GL_APICALL void           (* GL_APIENTRY glDrawElementsInstanced) (GLenum mode, GLsizei count, GLenum type, const GLvoid* indices, GLsizei instanceCount);
GL_APICALL GLsync         (* GL_APIENTRY glFenceSync) (GLenum condition, GLbitfield flags);
GL_APICALL GLboolean      (* GL_APIENTRY glIsSync) (GLsync sync);
GL_APICALL void           (* GL_APIENTRY glDeleteSync) (GLsync sync);
GL_APICALL GLenum         (* GL_APIENTRY glClientWaitSync) (GLsync sync, GLbitfield flags, GLuint64 timeout);
GL_APICALL void           (* GL_APIENTRY glWaitSync) (GLsync sync, GLbitfield flags, GLuint64 timeout);
GL_APICALL void           (* GL_APIENTRY glGetInteger64v) (GLenum pname, GLint64* params);
GL_APICALL void           (* GL_APIENTRY glGetSynciv) (GLsync sync, GLenum pname, GLsizei bufSize, GLsizei* length, GLint* values);
GL_APICALL void           (* GL_APIENTRY glGetInteger64i_v) (GLenum target, GLuint index, GLint64* data);
GL_APICALL void           (* GL_APIENTRY glGetBufferParameteri64v) (GLenum target, GLenum pname, GLint64* params);
GL_APICALL void           (* GL_APIENTRY glGenSamplers) (GLsizei count, GLuint* samplers);
GL_APICALL void           (* GL_APIENTRY glDeleteSamplers) (GLsizei count, const GLuint* samplers);
GL_APICALL GLboolean      (* GL_APIENTRY glIsSampler) (GLuint sampler);
GL_APICALL void           (* GL_APIENTRY glBindSampler) (GLuint unit, GLuint sampler);
GL_APICALL void           (* GL_APIENTRY glSamplerParameteri) (GLuint sampler, GLenum pname, GLint param);
GL_APICALL void           (* GL_APIENTRY glSamplerParameteriv) (GLuint sampler, GLenum pname, const GLint* param);
GL_APICALL void           (* GL_APIENTRY glSamplerParameterf) (GLuint sampler, GLenum pname, GLfloat param);
GL_APICALL void           (* GL_APIENTRY glSamplerParameterfv) (GLuint sampler, GLenum pname, const GLfloat* param);
GL_APICALL void           (* GL_APIENTRY glGetSamplerParameteriv) (GLuint sampler, GLenum pname, GLint* params);
GL_APICALL void           (* GL_APIENTRY glGetSamplerParameterfv) (GLuint sampler, GLenum pname, GLfloat* params);
GL_APICALL void           (* GL_APIENTRY glVertexAttribDivisor) (GLuint index, GLuint divisor);
GL_APICALL void           (* GL_APIENTRY glBindTransformFeedback) (GLenum target, GLuint id);
GL_APICALL void           (* GL_APIENTRY glDeleteTransformFeedbacks) (GLsizei n, const GLuint* ids);
GL_APICALL void           (* GL_APIENTRY glGenTransformFeedbacks) (GLsizei n, GLuint* ids);
GL_APICALL GLboolean      (* GL_APIENTRY glIsTransformFeedback) (GLuint id);
GL_APICALL void           (* GL_APIENTRY glPauseTransformFeedback) (void);
GL_APICALL void           (* GL_APIENTRY glResumeTransformFeedback) (void);
GL_APICALL void           (* GL_APIENTRY glGetProgramBinary) (GLuint program, GLsizei bufSize, GLsizei* length, GLenum* binaryFormat, GLvoid* binary);
GL_APICALL void           (* GL_APIENTRY glProgramBinary) (GLuint program, GLenum binaryFormat, const GLvoid* binary, GLsizei length);
GL_APICALL void           (* GL_APIENTRY glProgramParameteri) (GLuint program, GLenum pname, GLint value);
GL_APICALL void           (* GL_APIENTRY glInvalidateFramebuffer) (GLenum target, GLsizei numAttachments, const GLenum* attachments);
GL_APICALL void           (* GL_APIENTRY glInvalidateSubFramebuffer) (GLenum target, GLsizei numAttachments, const GLenum* attachments, GLint x, GLint y, GLsizei width, GLsizei height);
GL_APICALL void           (* GL_APIENTRY glTexStorage2D) (GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height);
GL_APICALL void           (* GL_APIENTRY glTexStorage3D) (GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth);
GL_APICALL void           (* GL_APIENTRY glGetInternalformativ) (GLenum target, GLenum internalformat, GLenum pname, GLsizei bufSize, GLint* params);

// GL_EXT_texture_border_clamp

GL_APICALL void           (* GL_APIENTRY  glTexParameterIivEXT) (GLenum target, GLenum pname, const GLint *params);
GL_APICALL void           (* GL_APIENTRY  glTexParameterIuivEXT) (GLenum target, GLenum pname, const GLuint *params);
GL_APICALL void           (* GL_APIENTRY  glGetTexParameterIivEXT) (GLenum target, GLenum pname, GLint *params);
GL_APICALL void           (* GL_APIENTRY  glGetTexParameterIuivEXT) (GLenum target, GLenum pname, GLuint *params);
GL_APICALL void           (* GL_APIENTRY  glSamplerParameterIivEXT) (GLuint sampler, GLenum pname, const GLint *param);
GL_APICALL void           (* GL_APIENTRY  glSamplerParameterIuivEXT) (GLuint sampler, GLenum pname, const GLuint *param);
GL_APICALL void           (* GL_APIENTRY  glGetSamplerParameterIivEXT) (GLuint sampler, GLenum pname, GLint *params);
GL_APICALL void           (* GL_APIENTRY  glGetSamplerParameterIuivEXT) (GLuint sampler, GLenum pname, GLuint *params);

// End GL_EXT_texture_border_clamp

// clang-format on
