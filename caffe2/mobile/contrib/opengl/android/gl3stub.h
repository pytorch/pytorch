
#ifndef __gl3_h_
#define __gl3_h_

/*
 * stub gl3.h for dynamic loading, based on:
 * gl3.h last updated on $Date: 2013-02-12 14:37:24 -0800 (Tue, 12 Feb 2013) $
 *
 * Changes:
 * - Added #include <GLES2/gl2.h>
 * - Removed duplicate OpenGL ES 2.0 declarations
 * - Converted OpenGL ES 3.0 function prototypes to function pointer
 *   declarations
 * - Added gl3stubInit() declaration
 */

#include <GLES2/gl2.h>
#include <android/api-level.h>

// clang-format off
#ifdef __cplusplus
extern "C" {
#endif

/* Call this function before calling any OpenGL ES 3.0 functions. It will
 * return GL_TRUE if the OpenGL ES 3.0 was successfully initialized, GL_FALSE
 * otherwise. */
GLboolean gl3stubInit();

/*-------------------------------------------------------------------------
 * Data type definitions
 *-----------------------------------------------------------------------*/

/* OpenGL ES 3.0 */

typedef unsigned short   GLhalf;
#if __ANDROID_API__ <= 19
typedef khronos_int64_t  GLint64;
typedef khronos_uint64_t GLuint64;
typedef struct __GLsync *GLsync;
#endif

/*-------------------------------------------------------------------------
 * Token definitions
 *-----------------------------------------------------------------------*/

/* OpenGL ES core versions */
#define GL_ES_VERSION_3_0                                1

/* OpenGL ES 3.0 */

#define GL_READ_BUFFER                                   0x0C02
#define GL_UNPACK_ROW_LENGTH                             0x0CF2
#define GL_UNPACK_SKIP_ROWS                              0x0CF3
#define GL_UNPACK_SKIP_PIXELS                            0x0CF4
#define GL_PACK_ROW_LENGTH                               0x0D02
#define GL_PACK_SKIP_ROWS                                0x0D03
#define GL_PACK_SKIP_PIXELS                              0x0D04
#define GL_COLOR                                         0x1800
#define GL_DEPTH                                         0x1801
#define GL_STENCIL                                       0x1802
#define GL_RED                                           0x1903
#define GL_RGB8                                          0x8051
#define GL_RGBA8                                         0x8058
#define GL_RGB10_A2                                      0x8059
#define GL_TEXTURE_BINDING_3D                            0x806A
#define GL_UNPACK_SKIP_IMAGES                            0x806D
#define GL_UNPACK_IMAGE_HEIGHT                           0x806E
#define GL_TEXTURE_3D                                    0x806F
#define GL_TEXTURE_WRAP_R                                0x8072
#define GL_MAX_3D_TEXTURE_SIZE                           0x8073
#define GL_UNSIGNED_INT_2_10_10_10_REV                   0x8368
#define GL_MAX_ELEMENTS_VERTICES                         0x80E8
#define GL_MAX_ELEMENTS_INDICES                          0x80E9
#define GL_TEXTURE_MIN_LOD                               0x813A
#define GL_TEXTURE_MAX_LOD                               0x813B
#define GL_TEXTURE_BASE_LEVEL                            0x813C
#define GL_TEXTURE_MAX_LEVEL                             0x813D
#define GL_MIN                                           0x8007
#define GL_MAX                                           0x8008
#define GL_DEPTH_COMPONENT24                             0x81A6
#define GL_MAX_TEXTURE_LOD_BIAS                          0x84FD
#define GL_TEXTURE_COMPARE_MODE                          0x884C
#define GL_TEXTURE_COMPARE_FUNC                          0x884D
#define GL_CURRENT_QUERY                                 0x8865
#define GL_QUERY_RESULT                                  0x8866
#define GL_QUERY_RESULT_AVAILABLE                        0x8867
#define GL_BUFFER_MAPPED                                 0x88BC
#define GL_BUFFER_MAP_POINTER                            0x88BD
#define GL_STREAM_READ                                   0x88E1
#define GL_STREAM_COPY                                   0x88E2
#define GL_STATIC_READ                                   0x88E5
#define GL_STATIC_COPY                                   0x88E6
#define GL_DYNAMIC_READ                                  0x88E9
#define GL_DYNAMIC_COPY                                  0x88EA
#define GL_MAX_DRAW_BUFFERS                              0x8824
#define GL_DRAW_BUFFER0                                  0x8825
#define GL_DRAW_BUFFER1                                  0x8826
#define GL_DRAW_BUFFER2                                  0x8827
#define GL_DRAW_BUFFER3                                  0x8828
#define GL_DRAW_BUFFER4                                  0x8829
#define GL_DRAW_BUFFER5                                  0x882A
#define GL_DRAW_BUFFER6                                  0x882B
#define GL_DRAW_BUFFER7                                  0x882C
#define GL_DRAW_BUFFER8                                  0x882D
#define GL_DRAW_BUFFER9                                  0x882E
#define GL_DRAW_BUFFER10                                 0x882F
#define GL_DRAW_BUFFER11                                 0x8830
#define GL_DRAW_BUFFER12                                 0x8831
#define GL_DRAW_BUFFER13                                 0x8832
#define GL_DRAW_BUFFER14                                 0x8833
#define GL_DRAW_BUFFER15                                 0x8834
#define GL_MAX_FRAGMENT_UNIFORM_COMPONENTS               0x8B49
#define GL_MAX_VERTEX_UNIFORM_COMPONENTS                 0x8B4A
#define GL_SAMPLER_3D                                    0x8B5F
#define GL_SAMPLER_2D_SHADOW                             0x8B62
#define GL_FRAGMENT_SHADER_DERIVATIVE_HINT               0x8B8B
#define GL_PIXEL_PACK_BUFFER                             0x88EB
#define GL_PIXEL_UNPACK_BUFFER                           0x88EC
#define GL_PIXEL_PACK_BUFFER_BINDING                     0x88ED
#define GL_PIXEL_UNPACK_BUFFER_BINDING                   0x88EF
#define GL_FLOAT_MAT2x3                                  0x8B65
#define GL_FLOAT_MAT2x4                                  0x8B66
#define GL_FLOAT_MAT3x2                                  0x8B67
#define GL_FLOAT_MAT3x4                                  0x8B68
#define GL_FLOAT_MAT4x2                                  0x8B69
#define GL_FLOAT_MAT4x3                                  0x8B6A
#define GL_SRGB                                          0x8C40
#define GL_SRGB8                                         0x8C41
#define GL_SRGB8_ALPHA8                                  0x8C43
#define GL_COMPARE_REF_TO_TEXTURE                        0x884E
#define GL_MAJOR_VERSION                                 0x821B
#define GL_MINOR_VERSION                                 0x821C
#define GL_NUM_EXTENSIONS                                0x821D
#define GL_RGBA32F                                       0x8814
#define GL_RGB32F                                        0x8815
#define GL_RGBA16F                                       0x881A
#define GL_RGB16F                                        0x881B
#define GL_VERTEX_ATTRIB_ARRAY_INTEGER                   0x88FD
#define GL_MAX_ARRAY_TEXTURE_LAYERS                      0x88FF
#define GL_MIN_PROGRAM_TEXEL_OFFSET                      0x8904
#define GL_MAX_PROGRAM_TEXEL_OFFSET                      0x8905
#define GL_MAX_VARYING_COMPONENTS                        0x8B4B
#define GL_TEXTURE_2D_ARRAY                              0x8C1A
#define GL_TEXTURE_BINDING_2D_ARRAY                      0x8C1D
#define GL_R11F_G11F_B10F                                0x8C3A
#define GL_UNSIGNED_INT_10F_11F_11F_REV                  0x8C3B
#define GL_RGB9_E5                                       0x8C3D
#define GL_UNSIGNED_INT_5_9_9_9_REV                      0x8C3E
#define GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH         0x8C76
#define GL_TRANSFORM_FEEDBACK_BUFFER_MODE                0x8C7F
#define GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS    0x8C80
#define GL_TRANSFORM_FEEDBACK_VARYINGS                   0x8C83
#define GL_TRANSFORM_FEEDBACK_BUFFER_START               0x8C84
#define GL_TRANSFORM_FEEDBACK_BUFFER_SIZE                0x8C85
#define GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN         0x8C88
#define GL_RASTERIZER_DISCARD                            0x8C89
#define GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS 0x8C8A
#define GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS       0x8C8B
#define GL_INTERLEAVED_ATTRIBS                           0x8C8C
#define GL_SEPARATE_ATTRIBS                              0x8C8D
#define GL_TRANSFORM_FEEDBACK_BUFFER                     0x8C8E
#define GL_TRANSFORM_FEEDBACK_BUFFER_BINDING             0x8C8F
#define GL_RGBA32UI                                      0x8D70
#define GL_RGB32UI                                       0x8D71
#define GL_RGBA16UI                                      0x8D76
#define GL_RGB16UI                                       0x8D77
#define GL_RGBA8UI                                       0x8D7C
#define GL_RGB8UI                                        0x8D7D
#define GL_RGBA32I                                       0x8D82
#define GL_RGB32I                                        0x8D83
#define GL_RGBA16I                                       0x8D88
#define GL_RGB16I                                        0x8D89
#define GL_RGBA8I                                        0x8D8E
#define GL_RGB8I                                         0x8D8F
#define GL_RED_INTEGER                                   0x8D94
#define GL_RGB_INTEGER                                   0x8D98
#define GL_RGBA_INTEGER                                  0x8D99
#define GL_SAMPLER_2D_ARRAY                              0x8DC1
#define GL_SAMPLER_2D_ARRAY_SHADOW                       0x8DC4
#define GL_SAMPLER_CUBE_SHADOW                           0x8DC5
#define GL_UNSIGNED_INT_VEC2                             0x8DC6
#define GL_UNSIGNED_INT_VEC3                             0x8DC7
#define GL_UNSIGNED_INT_VEC4                             0x8DC8
#define GL_INT_SAMPLER_2D                                0x8DCA
#define GL_INT_SAMPLER_3D                                0x8DCB
#define GL_INT_SAMPLER_CUBE                              0x8DCC
#define GL_INT_SAMPLER_2D_ARRAY                          0x8DCF
#define GL_UNSIGNED_INT_SAMPLER_2D                       0x8DD2
#define GL_UNSIGNED_INT_SAMPLER_3D                       0x8DD3
#define GL_UNSIGNED_INT_SAMPLER_CUBE                     0x8DD4
#define GL_UNSIGNED_INT_SAMPLER_2D_ARRAY                 0x8DD7
#define GL_BUFFER_ACCESS_FLAGS                           0x911F
#define GL_BUFFER_MAP_LENGTH                             0x9120
#define GL_BUFFER_MAP_OFFSET                             0x9121
#define GL_DEPTH_COMPONENT32F                            0x8CAC
#define GL_DEPTH32F_STENCIL8                             0x8CAD
#define GL_FLOAT_32_UNSIGNED_INT_24_8_REV                0x8DAD
#define GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING         0x8210
#define GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE         0x8211
#define GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE               0x8212
#define GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE             0x8213
#define GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE              0x8214
#define GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE             0x8215
#define GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE             0x8216
#define GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE           0x8217
#define GL_FRAMEBUFFER_DEFAULT                           0x8218
#define GL_FRAMEBUFFER_UNDEFINED                         0x8219
#define GL_DEPTH_STENCIL_ATTACHMENT                      0x821A
#define GL_DEPTH_STENCIL                                 0x84F9
#define GL_UNSIGNED_INT_24_8                             0x84FA
#define GL_DEPTH24_STENCIL8                              0x88F0
#define GL_UNSIGNED_NORMALIZED                           0x8C17
#define GL_DRAW_FRAMEBUFFER_BINDING                      GL_FRAMEBUFFER_BINDING
#define GL_READ_FRAMEBUFFER                              0x8CA8
#define GL_DRAW_FRAMEBUFFER                              0x8CA9
#define GL_READ_FRAMEBUFFER_BINDING                      0x8CAA
#define GL_RENDERBUFFER_SAMPLES                          0x8CAB
#define GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER          0x8CD4
#define GL_MAX_COLOR_ATTACHMENTS                         0x8CDF
#define GL_COLOR_ATTACHMENT1                             0x8CE1
#define GL_COLOR_ATTACHMENT2                             0x8CE2
#define GL_COLOR_ATTACHMENT3                             0x8CE3
#define GL_COLOR_ATTACHMENT4                             0x8CE4
#define GL_COLOR_ATTACHMENT5                             0x8CE5
#define GL_COLOR_ATTACHMENT6                             0x8CE6
#define GL_COLOR_ATTACHMENT7                             0x8CE7
#define GL_COLOR_ATTACHMENT8                             0x8CE8
#define GL_COLOR_ATTACHMENT9                             0x8CE9
#define GL_COLOR_ATTACHMENT10                            0x8CEA
#define GL_COLOR_ATTACHMENT11                            0x8CEB
#define GL_COLOR_ATTACHMENT12                            0x8CEC
#define GL_COLOR_ATTACHMENT13                            0x8CED
#define GL_COLOR_ATTACHMENT14                            0x8CEE
#define GL_COLOR_ATTACHMENT15                            0x8CEF
#define GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE            0x8D56
#define GL_MAX_SAMPLES                                   0x8D57
#define GL_HALF_FLOAT                                    0x140B
#define GL_MAP_READ_BIT                                  0x0001
#define GL_MAP_WRITE_BIT                                 0x0002
#define GL_MAP_INVALIDATE_RANGE_BIT                      0x0004
#define GL_MAP_INVALIDATE_BUFFER_BIT                     0x0008
#define GL_MAP_FLUSH_EXPLICIT_BIT                        0x0010
#define GL_MAP_UNSYNCHRONIZED_BIT                        0x0020
#define GL_RG                                            0x8227
#define GL_RG_INTEGER                                    0x8228
#define GL_R8                                            0x8229
#define GL_RG8                                           0x822B
#define GL_R16F                                          0x822D
#define GL_R32F                                          0x822E
#define GL_RG16F                                         0x822F
#define GL_RG32F                                         0x8230
#define GL_R8I                                           0x8231
#define GL_R8UI                                          0x8232
#define GL_R16I                                          0x8233
#define GL_R16UI                                         0x8234
#define GL_R32I                                          0x8235
#define GL_R32UI                                         0x8236
#define GL_RG8I                                          0x8237
#define GL_RG8UI                                         0x8238
#define GL_RG16I                                         0x8239
#define GL_RG16UI                                        0x823A
#define GL_RG32I                                         0x823B
#define GL_RG32UI                                        0x823C
#define GL_VERTEX_ARRAY_BINDING                          0x85B5
#define GL_R8_SNORM                                      0x8F94
#define GL_RG8_SNORM                                     0x8F95
#define GL_RGB8_SNORM                                    0x8F96
#define GL_RGBA8_SNORM                                   0x8F97
#define GL_SIGNED_NORMALIZED                             0x8F9C
#define GL_PRIMITIVE_RESTART_FIXED_INDEX                 0x8D69
#define GL_COPY_READ_BUFFER                              0x8F36
#define GL_COPY_WRITE_BUFFER                             0x8F37
#define GL_COPY_READ_BUFFER_BINDING                      GL_COPY_READ_BUFFER
#define GL_COPY_WRITE_BUFFER_BINDING                     GL_COPY_WRITE_BUFFER
#define GL_UNIFORM_BUFFER                                0x8A11
#define GL_UNIFORM_BUFFER_BINDING                        0x8A28
#define GL_UNIFORM_BUFFER_START                          0x8A29
#define GL_UNIFORM_BUFFER_SIZE                           0x8A2A
#define GL_MAX_VERTEX_UNIFORM_BLOCKS                     0x8A2B
#define GL_MAX_FRAGMENT_UNIFORM_BLOCKS                   0x8A2D
#define GL_MAX_COMBINED_UNIFORM_BLOCKS                   0x8A2E
#define GL_MAX_UNIFORM_BUFFER_BINDINGS                   0x8A2F
#define GL_MAX_UNIFORM_BLOCK_SIZE                        0x8A30
#define GL_MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS        0x8A31
#define GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS      0x8A33
#define GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT               0x8A34
#define GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH          0x8A35
#define GL_ACTIVE_UNIFORM_BLOCKS                         0x8A36
#define GL_UNIFORM_TYPE                                  0x8A37
#define GL_UNIFORM_SIZE                                  0x8A38
#define GL_UNIFORM_NAME_LENGTH                           0x8A39
#define GL_UNIFORM_BLOCK_INDEX                           0x8A3A
#define GL_UNIFORM_OFFSET                                0x8A3B
#define GL_UNIFORM_ARRAY_STRIDE                          0x8A3C
#define GL_UNIFORM_MATRIX_STRIDE                         0x8A3D
#define GL_UNIFORM_IS_ROW_MAJOR                          0x8A3E
#define GL_UNIFORM_BLOCK_BINDING                         0x8A3F
#define GL_UNIFORM_BLOCK_DATA_SIZE                       0x8A40
#define GL_UNIFORM_BLOCK_NAME_LENGTH                     0x8A41
#define GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS                 0x8A42
#define GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES          0x8A43
#define GL_UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER     0x8A44
#define GL_UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER   0x8A46
#define GL_INVALID_INDEX                                 0xFFFFFFFFu
#define GL_MAX_VERTEX_OUTPUT_COMPONENTS                  0x9122
#define GL_MAX_FRAGMENT_INPUT_COMPONENTS                 0x9125
#define GL_MAX_SERVER_WAIT_TIMEOUT                       0x9111
#define GL_OBJECT_TYPE                                   0x9112
#define GL_SYNC_CONDITION                                0x9113
#define GL_SYNC_STATUS                                   0x9114
#define GL_SYNC_FLAGS                                    0x9115
#define GL_SYNC_FENCE                                    0x9116
#define GL_SYNC_GPU_COMMANDS_COMPLETE                    0x9117
#define GL_UNSIGNALED                                    0x9118
#define GL_SIGNALED                                      0x9119
#define GL_ALREADY_SIGNALED                              0x911A
#define GL_TIMEOUT_EXPIRED                               0x911B
#define GL_CONDITION_SATISFIED                           0x911C
#define GL_WAIT_FAILED                                   0x911D
#define GL_SYNC_FLUSH_COMMANDS_BIT                       0x00000001
#define GL_TIMEOUT_IGNORED                               0xFFFFFFFFFFFFFFFFull
#define GL_VERTEX_ATTRIB_ARRAY_DIVISOR                   0x88FE
#define GL_ANY_SAMPLES_PASSED                            0x8C2F
#define GL_ANY_SAMPLES_PASSED_CONSERVATIVE               0x8D6A
#define GL_SAMPLER_BINDING                               0x8919
#define GL_RGB10_A2UI                                    0x906F
#define GL_TEXTURE_SWIZZLE_R                             0x8E42
#define GL_TEXTURE_SWIZZLE_G                             0x8E43
#define GL_TEXTURE_SWIZZLE_B                             0x8E44
#define GL_TEXTURE_SWIZZLE_A                             0x8E45
#define GL_GREEN                                         0x1904
#define GL_BLUE                                          0x1905
#define GL_INT_2_10_10_10_REV                            0x8D9F
#define GL_TRANSFORM_FEEDBACK                            0x8E22
#define GL_TRANSFORM_FEEDBACK_PAUSED                     0x8E23
#define GL_TRANSFORM_FEEDBACK_ACTIVE                     0x8E24
#define GL_TRANSFORM_FEEDBACK_BINDING                    0x8E25
#define GL_PROGRAM_BINARY_RETRIEVABLE_HINT               0x8257
#define GL_PROGRAM_BINARY_LENGTH                         0x8741
#define GL_NUM_PROGRAM_BINARY_FORMATS                    0x87FE
#define GL_PROGRAM_BINARY_FORMATS                        0x87FF
#define GL_COMPRESSED_R11_EAC                            0x9270
#define GL_COMPRESSED_SIGNED_R11_EAC                     0x9271
#define GL_COMPRESSED_RG11_EAC                           0x9272
#define GL_COMPRESSED_SIGNED_RG11_EAC                    0x9273
#define GL_COMPRESSED_RGB8_ETC2                          0x9274
#define GL_COMPRESSED_SRGB8_ETC2                         0x9275
#define GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2      0x9276
#define GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2     0x9277
#define GL_COMPRESSED_RGBA8_ETC2_EAC                     0x9278
#define GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC              0x9279
#define GL_TEXTURE_IMMUTABLE_FORMAT                      0x912F
#define GL_MAX_ELEMENT_INDEX                             0x8D6B
#define GL_NUM_SAMPLE_COUNTS                             0x9380
#define GL_TEXTURE_IMMUTABLE_LEVELS                      0x82DF

/*-------------------------------------------------------------------------
 * Entrypoint definitions
 *-----------------------------------------------------------------------*/

/* OpenGL ES 3.0 */

extern GL_APICALL void           (* GL_APIENTRY glReadBuffer) (GLenum mode);
extern GL_APICALL void           (* GL_APIENTRY glDrawRangeElements) (GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const GLvoid* indices);
extern GL_APICALL void           (* GL_APIENTRY glTexImage3D) (GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const GLvoid* pixels);
extern GL_APICALL void           (* GL_APIENTRY glTexSubImage3D) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const GLvoid* pixels);
extern GL_APICALL void           (* GL_APIENTRY glCopyTexSubImage3D) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);
extern GL_APICALL void           (* GL_APIENTRY glCompressedTexImage3D) (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const GLvoid* data);
extern GL_APICALL void           (* GL_APIENTRY glCompressedTexSubImage3D) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const GLvoid* data);
extern GL_APICALL void           (* GL_APIENTRY glGenQueries) (GLsizei n, GLuint* ids);
extern GL_APICALL void           (* GL_APIENTRY glDeleteQueries) (GLsizei n, const GLuint* ids);
extern GL_APICALL GLboolean      (* GL_APIENTRY glIsQuery) (GLuint id);
extern GL_APICALL void           (* GL_APIENTRY glBeginQuery) (GLenum target, GLuint id);
extern GL_APICALL void           (* GL_APIENTRY glEndQuery) (GLenum target);
extern GL_APICALL void           (* GL_APIENTRY glGetQueryiv) (GLenum target, GLenum pname, GLint* params);
extern GL_APICALL void           (* GL_APIENTRY glGetQueryObjectuiv) (GLuint id, GLenum pname, GLuint* params);
extern GL_APICALL GLboolean      (* GL_APIENTRY glUnmapBuffer) (GLenum target);
extern GL_APICALL void           (* GL_APIENTRY glGetBufferPointerv) (GLenum target, GLenum pname, GLvoid** params);
extern GL_APICALL void           (* GL_APIENTRY glDrawBuffers) (GLsizei n, const GLenum* bufs);
extern GL_APICALL void           (* GL_APIENTRY glUniformMatrix2x3fv) (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
extern GL_APICALL void           (* GL_APIENTRY glUniformMatrix3x2fv) (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
extern GL_APICALL void           (* GL_APIENTRY glUniformMatrix2x4fv) (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
extern GL_APICALL void           (* GL_APIENTRY glUniformMatrix4x2fv) (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
extern GL_APICALL void           (* GL_APIENTRY glUniformMatrix3x4fv) (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
extern GL_APICALL void           (* GL_APIENTRY glUniformMatrix4x3fv) (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
extern GL_APICALL void           (* GL_APIENTRY glBlitFramebuffer) (GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);
extern GL_APICALL void           (* GL_APIENTRY glRenderbufferStorageMultisample) (GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height);
extern GL_APICALL void           (* GL_APIENTRY glFramebufferTextureLayer) (GLenum target, GLenum attachment, GLuint texture, GLint level, GLint layer);
extern GL_APICALL GLvoid*        (* GL_APIENTRY glMapBufferRange) (GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access);
extern GL_APICALL void           (* GL_APIENTRY glFlushMappedBufferRange) (GLenum target, GLintptr offset, GLsizeiptr length);
extern GL_APICALL void           (* GL_APIENTRY glBindVertexArray) (GLuint array);
extern GL_APICALL void           (* GL_APIENTRY glDeleteVertexArrays) (GLsizei n, const GLuint* arrays);
extern GL_APICALL void           (* GL_APIENTRY glGenVertexArrays) (GLsizei n, GLuint* arrays);
extern GL_APICALL GLboolean      (* GL_APIENTRY glIsVertexArray) (GLuint array);
extern GL_APICALL void           (* GL_APIENTRY glGetIntegeri_v) (GLenum target, GLuint index, GLint* data);
extern GL_APICALL void           (* GL_APIENTRY glBeginTransformFeedback) (GLenum primitiveMode);
extern GL_APICALL void           (* GL_APIENTRY glEndTransformFeedback) (void);
extern GL_APICALL void           (* GL_APIENTRY glBindBufferRange) (GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size);
extern GL_APICALL void           (* GL_APIENTRY glBindBufferBase) (GLenum target, GLuint index, GLuint buffer);
extern GL_APICALL void           (* GL_APIENTRY glTransformFeedbackVaryings) (GLuint program, GLsizei count, const GLchar* const* varyings, GLenum bufferMode);
extern GL_APICALL void           (* GL_APIENTRY glGetTransformFeedbackVarying) (GLuint program, GLuint index, GLsizei bufSize, GLsizei* length, GLsizei* size, GLenum* type, GLchar* name);
extern GL_APICALL void           (* GL_APIENTRY glVertexAttribIPointer) (GLuint index, GLint size, GLenum type, GLsizei stride, const GLvoid* pointer);
extern GL_APICALL void           (* GL_APIENTRY glGetVertexAttribIiv) (GLuint index, GLenum pname, GLint* params);
extern GL_APICALL void           (* GL_APIENTRY glGetVertexAttribIuiv) (GLuint index, GLenum pname, GLuint* params);
extern GL_APICALL void           (* GL_APIENTRY glVertexAttribI4i) (GLuint index, GLint x, GLint y, GLint z, GLint w);
extern GL_APICALL void           (* GL_APIENTRY glVertexAttribI4ui) (GLuint index, GLuint x, GLuint y, GLuint z, GLuint w);
extern GL_APICALL void           (* GL_APIENTRY glVertexAttribI4iv) (GLuint index, const GLint* v);
extern GL_APICALL void           (* GL_APIENTRY glVertexAttribI4uiv) (GLuint index, const GLuint* v);
extern GL_APICALL void           (* GL_APIENTRY glGetUniformuiv) (GLuint program, GLint location, GLuint* params);
extern GL_APICALL GLint          (* GL_APIENTRY glGetFragDataLocation) (GLuint program, const GLchar *name);
extern GL_APICALL void           (* GL_APIENTRY glUniform1ui) (GLint location, GLuint v0);
extern GL_APICALL void           (* GL_APIENTRY glUniform2ui) (GLint location, GLuint v0, GLuint v1);
extern GL_APICALL void           (* GL_APIENTRY glUniform3ui) (GLint location, GLuint v0, GLuint v1, GLuint v2);
extern GL_APICALL void           (* GL_APIENTRY glUniform4ui) (GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3);
extern GL_APICALL void           (* GL_APIENTRY glUniform1uiv) (GLint location, GLsizei count, const GLuint* value);
extern GL_APICALL void           (* GL_APIENTRY glUniform2uiv) (GLint location, GLsizei count, const GLuint* value);
extern GL_APICALL void           (* GL_APIENTRY glUniform3uiv) (GLint location, GLsizei count, const GLuint* value);
extern GL_APICALL void           (* GL_APIENTRY glUniform4uiv) (GLint location, GLsizei count, const GLuint* value);
extern GL_APICALL void           (* GL_APIENTRY glClearBufferiv) (GLenum buffer, GLint drawbuffer, const GLint* value);
extern GL_APICALL void           (* GL_APIENTRY glClearBufferuiv) (GLenum buffer, GLint drawbuffer, const GLuint* value);
extern GL_APICALL void           (* GL_APIENTRY glClearBufferfv) (GLenum buffer, GLint drawbuffer, const GLfloat* value);
extern GL_APICALL void           (* GL_APIENTRY glClearBufferfi) (GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil);
extern GL_APICALL const GLubyte* (* GL_APIENTRY glGetStringi) (GLenum name, GLuint index);
extern GL_APICALL void           (* GL_APIENTRY glCopyBufferSubData) (GLenum readTarget, GLenum writeTarget, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size);
extern GL_APICALL void           (* GL_APIENTRY glGetUniformIndices) (GLuint program, GLsizei uniformCount, const GLchar* const* uniformNames, GLuint* uniformIndices);
extern GL_APICALL void           (* GL_APIENTRY glGetActiveUniformsiv) (GLuint program, GLsizei uniformCount, const GLuint* uniformIndices, GLenum pname, GLint* params);
extern GL_APICALL GLuint         (* GL_APIENTRY glGetUniformBlockIndex) (GLuint program, const GLchar* uniformBlockName);
extern GL_APICALL void           (* GL_APIENTRY glGetActiveUniformBlockiv) (GLuint program, GLuint uniformBlockIndex, GLenum pname, GLint* params);
extern GL_APICALL void           (* GL_APIENTRY glGetActiveUniformBlockName) (GLuint program, GLuint uniformBlockIndex, GLsizei bufSize, GLsizei* length, GLchar* uniformBlockName);
extern GL_APICALL void           (* GL_APIENTRY glUniformBlockBinding) (GLuint program, GLuint uniformBlockIndex, GLuint uniformBlockBinding);
extern GL_APICALL void           (* GL_APIENTRY glDrawArraysInstanced) (GLenum mode, GLint first, GLsizei count, GLsizei instanceCount);
extern GL_APICALL void           (* GL_APIENTRY glDrawElementsInstanced) (GLenum mode, GLsizei count, GLenum type, const GLvoid* indices, GLsizei instanceCount);
extern GL_APICALL GLsync         (* GL_APIENTRY glFenceSync) (GLenum condition, GLbitfield flags);
extern GL_APICALL GLboolean      (* GL_APIENTRY glIsSync) (GLsync sync);
extern GL_APICALL void           (* GL_APIENTRY glDeleteSync) (GLsync sync);
extern GL_APICALL GLenum         (* GL_APIENTRY glClientWaitSync) (GLsync sync, GLbitfield flags, GLuint64 timeout);
extern GL_APICALL void           (* GL_APIENTRY glWaitSync) (GLsync sync, GLbitfield flags, GLuint64 timeout);
extern GL_APICALL void           (* GL_APIENTRY glGetInteger64v) (GLenum pname, GLint64* params);
extern GL_APICALL void           (* GL_APIENTRY glGetSynciv) (GLsync sync, GLenum pname, GLsizei bufSize, GLsizei* length, GLint* values);
extern GL_APICALL void           (* GL_APIENTRY glGetInteger64i_v) (GLenum target, GLuint index, GLint64* data);
extern GL_APICALL void           (* GL_APIENTRY glGetBufferParameteri64v) (GLenum target, GLenum pname, GLint64* params);
extern GL_APICALL void           (* GL_APIENTRY glGenSamplers) (GLsizei count, GLuint* samplers);
extern GL_APICALL void           (* GL_APIENTRY glDeleteSamplers) (GLsizei count, const GLuint* samplers);
extern GL_APICALL GLboolean      (* GL_APIENTRY glIsSampler) (GLuint sampler);
extern GL_APICALL void           (* GL_APIENTRY glBindSampler) (GLuint unit, GLuint sampler);
extern GL_APICALL void           (* GL_APIENTRY glSamplerParameteri) (GLuint sampler, GLenum pname, GLint param);
extern GL_APICALL void           (* GL_APIENTRY glSamplerParameteriv) (GLuint sampler, GLenum pname, const GLint* param);
extern GL_APICALL void           (* GL_APIENTRY glSamplerParameterf) (GLuint sampler, GLenum pname, GLfloat param);
extern GL_APICALL void           (* GL_APIENTRY glSamplerParameterfv) (GLuint sampler, GLenum pname, const GLfloat* param);
extern GL_APICALL void           (* GL_APIENTRY glGetSamplerParameteriv) (GLuint sampler, GLenum pname, GLint* params);
extern GL_APICALL void           (* GL_APIENTRY glGetSamplerParameterfv) (GLuint sampler, GLenum pname, GLfloat* params);
extern GL_APICALL void           (* GL_APIENTRY glVertexAttribDivisor) (GLuint index, GLuint divisor);
extern GL_APICALL void           (* GL_APIENTRY glBindTransformFeedback) (GLenum target, GLuint id);
extern GL_APICALL void           (* GL_APIENTRY glDeleteTransformFeedbacks) (GLsizei n, const GLuint* ids);
extern GL_APICALL void           (* GL_APIENTRY glGenTransformFeedbacks) (GLsizei n, GLuint* ids);
extern GL_APICALL GLboolean      (* GL_APIENTRY glIsTransformFeedback) (GLuint id);
extern GL_APICALL void           (* GL_APIENTRY glPauseTransformFeedback) (void);
extern GL_APICALL void           (* GL_APIENTRY glResumeTransformFeedback) (void);
extern GL_APICALL void           (* GL_APIENTRY glGetProgramBinary) (GLuint program, GLsizei bufSize, GLsizei* length, GLenum* binaryFormat, GLvoid* binary);
extern GL_APICALL void           (* GL_APIENTRY glProgramBinary) (GLuint program, GLenum binaryFormat, const GLvoid* binary, GLsizei length);
extern GL_APICALL void           (* GL_APIENTRY glProgramParameteri) (GLuint program, GLenum pname, GLint value);
extern GL_APICALL void           (* GL_APIENTRY glInvalidateFramebuffer) (GLenum target, GLsizei numAttachments, const GLenum* attachments);
extern GL_APICALL void           (* GL_APIENTRY glInvalidateSubFramebuffer) (GLenum target, GLsizei numAttachments, const GLenum* attachments, GLint x, GLint y, GLsizei width, GLsizei height);
extern GL_APICALL void           (* GL_APIENTRY glTexStorage2D) (GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height);
extern GL_APICALL void           (* GL_APIENTRY glTexStorage3D) (GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth);
extern GL_APICALL void           (* GL_APIENTRY glGetInternalformativ) (GLenum target, GLenum internalformat, GLenum pname, GLsizei bufSize, GLint* params);

#ifndef GL_EXT_texture_border_clamp
#define GL_EXT_texture_border_clamp 1
#define GL_TEXTURE_BORDER_COLOR_EXT       0x1004
#define GL_CLAMP_TO_BORDER_EXT            0x812D
extern GL_APICALL void           (* GL_APIENTRY  glTexParameterIivEXT) (GLenum target, GLenum pname, const GLint *params);
extern GL_APICALL void           (* GL_APIENTRY  glTexParameterIuivEXT) (GLenum target, GLenum pname, const GLuint *params);
extern GL_APICALL void           (* GL_APIENTRY  glGetTexParameterIivEXT) (GLenum target, GLenum pname, GLint *params);
extern GL_APICALL void           (* GL_APIENTRY  glGetTexParameterIuivEXT) (GLenum target, GLenum pname, GLuint *params);
extern GL_APICALL void           (* GL_APIENTRY  glSamplerParameterIivEXT) (GLuint sampler, GLenum pname, const GLint *param);
extern GL_APICALL void           (* GL_APIENTRY  glSamplerParameterIuivEXT) (GLuint sampler, GLenum pname, const GLuint *param);
extern GL_APICALL void           (* GL_APIENTRY  glGetSamplerParameterIivEXT) (GLuint sampler, GLenum pname, GLint *params);
extern GL_APICALL void           (* GL_APIENTRY  glGetSamplerParameterIuivEXT) (GLuint sampler, GLenum pname, GLuint *params);
#endif /* GL_EXT_texture_border_clamp */

#ifdef __cplusplus
}
#endif
// clang-format on

#endif
