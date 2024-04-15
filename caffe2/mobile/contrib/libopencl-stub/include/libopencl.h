#ifndef LIBOPENCL_STUB_H
#define LIBOPENCL_STUB_H

#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*f_pfn_notify)(const char *, const void *, size_t, void *);

typedef cl_int (*f_clGetPlatformIDs) (cl_uint, cl_platform_id *, cl_uint *);

typedef cl_int (*f_clGetPlatformInfo) (cl_platform_id, cl_platform_info, size_t, void *, size_t *);

typedef cl_int (*f_clGetDeviceIDs) (cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);

typedef cl_int (*f_clGetDeviceInfo) (cl_device_id, cl_device_info, size_t, void *, size_t *);

typedef cl_int (*f_clCreateSubDevices) (cl_device_id, const cl_device_partition_property *,
					cl_uint, cl_device_id *, cl_uint *);

typedef cl_int (*f_clRetainDevice) (cl_device_id);

typedef cl_int (*f_clReleaseDevice) (cl_device_id);

typedef cl_context (*f_clCreateContext) (const cl_context_properties *, cl_uint, const cl_device_id *,
                			f_pfn_notify, void *, cl_int *);

typedef cl_context (*f_clCreateContextFromType) (const cl_context_properties *, cl_device_type,
                        		f_pfn_notify, void *, cl_int *);

typedef cl_int (*f_clRetainContext) (cl_context);

typedef cl_int (*f_clReleaseContext) (cl_context);

typedef cl_int (*f_clGetContextInfo) (cl_context, cl_context_info, size_t, void *, size_t *);

typedef cl_command_queue (*f_clCreateCommandQueue) (cl_context, cl_device_id, cl_command_queue_properties, cl_int *);

typedef cl_int (*f_clRetainCommandQueue) (cl_command_queue);

typedef cl_int (*f_clReleaseCommandQueue) (cl_command_queue);

typedef cl_int (*f_clGetCommandQueueInfo) (cl_command_queue, cl_command_queue_info, size_t, void *, size_t *);

typedef cl_mem (*f_clCreateBuffer) (cl_context, cl_mem_flags, size_t, void *, cl_int *);

typedef cl_mem (*f_clCreateSubBuffer) (cl_mem, cl_mem_flags, cl_buffer_create_type, const void *, cl_int *);

typedef cl_mem (*f_clCreateImage) (cl_context, cl_mem_flags, const cl_image_format *, const cl_image_desc *, void *, cl_int *);

typedef cl_int (*f_clRetainMemObject) (cl_mem);

typedef cl_int (*f_clReleaseMemObject) (cl_mem);

typedef cl_int (*f_clGetMemObjectInfo) (cl_mem, cl_mem_info, size_t, void *, size_t *);

typedef cl_int (*f_clGetImageInfo) (cl_mem, cl_image_info, size_t, void *, size_t *);

typedef cl_int (*f_clSetMemObjectDestructorCallback) (cl_mem, void (*pfn_notify)( cl_mem memobj, void* user_data), void *);

typedef cl_int (*f_clGetSupportedImageFormats) (cl_context, cl_mem_flags, cl_mem_object_type, cl_uint, cl_image_format *, cl_uint *);

typedef cl_sampler (*f_clCreateSampler) (cl_context, cl_bool, cl_addressing_mode, cl_filter_mode, cl_int *);

typedef cl_int (*f_clRetainSampler) (cl_sampler);

typedef cl_int (*f_clReleaseSampler) (cl_sampler);

typedef cl_int (*f_clGetSamplerInfo) (cl_sampler, cl_sampler_info, size_t, void *, size_t *);

typedef cl_program (*f_clCreateProgramWithSource) (cl_context, cl_uint, const char **, const size_t *, cl_int *);

typedef cl_program (*f_clCreateProgramWithBinary) (cl_context, cl_uint, const cl_device_id *,
        const size_t *, const unsigned char **, cl_int *, cl_int *);

typedef cl_program (*f_clCreateProgramWithBuiltInKernels) (cl_context, cl_uint, const cl_device_id *, const char *, cl_int *);

typedef cl_int (*f_clRetainProgram) (cl_program);

typedef cl_int (*f_clReleaseProgram) (cl_program);

typedef cl_int (*f_clBuildProgram) (cl_program, cl_uint, const cl_device_id *, const char *,
        void (*pfn_notify)(cl_program program, void * user_data), void *);

typedef cl_int (*f_clCompileProgram) (cl_program, cl_uint, const cl_device_id *, const char *, cl_uint, const cl_program *,
        const char **, void (*pfn_notify)(cl_program program, void * user_data), void *);

typedef cl_program (*f_clLinkProgram) (cl_context, cl_uint, const cl_device_id *, const char *, cl_uint, const cl_program *,
                    void (*pfn_notify)(cl_program program, void * user_data), void *, cl_int *);

typedef cl_int (*f_clUnloadPlatformCompiler)(cl_platform_id);

typedef cl_int (*f_clGetProgramInfo) (cl_program, cl_program_info, size_t, void *, size_t *);

typedef cl_int (*f_clGetProgramBuildInfo) (cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);

typedef cl_kernel (*f_clCreateKernel) (cl_program, const char *, cl_int *);

typedef cl_int (*f_clCreateKernelsInProgram) (cl_program, cl_uint, cl_kernel *, cl_uint *);

typedef cl_int (*f_clRetainKernel) (cl_kernel);

typedef cl_int (*f_clReleaseKernel) (cl_kernel);

typedef cl_int (*f_clSetKernelArg) (cl_kernel, cl_uint, size_t,const void *);

typedef cl_int (*f_clGetKernelInfo) (cl_kernel, cl_kernel_info, size_t, void *, size_t *);

typedef cl_int (*f_clGetKernelArgInfo) (cl_kernel, cl_uint, cl_kernel_arg_info, size_t, void *, size_t *);

typedef cl_int (*f_clGetKernelWorkGroupInfo) (cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void *, size_t *);

typedef cl_int (*f_clWaitForEvents) (cl_uint, const cl_event *);

typedef cl_int (*f_clGetEventInfo) (cl_event, cl_event_info, size_t, void *, size_t *);

typedef cl_event (*f_clCreateUserEvent) (cl_context, cl_int *);

typedef cl_int (*f_clRetainEvent) (cl_event);

typedef cl_int (*f_clReleaseEvent) (cl_event);

typedef cl_int (*f_clSetUserEventStatus) (cl_event, cl_int);

typedef cl_int (*f_clSetEventCallback) (cl_event, cl_int, void (*pfn_notify)(cl_event, cl_int, void *), void *);

typedef cl_int (*f_clGetEventProfilingInfo) (cl_event, cl_profiling_info, size_t, void *, size_t *);

typedef cl_int (*f_clFlush) (cl_command_queue);

typedef cl_int (*f_clFinish) (cl_command_queue);

typedef cl_int (*f_clEnqueueReadBuffer) (cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);

typedef cl_int (*f_clEnqueueReadBufferRect) (cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, const size_t *,
                            size_t, size_t, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);

typedef cl_int (*f_clEnqueueWriteBuffer) (cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);

typedef cl_int (*f_clEnqueueWriteBufferRect) (cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, const size_t *,
                            size_t, size_t, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);

typedef cl_int (*f_clEnqueueFillBuffer) (cl_command_queue, cl_mem, const void *, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *);

typedef cl_int (*f_clEnqueueCopyBuffer) (cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *);

typedef cl_int (*f_clEnqueueCopyBufferRect) (cl_command_queue, cl_mem, cl_mem, const size_t *, const size_t *, const size_t *,
                            size_t, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *);

typedef cl_int (*f_clEnqueueReadImage) (cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *,
							size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);

typedef cl_int (*f_clEnqueueWriteImage) (cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *,
							size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);

typedef cl_int (*f_clEnqueueFillImage) (cl_command_queue, cl_mem, const void *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);

typedef cl_int (*f_clEnqueueCopyImage) (cl_command_queue, cl_mem, cl_mem, const size_t *, const size_t *, const size_t *,
          cl_uint, const cl_event *, cl_event *);

typedef cl_int (*f_clEnqueueCopyImageToBuffer) (cl_command_queue, cl_mem, cl_mem, const size_t *, const size_t *, size_t, cl_uint, const cl_event *, cl_event *);

typedef cl_int (*f_clEnqueueCopyBufferToImage) (cl_command_queue, cl_mem, cl_mem, size_t, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);

typedef void * (*f_clEnqueueMapBuffer) (cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t,
						size_t, cl_uint, const cl_event *, cl_event *, cl_int *);

typedef void * (*f_clEnqueueMapImage) (cl_command_queue, cl_mem, cl_bool, cl_map_flags, const size_t *, const size_t *,
                  size_t *, size_t *, cl_uint, const cl_event *, cl_event *, cl_int *);

typedef cl_int (*f_clEnqueueUnmapMemObject) (cl_command_queue, cl_mem, void *, cl_uint, const cl_event *, cl_event *);

typedef cl_int (*f_clEnqueueMigrateMemObjects)(cl_command_queue, cl_uint, const cl_mem *, cl_mem_migration_flags,
						cl_uint, const cl_event *, cl_event *);

typedef cl_int (*f_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *,
                       const size_t *, cl_uint, const cl_event *, cl_event *);

typedef cl_int (*f_clEnqueueTask)(cl_command_queue, cl_kernel, cl_uint, const cl_event *, cl_event *);

typedef cl_int (*f_clEnqueueNativeKernel)(cl_command_queue, void (*user_func)(void *),  void *, size_t,
                      cl_uint, const cl_mem *, const void **, cl_uint, const cl_event *, cl_event *);

typedef cl_int (*f_clEnqueueMarkerWithWaitList)(cl_command_queue, cl_uint, const cl_event *, cl_event *);

typedef cl_int (*f_clEnqueueBarrierWithWaitList)(cl_command_queue, cl_uint, const cl_event *, cl_event *);

typedef void * (*f_clGetExtensionFunctionAddressForPlatform)(cl_platform_id, const char *);

typedef cl_mem (*f_clCreateImage2D)(cl_context, cl_mem_flags,const cl_image_format *, size_t, size_t,
                				size_t, void *, cl_int *);

typedef cl_mem (*f_clCreateImage3D)(cl_context, cl_mem_flags, const cl_image_format *, size_t,
                		size_t, size_t, size_t, size_t, void *, cl_int *);

typedef cl_int (*f_clEnqueueMarker)(cl_command_queue, cl_event *);

typedef cl_int(*f_clEnqueueWaitForEvents)(cl_command_queue, cl_uint, const cl_event *);

typedef cl_int (*f_clEnqueueBarrier)(cl_command_queue);

typedef cl_int (*f_clUnloadCompiler)(void);

typedef void * (*f_clGetExtensionFunctionAddress)(const char *);

typedef cl_mem (*f_clCreateFromGLBuffer) (cl_context, cl_mem_flags, cl_GLuint, int *);

typedef cl_mem (*f_clCreateFromGLTexture) (cl_context, cl_mem_flags, cl_GLenum, cl_GLint, cl_GLuint, cl_int *);

typedef cl_mem (*f_clCreateFromGLRenderbuffer) (cl_context, cl_mem_flags, cl_GLuint, cl_int *);

typedef cl_int (*f_clGetGLObjectInfo) (cl_mem memobj, cl_gl_object_type *, cl_GLuint *);

typedef cl_int (*f_clGetGLTextureInfo) (cl_mem, cl_gl_texture_info, size_t, void *, size_t *);

typedef cl_int (*f_clEnqueueAcquireGLObjects) (cl_command_queue, cl_uint, const cl_mem *, cl_uint,
                                        const cl_event *, cl_event *);

typedef cl_int (*f_clEnqueueReleaseGLObjects) (cl_command_queue, cl_uint, const cl_mem *, cl_uint,
                                        const cl_event *, cl_event *);

typedef cl_mem (*f_clCreateFromGLTexture2D) (cl_context, cl_mem_flags, cl_GLenum, cl_GLint, cl_GLuint, cl_int *);

typedef cl_mem (*f_clCreateFromGLTexture3D) (cl_context, cl_mem_flags, cl_GLenum, cl_GLint, cl_GLuint, cl_int *);

//typedef cl_uint     cl_gl_context_info;
typedef cl_int (*f_clGetGLContextInfoKHR) (const cl_context_properties *, cl_gl_context_info, size_t,
                                        void *, size_t *);

// Additional api to reset currently opened opencl shared-object
// Subsequent calls will use newly set environment variables
void stubOpenclReset();

// Helper function to get the path to libOpenCL.so
int open_libopencl_so();
cl_int get_libopencl_path(char** cl_path);

#ifdef __cplusplus
}
#endif

#endif    // LIBOPENCL_STUB_H
