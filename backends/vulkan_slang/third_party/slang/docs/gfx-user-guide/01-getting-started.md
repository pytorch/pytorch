---
layout: user-guide
---

Getting Started with Slang Graphics Layer
============================================

[//]: # (ShortTitle: Getting Started)

In this article, we provide instructions on installing the graphics layer into your application, and demonstrate the basic use of the graphics layer via a simple compute shader example. We will use the same [hello-world.slang](https://github.com/shader-slang/slang/blob/master/examples/hello-world/hello-world.slang) shader from the `hello-world` example in the [Slang getting started tutorial](../user-guide/01-get-started.html).

Installation
------------------

### Obtain Release Package

The Slang graphics library is implemented in `gfx.dll` (`libgfx.so` in unix systems). Since Slang is tightly integrated into the graphics layer, you need to include both `slang.dll` and `gfx.dll` in your application. Official Slang releases provide prebuilt binaries for both libraries as well as the header files to use them. If you prefer to build the libraries yourself, please follow [build instructions](../building).

### Install Header Files
Once you have built or obtained a Slang release, make the following header files from the release package accessible to your application:
- `slang-gfx.h`
- `slang.h`
- `slang-com-ptr.h`
- `slang-com-helper.h`

### Linking the Library
On Windows (with `msvc`), make sure that `gfx.lib` is provided as linker input via the `Linker->Input->Additional Dependencies` project configuration. On Unix systems, make sure to pass `-lgfx` when compiling your application.

Creating a GPU Device
---------------------------

To start using the graphics layer, create an `IDevice` object by calling `gfxCreateDevice`. The `IDevice` interface is the main entry-point to interact with the graphics layer. It represent GPU device context where all interactions with the GPU take place.

```cpp
#include "slang-gfx.h"

using namespace gfx;

IDevice* gDevice = nullptr;

void initGfx()
{
    IDevice::Desc deviceDesc = {};
    gfxCreateDevice(deviceDesc, &gDevice);
}
```

The `IDevice::Desc` struct passed to `gfxCreateDevice` defines many configurations on how a device shall be created. Most notably, the `deviceType` field specifies what underlying graphics API to use. By default, `gfxCreateDevice` will attempt to use the best API available on current platform. On Windows, the layer will prefer to use `D3D12` but will also try to use `Vulkan`, `D3D11`, `OpenGL` in order, in case the former API isn't available. On Unix systems, it will always default to `Vulkan` since this is the only API that supports full Graphics capabilities. A user can always specify the `deviceType` field to force the layer to use a specific API. If the device creation succeeds, `gfxCreateDevice` will return `SLANG_OK(0)`.

Similar to the Slang API, objects created by the graphics layer also conforms to the COM standard. The user to responsible for calling `release` method on every object returned to the user by the layer to prevent memory leaks.

Enabling the Debug Layer
--------------------------

The Slang Graphics Layer provides a debug layer that can be enabled to perform additional validations to ensure correctness. To enable the debug layer, simply call `gfxEnableDebugLayer` before calling `gfxCreateDevice`.

To receive diagnostic messages, you need to create a class that implements the `IDebugCallback` interface, and call `gfxSetDebugCallback` to provide the callback instance to the graphics layer. For example:

```cpp
struct MyDebugCallback : public IDebugCallback
{
    virtual SLANG_NO_THROW void SLANG_MCALL handleMessage(
        DebugMessageType type,
        DebugMessageSource source,
        const char* message) override
    {
        printf("%s\n", message);
    }
};

MyDebugCallback gCallback;
void initGfx()
{
    gfxEnableDebugLayer();
    gfxSetDebugCallback(&gCallback);

    IDevice::Desc deviceDesc = {};
    gfxCreateDevice(&deviceDesc, &gDevice);
}
```


Creating a Command Queue
------------------------------
A command queue is where the GPU device takes commands from the application to execute. To create a command queue, call `IDevice::createCommandQueue`.
```cpp
ICommandQueue* gQueue = nullptr;

ICommandQueue::Desc queueDesc = {ICommandQueue::QueueType::Graphics};
device->createCommandQueue(queueDesc, &gQueue);
```

Allocating a Command Buffer
------------------------------
A command buffer is treated as a _transient_ resource by the graphics layer. A transient resource is required by the GPU during execution of a task, and are no longer needed when the execution has completed. Slang graphics layer provides an `ITransientResourceHeap` object to efficiently manage the life cycle of transient resources. In order to allocate a command buffer, we need to create an `ITransientResourceHeap` object first by calling `IDevice::createTransientResourceHeap`.

```cpp
ITransientResourceHeap* gTransientHeap;

ITransientResourceHeap::Desc transientHeapDesc = {};
transientHeapDesc.constantBufferSize = 4096;
device->createTransientResourceHeap(transientHeapDesc, &gTransientHeap);
```

With a `TransientResourceHeap`, we can call `createCommandBuffer` method to allocate a command buffer:

```cpp
ICommandBuffer* commandBuffer;
gTransientHeap->createCommandBuffer(&commandBuffer);
```

A user should regularly call `ITransientResourceHeap::synchronizeAndReset` to recycle all previously allocated transient resources. A standard practice is to create two `TransientResourceHeap`s in a double-buffered renderer, and alternate the transient heap on each frame to allocate command buffers and other transient resources. With this setup, the application can call `synchronizeAndReset` at start of each frame on the corresponding transient resource heap to make sure all transient resources are timely recycled.

Creating Buffer Resource
------------------------------
We need to create the buffer resources used our `hello-world` shader as input and output. This can be done via `IDevice::createBufferResource` method. When creating a resource, the user must specify a resource state that the resource will be in by default, as well as all allowed resource states the resource can be in. Resource states in the graphics layer follows the same model of resource states in D3D12, and the user can also assume the same automatic resource promotion/demotion behavior in D3D12.

```cpp
const int numberCount = 4;
float initialData[] = {0.0f, 1.0f, 2.0f, 3.0f};
IBufferResource::Desc bufferDesc = {};
bufferDesc.sizeInBytes = numberCount * sizeof(float);
bufferDesc.format = Format::Unknown;
bufferDesc.elementSize = sizeof(float);
bufferDesc.defaultState = ResourceState::UnorderedAccess;
bufferDesc.allowedStates = ResourceStateSet(ResourceState::UnorderedAccess,
                                            ResourceState::ShaderResource);
IBufferResource* inputBuffer0;
SLANG_RETURN_ON_FAIL(device->createBufferResource(
    bufferDesc,
    (void*)initialData,
    &inputBuffer0));
```


Creating a Pipeline State
---------------------------

A pipeline state object encapsulates the shader program to execute on the GPU device, as well as other fix function states for graphics rendering. In this example, we will be compiling and running a simple compute shader written in Slang. To do that we need to create a compute pipeline state from a Slang `IComponentType`. We refer the reader to the (Slang getting started tutorial)[../user-guide/01-getting-started.html] on how to create a Slang `IComponentType` from a shader file. The following source creates a Graphics layer `IPipelineState` object from a shader module represented by a `slang::IComponentType` object:

```cpp
void createComputePipelineFromShader(
    IComponentType* slangProgram, 
    IPipelineState*& outPipelineState)
{
    // The `IComponentType` parameter that represents the compute
    // kernel, we can use it to create a `IShaderProgram` object in the graphics
    // layer.
    IShaderProgram* shaderProgram = nullptr;
    IShaderProgram::Desc programDesc = {};
    programDesc.pipelineType = PipelineType::Compute;
    programDesc.slangProgram = slangProgram;
    gDevice->createShaderProgram(programDesc, &shaderProgram);
    
    // Create a compute pipeline state from `shaderProgram`.
    ComputePipelineStateDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram;
    gDevice->createComputePipelineState(pipelineDesc, &outPipelineState);

    // Since we no longer need to use `shaderProgram` after creating
    // a pipeline state, we should release it to prevent memory leaks.
    shaderProgram->release();
}
```

Recording Commands to Run a Compute Shader
------------------------------------

[//]: # (ShortTitle: Recording Commands)

Now that we have created all the resources and allocated a command buffer, we can start recording commands to
set the compute pipeline state, bind shader parameters, and dispatch a kernel launch.

Since we are only using compute commands, we begin the recording by calling `ICommandBuffer::encodeComputeCommands`. This methods returns a transient `IComputeCommandEncoder` object for accepting actual compute commands.

```cpp
IComputeCommandEncoder* encoder = commandBuffer->encodeComputeCommands();
```

The first command is to bind the pipeline state we created earlier:

```cpp
IShaderObject* rootObject = encoder->bindPipeline(pipelineState);
```

Binding a pipeline state yields a transient `IShaderObject` object. We can use the `IShaderObject` instance to bind shader parameters. For the `hello-world` shader, we need to bind three parameters: `buffer0`, `buffer1` and `result`.

```cpp
// Create a resource view for buffer0.
IBufferView* buffer0View;
{
    IResourceView::Desc viewDesc = {};
    viewDesc.type = IResourceView::Type::ShaderResource;
    viewDesc.format = Format::Unknown;
    SLANG_RETURN_ON_FAIL(device->createBufferView(inputBuffer0, viewDesc, &buffer0View));
}
// Bind the resource view to shader.
rootObject->setResource(ShaderOffset{0,0,0}, buffer0View);

// Create a resource view for buffer1.
IBufferView* buffer1View;
{
    IResourceView::Desc viewDesc = {};
    viewDesc.type = IResourceView::Type::ShaderResource;
    viewDesc.format = Format::Unknown;
    SLANG_RETURN_ON_FAIL(device->createBufferView(inputBuffer1, viewDesc, &buffer1View));
}
// Bind the resource view to shader.
rootObject->setResource(ShaderOffset{0,1,0}, buffer1View);

// Create a resource view for resultBuffer.
IBufferView* resultView;
{
    IResourceView::Desc viewDesc = {};
    viewDesc.type = IResourceView::Type::UnorderedAccess;
    viewDesc.format = Format::Unknown;
    SLANG_RETURN_ON_FAIL(device->createBufferView(resultBuffer, viewDesc, &resultView));
}
rootObject->setResource(ShaderOffset{0,2,0}, resultView);
```

> #### Note
> Since `rootObject` is a transient object returned by the command encoder, it is automatically released
> with the command encoder. Calling `release` on `rootObject` is OK but not needed.

After binding all shader parameters, we can now dispatch the kernel:

```cpp
encoder->dispatchCompute(1, 1, 1);
```

> #### Note
> Command encoders are transient objects managed by a command buffer, it is automatically released
> with the command buffer. Calling `release` on `rootObject` is OK but not needed.

When we are done recording commands, we need to close the command encoder and the command buffer.

```cpp
encoder->endEncoding();
commandBuffer->close();
```

Now we are ready to submit the command buffer to the command queue, and wait for the GPU execution to finish.
```cpp
gQueue->executeCommandBuffer(commandBuffer);
gQueue->wait();
```

Cleaning Up
----------------

At the end of our example, we need to make sure all created objects are released by calling the `release` method:

```cpp
commandBuffer->release();
gQueue->release();
gTransientResourceHeap->release();
inputBuffer0->release();
buffer0View->release();
...
gDevice->release();
```

The order of calls to `release` does not matter, as long as all objects are released from the user.
