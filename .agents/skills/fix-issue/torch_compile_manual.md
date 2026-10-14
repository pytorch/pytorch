# torch.compile, the missing manual

You are here because you want to use [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) to make your PyTorch model run faster. torch.compile is a complex and relatively new piece of software, and so you are likely to have growing pains. This manual is all about how to resolve problems that may arise when working with torch.compile, including both bugs in PyTorch itself, as well as fundamentally difficult problems that require some care from the user.

This manual’s focus is for technical end users who don’t know much about PyTorch’s internals, but do understand their model and are willing to interact with PyTorch developers via GitHub. You don’t need to read this manual end-to-end; use “View \> Show outline” to skip to the sections relevant to you.

**NOTICE:** As of Jul 5, 2024, this document is about bleeding edge PyTorch nightlies. Although some of the guidance here will work for older versions of PyTorch, I often find myself submitting patches to PyTorch while I am working on this manual to make things better. At some point, we’ll change this notice to a particular version of PyTorch and start versioning advice, but right now assume you want the latest PyTorch.

# Setting expectations

## The three regimes of enablement

There is a huge amount of variety of model architectures in PyTorch. While torch.compile is intended to be a general purpose compiler, it’s very easy to end up “out of distribution”; when helping people enable their models with torch.compile, I’ve noticed that models tend to fall into one of three different regimes:

1. **It just works.** There are a few cases when this tends to happen: (1) your model is one we’ve specifically spent time making sure works well, (2) your model is not too complicated and is written in simple, idiomatic code  or (3) the model was written from the ground up to be torch.compile friendly (e.g., gpt-fast, torchao).

2. **It works with a little work.** In real world settings, you may have all sort of fuzz that makes things a little less likely to work. Maybe you’re using some third party libraries that are using fancy Python features, or you’re a standard transformer model but you’ve got some modeling innovations in one part. But fundamentally, your model is similar to ones that work well. In this case, you may have to do some work, rewrite some code to work around compiler bugs, but you should be able to get to torch.compile with minimal investment.

3. **It’s going to be a slog.** Your model is doing strange things. You’re trying to highly optimized eager code that is doing strange things in backward hooks, or you are trying to take advantage of sparsity with data-dependent computation, or your code has been written taking full advantage of PyTorch’s eager nature, weaving in and out of Tensor and plain Python compute.

If you are in regime (3), expect to spend a lot of time working with the PyTorch team fixing bugs. There may be easily dozens of bugs that may need to be fixed before you can even get performance numbers, and even after the compiler is no longer crashing, you may still need to do more work to ensure accurate results and good performance. It is totally reasonable to give up and go do something else. However, I will say that a lot of the ongoing work on torch.compile in PyTorch is about making these sorts of use cases well supported (a lot of Meta’s recommendation models fall into category (3)\!) and so, if you are willing to put in the investment, maybe do some codevelopment with torch.compile, there might be some really nice speedups waiting for you at the end of the tunnel.

## What you should expect to be compilable

Within a training workflow, there are distinct parts to your training script. While we normally think of the model as the forwards definition in an nn.Module, there is also the backwards, optimizer step, etc. Actually, you can compile most of these things. Here is some more details:

* The traditional, bread and butter use case for torch.compile is compiling the **nn.Module** definition of your model. A model compiled this way will produce an optimized forward and backward, analogous to if you had taken the nn.Module and replaced it with a custom autograd function with an optimized forward and backward. This is the very first thing in torch.compile that we got working and it’s the one that works the best. You can compile a module by doing torch.compile(...)(nn\_module) (which constructs a wrapper NN module around the original), but another common pattern is to compile the forward method only with nn\_module.forward \= torch.compile(...)(nn\_module.forward).

* It is also supported to run **compiled optimizers**, by putting torch.compile around the optimizer step call (e.g., as seen in [https://pytorch.org/tutorials/recipes/compiling_optimizer.html](https://pytorch.org/tutorials/recipes/compiling_optimizer.html) and [https://pytorch.org/tutorials/recipes/compiling_optimizer_lr_scheduler.html](https://pytorch.org/tutorials/recipes/compiling_optimizer_lr_scheduler.html)). Optimizers have some peculiarities which took us some time to work out (most of it landed by now--Adam, AdamW, SGD, RAdam and Adamax are all expected to work):
  * It’s pretty common for optimizers to do a lot of computation on plain Python int/float. As of Jul 7, 2024, Dynamo doesn’t do a great job capturing these. Most optimizers have capturable variants that do computation on tensors instead (capturable optimizers are also good for eager mode, as CUDA graphs works on them). Similarly, LR schedulers typically have a float learning rate; this must be [wrapped a Tensor to work with torch.compile](https://pytorch.org/tutorials/recipes/compiling_optimizer_lr_scheduler.html). There is work scheduled in H2 2024 to remove this limitation: [https://github.com/pytorch/pytorch/issues/107277](https://github.com/pytorch/pytorch/issues/107277)
  * Hypothetically, an optimizer can be written naively updating each parameter one-by-one, and a compiler should be able to horizontally fuse these updates. In practice, it is much better if the optimizers directly use foreach kernels, as forcing Inductor to reverse engineer the horizontal fusion is quite bad for compile time (a double whammy of there being *lots* of parameters that need to be fused together, and non-linear asymptotics associated with the fusion pass.) You should expect torch.compile to also perform vertical fusion on these updates (the usual source of performance improvements on compiled optimizers.)

* Typically, the act of compiling a module also gives you the compiled backwards. However, you can also directly run **compiled autograd** with [torch.\_dynamo.compiled\_autograd](https://github.com/pytorch/pytorch/blob/8d6e34b21b94df5b25ba5904e387d3261faecc55/test/inductor/test_compiled_autograd.py) which will directly compile the autograd graph executed by a backward() call. There are three situations when you might want to use this: (1) you will get some performance out of the box as accumulate grad nodes can be fused into the compiled region (they can’t with traditional AOTAutograd), (2) your forwards graph cannot be compiled all in one go because it has some dynamism, but your backwards graph is always the same each iteration, (3) you are making use of nontrivial autograd features like hooks which cannot be compiled ahead of time, and whose compilation must be deferred until you actually run backward(). This is especially common when compiling distributed wrappers like FSDP.

* **Logging** typically induces a graph break, since they represent side effects that are not currently representable in PT2 IR. However, with the configuration setting torch.\_dynamo.config.reorderable\_logging\_functions you can specify some functions (like print or warnings.warn) to be reorderable to the end of a compiled region. This includes logging functions which print out Tensors. Note that logging in this way can change the performance characteristics of your code, since Tensors that previously were not materialized may need to be materialized for logging purposes. Additionally, the logs are only ever printed at the *end* of execution, so if a buffer is mutated, you will see the output after mutation. See also [https://github.com/pytorch/pytorch/pull/116106](https://github.com/pytorch/pytorch/pull/116106)

Some things that don’t work:

* **Single step capture** refers to a hypothetical Dynamo capture strategy which is able to capture the forwards, backwards and optimizer step in a single graph. There is an RFC for this functionality [https://github.com/pytorch/pytorch/issues/117394](https://github.com/pytorch/pytorch/issues/117394) but implementation is still in progress as of Jul 7, 2024\.

* **Preprocessing**, in our experience, often involves a lot of bespoke custom operators for doing domain specific operations. While PT2 can in principle capture custom operators, you aren’t necessarily going to get any runtime speedup from doing so. Hypothetically, PT2 might be pretty good at certain kinds of preproc, but we haven’t seen all that much usage in this niche. Update: As of Aug 15, 2024, there’s quite a lot of interest in “native PyTorch preprocessing” at Meta, so get in touch if this is something you do want to compile\!

# Common debugging strategies

In the next section, I’m going to talk through some common classes of problems that you might run into when using torch.compile. This section, however, talks about common debugging strategies that are applicable to *all* different problems.  I’ve ordered them from least effort to most effort, so do things that come earlier first\!

## Run TORCH\_TRACE on the model and look at it in tlparse

TORCH\_TRACE/tlparse are a pair of tools that produce compilation reports that look like this: [https://web.mit.edu/~ezyang/Public/bhack-20240609-tlparse/index.html](https://web.mit.edu/~ezyang/Public/bhack-20240609-tlparse/index.html)

Traces are very easy to collect. To collect a trace, run your reproduction command with

```py
TORCH_TRACE="/tmp/tracedir" python foo.py
pip install tlparse # if needed (in Meta, do `feature install tlparse`)
tlparse /tmp/tracedir
```

This will work even if you are running a distributed job (you will get a trace per rank). This will open up your browser with HTML like generated above. If you are making a bug report for a complicated problem that you don’t have a standalone reproduction for, you can still greatly assist PyTorch developers by attaching the trace log generated in /tmp/tracedir. **WARNING: The trace log contains all of your model code.** Do not share the trace log if the model you are working on is sensitive. The trace log does NOT contain weights. The tracing process is low enough overhead that you should also consider enabling it by default for production runs using torch.compile (this is what we do internally at Meta).

The output of tlparse is mostly oriented at PyTorch developers, and the log format is very easy to upload and share on GitHub. However, you can still get some useful information from it as a non-PyTorch developer. First, I recommend reading the help text that is inline in the report, it helps explain what the report means. Here are some things you can get from a tlparse:

* What model code did I actually torch.compile, by looking at the stack trie? (This is especially useful if you’re not familiar with the codebase being compiled\!)

* How many graph breaks / distinct compilation regions do I have? (Each distinct compile is its own color coded block like [\[0/0\]](https://web.mit.edu/~ezyang/Public/bhack-20240609-tlparse/index.html#[0/0])). Frames that are potentially graph break’ed are light green [\[2/4\]](https://web.mit.edu/~ezyang/Public/bhack-20240609-tlparse/index.html#[2/4]). If there are a lot of frames, that is suspicious, and suggests that you have had some catastrophic graph breaks, or maybe your code isn’t a good match for torch.compile.

* How many times did I recompile a particular frame? Something that recompiled a lot will look like: [\[10/0\]](https://web.mit.edu/~ezyang/Public/bhack-20240609-tlparse/index.html#[10/0]) [\[10/1\]](https://web.mit.edu/~ezyang/Public/bhack-20240609-tlparse/index.html#[10/2]) [\[10/2\]](https://web.mit.edu/~ezyang/Public/bhack-20240609-tlparse/index.html#[10/4]) \-- if something is being recompiled a lot, that is very suspicious and worth looking into, even if it isn’t the root cause of your problem.

* Was there a compilation error?  Frames that errored will look like [\[0/1\]](https://web.mit.edu/~ezyang/Public/bhack-20240609-tlparse/index.html#[0/1])

* What intermediate compiler products did I generate for a given frame? I find people especially like looking at the Triton code in inductor\_output\_code\_\*, since if you can correlate a given Triton kernel with the user code that generated it, expert users can simply look and see what code PT2 is generating.

* Are there relevant information for a particular frame? You can find these in compilation\_metrics.

If you already have a PT2 model deployed in production, it is possible for it to stop working when you upgrade to another PyTorch release/nightly. Comparing the tlparses of the working and non-working versions is a good way to get started on debugging, I usually put them in two side-by-side browser windows and scroll down them successively, doing a visual diff.

**Meta only:** TORCH\_TRACE is enabled by default in fbcode on MAST jobs. After running sudo feature install tlparse to install the fbcode specific version of tlparse, you can get the tlparse of a job by running tlparse $MAST\_ID or tlparse $MAST\_URL, this will automatically upload the report to a CDN URL which you can use to view it directly.  If you are running Conda on MAST, we only recently (Jul 2, 2024\) turned on TORCH\_TRACE by default; if your PyTorch package is too old, you can run your command with \--env="TORCH\_TRACE=/logs". Once you have run with this, you can use tlparse like above to get a report.

## Do an ablation {#do-an-ablation}

If PyTorch crashes, it’s usually pretty obvious what caused the crash. But if your outputs are garbage, this is kind of the nightmare scenario where there’s no where to start looking for the problem. It also is a situation where you’re unlikely to be able to create a minimal reproducer.

The very first thing you should do in this situation is an ablation: rerun your workload disabling various pieces of the compiler stack and isolate which component is causing the problem.

There are two main ways you can ablate:

* You can disable layers of the compiler stack (e.g., disable inductor) and try to localize whose fault it is. To ablate in this way, modify the backend kwarg of your torch.compile invocation. There are three settings we recommend testing:

  * backend=”eager”: if this fails, this tells us it is a Dynamo problem
  * backend=”aot\_eager”: if this fails, but not eager, this tells us it is an AOTAutograd problem
  * backend=”aot\_eager\_decomp\_partition”: if this fails, but not aot\_eager, this tells us it is a problem related to our decompositions/partitioner

  Additionally, if you are running with mode=”reduce-overhead”, you should try without it (if it’s a CUDA graphs problem). If you are running with dynamic=True, you should try without it (if it’s a dynamic shapes problem.) If you suspect that your problem is related to a particular FX pass, you can also manually disable the pass by commenting it out or disabling the relevant config, e.g., torch/\_inductor/fx\_passes/joint\_graph.py or torch/\_inductor/fx\_passes/post\_grad.py (TODO: optimization fuel so you don’t have to know the internals to do this)

* You can disable the compiler of layers of your model. You can do this manually by pushing your torch.compile invocation inside your model, or you can do it programmatically using a patch like [https://gist.github.com/ezyang/4a5138b11327335e618dd37ad2fd0a4e](https://gist.github.com/ezyang/4a5138b11327335e618dd37ad2fd0a4e)  (TODO: land this properly)

There are other ways to diagnose accuracy problems (e.g., comparing outputs layer by layer), but ablations are easy to do and not labor intensive, so you should do them first.

## The minifier / automatic repro generator

I’m almost loathe to put this here, because most of the time the repro generator doesn’t work and you’ll have to do something else. (I like to think that this is because bugs that can have repros automatically generated this way are all easy to fix, so we’ve fixed them all, and that leaves the hard bugs that don’t repro easily.) But it’s very easy to try, so you might as well try it and cry when it doesn’t work.

Instructions for operating the minifier are at [https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html](https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html) . If the compiler is crashing, you can set TORCHDYNAMO\_REPRO\_AFTER="dynamo" or TORCHDYNAMO\_REPRO\_AFTER="aot" (aot is more likely to work, but it won’t catch AOTAutograd bugs) and then pray that the generated repro.py actually has your problem. If it’s an accuracy problem, you can try TORCHDYNAMO\_REPRO\_LEVEL=4 (and cry when it fails to find the actual subgraph that has a problem).

## Check for recent feature flag changes

If a PT2 run used to be working, and suddenly it is not working, even though you didn’t do a code deploy, chances are someone changed a feature flag that changed some behavior. By default, OSS PyTorch isn’t configured to point at feature flags, but we have extension points which you could use to integrate with your own feature flag system.

**Meta-only:** Our feature flags can be viewed at [https://www.internalfb.com/intern/justknobs/?name=pytorch](https://www.internalfb.com/intern/justknobs/?name=pytorch) ; specifically, you are likely to be most interested in [https://www.internalfb.com/intern/justknobs/?name=pytorch%2Fdynamo](https://www.internalfb.com/intern/justknobs/?name=pytorch%2Fdynamo) and [https://www.internalfb.com/intern/justknobs/?name=pytorch%2Fremote_cache](https://www.internalfb.com/intern/justknobs/?name=pytorch%2Fremote_cache) . Some of our knobs use a “version” concept, which is that rather than a TRUE/FALSE killswitch, they are associated with a number, which is bumped up in source code when we want to do a new rollout. In this case, to undo a rollout, you want to revert the number to its previous value. You can view the changelog across all JKs in pytorch/ at [https://www.internalfb.com/code/configerator/[history\]/source/justknobs/pytorch/](https://www.internalfb.com/code/configerator/[history]/source/justknobs/pytorch/) and you can also use landline to plot out changes to JKs, as well as other things: [https://fburl.com/canvas/29uskthn](https://fburl.com/canvas/29uskthn)

## Bisect

Bisecting is a good idea if something used to work and now it doesn’t. Bisecting is kind of painful in open source, since you have to figure out where to download old nightlies or build PyTorch from source, but we use this strategy a lot at Meta and so I would remiss not to mention it.

## Create a standalone reproducer of your workflow

Creating reproducers is a lot of work, and it is 100% OK if you do not have time to do it. But if you are a very motivated user who doesn’t know very much about PT2, creating a standalone reproducer can have a huge impact on our ability to fix the bug. Without a reproducer, your bug report has to have enough information that we can root cause the problem and write a reproducer from scratch.

Not all reproducers are created equal, but even a crappy reproduction can be helpful in the right circumstances. Here’s a list of useful reproducers, with the most preferred first.

1. A self-contained (no external dependencies), small (less than 100 LOC) reproduction script that when run produces the problem.

2. A self-contained but large reproducer. Being self-contained is a huge win\!

3. A not self-contained reproducer that is not too sensitive to the dependencies used. For example, if you can reproduce a problem if you first \`pip install transformers\` and then run a script and it will produce the problem, that’s not too bad, we will probably be able to run it and check things out.

4. A not self-contained reproducer that requires substantial environmental setup / a Docker image to reproduce. For example, maybe you need us to download a dataset from some URL, or do multiple nontrivial environment setup steps, or the it is very important to have very particular versions of system libraries so a Docker image is required. The more difficult it is to setup the environment, the harder it is for us to recreate it and setup the problem. NB: Docker makes it “easier” to setup the environment, but it makes it more difficult to change things about the environment / use our preferred development environment, so it’s not really a magic bullet, although we’ll take it in a pinch.

Somewhat orthogonally, a reproducer that can be run in a single process is better than a reproducer that requires multiprocess training (but once again, if you only have a multiprocess reproducer, we’ll take it\!)

For Dynamo related problems (e.g., we don’t support some Python construct), it’s often not necessary to have a reproducer, it’s usually self explanatory what is not implemented.

TODO: Do some of the ideas in [https://github.com/pytorch/pytorch/issues/126644](https://github.com/pytorch/pytorch/issues/126644) to make getting these out of tlparse easier

## Create a test case from the bug report

Sometimes, once we have a hypothesis for the root cause of a bug report, it’s often possible to write a test case from scratch to tickle the problem in question. This requires some more inside knowledge about PT2, especially what tools are available to exercise certain codepaths in PT2, but it’s also an extremely useful thing to do. As a PyTorch developer, I often find it is more time consuming to write a test case for a fix than it is to do the fix itself.

tlparse is your best friend for writing test cases, as you can use it to conveniently inspect the IR generated in the real world occurrence of the bug, and it may give you ideas for how to trigger it on its own. Here is a nonexhaustive list of things to check if you think the bug is related to certain subsystems:

* **Autograd.** Did you have tensor inputs with requires\_grad=True? Did you call backward() on the output?

* **Dynamic shapes.** Did you set dynamic=True? Or did you run the test code multiple times with varying shapes?

* **Custom operators.** Is there a custom operator involved in the real workflow? Can you replicate some of its important characteristics using the [Python custom operator API](https://pytorch.org/docs/main/notes/custom_operators.html)?

* **Configuration.** Did you set all the same configuration? This includes torch.\_dynamo.config and torch.\_inductor.config settings, as well as arguments to torch.compile like backend/mode.

* **Context managers.** Did you replicate any active context managers? This could be torch.no\_grad, automatic mixed precision, TorchFunctionMode/TorchDispatchMode, activation checkpointing, compiled autograd etc.

* **Tensor subclasses.** Is there a tensor subclass involved?

* **Partitioner.** If the bug is related to what the partitioner decides to save for backwards, consider writing a custom op with its own backward rule so you can control what is saved for backwards.

# The compiler crashes

## Should you even expect your program to compile?

At a high level, you should only expect programs that correspond to a fixed sequence of PyTorch tensor operations to compile in a useful way. Importantly, this sequence of tensor operations must stay the same from run to run. This is different from TorchScript, where if you made your code TorchScriptable, you had some access to nontrivial Python language features such as loops and Python data structures like lists, which you could reasonably expect to be captured by the compiler and execute in the runtime. In PT2, our expectation is that those parts of the program are just run by the *normal* CPython interpreter, and torch.compile is used on the “tensor compute.” Even simple things like indexing into a Python list by a dynamic index, or taking a list of Tensors which may vary in size, do not work with PT2. If this is a sticking point for your model, you should perhaps consider using something like Mojo.

This is an important question to ask in the context of compiler crashes, because code that is using fancy Python features or manipulating Python data structures is much more likely to cause a Dynamo crash, and here, the correct answer is not “try to bash through the crash” but “reevaluate what you should actually expect to compile.” So better ask this question before you spend a bunch of time trying to make the compiler happy\! (Even if you *did* manage to make it compile, chances are you would keep recompiling on every iteration, thereby making compilation useless.)

I have found that sometimes, it’s not so easy to tell if you should expect your model to be compilable. Fortunately, there’s also a mechanical way to check. Use something like LoggingMode ([https://github.com/albanD/subclass_zoo/blob/main/logging_mode.py](https://github.com/albanD/subclass_zoo/blob/main/logging_mode.py)) to trace the tensor operations on your model; do this multiple times over a representative distribution of inputs. If all the traces are the same, you probably can compile (and Dynamo’s job is to capture this trace in a sound way); if the traces are different, you probably can’t.

If your code is using Python data structures in a nontrivial way, but you are very motivated to use torch.compile, the name of the game is figuring out how to represent those data structures as just plain Tensors. For example, variable length sequence inputs could naively be represented as a Python list of tensors with varying length. This cannot be compiled with a variable batch size, since varying the batch size would vary the length of the Python list (not allowed). However, an abstraction like nested tensor packs the tensors into a single tensor with extra metadata recording the boundaries of each sequence; this packed representation *is* torch.compile’able\! (As a side note: it *is* possible to compile a list of variable length tensors if you fix the size of this list. But if the list is large, this will end up being a lot of tensors and a lot of compute on each tensor, and the compiler may run quite slowly in this case. See also [Loops are unrolled](#loops-are-unrolled!))

## Types of compiler crashes

It’s important to distinguish between three types of compiler crashes:

* **Crashes due to incompleteness.**  These errors typically look like “torch.\_dynamo.exc.Unsupported: call\_method”, although sometimes you’ll see an AttributeError or something similar raised from Dynamo code, where on closer inspection it looks like we just forgot to account for a case. These problems are very common, but they don’t indicate some deep problem, just some feature work that needs to be done. You can often just file the error message and the exception and that will be enough for us to do something about it. You can workaround these by permitting graph breaks with fullgraph=False, or by selectively [disabling compilation](https://pytorch.org/docs/stable/generated/torch.compiler.disable.html) from parts of your model.

* **Crashes due to bugs.** Something went wrong. We’d like to fix it. As much information / reproducer is helpful in the bug report here.

* **Crashes due to user problems.** These error messages tend to be long and have lots of information. Some user education is required. For example, when working with data dependent operators, you may need to consult [Dealing with GuardOnDataDependentSymNode errors](https://docs.google.com/document/d/1HSuTTVvYH1pTew89Rtpeu84Ht3nQEFTYhAX3Ypa_xJs/edit#heading=h.44gwi83jepaj). Sometimes, these require framework fixes, but other times, you need to write your code differently.

The number of **graph breaks** in your model can materially affect what kinds of crashes you see. A model that graph breaks a lot is more likely to end up with lots of small, dynamic fragments of code to be compiled, and stresses the compiler in different ways than a single model that has captured everything. The logic in Dynamo for inserting graph breaks and reconstructing Python state in this case is more complicated, and so you are more likely to run into bugs here. If possible, reducing the number of graph breaks in your model can also make it less likely for us to crash.

It is also possible for us to **crash at runtime** due to faulty generated code. TORCH\_TRACE/tlparse can make it more convenient to inspect the generated code and look at what exactly has gone wrong; a bug report with the misgenerated code is often enough to diagnose the problem, without a full reproducer.

## Working around compiler crashes

For some model architectures, there are portions of the model which are particularly difficult to compile. You may want to *explicitly* disable these portions of the model which are problematic so that you can apply PT2 to the parts that work. There are a few ways to do this:

1. You can move torch.compile inside your model, so that you are only compiling specific modules rather than the entire top level model. For example, PT2 doesn’t have great support today for distributed wrapper modules like DDP or FSDP, so you just torch.compile the inner module you pass into these wrappers. Similarly, if it’s unrealistic to use torch.compile for your entire module, you may find specific modules like transformer blocks which you could compile and get wins just for those segments.

2. Alternatively, you can continue to use torch.compile at the very top-level of your module, and instead use @torch.\_dynamo.disable() decorator to disable PT2 on specific portions of your model. While this is, in some sense, equivalent to compiling only specific modules (because all disable does is induce a graph break when you reach the disabled module), it can be more convenient to say what is NOT compilable as opposed to what IS compilable. (Also, when we fix [https://github.com/pytorch/pytorch/issues/111003](https://github.com/pytorch/pytorch/issues/111003) you could end up with more fusion opportunities with a top-level torch.compile than with individual torch.compile annotations, as torch.compile annotations have to respect the call stack structure.) For example, we use this to disable PT2 on sparse architecture in recommendation models, as the sparse arch is difficult to compile.

There are some configuration flags which are more likely to cause compiler crashes. In particular, if you are crashing due to dynamic=True, try removing this flag (and relying on automatic dynamic to detect if you should be dynamic).

If you want to YOLO keep going even when the compiler crashes, you can set torch.\_dynamo.config.suppress\_errors \= True. Whenever the compiler crashes, we will just skip the frame and try again later. It’s best to eventually manually add disable annotations as necessary.

## Printing things out at compile time

Normally, if you add a print statement to compiled code, this will cause a graph break. However, if you are interested in some quantity which is knowable at compile time, you can use comptime.print to print it out when we are compiling. You can use this to find out what the symbolic expression corresponding to a symbol is, or what Dynamo thinks the type/size/etc of a variable is.

| from torch.\_dynamo.comptime import comptime*\# ... and in your code ...*comptime.print(...anything...) |
| :---- |

# Compile time is too long

The most important thing to figure out is if you are recompiling a lot (you can find this out with TORCH\_TRACE/tlparse or TORCH\_LOGS=recompiles; you should also see warnings about cache\_size\_limit exceeded), or the compiler is just generally slow.

## Compiler is recompiling too much {#compiler-is-recompiling-too-much}

If you are recompiling a lot, some of the entries in your tlparse will look like [\[10/0\]](https://web.mit.edu/~ezyang/Public/bhack-20240609-tlparse/index.html#[10/0]) [\[10/1\]](https://web.mit.edu/~ezyang/Public/bhack-20240609-tlparse/index.html#[10/2]) [\[10/2\]](https://web.mit.edu/~ezyang/Public/bhack-20240609-tlparse/index.html#[10/4])
 ... (and so forth). If the recompilations go all the way to 7 (or whatever your max cache size is), that’s a sure sign that something has gone wrong. Note that by default, if we hit the cache size limit for a frame, we will stop attempting to compile that frame, but we we still try to recursively compile frames inside of it--to address the compile time problem, it is usually best to try to fix the outermost frame with a problem first. (TODO: Maybe this should not be the default [https://github.com/pytorch/pytorch/issues/128954](https://github.com/pytorch/pytorch/issues/128954) ).

Your first line of defense for too many recompilations is to look at the recompilation reasons using TORCH\_LOGS=recompiles (TODO: tlparse should have this information by default too). These logs will tell you what Dynamo had specialized on the first compilation, which was no longer true on a subsequent run (thereby requiring a recompilation). What you should do depends on what the cause of recompilation is. One caveat: currently the recompiles logs only reports the FIRST guard that failed. You may resolve this problem, only to discover there is a *different* problem that is also causing specialization. You can get some sense for whether or not this is happening by diffing the generated guards for two compilations (you can get guards with TORCH\_LOGS=guards or with tlparse (TODO: tlparse isn’t actually rendering guards right now)).

### Size related recompilation

Recompilation due to size mismatch will be due to a guard like L\['x'\].size()\[0\] \== 2954. The most common explanation for this is that something is forcing specialization. If the specialization is to a specific integer value, you can find the reason for the specialization in “Symbolic shape specializations” section in the compilation\_metrics entry for the frame in question. Otherwise, you can look for the guard in question using TORCH\_LOGS=dynamic (if you know the exact string of the guard you are looking for, you can also use TORCHDYNAMO\_EXTENDED\_DEBUG\_GUARD\_ADDED=”s0 \== 2” or similar to get detailed information for only that guard. The TORCH\_LOGS=dynamic output will suggest what environment variables you can use.) More debugging information can be found at [The dynamic shapes manual](https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit#heading=h.fh8zzonyw8ng). The most common reasons for specialization are:

1. The very first time we compile a graph, by default we assume that all sizes are static; only when we discover that a size has changed do we recompile and attempt to make it dynamic. In TORCH\_LOGS=dynamic logs, this is likely to be the case if you don’t see the quantity get allocated a symbolic variable at all. This results in a tell-tale two compilations \[0/0\] (static) and \[0/1\] (dynamic), but it can potentially lead to more recompilations if you have a lot of input sizes that need to be dynamic, but some of them don’t vary until even later inputs. To bypass this, you can use torch.\_dynamo.mark\_dynamic(tensor, dim) to mark a particular dimension of a tensor as dynamic, so we will immediately compile it dynamically. However, this won’t work if the size of the input tensor is 0/1, in this case, you must use torch.\_dynamo.mark\_unbacked(tensor, dim), but this can cause different errors which you need to consult [Dealing with GuardOnDataDependentSymNode errors](https://docs.google.com/document/d/1HSuTTVvYH1pTew89Rtpeu84Ht3nQEFTYhAX3Ypa_xJs/edit#heading=h.44gwi83jepaj) about (you will start getting errors whenever you try to guard on this size quantity at all, which may or may not be what you wanted). Also, both of these strategies work poorly when there are graph breaks, as mark\_dynamic annotations are not propagated through eager code, and you need to have accurately annotated the inputs into every graph break region, which is basically impossible to do unless you’re an expert. If you’re looking for a better long term fix, perhaps check [https://github.com/pytorch/pytorch/issues/121111](https://github.com/pytorch/pytorch/issues/121111)

2. You have hit some framework code that doesn’t know how to work symbolically, and is forcing specialization. For example, you may need to SymInt’ify the schema of an operator, or an operator has a meta implementation written in C++ that only supports int64\_t input rather than SymInt, or some code is written in a crappy manner that causes a specialization. [The dynamic shapes manual](https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit#heading=h.fh8zzonyw8ng) contains guidance for how to fix these situations, but if you can file an issue with the log line corresponding to the specialization (by default, it also gives its best guess for what framework code caused the specialization), it’s usually pretty quick for a PyTorch developer to fix. You can also find the stack trace that caused a specialization in tlparse under compilation\_metrics look in the section “Symbolic shape specializations” (NB: this section will only show up if a size gets specialized to a single value. In more obscure situations, you may be recompiling because of a nontrivial guard like s0 % 2 \== 0; in this case, you will have to use TORCH\_LOGS=dynamic to find out more about where the guard came from.)

3. You have some fundamental problem in your user code that makes it hard to compile the program in a generic way. For example, if you an integer that you use to index into a Python list, this is going to force a specialization as PT2 does not support dynamic accesses into Python data structures. How to resolve this sort of situation varies, consider asking for help, but you may need to make some fairly involved architectural changes in this case.

### NN module (ID\_MATCH) related recompilation

A very common cause for recompilation is when we have an ID\_MATCH guard on some object, and that object is continuously getting regenerated every iteration--usually, the object in question is an NN module, which by default we install ID\_MATCH guards for. You can force Dynamo to instead trace into an NN module with torch.\_dynamo.config.inline\_inbuilt\_nn\_modules \= True, which will eliminate the ID\_MATCH guard in favor of more granular guards. We intend to turn this on by default, but it has exposed a lot of Dynamo bugs which as of Jul 3, 2024, we are still working through.

### Generic troubleshooting advice

If you are recompiling and it is not due to one of the common reasons above, you will need to understand where the guard is coming from. In the TORCH\_LOGS=guards output for a frame (also found in dynamo\_cpp\_guards\_str artifact in tlparse) there are annotations describing where any given guard came from:

| ID\_MATCH: \_\_\_check\_obj\_id(G\['g'\], 7665376) \# if g:  \# b.py:7 in f |
| :---- |

Here, you can see that the ID\_MATCH on the global named g (this is what G\[‘g’\] means; if you are referencing a local, you’ll instead see L\[‘varname’\]) was installed by Dynamo when tracing line 7 of b.py. The line in question was an if statement on g; so in fact, this is just guarding that g is True, which makes sense, as we needed to know the value of g to tell which way the branch would go.

TODO: recompiles should also have this information. The recompiles log doesn’t have this information, so once you see that a guard triggered recompilation, you must go to the guards output to find out more about the guard. The guards output is organized as a tree, since some guards must evaluate to True before other guards are safe to evaluate, but usually the output is self explanatory. Here is a more complete view of the guard tree from above. I’ve highlighted the leafs, which are the actual logic that gets executed.

| TREE\_GUARD\_MANAGER:\+- RootGuardManager| \+- DEFAULT\_DEVICE: utils\_device.CURRENT\_DEVICE \== None| \+- GLOBAL\_STATE: \_\_\_check\_global\_state()| \+- GuardManager: source=L\['x'\], accessed\_by=DictGetItemGuardAccessor(x)| | \+- TENSOR\_MATCH: check\_tensor(L\['x'\], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires\_grad=False, size=\[None\], stride=\[1\])| | \+- NO\_HASATTR: hasattr(L\['x'\], '\_dynamo\_dynamic\_indices') \== False| \+- GuardManager: source=G, accessed\_by=GlobalsGuardAccessor| | \+- GuardManager: source=G\['g'\], accessed\_by=DictGetItemGuardAccessor(g)| | | \+- ID\_MATCH: \_\_\_check\_obj\_id(G\['g'\], 7665376)\+- LAMBDA\_GUARD: 2 \<= L\['x'\].size()\[0\] |
| :---- |

Another thing you can do is take the dynamo\_cpp\_guards\_str for two compilations of the same frame, and diff them (e.g., with your favorite textual diff tool). Where the guards differ is a good explanation of why the two compilations are different, and unlike TORCH\_LOGS=recompiles, this will give you the FULL set of guards that differ, not just the first guard that failed.

Once you’ve found the code that is triggering the guard, you should see if you can rewrite the code so that it doesn’t trigger a guard, or perhaps file a bug to PyTorch if you think we should be able to work with your code without adding a guard (e.g., Dynamo is over-guarding.) Remember that fundamentally, Dynamo is only capable of working with straight-line traces of your program, so if you have a branching conditional and you need both branches to be compiled, we MUST compile twice, or you must figure out if you can use torch.cond or another higher order operator to rewrite your program.

## Compiler is generally slow

### Loops are unrolled\! {#loops-are-unrolled!}

Unlike a classic compiler, the time it takes to torch.compile your code depends on the length of the execution trace of your code, *not* the input Python program size. So for example, if you have some code that loops over a size 1024 dimension, you will end up with 1024 copies of the loop body, which is going to be slow to compile. Also, we don’t do autovectorization, so the result won’t be fast either\!  The remedy varies on your situation:

* PyTorch, with einops and broadcasting, is pretty expressive. So it might be possible to represent your computation in a batched fashion. If you are having difficulty transforming your code from single batch to multi-batch, facilities like functorch vmap can help you express your code in a single batch and batch it.

* You can use a library like Numba [https://numba.pydata.org/](https://numba.pydata.org/) whose goal in life is expressly to take code written as loops and turn them into optimized kernels.

Note that the compiler has substantially worse constant factors than eager mode, so code that, while not performant in eager mode, ran fast enough for prototyping, may be unacceptably slow when you torch.compile it. One example that we’ve run into is compiling programs with \~1500 input tensors, each representing a particular sparse feature. It’s easy to end up spending hours in compilation if you have tons and tons of itty bitty tensor compute in your graph.

### Speeding up compilation with caching

We have made substantial investments into caching as a way to reduce compile time overheads, especially in situations where you are repeatedly running the same model over and over again. Currently Jul 3, 2024 at Meta we have deployed remote Inductor caching and see 50%/80% improvement to warm start time for training/inference respectively. (Meta only: [https://fb.workplace.com/groups/257735836456307/posts/705138451716041/](https://fb.workplace.com/groups/257735836456307/posts/705138451716041/)) Although caches can be operationally complicated to run (and we are still working out the bugs, see also [Stuck ranks are compiling](#stuck-ranks-are-compiling)) we think deploying a cache is well worth it. There is some guidance for how to setup caches at [https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html) although as of Jul 3, 2024, we do not have too much signal from OSS on how well our caching developments work.

One of our priorities for H2 2024 is extending the cache to cover AOTAutograd as well, which promises to enable further reductions in compiling time.

By default, we also have a file system cache which is saved to /tmp. If retention on /tmp in your system is not long enough, change the cache directory with TORCHINDUCTOR\_CACHE\_DIR. (See also [https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html) )

### Speeding up compilation with hierarchical compilation

By default, PT2 inlines all your model code into a single function which it then compiles. On some architectures, this means reused blocks (e.g., transformer blocks) will get inlined and compiled repeatedly. If you are not getting benefit from being able to fuse across block boundaries, you can potentially greatly reduce compile time by only torch.compile’ing the block itself, in which case it only gets compiled once and reused for every instance. You may need to set torch.\_dynamo.config.inline\_inbuilt\_nn\_modules \= True to ensure recompilation doesn’t occur when the self instance changes.  In H2 2024 we plan to also work on letting you noinline blocks inside of a larger torch.compile region, allowing for true hierarchical compilation.

### Profiling the compiler

Although it is often not easy to make a compiler faster, sometimes there is a big, obvious inefficiency that profiling can reveal. You can conveniently get a cProfile trace of compilation using TORCH\_COMPILE\_CPROFILE=1, this gives a nice dot diagram like this:
![][image1]

which you can look for hot spots in (the more red, the more hot it is). Sampling profilers like py-spy can also help identify opportunities. For example, a profile can tell you if you are getting hosed because forking processes for parallel processes is taking a long time.

**Meta only:** If you are profiling within fbcode, we have built-in support for Strobelight profiling. See [https://fb.workplace.com/groups/257735836456307/?multi_permalinks=669969978566222&hoisted_section_header_type=recently_seen](https://fb.workplace.com/groups/257735836456307/?multi_permalinks=669969978566222&hoisted_section_header_type=recently_seen)

## Compiled results are slow on first run but not subsequently {#compiled-results-are-slow-on-first-run-but-not-subsequently}

We typically think of most of PT2’s work as being done during the compile step (e.g., the things that show up on tlparse), and once we are done compiling, we are all done. However, there are some compilation steps which are delayed until the first time we actually run the compiled code. There is currently not a very good accounting for these steps in tlparse, but you can get some information with TORCH\_LOGS=inductor.

Most notably, CUDA graphs (activated by mode=”reduce-overhead” or mode=”max-autotune”) require a warm up step before the actual CUDA graph recording step. This results in a tell tale compile time profile which looks like:

1. (100s) First run, we do PT2 compilation, but then we run the result as is without CUDA graphs
2. (10s) Second run, no PT2 compilation is needed, but we do CUDA graph recording.
3. (\<1s) Third run, actual fast run.

Additionally, there may be multiple CUDA graphs recorded for a single compiled product; for example, if your model has dynamic shapes, we will compile the model once in a dynamic way, but then CUDA graph for each distinct size. Similarly, if you compiled an NN module that is used multiple times with different parameters (e.g., a transformer block that has multiple copies in a graph), we will rerecord CUDA graphs for each distinct set of parameters. (TODO: Actually, did we implement this yet as of Jul 3, 2024? Well, it will eventually do this\!) This recording process can be an order of magnitude slower than regular inference. By deleting the mode argument or using mode=”max-autotune-no-cudagraphs” you can eliminate this warmup overhead, at the cost of slower runtime execution.

TODO: There may be other things that can show up at runtime; we hope to also display them in tlparse if feasible to do without high overhead

# It runs, but the performance is not what I expect

## Start here: tips and tricks

Read this first\!

* If your network can deal with reduced matmul precision, makes sure to enable TF32\! [https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices) This is the number one reason why naive benchmarking often finds PyTorch slower than other things.

* Inductor is optimized towards A100 and H100 performance. Even V100 are out of the happy path, and consumer cards are especially likely to have problems. A lot of this is due to targeting Triton, which has a relatively limited set of cards it is highly optimized for. We would like to do better, especially on consumer cards, but this is a big project, external contributions welcome\!

* If you are benchmarking, make sure you CUDA synchronize before you measure walltime, otherwise you may underestimate the time your kernels take. Also, make sure you warmup at least two iterations (but preferably something more like ten), especially if you’re using CUDA graphs.

* If you are benchmarking, and you specifically want to measure static shape performance across multiple sizes, make sure you explicitly disable dynamic shapes with torch.compile(dynamic=False). Otherwise, we will automatically recompile your kernel to be dynamic, and you will find the first size is fast but subsequent sizes are slower.

* If your network is overhead bound, consider using mode=”reduce-overhead” which will enable CUDA graphs (for more CUDA graphs guidance, see [CUDA graphs specific performance advice](#cuda-graphs-advice). (Make sure you warm up at least two iterations\!)

* If you are willing to wait at compile time, mode=”max-autotune” will spend time autotuning templates and block sizes for kernels, which is usually worth some percentage worth of performance. Autotuning results can be cached so you don’t have to repeatedly autotune every invocation.

* PT2 works well with custom kernels that you may have written by hand. For help integrating them into PT2, see [https://pytorch.org/docs/main/notes/custom_operators.html](https://pytorch.org/docs/main/notes/custom_operators.html) . If your operator has stringent requirements on the striding of its input, consider setting needs\_fixed\_stride\_order.

## Too many graph breaks

You can see how many individual graphs PT2 has compiled in tlparse output. You can also use TORCH\_LOGS=graph\_breaks or profiling (next section) to identify them.

Graph breaks make performance worse for a number of reasons:

* Smaller graphs mean less fusion opportunities; they also mean you suffer more from fixed overheads from PT2 (e.g., guards, AOTAutograd runtime code). CUDA graphs (from mode=”reduce-overhead”) in particular suffers from lots of small graphs, as tensors must be copied in/out of the fixed CUDA graph addresses.

* Smaller graphs are more likely to exhibit dynamic behavior (seeing different sizes), and in general dynamic shapes codegen is less performant than static shapes codegen.

Often, the graph breaks are just missing functionality in Dynamo, please submit bugs for these. Sometimes, you can work around these by simplifying your Python code (we still encourage to submit a bug though, we would like to support all of these patterns\!) However, there are some particular situations which are likely to result in lots of graph breaks and need some different treatment.

* By default, if you have any data-dependent computation, e.g., boolean masking, item() call, nonzero(), etc, this will trigger a graph break. If you are feeling brave, you can get past these problems by setting torch.\_dynamo.config.capture\_scalar\_outputs \= True and torch.\_dynamo.config.capture\_dynamic\_output\_shape\_ops \= True. You will want to read [Dealing with GuardOnDataDependentSymNode errors](https://docs.google.com/document/d/1HSuTTVvYH1pTew89Rtpeu84Ht3nQEFTYhAX3Ypa_xJs/edit#heading=h.44gwi83jepaj) next.

* If torch.compile fails to compile a frame, it will try again to compile every inner frame. But if the graph break was nested deep inside of one of the inner frames, this will just end up generating lots of itty-bitty frames. It may be better to just torch.\_dynamo.disable a function relatively high up in the call stack so as to prevent PT2 from churning through lots of small graphs.

## How do I map a kernel back to Inductor code and graph?

In general, we attempt to name kernels in a meaningful way corresponding to the operators that were fused into them: there are some configuration knobs in torch.\_inductor.config like torch.\_inductor.config.triton.descriptive\_names which you can use to tweak the naming conventions. In many cases, you can then map the kernels to original userland code by inspecting the output\_code in tlparse. If you are still having trouble, use TORCHINDUCTOR\_UNIQUE\_KERNEL\_NAMES=1 to force each kernel produced by Inductor to have a unique name.

Once you’ve gotten the Inductor generated Triton code, there is extra information on each kernel relating it back to the source code, e.g., above each Triton kernel:

| *\# kernel path: /tmp/torchinductor\_ezyang/tc/ctcg6vrb3wgwmuh625mqucd4gfgje4wzwya3gslwtfgvk74ucyl5.py \# Source Nodes: \[gelu\], Original ATen: \[aten.gelu, aten.gelu\_backward\] \# gelu \=\> add, erf, mul\_1* |
| :---- |

Here, “Source Nodes” identifies the names of nodes from dynamo\_output\_graph (also available in tlparse) corresponding to the kernel; this may be empty if there is no obvious Dynamo source node that created this node--e.g., if the kernel is generate for backwards. “Original ATen” gives the name of the original (prior to decomposition) ATen kernels which contributed to this kernel, and the “mlps\_0\_1 \=\> add\_9, erf, mul\_7” maps the Dynamo source node to inductor\_post\_grad\_graph nodes. In this example, we can see the ATen nodes are:

|          *\# File: /data/users/ezyang/a/pytorch/b.py:5 in f, code: return torch.nn.functional.gelu(x)*        mul\_1: "f32\[4\]\[1\]cuda:0" \= torch.ops.aten.mul.Tensor(primals\_1, 0.7071067811865476)        erf: "f32\[4\]\[1\]cuda:0" \= torch.ops.aten.erf.default(mul\_1);  mul\_1 \= None        add: "f32\[4\]\[1\]cuda:0" \= torch.ops.aten.add.Tensor(erf, 1);  erf \= None |
| :---- |

The post\_grad\_graph also reports the source code of the user code that generated these nodes.

Note: It’s not guaranteed that the Inductor kernel corresponds *exactly* to the source nodes here. Specifically, a single source node may potentially lower into multiple Inductor IR nodes, which then could potentially be scheduled into separate kernels. If so, a node will show up multiple times in different kernels. Another reason a node may show up multiple times is if we decided to perform recomputation (esp., recomputing a forwards operation in backwards.)

**Warning:** As of Jul 5, 2024, we do NOT print out post grad graph nodes corresponding to backwards (see [https://github.com/pytorch/pytorch/issues/130147](https://github.com/pytorch/pytorch/issues/130147), all the gelu\_backward nodes are missing). Forwards prints should be accurate though.

## Interpreting profiles of compiled code {#interpreting-profiles-of-compiled-code}

You can use PyTorch profiler [https://pytorch.org/docs/stable/profiler.html](https://pytorch.org/docs/stable/profiler.html) on torch.compile’d code. This is a good thing to if you have a large model and you’re not really sure what you’re looking for; the profiler can make it obvious if you’re blocked on a device-to-host sync and give you ideas for what kernels you might want to look into optimizing.

Some other tutorials you may find useful: [https://pytorch.org/docs/stable/torch.compiler_profiling_torch_compile.html](https://pytorch.org/docs/stable/torch.compiler_profiling_torch_compile.html) Note that [https://pytorch.org/docs/stable/generated/torch.autograd.profiler.record_function.html](https://pytorch.org/docs/stable/generated/torch.autograd.profiler.record_function.html) does not currently work inside torch.compile regions, but you can still use it outside of your torch.compile block.

Credit: The rest of this section is adapted from a [Meta internal post](https://fb.workplace.com/groups/257735836456307/posts/647088057521081) authored by David Berard, Aaron Shi and Animesh Jain.

When an operator is launched, we expect to see a few events:

1. CPU-side event
2. Kernel launch (if dealing with a GPU kernel)
3. GPU-side event

For example, this is what a traditional eager mode kernel looks like in the profiler.
![][image2]
torch.compile generated kernels will also have the same pattern, but they will look slightly different.

**What does an Inductor generated Triton kernel look like in profiler?**

1. The **CPU-side event** should appear as an event prefixed with “triton\_”. The events currently have minimal information \- the kernel name and a launch, but less information than typical aten kernel launches (which contain input shapes, types, etc.).
2. The **kernel launch** should appear as cuLaunchKernel instead of cudaLaunchKernel (cudaLaunchKernel is typical for aten ops)
3. The **GPU-side event** should appear, and how descriptive the name will be depends on the [inductor config for unique\_kernel\_names](https://github.com/pytorch/pytorch/blob/ca9678405a76fe7f9087d1cde16c906b795c0774/torch/_inductor/config.py#L546)

![][image3]

**What does a non-Inductor generated Triton kernel look like in profiler?**

E.g., if you use a hand written custom Triton kernel.

1. The **CPU-side event** may not appear in traces; the machinery for automatically inserting a profiler event is currently implemented at the Inductor level, so Triton kernels that bypass Inductor may not appear in traces, unless users have annotated them manually
2. The **kernel launch** should appear s cuLaunchKernel instead of cudaLaunchKernel (cudaLaunchKernel is typical for aten ops)
3. The **GPU-side event** should appear, named similarly to the triton kernel that was authored.

![][image4]

**What does an Inductor-generated CPU kernels look like in profiler?**

1. The **CPU-side** event will not appear in traces; we haven’t added profiling for this yet.
2. The **kernel launch** and **GPU-side** events don’t exist

**What is a Torch-Compiled Region?**

“Torch-Compiled Region” is a profiler event added for each frame handled by Dynamo.

**Why does my graph have a lot of nested Torch-Compiled regions?**

Here is an example:

![][image5]

Frames handled by Dynamo will recursively call into a continuation frame when there is a graph break. Thus, you see this nesting.

If you run two separate functions with torch.compile() applied independently on each of them, you should generally expect to see two adjacent (i.e NOT stacked/nested) Torch-Compiled regions. Meanwhile, if you encounter graph breaks (or disable()’ed/skipped regions), expect nested “Torch-Compiled Region” events.

**What does autograd with torch.compile look like in profiler?**

Sometimes, an event called “**CompiledFunction**” is inserted. **This only happens when Autograd is involved, i.e. some of the input tensors to the graph have requires\_grad=True.**

**![][image6]**
*For more context: [CompiledFunction](https://www.internalfb.com/code/fbsource/[8a62a43f21b416aea2161aa5235fb8968791666a]/fbcode/caffe2/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py?lines=463) is actually an [autograd.Function](https://pytorch.org/docs/stable/autograd.html#function) used in the implementation of the PT2 compiler in order to stitch together a compiled forward and compiled backward implementation. It is only used when some inputs require grad; otherwise, the compiler uses a wrapper that doesn’t include this event. Op-level and kernel-level profiler events should still appear even if a CompiledFunction doesn’t appear.*

When a CompiledFunction appears in a trace, it is typically paired with a CompiledFunctionBackward event in the backward pass. A “fwd-bwd link” should appear in the trace connecting the two, if the backward function is called.

If your use case includes a graph that doesn’t require grad, it can be more difficult to identify whether torch.compile is being applied correctly. Some clues include Torch-Compiled Regions or the existence of Inductor-generated Triton kernels. In particular, you cannot rely on the existence of CompiledFunction to tell that compilation has occurred.

**What does DDP Optimizer look like in the profiler?**

If [DDP Optimizer](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860) is [enabled](https://github.com/pytorch/pytorch/blob/85c807b3fd0f68c0f9ef0b64e54e37e792b25cea/torch/_dynamo/config.py#L242), it will introduce graph breaks that do not display nested “Torch-Compiled Region” events. This is because the graph breaks introduced by DDP Optimizer are introduced after Dynamo has applied.

**Can I count number of graph breaks from a profile?**

Ignoring the case where you have more than one independent torch.compile-d section of code, I think the number of "Torch-Compiled Region"s minus 1 (aka, the number of nested Torch-Compiled regions) is roughly accurate. But it’s better to use tlparse for this sort of thing.

**Why do I sometimes see ATen operators under Torch-Compiled Region?**

Example:

![][image7]

Most aten ops will be converted into triton kernels by inductor. However, less-common ops (and most custom ops) might not have lowerings in inductor, in which case inductor will fall back to using the original aten implementation. Additionally, matmuls frequently fall back to aten for two reasons: first, if you don't turn on autotuning for gemms, Inductor will automatically fall back to ATen; and second, sometimes during autotuning for gemms, Inductor will find that the original ATen implementation is still faster than the Triton kernel generated by inductor and choose to use the original ATen implementation instead. This is not a graph break, it’s just Inductor picking the best code for the job.

## Dealing with fusion

### Visualizing fusion choices

Although you can technically determine what ATen nodes have been fused purely by inspecting the comments on the generated Inductor kernels, a graph representation of the inductor\_post\_grad\_graph can make it more clear what the data dependencies between various kernels are. You can generate a graph of ATen nodes annotated with which nodes Inductor has chosen to fuse together with TORCH\_COMPILE\_DEBUG=1 INDUCTOR\_ORIG\_FX\_SVG=1. This will generate svg files that look like this:

![][image8]

Each ATen node is represented into the graph, and then they are grouped together into fused kernels (op0). Note that on a real world model, the resulting diagram can be intractably huge, so this works best if you’re working on a small fragment of code you’re trying to understand the compilation behavior of.

### Why did Inductor fuse it that way?

A common question that arises when diagnosing the performance of Inductor compiled models is why Inductor chose to do a fusion a particular way. In particular, you may have observed in a profile that a block of code you expected to compile into a single kernel was compiled into multiple kernels. The important diagnostics are accessible from TORCH\_LOGS=fusion, which prints explanations why two buffers were not fused.

To understand the output of these logs, we first have to understand what kinds of fusions we can reasonably expect Inductor to perform.

* **Pointwise fusion**: if you have multiple pointwise operations operating on the same shape, they will be fused together. In fact, this fusion happens on the fly during lowering not scheduling (this manifests as all the pointwise operations showing up in the same Pointwise IR node.)

* **View removal**: views generally do not need to be code generated. Instead, we simply modify the indexing expressions of subsequent accesses to the Tensor, so that we can directly index into whatever the original output was

* **Pattern matching**: we have a number of FX passes which look for patterns of several FX nodes together, replacing them with a single operator. TODO: Make it easier to identify which FX pass performed this fusion, see [https://github.com/pytorch/pytorch/issues/118123](https://github.com/pytorch/pytorch/issues/118123)

* **Epilogue fusion**: for some kernels for which we have Triton templates (e.g., matmul), we can potentially fuse extra pointwise operations into the epilogue of that kernel. Epilogue fusion is not always profitable, as complicated kernels like matrix multiply are often finely tuned and adding extra compute can make them perform worse than their unfused counterparts.

* **Vertical fusion**: a consumer can be fused into its producer, if all reads in the consumer either match corresponding writes in the producer, or are written by nodes that can be scheduled before the fusion of these two nodes. These decisions are reported in TORCH\_LOGS=fusion.

* **Horizontal fusion**: two nodes that don’t depend on each other but share reads can also be fused. These decisions are reported in TORCH\_LOGS=fusion.

There are a lot of reasons why a fusion potentially may not happen; furthermore, even if a fusion is possible, there may be multiple mutually exclusive fusions which Inductor must choose between. So it is best to consult TORCH\_LOGS=fusion for the source of truth about what particular decisions Inductor made. For Triton code generation, the main algorithm lives in can\_fuse in torch/\_inductor/codegen/simd.py (it’s in this file so it can be used for other Triton-like backends). One of the important things is that if you want horizontal fusion to occur, the nodes need to have compatible tiling. [Horace He](mailto:chilli@fb.com) has a nice Twitter thread explaining what tiling is [https://x.com/cHHillee/status/1620878972547665921](https://x.com/cHHillee/status/1620878972547665921) which is helpful for understanding why it can occur.

Here are a few concrete examples of reasons why things unexpectedly didn’t fuse that we’ve observed in the wild:

* While working on float8 Linear layer, a max(abs(tensor)) and a to\_fp8\_cast (with no data dependency on the max-abs operator) were expected to be fused together in forwards but were not. From the "fusion" log, these two buffers were not fused due to "invalid tiling for reduction". The tiling of the pointwise operator is (4096, 4096), while the tiling of the reduction operator is (512, 32768). Counterintuitively, however, both operators took the same contiguous 4096x4096 tensor as input, so it looks like they should be able to have the same tiling and be fused.

  The root cause of the problem is that the reduction from \[4096 x 4096\] to \[1\] (amax op) was split into two layers to be more efficient: the first layer reduced \[4096 x 4096\] to \[X\] and the second layer reduced \[X\] to \[1\]. When Inductor made a choice of X, it did not realize that picking 4096 would make it possible to fuse the first layer reduction with the following pointwise op. In this case, inductor picks X other than 4096 (512 here).

  In the end, it turned out that fusing these two kernels together was not actually a good idea, as directly reducing from 4096 to 1 in the second reduction is less efficient. There is also a tracking issue for helping Inductor delay decisions about loop ordering until after fusion, so Inductor doesn’t pick loop orders that impede fusion: [https://github.com/pytorch/pytorch/issues/126255](https://github.com/pytorch/pytorch/issues/126255)

  Another algorithmic change that would permit the second-level amax reduction and the fp8 conversion to be fused is to make the amax row-wise rather than tensor-wise. However, this requires cuBLAS to support row-wise scales, which as of Jan 11, 2024 was not supported.

* A user observed in profiles that add/view Triton kernels after QKV projection in MHA took a long time and were clearly memory bound. Here, the expectation was that the add should have been fused into the mm as an addmm; however, in this custom MHA layer, the view happened between the mm and the add, thereby impeding the fusion. A POC reordering mm \-\> view \-\> add into addmm \-\> view ([https://github.com/pytorch/pytorch/pull/121059](https://github.com/pytorch/pytorch/pull/121059)) significantly improved performance. The longer term resolution was to look into optimization passes that normalized views to move them out of the way of compute ops as much as possible.

* A user was expecting pointwise operations to be fused with amax, but in microbenchmarking found that sometimes this fusion did not occur. In fact, fusion did not occur when there were multiple dynamic dimensions in the reduction ranges, as this made indexing expressions sufficiently complicated that fusion could not occur. When only one dynamic dimension occurred (e.g., the batch dimension), fusion was able to happen.

In principle, more autotuning can also help Inductor make better choices, as Inductor no longer needs to rely on heuristics but instead can try various fusions. In practice, today, autotuning is only really used to make decisions about block sizes in Triton. There is an interesting proposed project from [Jason Ansel](mailto:jansel@meta.com) to extend our autotuning decisions to more of the compiler, including fusion decisions, but as of Jul 5, 2024 no one has committed to working on it. Meta only: [Autotuning in PT2/TorchInductor](https://docs.google.com/document/d/1PtRQ4QxKUfCZbOsTBo4p0sddC7Wod1hYqE1HxqrOwzI/edit#heading=h.84qzljq14miv)

## Tuning Inductor generated kernels

Inductor try to make good decisions in its generated Triton code, including automating the drudgery of finding good block sizes away with autotuning, but sometimes it just makes bad decisions. One popular use case for Inductor among expert users is to use it as the basis for a more tuned manual Triton implementation.

There are a number of tools you have to understand why a kernel may be performing poorly.

* The generated Inductor code for kernels comes with a built in benchmark harness “benchmark\_kernel”, which is invoked when you run the inductor\_output\_code Python file directly. The outputted Python files in tlparse are directly runnable and you can use this as the basis to do optimization. Here is worked example of using this capability: [https://pytorch.org/docs/stable/torch.compiler_inductor_profiling.html#benchmark-individual-triton-kernel](https://pytorch.org/docs/stable/torch.compiler_inductor_profiling.html#benchmark-individual-triton-kernel)

* You can use TORCHINDUCTOR\_PROFILE=1 to have Inductor estimate kernel bandwidth numbers, which look like this:
  ![][image9]

* ncu ([https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)) is a great tool for figuring out if a kernel is inefficient. You could use a script like [https://gist.github.com/shunting314/0667d69299a5d9435b2931da2d2df476](https://gist.github.com/shunting314/0667d69299a5d9435b2931da2d2df476) to conveniently ncu every Triton kernel generated by Inductor. For example, if ncu\_mem\_bw\_gbps is low, that means the kernel is not saturating memory bandwidth; if the kernel takes a long time to run, it’s a good candidate for compute optimization. (This is different from the Inductor estimate of memory bandwidth from the previous bullet; the NCU number is more accurate as it represents the true memory bandwidth.) An example of a problem we fixed with this methodology is identifying tl.rand being slow ([https://gist.github.com/shunting314/2a0bb5b1668f79caeb5b7d3df2ce5561](https://gist.github.com/shunting314/2a0bb5b1668f79caeb5b7d3df2ce5561)); the problem here being that we were using tl.rand which generates four samples and throwing three away.

Also, just reading the kernel can be pretty informative. For example, the debugging process that lead to [https://github.com/pytorch/pytorch/pull/124131](https://github.com/pytorch/pytorch/pull/124131) arose because a user was looking at the generated Triton kernels and noticed that Inductor was generating uncoalesced writes, comparing:

| buf6 \= empty\_strided\_cuda((26624, 1024), (1024, 1), torch.float8\_e4m3fn) ... tl.store(out\_ptr2 \+ (r1 \+ (1024\*x0)), tmp15, rmask) |
| :---- |

vs

| buf6 \= empty\_strided\_cuda((1024, 26624), (26624, 1), torch.float8\_e4m3fn) ... tl.store(out\_ptr2 \+ (x0 \+ (26624\*r1)), tmp15, rmask) |
| :---- |

Another cheat code is to just see what [Shunting Zhang](mailto:shunting@fb.com) has been up to, as of Jul 5, 2024,  many of our major, broad base Inductor performance improvements have come out of things he’s been working on.

## Dealing with GPU dead time

A commonly used proxy for performance is how well your GPU is utilized: if there are lots of gaps where no GPU compute is happening, this is bad and you want to fix it. Code that is well utilized without torch.compile can exhibit gaps after torch.compile, because Inductor optimized kernels are faster and spend less time on memory bandwidth and so your critical path can end up being different.

GPU dead time typically arises when you have DtoH syncs or distributed collectives. It also happens when you have lots of CPU overhead, but PT2 generally reduces CPU overhead, and you can also use mode=”reduce-overhead” to reduce it even more. The best remedy for DtoH sync bubbles is to eliminate the DtoH sync. For example, if you are using a boolean mask, consider instead using torch.where.

Sometimes, the gap is unavoidable; e.g., you need a distributed collective, or it is very important to to inspect the data because you want to reduce the amount of compute being used downstream. In these cases, the usual name of the game is to find some other compute that doesn’t have a data dependency on the blocking operation, and move it earlier so that it can be scheduled while you are waiting on communication. In principle, PT2 could perform this sort of optimization; it doesn’t as of Jul 5, 2024 but we are very interested in this class of optimizations.

## Custom FX passes

If there is some code in a single model that performs poorly, you might be able to rewrite it so that it does better. But what if you have a lot of models and they all have the same problem? You can write a custom graph pass to identify and replace these patterns. Custom graph passes are configurable in torch.\_inductor.config:

| *\# register custom graph optimization pass hook. so far, pre/post passes are**\# only applied before/after pattern\_matcher in post\_grad\_passes.**\#**\# def my\_custom\_pre\_pass(graph: torch.fx.graph.Graph):**\#     \# my custom graph optimization pass**\#     ...**\#**\# def my\_custom\_post\_pass(graph: torch.fx.graph.Graph):**\#     \# my custom graph optimization pass**\#     ...**\#**\# torch.\_inductor.config.post\_grad\_custom\_pre\_pass \= my\_custom\_pre\_pass**\# torch.\_inductor.config.post\_grad\_custom\_post\_pass \= my\_custom\_post\_pass*post\_grad\_custom\_pre\_pass: Optional\[Callable\[\[torch.fx.graph.Graph\], None\]\] \= Nonepost\_grad\_custom\_post\_pass: Optional\[Callable\[\[torch.fx.graph.Graph\], None\]\] \= None*\# Registers a custom joint graph pass.*joint\_custom\_pre\_pass: Optional\[Callable\[\[torch.fx.Graph\], None\]\] \= Nonejoint\_custom\_post\_pass: Optional\[Callable\[\[torch.fx.Graph\], None\]\] \= None |
| :---- |

For information on how to write FX passes, consult [https://pytorch.org/docs/stable/fx.html](https://pytorch.org/docs/stable/fx.html) documentation. The operators you will see in graphs vary depending on when a pass is performed, but in general, you should expect to see ATen operations post decomposition.

## CUDA graphs advice {#cuda-graphs-advice}

When working with mode=”reduce-overhead”, it’s important to understand some basic characteristics about CUDA graphs:

* CUDA graphs is only capable of capturing CUDA computation. If you have CPU compute inside your graph, CUDA graphs isn’t going to work. Sometimes, the CPU compute can be moved to GPU; in principle, Inductor could do this automatically for you but it only does so in limited cases. While random numbers work with CUDA graphs, checkpointing and RNG does not work [https://github.com/pytorch/pytorch/issues/130123](https://github.com/pytorch/pytorch/issues/130123) , see [https://github.com/pytorch/pytorch/pull/114068](https://github.com/pytorch/pytorch/pull/114068) for a potential strategy how to fix this.

* A recorded CUDA graph operates on fixed CUDA addresses. This means that, in the worst case scenario, inputs/outputs must be copied in/out of the CUDA addresses the CUDA graphs is expected to run. In practice, we have a number of optimizations which are intended to reduce the cost of this: parameters are treated as static addresses and assumed not to move, outputs by default are left in CUDA graph memory (they must be freed before the CUDA graph can be rerun). Running afoul these constraints could mean that we need to rerecord CUDA graphs (e.g., if parameter addresses are changing) or we need to do excess copying (torch.\_dynamo.mark\_static\_address could be used to avoid this copying if the memory in question truly never moves).

* A recorded CUDA graph must maintain the allocated CUDA memory, since it only works on particular addresses. This means that CUDA graphs can overall increase the resident memory usage of your system, as live CUDA graph’s memory cannot be used by eager mode, even if the CUDA graph itself is not being used. Reuse of memory is possible across multiple CUDA graphs over graph breaks due to CUDA graph trees.

* CUDA graphs don’t support dynamic shapes. We actually do support mode=”reduce-overhead” in conjunction with dynamic=True, but the way this is implemented is by recording a distinct CUDA graph for each size you see. If this is too many CUDA graphs, you will want to pad sizes to multiples to reduce the number of CUDA graphs you need to record. See also notes at [Compiled results are slow on first run](#compiled-results-are-slow-on-first-run-but-not-subsequently)

* If you are failing with “Exception Found: These live storage data ptrs are in the cudagraph pool but not accounted for as an output of cudagraph trees”, try running with torch.\_inductor.triton.cudagraph\_trees\_history\_recording \= True

The most common complaint with CUDA graphs is that they are not actually working. You can explanations for why CUDA graphs did not work by running with TORCH\_LOGS=perf\_hints. One common remediation is to split a model into a CUDA graphable part, and a non-CUDA graphable part. Inductor could potentially do this automatically, see [https://github.com/pytorch/pytorch/issues/125864](https://github.com/pytorch/pytorch/issues/125864)

TODO: Explain mark\_step

# It runs, but uses too much memory

torch.compile is typically expected to cause models to use less memory, unless you are using mode=”reduce-overhead” (as CUDA graphs typically increases memory usage, due to CUDA graphs memory cost as well as copying inputs into CUDA graphs). Sometimes, a bug in PT2 means that we use too much memory. Some culprits we have seen in the past include:

* When Inductor compiles a graph assuming aligned inputs, if you pass it unaligned inputs, we have to copy them so that they are aligned. This copy can increase the overall used memory.

* A tensor which is no longer being used is still being kept live. One particular pernicious way this can happen in Python is if a tensor is an argument to a function; by default, arguments to functions are kept live for the entirety of the function call, even if they become unused midway through. We have a boxed calling convention (pass in a list of arguments which is cleared inside the function) to workaround this, but sometimes we do this incorrectly.

* Bad fusion decisions can reorder nodes so that high memory watermark increases. Inductor is supposed to not do fusions that increase high memory water mark.

## Memory profiler

You can visualize the state of allocated memory in PyTorch using memory snapshots. This works with PT2. Check [https://zdevito.github.io/2022/08/16/memory-snapshots.html](https://zdevito.github.io/2022/08/16/memory-snapshots.html) for instructios

## (Selective) activation checkpointing

Activation checkpointing is a classic approach to trading of compute for memory usage. The standard checkpointing API [https://pytorch.org/docs/stable/checkpoint.html](https://pytorch.org/docs/stable/checkpoint.html) works with torch.compile. An example case study of this can be found at [https://pytorch.org/blog/maximizing-training-throughput/](https://pytorch.org/blog/maximizing-training-throughput/)

# It runs, but it fails with NCCL timeout

In general, NCCL timeouts occur when not all nodes actually perform the same NCCL collective in a timely fashion. NCCL timeouts are the bane of distributed training even when PT2 is not involved, since it could really be *anything* that caused the problem: intermittent hardware problem? Some issue with the network switch? Some nodes nondeterministically running Python GC? Statistical profiler unluckily turned on and only hit rank 0? Bug in the framework? You name it, we’ve seen it. Luckily, PT2 introduces exciting, new failure modes on top of all of these.

The most important diagnostic you have for dealing with NCCL timeout is inspecting the stacks of the ranks at the time you hung; in particular, you want to know what the ranks that didn’t make it to the NCCL collective were doing. The sections below are organized based on what the stuck ranks are doing.

## Stuck ranks are compiling {#stuck-ranks-are-compiling}

There is currently a known problem with PT2 which is that compilation of a model can take a long time, typically longer than most NCCL timeouts are configured. If you are unlucky and only some ranks are compiling while other ranks are not, or if some ranks cache hit while others do not, the difference in compile time can easily cause a NCCL timeout.

We plan to fix this [https://github.com/pytorch/pytorch/issues/108971](https://github.com/pytorch/pytorch/issues/108971) but while you are waiting for the fix, here are some things you can do in userland to mitigate:

* Increase the NCCL timeout until it can account for compile time. This is not a great solution (since it also means you take that long to timeout when there is an actual problem), but assuming there aren’t other problems, it can get you unblocked / confirm that compilation time is in fact the issue. If divergence only happens on the first N iterations, you can also temporarily set the timeout to long during warmup, and then set it to a more reasonable value after quiescence.

* If you are using our remote cache, ([https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)), rank imbalance can occur when some ranks cache hit the remote cache, while other ranks do not. (Internally, we have observed this happening on pyper models). Disabling the remote cache in this situation will ensure that all ranks uniformly compile at the same time, so they all reach the NCCL collective at the same time. To disable remote caching only, run with TORCHINDUCTOR\_FX\_GRAPH\_REMOTE\_CACHE=0 or torch.\_inductor.config.fx\_graph\_remote\_cache \= False, alternatively, to locally disable all caches, run with TORCHINDUCTOR\_FORCE\_DISABLE\_CACHES=1 or torch.\_inductor.config.force\_disable\_caches \= True
  * **Meta only:** Remote cache can be disabled on a per-PG basis by adding rules to the JK. See [https://www.internalfb.com/intern/justknobs/?name=pytorch%2Fremote_cache#fx_graph_memcache_version](https://www.internalfb.com/intern/justknobs/?name=pytorch%2Fremote_cache#fx_graph_memcache_version) as of Jul 5, 2024 for a live example of this. This should only be used in an emergency and env variable disable should be preferred otherwise.

* If one rank is recompiling, a reasonable strategy is to prevent recompilation in the first place, so that each rank compiles only once at the beginning of the training run, and then is guaranteed to never recompile again.  The section [“Compiler is recompiling too much”](#compiler-is-recompiling-too-much) contains guidance for how to do this.

## Stuck ranks are running compiled code

We are aware of some theoretical problems which could be introduced by PT2.

* PT2 can change the performance characteristics of code. If it changes performance in a way that diverges between ranks (e.g., some sort of data-dependent autotuning), then data dependent performance could introduce enough variance that you start NCCL timing out. It is probably right to increase the timeout in this situation. You can investigate if this is the case by inspecting GPU traces (see [Working with the profile on compiled code](#interpreting-profiles-of-compiled-code)) and seeing if there is imbalance, in the same way you would diagnose imbalance without torch.compile. TODO: Better guidance for visualizing performance traces over many ranks at the same time.

* PT2 supports capturing distributed collectives. PT2 potentially has an opportunity to optimize these collectives, e.g., by reordering them. If PT2 reorders collectives, it must reorder all collectives in exactly the same way on all ranks. If this reordering process is nondetermistic, or if the input graphs across ranks differ, there is no way to verify that PT2 did this correctly (as we currently do not do any cross-node communication during compilation).

* The configuration option torch.\_functorch.config.fake\_tensor\_propagate\_real\_tensors can cause hangs, as propagating real tensors involves running real tensor operations during compile-time, *including* collectives. See [https://github.com/pytorch/pytorch/issues/126846](https://github.com/pytorch/pytorch/issues/126846)

# It runs, but my outputs are garbage

Our debugging story for this currently sucks. Start with [ablation](#do-an-ablation), and good luck.

One of the bigger missing pieces here is a way of logging out intermediate values in between modules and using it to triangulate where things diverge compared to eager. This tooling doesn’t exist in OSS.

There are some configuration options which make Dynamo more pedantically correct. They are worth trying:

* ~~torch.\_dynamo.config.guard\_nn\_modules \= True \- a lot of accuracy errors related to switching between eval/train on NN modules, or mutating NN modules between compiled function calls, are due to lack of guards on NN modules.~~ This is currently enabled by default on OSS (as of Jul 3, 2024\) and enabled by default in fbcode (as of Sep 27, 2024).

* torch.\_inductor.config.emulate\_precision\_casts=True \- when we compile low precision code, we omit conversions to smaller types if it is not necessary (e.g., two operations have been fused together). This can change the numerics of your code, and in some situations, running a computation in higher precision CAN harm the overall accuracy of your model. This config flag forces us to exactly emulate the precision casts even if it slows down kernels to reduce numeric divergence.

* torch.\_dynamo.config.optimize\_ddp \= False ; DDP optimizer is kind of complicated so it’s something good to try eliminating, especially if you’re failing with backend=”eager”.

TODO: Lessons learned from [https://github.com/pytorch/pytorch/issues/96693](https://github.com/pytorch/pytorch/issues/96693)

## Stride divergence

PT2 is not guaranteed to produce tensors with exactly the same stride as the eager variants. This can potentially cause correctness problems (or runtime crashes) outside of the torch.compile region if subsequent code is expecting a specific striding without testing, or is using the limited subset of APIs (mostly torch.as\_strided, and also obscure situations involving mutating through .reshape()/.contiguous()) which are sensitive to input of strides.

TODO: There is supposed to be a way to fix strides, similar to how we can do this for custom ops.

# Changelog

* Sep 27, 2024 Added note about torch.\_dynamo.config.optimize\_ddp \= False to accuracy section; guard\_nn\_modules is now obsolete as it’s True by default.
* Aug 13, 2024 Added note on torch.\_inductor.config.emulate\_precision\_casts=True and torch.\_inductor.triton.cudagraph\_trees\_history\_recording \= True
* Jul 11, 2024 Added note to CUDA synchronize when microbenchmarking CUDA code
* Jul 10, 2024 Added that CUDA graphs require two iterations of warmup (not one)
* Jul 9, 2024
  * Added section “Stride divergence”
  * Clarify that compiled autograd can give performance benefit by fusing accumulate grad nodes into the compiled region
* Jul 8, 2024
  * Note that local cache is saved to /tmp by default; can be changed with TORCHINDUCTOR\_CACHE\_DIR
* Jul 7, 2024 Added section on “What you should expect to compile”: [https://x.com/ezyang/status/1809766173849821669](https://x.com/ezyang/status/1809766173849821669)
