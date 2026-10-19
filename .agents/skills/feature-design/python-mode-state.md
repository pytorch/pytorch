# Python mode state and threads

**Applies when** a feature keeps a mode or an active context in Python-level
state: a `ContextVar`, a `threading.local`, or a module global (an active
capture context, a "currently inside X" flag that decorators check). It does
not cover C++ thread-local state propagated by `at::ThreadLocalState`.

## Which threads see the state

Check both directions. Each one fails differently.

- **Threads that don't inherit it.** The autograd engine runs backward nodes
  for accelerator devices on its own worker threads. It propagates C++
  thread-local state (grad mode, autocast, dispatch keys, saved tensor hooks)
  but not Python `ContextVar`s or `threading.local`. Code that checks the mode
  from a custom backward or a gradient hook sees "not in the mode" and silently
  takes the wrong path. The same holds for thread pools and C++-created
  threads.
- **Threads that do inherit it.** On free-threaded Python 3.14,
  `threading.Thread` copies the parent's context by default
  (`sys.flags.thread_inherit_context`). asyncio tasks and
  `contextvars.copy_context().run` copy it too. Copying a context copies
  references, so a mutable object stored in a `ContextVar` is then shared: a
  child thread can mutate, end, or corrupt the parent's active context.

## What to do

- Store immutable values in a `ContextVar`, or record the owning thread and
  check it before anyone mutates the context.
- Decide which threads are supported and enforce it. A call from an
  unsupported thread should raise an actionable error rather than silently
  bypass the mode.
- Test both directions: a mode check reached from backward, and a thread
  started inside the mode.
