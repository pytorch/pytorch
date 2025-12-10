from functools import wraps


class Callback:
    """
    Base class and interface for callback mechanism

    This class can be used directly for monitoring file transfers by
    providing ``callback=Callback(hooks=...)`` (see the ``hooks`` argument,
    below), or subclassed for more specialised behaviour.

    Parameters
    ----------
    size: int (optional)
        Nominal quantity for the value that corresponds to a complete
        transfer, e.g., total number of tiles or total number of
        bytes
    value: int (0)
        Starting internal counter value
    hooks: dict or None
        A dict of named functions to be called on each update. The signature
        of these must be ``f(size, value, **kwargs)``
    """

    def __init__(self, size=None, value=0, hooks=None, **kwargs):
        self.size = size
        self.value = value
        self.hooks = hooks or {}
        self.kw = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        self.close()

    def close(self):
        """Close callback."""

    def branched(self, path_1, path_2, **kwargs):
        """
        Return callback for child transfers

        If this callback is operating at a higher level, e.g., put, which may
        trigger transfers that can also be monitored. The function returns a callback
        that has to be passed to the child method, e.g., put_file,
        as `callback=` argument.

        The implementation uses `callback.branch` for compatibility.
        When implementing callbacks, it is recommended to override this function instead
        of `branch` and avoid calling `super().branched(...)`.

        Prefer using this function over `branch`.

        Parameters
        ----------
        path_1: str
            Child's source path
        path_2: str
            Child's destination path
        **kwargs:
            Arbitrary keyword arguments

        Returns
        -------
        callback: Callback
            A callback instance to be passed to the child method
        """
        self.branch(path_1, path_2, kwargs)
        # mutate kwargs so that we can force the caller to pass "callback=" explicitly
        return kwargs.pop("callback", DEFAULT_CALLBACK)

    def branch_coro(self, fn):
        """
        Wraps a coroutine, and pass a new child callback to it.
        """

        @wraps(fn)
        async def func(path1, path2: str, **kwargs):
            with self.branched(path1, path2, **kwargs) as child:
                return await fn(path1, path2, callback=child, **kwargs)

        return func

    def set_size(self, size):
        """
        Set the internal maximum size attribute

        Usually called if not initially set at instantiation. Note that this
        triggers a ``call()``.

        Parameters
        ----------
        size: int
        """
        self.size = size
        self.call()

    def absolute_update(self, value):
        """
        Set the internal value state

        Triggers ``call()``

        Parameters
        ----------
        value: int
        """
        self.value = value
        self.call()

    def relative_update(self, inc=1):
        """
        Delta increment the internal counter

        Triggers ``call()``

        Parameters
        ----------
        inc: int
        """
        self.value += inc
        self.call()

    def call(self, hook_name=None, **kwargs):
        """
        Execute hook(s) with current state

        Each function is passed the internal size and current value

        Parameters
        ----------
        hook_name: str or None
            If given, execute on this hook
        kwargs: passed on to (all) hook(s)
        """
        if not self.hooks:
            return
        kw = self.kw.copy()
        kw.update(kwargs)
        if hook_name:
            if hook_name not in self.hooks:
                return
            return self.hooks[hook_name](self.size, self.value, **kw)
        for hook in self.hooks.values() or []:
            hook(self.size, self.value, **kw)

    def wrap(self, iterable):
        """
        Wrap an iterable to call ``relative_update`` on each iterations

        Parameters
        ----------
        iterable: Iterable
            The iterable that is being wrapped
        """
        for item in iterable:
            self.relative_update()
            yield item

    def branch(self, path_1, path_2, kwargs):
        """
        Set callbacks for child transfers

        If this callback is operating at a higher level, e.g., put, which may
        trigger transfers that can also be monitored. The passed kwargs are
        to be *mutated* to add ``callback=``, if this class supports branching
        to children.

        Parameters
        ----------
        path_1: str
            Child's source path
        path_2: str
            Child's destination path
        kwargs: dict
            arguments passed to child method, e.g., put_file.

        Returns
        -------

        """
        return None

    def no_op(self, *_, **__):
        pass

    def __getattr__(self, item):
        """
        If undefined methods are called on this class, nothing happens
        """
        return self.no_op

    @classmethod
    def as_callback(cls, maybe_callback=None):
        """Transform callback=... into Callback instance

        For the special value of ``None``, return the global instance of
        ``NoOpCallback``. This is an alternative to including
        ``callback=DEFAULT_CALLBACK`` directly in a method signature.
        """
        if maybe_callback is None:
            return DEFAULT_CALLBACK
        return maybe_callback


class NoOpCallback(Callback):
    """
    This implementation of Callback does exactly nothing
    """

    def call(self, *args, **kwargs):
        return None


class DotPrinterCallback(Callback):
    """
    Simple example Callback implementation

    Almost identical to Callback with a hook that prints a char; here we
    demonstrate how the outer layer may print "#" and the inner layer "."
    """

    def __init__(self, chr_to_print="#", **kwargs):
        self.chr = chr_to_print
        super().__init__(**kwargs)

    def branch(self, path_1, path_2, kwargs):
        """Mutate kwargs to add new instance with different print char"""
        kwargs["callback"] = DotPrinterCallback(".")

    def call(self, **kwargs):
        """Just outputs a character"""
        print(self.chr, end="")


class TqdmCallback(Callback):
    """
    A callback to display a progress bar using tqdm

    Parameters
    ----------
    tqdm_kwargs : dict, (optional)
        Any argument accepted by the tqdm constructor.
        See the `tqdm doc <https://tqdm.github.io/docs/tqdm/#__init__>`_.
        Will be forwarded to `tqdm_cls`.
    tqdm_cls: (optional)
        subclass of `tqdm.tqdm`. If not passed, it will default to `tqdm.tqdm`.

    Examples
    --------
    >>> import fsspec
    >>> from fsspec.callbacks import TqdmCallback
    >>> fs = fsspec.filesystem("memory")
    >>> path2distant_data = "/your-path"
    >>> fs.upload(
            ".",
            path2distant_data,
            recursive=True,
            callback=TqdmCallback(),
        )

    You can forward args to tqdm using the ``tqdm_kwargs`` parameter.

    >>> fs.upload(
            ".",
            path2distant_data,
            recursive=True,
            callback=TqdmCallback(tqdm_kwargs={"desc": "Your tqdm description"}),
        )

    You can also customize the progress bar by passing a subclass of `tqdm`.

    .. code-block:: python

        class TqdmFormat(tqdm):
            '''Provides a `total_time` format parameter'''
            @property
            def format_dict(self):
                d = super().format_dict
                total_time = d["elapsed"] * (d["total"] or 0) / max(d["n"], 1)
                d.update(total_time=self.format_interval(total_time) + " in total")
                return d

    >>> with TqdmCallback(
            tqdm_kwargs={
                "desc": "desc",
                "bar_format": "{total_time}: {percentage:.0f}%|{bar}{r_bar}",
            },
            tqdm_cls=TqdmFormat,
        ) as callback:
            fs.upload(".", path2distant_data, recursive=True, callback=callback)
    """

    def __init__(self, tqdm_kwargs=None, *args, **kwargs):
        try:
            from tqdm import tqdm

        except ImportError as exce:
            raise ImportError(
                "Using TqdmCallback requires tqdm to be installed"
            ) from exce

        self._tqdm_cls = kwargs.pop("tqdm_cls", tqdm)
        self._tqdm_kwargs = tqdm_kwargs or {}
        self.tqdm = None
        super().__init__(*args, **kwargs)

    def call(self, *args, **kwargs):
        if self.tqdm is None:
            self.tqdm = self._tqdm_cls(total=self.size, **self._tqdm_kwargs)
        self.tqdm.total = self.size
        self.tqdm.update(self.value - self.tqdm.n)

    def close(self):
        if self.tqdm is not None:
            self.tqdm.close()
            self.tqdm = None

    def __del__(self):
        return self.close()


DEFAULT_CALLBACK = _DEFAULT_CALLBACK = NoOpCallback()
