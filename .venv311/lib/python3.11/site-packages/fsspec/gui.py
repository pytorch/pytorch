import ast
import contextlib
import logging
import os
import re
from collections.abc import Sequence
from typing import ClassVar

import panel as pn

from .core import OpenFile, get_filesystem_class, split_protocol
from .registry import known_implementations

pn.extension()
logger = logging.getLogger("fsspec.gui")


class SigSlot:
    """Signal-slot mixin, for Panel event passing

    Include this class in a widget manager's superclasses to be able to
    register events and callbacks on Panel widgets managed by that class.

    The method ``_register`` should be called as widgets are added, and external
    code should call ``connect`` to associate callbacks.

    By default, all signals emit a DEBUG logging statement.
    """

    # names of signals that this class may emit each of which must be
    # set by _register for any new instance
    signals: ClassVar[Sequence[str]] = []
    # names of actions that this class may respond to
    slots: ClassVar[Sequence[str]] = []

    # each of which must be a method name

    def __init__(self):
        self._ignoring_events = False
        self._sigs = {}
        self._map = {}
        self._setup()

    def _setup(self):
        """Create GUI elements and register signals"""
        self.panel = pn.pane.PaneBase()
        # no signals to set up in the base class

    def _register(
        self, widget, name, thing="value", log_level=logging.DEBUG, auto=False
    ):
        """Watch the given attribute of a widget and assign it a named event

        This is normally called at the time a widget is instantiated, in the
        class which owns it.

        Parameters
        ----------
        widget : pn.layout.Panel or None
            Widget to watch. If None, an anonymous signal not associated with
            any widget.
        name : str
            Name of this event
        thing : str
            Attribute of the given widget to watch
        log_level : int
            When the signal is triggered, a logging event of the given level
            will be fired in the dfviz logger.
        auto : bool
            If True, automatically connects with a method in this class of the
            same name.
        """
        if name not in self.signals:
            raise ValueError(f"Attempt to assign an undeclared signal: {name}")
        self._sigs[name] = {
            "widget": widget,
            "callbacks": [],
            "thing": thing,
            "log": log_level,
        }
        wn = "-".join(
            [
                getattr(widget, "name", str(widget)) if widget is not None else "none",
                thing,
            ]
        )
        self._map[wn] = name
        if widget is not None:
            widget.param.watch(self._signal, thing, onlychanged=True)
        if auto and hasattr(self, name):
            self.connect(name, getattr(self, name))

    def _repr_mimebundle_(self, *args, **kwargs):
        """Display in a notebook or a server"""
        try:
            return self.panel._repr_mimebundle_(*args, **kwargs)
        except (ValueError, AttributeError) as exc:
            raise NotImplementedError(
                "Panel does not seem to be set up properly"
            ) from exc

    def connect(self, signal, slot):
        """Associate call back with given event

        The callback must be a function which takes the "new" value of the
        watched attribute as the only parameter. If the callback return False,
        this cancels any further processing of the given event.

        Alternatively, the callback can be a string, in which case it means
        emitting the correspondingly-named event (i.e., connect to self)
        """
        self._sigs[signal]["callbacks"].append(slot)

    def _signal(self, event):
        """This is called by a an action on a widget

        Within an self.ignore_events context, nothing happens.

        Tests can execute this method by directly changing the values of
        widget components.
        """
        if not self._ignoring_events:
            wn = "-".join([event.obj.name, event.name])
            if wn in self._map and self._map[wn] in self._sigs:
                self._emit(self._map[wn], event.new)

    @contextlib.contextmanager
    def ignore_events(self):
        """Temporarily turn off events processing in this instance

        (does not propagate to children)
        """
        self._ignoring_events = True
        try:
            yield
        finally:
            self._ignoring_events = False

    def _emit(self, sig, value=None):
        """An event happened, call its callbacks

        This method can be used in tests to simulate message passing without
        directly changing visual elements.

        Calling of callbacks will halt whenever one returns False.
        """
        logger.log(self._sigs[sig]["log"], f"{sig}: {value}")
        for callback in self._sigs[sig]["callbacks"]:
            if isinstance(callback, str):
                self._emit(callback)
            else:
                try:
                    # running callbacks should not break the interface
                    ret = callback(value)
                    if ret is False:
                        break
                except Exception as e:
                    logger.exception(
                        "Exception (%s) while executing callback for signal: %s",
                        e,
                        sig,
                    )

    def show(self, threads=False):
        """Open a new browser tab and display this instance's interface"""
        self.panel.show(threads=threads, verbose=False)
        return self


class SingleSelect(SigSlot):
    """A multiselect which only allows you to select one item for an event"""

    signals = ["_selected", "selected"]  # the first is internal
    slots = ["set_options", "set_selection", "add", "clear", "select"]

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__()

    def _setup(self):
        self.panel = pn.widgets.MultiSelect(**self.kwargs)
        self._register(self.panel, "_selected", "value")
        self._register(None, "selected")
        self.connect("_selected", self.select_one)

    def _signal(self, *args, **kwargs):
        super()._signal(*args, **kwargs)

    def select_one(self, *_):
        with self.ignore_events():
            val = [self.panel.value[-1]] if self.panel.value else []
            self.panel.value = val
        self._emit("selected", self.panel.value)

    def set_options(self, options):
        self.panel.options = options

    def clear(self):
        self.panel.options = []

    @property
    def value(self):
        return self.panel.value

    def set_selection(self, selection):
        self.panel.value = [selection]


class FileSelector(SigSlot):
    """Panel-based graphical file selector widget

    Instances of this widget are interactive and can be displayed in jupyter by having
    them as the output of a cell,  or in a separate browser tab using ``.show()``.
    """

    signals = [
        "protocol_changed",
        "selection_changed",
        "directory_entered",
        "home_clicked",
        "up_clicked",
        "go_clicked",
        "filters_changed",
    ]
    slots = ["set_filters", "go_home"]

    def __init__(self, url=None, filters=None, ignore=None, kwargs=None):
        """

        Parameters
        ----------
        url : str (optional)
            Initial value of the URL to populate the dialog; should include protocol
        filters : list(str) (optional)
            File endings to include in the listings. If not included, all files are
            allowed. Does not affect directories.
            If given, the endings will appear as checkboxes in the interface
        ignore : list(str) (optional)
            Regex(s) of file basename patterns to ignore, e.g., "\\." for typical
            hidden files on posix
        kwargs : dict (optional)
            To pass to file system instance
        """
        if url:
            self.init_protocol, url = split_protocol(url)
        else:
            self.init_protocol, url = "file", os.getcwd()
        self.init_url = url
        self.init_kwargs = (kwargs if isinstance(kwargs, str) else str(kwargs)) or "{}"
        self.filters = filters
        self.ignore = [re.compile(i) for i in ignore or []]
        self._fs = None
        super().__init__()

    def _setup(self):
        self.url = pn.widgets.TextInput(
            name="url",
            value=self.init_url,
            align="end",
            sizing_mode="stretch_width",
            width_policy="max",
        )
        self.protocol = pn.widgets.Select(
            options=sorted(known_implementations),
            value=self.init_protocol,
            name="protocol",
            align="center",
        )
        self.kwargs = pn.widgets.TextInput(
            name="kwargs", value=self.init_kwargs, align="center"
        )
        self.go = pn.widgets.Button(name="⇨", align="end", width=45)
        self.main = SingleSelect(size=10)
        self.home = pn.widgets.Button(name="🏠", width=40, height=30, align="end")
        self.up = pn.widgets.Button(name="‹", width=30, height=30, align="end")

        self._register(self.protocol, "protocol_changed", auto=True)
        self._register(self.go, "go_clicked", "clicks", auto=True)
        self._register(self.up, "up_clicked", "clicks", auto=True)
        self._register(self.home, "home_clicked", "clicks", auto=True)
        self._register(None, "selection_changed")
        self.main.connect("selected", self.selection_changed)
        self._register(None, "directory_entered")
        self.prev_protocol = self.protocol.value
        self.prev_kwargs = self.storage_options

        self.filter_sel = pn.widgets.CheckBoxGroup(
            value=[], options=[], inline=False, align="end", width_policy="min"
        )
        self._register(self.filter_sel, "filters_changed", auto=True)

        self.panel = pn.Column(
            pn.Row(self.protocol, self.kwargs),
            pn.Row(self.home, self.up, self.url, self.go, self.filter_sel),
            self.main.panel,
        )
        self.set_filters(self.filters)
        self.go_clicked()

    def set_filters(self, filters=None):
        self.filters = filters
        if filters:
            self.filter_sel.options = filters
            self.filter_sel.value = filters
        else:
            self.filter_sel.options = []
            self.filter_sel.value = []

    @property
    def storage_options(self):
        """Value of the kwargs box as a dictionary"""
        return ast.literal_eval(self.kwargs.value) or {}

    @property
    def fs(self):
        """Current filesystem instance"""
        if self._fs is None:
            cls = get_filesystem_class(self.protocol.value)
            self._fs = cls(**self.storage_options)
        return self._fs

    @property
    def urlpath(self):
        """URL of currently selected item"""
        return (
            (f"{self.protocol.value}://{self.main.value[0]}")
            if self.main.value
            else None
        )

    def open_file(self, mode="rb", compression=None, encoding=None):
        """Create OpenFile instance for the currently selected item

        For example, in a notebook you might do something like

        .. code-block::

            [ ]: sel = FileSelector(); sel

            # user selects their file

            [ ]: with sel.open_file('rb') as f:
            ...      out = f.read()

        Parameters
        ----------
        mode: str (optional)
            Open mode for the file.
        compression: str (optional)
            The interact with the file as compressed. Set to 'infer' to guess
            compression from the file ending
        encoding: str (optional)
            If using text mode, use this encoding; defaults to UTF8.
        """
        if self.urlpath is None:
            raise ValueError("No file selected")
        return OpenFile(self.fs, self.urlpath, mode, compression, encoding)

    def filters_changed(self, values):
        self.filters = values
        self.go_clicked()

    def selection_changed(self, *_):
        if self.urlpath is None:
            return
        if self.fs.isdir(self.urlpath):
            self.url.value = self.fs._strip_protocol(self.urlpath)
        self.go_clicked()

    def go_clicked(self, *_):
        if (
            self.prev_protocol != self.protocol.value
            or self.prev_kwargs != self.storage_options
        ):
            self._fs = None  # causes fs to be recreated
            self.prev_protocol = self.protocol.value
            self.prev_kwargs = self.storage_options
        listing = sorted(
            self.fs.ls(self.url.value, detail=True), key=lambda x: x["name"]
        )
        listing = [
            l
            for l in listing
            if not any(i.match(l["name"].rsplit("/", 1)[-1]) for i in self.ignore)
        ]
        folders = {
            "📁 " + o["name"].rsplit("/", 1)[-1]: o["name"]
            for o in listing
            if o["type"] == "directory"
        }
        files = {
            "📄 " + o["name"].rsplit("/", 1)[-1]: o["name"]
            for o in listing
            if o["type"] == "file"
        }
        if self.filters:
            files = {
                k: v
                for k, v in files.items()
                if any(v.endswith(ext) for ext in self.filters)
            }
        self.main.set_options(dict(**folders, **files))

    def protocol_changed(self, *_):
        self._fs = None
        self.main.options = []
        self.url.value = ""

    def home_clicked(self, *_):
        self.protocol.value = self.init_protocol
        self.kwargs.value = self.init_kwargs
        self.url.value = self.init_url
        self.go_clicked()

    def up_clicked(self, *_):
        self.url.value = self.fs._parent(self.url.value)
        self.go_clicked()
