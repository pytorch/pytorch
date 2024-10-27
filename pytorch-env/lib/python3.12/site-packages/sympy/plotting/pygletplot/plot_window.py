from time import perf_counter


import pyglet.gl as pgl

from sympy.plotting.pygletplot.managed_window import ManagedWindow
from sympy.plotting.pygletplot.plot_camera import PlotCamera
from sympy.plotting.pygletplot.plot_controller import PlotController


class PlotWindow(ManagedWindow):

    def __init__(self, plot, antialiasing=True, ortho=False,
                 invert_mouse_zoom=False, linewidth=1.5, caption="SymPy Plot",
                 **kwargs):
        """
        Named Arguments
        ===============

        antialiasing = True
            True OR False
        ortho = False
            True OR False
        invert_mouse_zoom = False
            True OR False
        """
        self.plot = plot

        self.camera = None
        self._calculating = False

        self.antialiasing = antialiasing
        self.ortho = ortho
        self.invert_mouse_zoom = invert_mouse_zoom
        self.linewidth = linewidth
        self.title = caption
        self.last_caption_update = 0
        self.caption_update_interval = 0.2
        self.drawing_first_object = True

        super().__init__(**kwargs)

    def setup(self):
        self.camera = PlotCamera(self, ortho=self.ortho)
        self.controller = PlotController(self,
                invert_mouse_zoom=self.invert_mouse_zoom)
        self.push_handlers(self.controller)

        pgl.glClearColor(1.0, 1.0, 1.0, 0.0)
        pgl.glClearDepth(1.0)

        pgl.glDepthFunc(pgl.GL_LESS)
        pgl.glEnable(pgl.GL_DEPTH_TEST)

        pgl.glEnable(pgl.GL_LINE_SMOOTH)
        pgl.glShadeModel(pgl.GL_SMOOTH)
        pgl.glLineWidth(self.linewidth)

        pgl.glEnable(pgl.GL_BLEND)
        pgl.glBlendFunc(pgl.GL_SRC_ALPHA, pgl.GL_ONE_MINUS_SRC_ALPHA)

        if self.antialiasing:
            pgl.glHint(pgl.GL_LINE_SMOOTH_HINT, pgl.GL_NICEST)
            pgl.glHint(pgl.GL_POLYGON_SMOOTH_HINT, pgl.GL_NICEST)

        self.camera.setup_projection()

    def on_resize(self, w, h):
        super().on_resize(w, h)
        if self.camera is not None:
            self.camera.setup_projection()

    def update(self, dt):
        self.controller.update(dt)

    def draw(self):
        self.plot._render_lock.acquire()
        self.camera.apply_transformation()

        calc_verts_pos, calc_verts_len = 0, 0
        calc_cverts_pos, calc_cverts_len = 0, 0

        should_update_caption = (perf_counter() - self.last_caption_update >
                                 self.caption_update_interval)

        if len(self.plot._functions.values()) == 0:
            self.drawing_first_object = True

        iterfunctions = iter(self.plot._functions.values())

        for r in iterfunctions:
            if self.drawing_first_object:
                self.camera.set_rot_preset(r.default_rot_preset)
                self.drawing_first_object = False

            pgl.glPushMatrix()
            r._draw()
            pgl.glPopMatrix()

            # might as well do this while we are
            # iterating and have the lock rather
            # than locking and iterating twice
            # per frame:

            if should_update_caption:
                try:
                    if r.calculating_verts:
                        calc_verts_pos += r.calculating_verts_pos
                        calc_verts_len += r.calculating_verts_len
                    if r.calculating_cverts:
                        calc_cverts_pos += r.calculating_cverts_pos
                        calc_cverts_len += r.calculating_cverts_len
                except ValueError:
                    pass

        for r in self.plot._pobjects:
            pgl.glPushMatrix()
            r._draw()
            pgl.glPopMatrix()

        if should_update_caption:
            self.update_caption(calc_verts_pos, calc_verts_len,
                                calc_cverts_pos, calc_cverts_len)
            self.last_caption_update = perf_counter()

        if self.plot._screenshot:
            self.plot._screenshot._execute_saving()

        self.plot._render_lock.release()

    def update_caption(self, calc_verts_pos, calc_verts_len,
            calc_cverts_pos, calc_cverts_len):
        caption = self.title
        if calc_verts_len or calc_cverts_len:
            caption += " (calculating"
            if calc_verts_len > 0:
                p = (calc_verts_pos / calc_verts_len) * 100
                caption += " vertices %i%%" % (p)
            if calc_cverts_len > 0:
                p = (calc_cverts_pos / calc_cverts_len) * 100
                caption += " colors %i%%" % (p)
            caption += ")"
        if self.caption != caption:
            self.set_caption(caption)
