from pyglet.window import key
from pyglet.window.mouse import LEFT, RIGHT, MIDDLE
from sympy.plotting.pygletplot.util import get_direction_vectors, get_basis_vectors


class PlotController:

    normal_mouse_sensitivity = 4.0
    modified_mouse_sensitivity = 1.0

    normal_key_sensitivity = 160.0
    modified_key_sensitivity = 40.0

    keymap = {
        key.LEFT: 'left',
        key.A: 'left',
        key.NUM_4: 'left',

        key.RIGHT: 'right',
        key.D: 'right',
        key.NUM_6: 'right',

        key.UP: 'up',
        key.W: 'up',
        key.NUM_8: 'up',

        key.DOWN: 'down',
        key.S: 'down',
        key.NUM_2: 'down',

        key.Z: 'rotate_z_neg',
        key.NUM_1: 'rotate_z_neg',

        key.C: 'rotate_z_pos',
        key.NUM_3: 'rotate_z_pos',

        key.Q: 'spin_left',
        key.NUM_7: 'spin_left',
        key.E: 'spin_right',
        key.NUM_9: 'spin_right',

        key.X: 'reset_camera',
        key.NUM_5: 'reset_camera',

        key.NUM_ADD: 'zoom_in',
        key.PAGEUP: 'zoom_in',
        key.R: 'zoom_in',

        key.NUM_SUBTRACT: 'zoom_out',
        key.PAGEDOWN: 'zoom_out',
        key.F: 'zoom_out',

        key.RSHIFT: 'modify_sensitivity',
        key.LSHIFT: 'modify_sensitivity',

        key.F1: 'rot_preset_xy',
        key.F2: 'rot_preset_xz',
        key.F3: 'rot_preset_yz',
        key.F4: 'rot_preset_perspective',

        key.F5: 'toggle_axes',
        key.F6: 'toggle_axe_colors',

        key.F8: 'save_image'
    }

    def __init__(self, window, *, invert_mouse_zoom=False, **kwargs):
        self.invert_mouse_zoom = invert_mouse_zoom
        self.window = window
        self.camera = window.camera
        self.action = {
            # Rotation around the view Y (up) vector
            'left': False,
            'right': False,
            # Rotation around the view X vector
            'up': False,
            'down': False,
            # Rotation around the view Z vector
            'spin_left': False,
            'spin_right': False,
            # Rotation around the model Z vector
            'rotate_z_neg': False,
            'rotate_z_pos': False,
            # Reset to the default rotation
            'reset_camera': False,
            # Performs camera z-translation
            'zoom_in': False,
            'zoom_out': False,
            # Use alternative sensitivity (speed)
            'modify_sensitivity': False,
            # Rotation presets
            'rot_preset_xy': False,
            'rot_preset_xz': False,
            'rot_preset_yz': False,
            'rot_preset_perspective': False,
            # axes
            'toggle_axes': False,
            'toggle_axe_colors': False,
            # screenshot
            'save_image': False
        }

    def update(self, dt):
        z = 0
        if self.action['zoom_out']:
            z -= 1
        if self.action['zoom_in']:
            z += 1
        if z != 0:
            self.camera.zoom_relative(z/10.0, self.get_key_sensitivity()/10.0)

        dx, dy, dz = 0, 0, 0
        if self.action['left']:
            dx -= 1
        if self.action['right']:
            dx += 1
        if self.action['up']:
            dy -= 1
        if self.action['down']:
            dy += 1
        if self.action['spin_left']:
            dz += 1
        if self.action['spin_right']:
            dz -= 1

        if not self.is_2D():
            if dx != 0:
                self.camera.euler_rotate(dx*dt*self.get_key_sensitivity(),
                                         *(get_direction_vectors()[1]))
            if dy != 0:
                self.camera.euler_rotate(dy*dt*self.get_key_sensitivity(),
                                         *(get_direction_vectors()[0]))
            if dz != 0:
                self.camera.euler_rotate(dz*dt*self.get_key_sensitivity(),
                                         *(get_direction_vectors()[2]))
        else:
            self.camera.mouse_translate(0, 0, dx*dt*self.get_key_sensitivity(),
                                        -dy*dt*self.get_key_sensitivity())

        rz = 0
        if self.action['rotate_z_neg'] and not self.is_2D():
            rz -= 1
        if self.action['rotate_z_pos'] and not self.is_2D():
            rz += 1

        if rz != 0:
            self.camera.euler_rotate(rz*dt*self.get_key_sensitivity(),
                                     *(get_basis_vectors()[2]))

        if self.action['reset_camera']:
            self.camera.reset()

        if self.action['rot_preset_xy']:
            self.camera.set_rot_preset('xy')
        if self.action['rot_preset_xz']:
            self.camera.set_rot_preset('xz')
        if self.action['rot_preset_yz']:
            self.camera.set_rot_preset('yz')
        if self.action['rot_preset_perspective']:
            self.camera.set_rot_preset('perspective')

        if self.action['toggle_axes']:
            self.action['toggle_axes'] = False
            self.camera.axes.toggle_visible()

        if self.action['toggle_axe_colors']:
            self.action['toggle_axe_colors'] = False
            self.camera.axes.toggle_colors()

        if self.action['save_image']:
            self.action['save_image'] = False
            self.window.plot.saveimage()

        return True

    def get_mouse_sensitivity(self):
        if self.action['modify_sensitivity']:
            return self.modified_mouse_sensitivity
        else:
            return self.normal_mouse_sensitivity

    def get_key_sensitivity(self):
        if self.action['modify_sensitivity']:
            return self.modified_key_sensitivity
        else:
            return self.normal_key_sensitivity

    def on_key_press(self, symbol, modifiers):
        if symbol in self.keymap:
            self.action[self.keymap[symbol]] = True

    def on_key_release(self, symbol, modifiers):
        if symbol in self.keymap:
            self.action[self.keymap[symbol]] = False

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & LEFT:
            if self.is_2D():
                self.camera.mouse_translate(x, y, dx, dy)
            else:
                self.camera.spherical_rotate((x - dx, y - dy), (x, y),
                                             self.get_mouse_sensitivity())
        if buttons & MIDDLE:
            self.camera.zoom_relative([1, -1][self.invert_mouse_zoom]*dy,
                                      self.get_mouse_sensitivity()/20.0)
        if buttons & RIGHT:
            self.camera.mouse_translate(x, y, dx, dy)

    def on_mouse_scroll(self, x, y, dx, dy):
        self.camera.zoom_relative([1, -1][self.invert_mouse_zoom]*dy,
                                  self.get_mouse_sensitivity())

    def is_2D(self):
        functions = self.window.plot._functions
        for i in functions:
            if len(functions[i].i_vars) > 1 or len(functions[i].d_vars) > 2:
                return False
        return True
