import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

m, k, b, g = _sm.symbols('m k b g', real=True)
position, speed = _me.dynamicsymbols('position speed')
position_d, speed_d = _me.dynamicsymbols('position_ speed_', 1)
o = _me.dynamicsymbols('o')
force = o*_sm.sin(_me.dynamicsymbols._t)
frame_ceiling = _me.ReferenceFrame('ceiling')
point_origin = _me.Point('origin')
point_origin.set_vel(frame_ceiling, 0)
particle_block = _me.Particle('block', _me.Point('block_pt'), _sm.Symbol('m'))
particle_block.point.set_pos(point_origin, position*frame_ceiling.x)
particle_block.mass = m
particle_block.point.set_vel(frame_ceiling, speed*frame_ceiling.x)
force_magnitude = m*g-k*position-b*speed+force
force_block = (force_magnitude*frame_ceiling.x).subs({position_d:speed})
kd_eqs = [position_d - speed]
forceList = [(particle_block.point,(force_magnitude*frame_ceiling.x).subs({position_d:speed}))]
kane = _me.KanesMethod(frame_ceiling, q_ind=[position], u_ind=[speed], kd_eqs = kd_eqs)
fr, frstar = kane.kanes_equations([particle_block], forceList)
zero = fr+frstar
from pydy.system import System
sys = System(kane, constants = {m:1.0, k:1.0, b:0.2, g:9.8},
specifieds={_me.dynamicsymbols('t'):lambda x, t: t, o:2},
initial_conditions={position:0.1, speed:-1*1.0},
times = _np.linspace(0.0, 10.0, 10.0/0.01))

y=sys.integrate()
