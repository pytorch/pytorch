import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

g, lb, w, h = _sm.symbols('g lb w h', real=True)
theta, phi, omega, alpha = _me.dynamicsymbols('theta phi omega alpha')
theta_d, phi_d, omega_d, alpha_d = _me.dynamicsymbols('theta_ phi_ omega_ alpha_', 1)
theta_dd, phi_dd = _me.dynamicsymbols('theta_ phi_', 2)
frame_n = _me.ReferenceFrame('n')
body_a_cm = _me.Point('a_cm')
body_a_cm.set_vel(frame_n, 0)
body_a_f = _me.ReferenceFrame('a_f')
body_a = _me.RigidBody('a', body_a_cm, body_a_f, _sm.symbols('m'), (_me.outer(body_a_f.x,body_a_f.x),body_a_cm))
body_b_cm = _me.Point('b_cm')
body_b_cm.set_vel(frame_n, 0)
body_b_f = _me.ReferenceFrame('b_f')
body_b = _me.RigidBody('b', body_b_cm, body_b_f, _sm.symbols('m'), (_me.outer(body_b_f.x,body_b_f.x),body_b_cm))
body_a_f.orient(frame_n, 'Axis', [theta, frame_n.y])
body_b_f.orient(body_a_f, 'Axis', [phi, body_a_f.z])
point_o = _me.Point('o')
la = (lb-h/2)/2
body_a_cm.set_pos(point_o, la*body_a_f.z)
body_b_cm.set_pos(point_o, lb*body_a_f.z)
body_a_f.set_ang_vel(frame_n, omega*frame_n.y)
body_b_f.set_ang_vel(body_a_f, alpha*body_a_f.z)
point_o.set_vel(frame_n, 0)
body_a_cm.v2pt_theory(point_o,frame_n,body_a_f)
body_b_cm.v2pt_theory(point_o,frame_n,body_a_f)
ma = _sm.symbols('ma')
body_a.mass = ma
mb = _sm.symbols('mb')
body_b.mass = mb
iaxx = 1/12*ma*(2*la)**2
iayy = iaxx
iazz = 0
ibxx = 1/12*mb*h**2
ibyy = 1/12*mb*(w**2+h**2)
ibzz = 1/12*mb*w**2
body_a.inertia = (_me.inertia(body_a_f, iaxx, iayy, iazz, 0, 0, 0), body_a_cm)
body_b.inertia = (_me.inertia(body_b_f, ibxx, ibyy, ibzz, 0, 0, 0), body_b_cm)
force_a = body_a.mass*(g*frame_n.z)
force_b = body_b.mass*(g*frame_n.z)
kd_eqs = [theta_d - omega, phi_d - alpha]
forceList = [(body_a.masscenter,body_a.mass*(g*frame_n.z)), (body_b.masscenter,body_b.mass*(g*frame_n.z))]
kane = _me.KanesMethod(frame_n, q_ind=[theta,phi], u_ind=[omega, alpha], kd_eqs = kd_eqs)
fr, frstar = kane.kanes_equations([body_a, body_b], forceList)
zero = fr+frstar
from pydy.system import System
sys = System(kane, constants = {g:9.81, lb:0.2, w:0.2, h:0.1, ma:0.01, mb:0.1},
specifieds={},
initial_conditions={theta:_np.deg2rad(90), phi:_np.deg2rad(0.5), omega:0, alpha:0},
times = _np.linspace(0.0, 10, 10/0.02))

y=sys.integrate()
