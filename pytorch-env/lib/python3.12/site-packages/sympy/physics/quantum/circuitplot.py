"""Matplotlib based plotting of quantum circuits.

Todo:

* Optimize printing of large circuits.
* Get this to work with single gates.
* Do a better job checking the form of circuits to make sure it is a Mul of
  Gates.
* Get multi-target gates plotting.
* Get initial and final states to plot.
* Get measurements to plot. Might need to rethink measurement as a gate
  issue.
* Get scale and figsize to be handled in a better way.
* Write some tests/examples!
"""

from __future__ import annotations

from sympy.core.mul import Mul
from sympy.external import import_module
from sympy.physics.quantum.gate import Gate, OneQubitGate, CGate, CGateS


__all__ = [
    'CircuitPlot',
    'circuit_plot',
    'labeller',
    'Mz',
    'Mx',
    'CreateOneQubitGate',
    'CreateCGate',
]

np = import_module('numpy')
matplotlib = import_module(
    'matplotlib', import_kwargs={'fromlist': ['pyplot']},
    catch=(RuntimeError,))  # This is raised in environments that have no display.

if np and matplotlib:
    pyplot = matplotlib.pyplot
    Line2D = matplotlib.lines.Line2D
    Circle = matplotlib.patches.Circle

#from matplotlib import rc
#rc('text',usetex=True)

class CircuitPlot:
    """A class for managing a circuit plot."""

    scale = 1.0
    fontsize = 20.0
    linewidth = 1.0
    control_radius = 0.05
    not_radius = 0.15
    swap_delta = 0.05
    labels: list[str] = []
    inits: dict[str, str] = {}
    label_buffer = 0.5

    def __init__(self, c, nqubits, **kwargs):
        if not np or not matplotlib:
            raise ImportError('numpy or matplotlib not available.')
        self.circuit = c
        self.ngates = len(self.circuit.args)
        self.nqubits = nqubits
        self.update(kwargs)
        self._create_grid()
        self._create_figure()
        self._plot_wires()
        self._plot_gates()
        self._finish()

    def update(self, kwargs):
        """Load the kwargs into the instance dict."""
        self.__dict__.update(kwargs)

    def _create_grid(self):
        """Create the grid of wires."""
        scale = self.scale
        wire_grid = np.arange(0.0, self.nqubits*scale, scale, dtype=float)
        gate_grid = np.arange(0.0, self.ngates*scale, scale, dtype=float)
        self._wire_grid = wire_grid
        self._gate_grid = gate_grid

    def _create_figure(self):
        """Create the main matplotlib figure."""
        self._figure = pyplot.figure(
            figsize=(self.ngates*self.scale, self.nqubits*self.scale),
            facecolor='w',
            edgecolor='w'
        )
        ax = self._figure.add_subplot(
            1, 1, 1,
            frameon=True
        )
        ax.set_axis_off()
        offset = 0.5*self.scale
        ax.set_xlim(self._gate_grid[0] - offset, self._gate_grid[-1] + offset)
        ax.set_ylim(self._wire_grid[0] - offset, self._wire_grid[-1] + offset)
        ax.set_aspect('equal')
        self._axes = ax

    def _plot_wires(self):
        """Plot the wires of the circuit diagram."""
        xstart = self._gate_grid[0]
        xstop = self._gate_grid[-1]
        xdata = (xstart - self.scale, xstop + self.scale)
        for i in range(self.nqubits):
            ydata = (self._wire_grid[i], self._wire_grid[i])
            line = Line2D(
                xdata, ydata,
                color='k',
                lw=self.linewidth
            )
            self._axes.add_line(line)
            if self.labels:
                init_label_buffer = 0
                if self.inits.get(self.labels[i]): init_label_buffer = 0.25
                self._axes.text(
                    xdata[0]-self.label_buffer-init_label_buffer,ydata[0],
                    render_label(self.labels[i],self.inits),
                    size=self.fontsize,
                    color='k',ha='center',va='center')
        self._plot_measured_wires()

    def _plot_measured_wires(self):
        ismeasured = self._measurements()
        xstop = self._gate_grid[-1]
        dy = 0.04 # amount to shift wires when doubled
        # Plot doubled wires after they are measured
        for im in ismeasured:
            xdata = (self._gate_grid[ismeasured[im]],xstop+self.scale)
            ydata = (self._wire_grid[im]+dy,self._wire_grid[im]+dy)
            line = Line2D(
                xdata, ydata,
                color='k',
                lw=self.linewidth
            )
            self._axes.add_line(line)
        # Also double any controlled lines off these wires
        for i,g in enumerate(self._gates()):
            if isinstance(g, (CGate, CGateS)):
                wires = g.controls + g.targets
                for wire in wires:
                    if wire in ismeasured and \
                           self._gate_grid[i] > self._gate_grid[ismeasured[wire]]:
                        ydata = min(wires), max(wires)
                        xdata = self._gate_grid[i]-dy, self._gate_grid[i]-dy
                        line = Line2D(
                            xdata, ydata,
                            color='k',
                            lw=self.linewidth
                            )
                        self._axes.add_line(line)
    def _gates(self):
        """Create a list of all gates in the circuit plot."""
        gates = []
        if isinstance(self.circuit, Mul):
            for g in reversed(self.circuit.args):
                if isinstance(g, Gate):
                    gates.append(g)
        elif isinstance(self.circuit, Gate):
            gates.append(self.circuit)
        return gates

    def _plot_gates(self):
        """Iterate through the gates and plot each of them."""
        for i, gate in enumerate(self._gates()):
            gate.plot_gate(self, i)

    def _measurements(self):
        """Return a dict ``{i:j}`` where i is the index of the wire that has
        been measured, and j is the gate where the wire is measured.
        """
        ismeasured = {}
        for i,g in enumerate(self._gates()):
            if getattr(g,'measurement',False):
                for target in g.targets:
                    if target in ismeasured:
                        if ismeasured[target] > i:
                            ismeasured[target] = i
                    else:
                        ismeasured[target] = i
        return ismeasured

    def _finish(self):
        # Disable clipping to make panning work well for large circuits.
        for o in self._figure.findobj():
            o.set_clip_on(False)

    def one_qubit_box(self, t, gate_idx, wire_idx):
        """Draw a box for a single qubit gate."""
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        self._axes.text(
            x, y, t,
            color='k',
            ha='center',
            va='center',
            bbox={"ec": 'k', "fc": 'w', "fill": True, "lw": self.linewidth},
            size=self.fontsize
        )

    def two_qubit_box(self, t, gate_idx, wire_idx):
        """Draw a box for a two qubit gate. Does not work yet.
        """
        # x = self._gate_grid[gate_idx]
        # y = self._wire_grid[wire_idx]+0.5
        print(self._gate_grid)
        print(self._wire_grid)
        # unused:
        # obj = self._axes.text(
        #     x, y, t,
        #     color='k',
        #     ha='center',
        #     va='center',
        #     bbox=dict(ec='k', fc='w', fill=True, lw=self.linewidth),
        #     size=self.fontsize
        # )

    def control_line(self, gate_idx, min_wire, max_wire):
        """Draw a vertical control line."""
        xdata = (self._gate_grid[gate_idx], self._gate_grid[gate_idx])
        ydata = (self._wire_grid[min_wire], self._wire_grid[max_wire])
        line = Line2D(
            xdata, ydata,
            color='k',
            lw=self.linewidth
        )
        self._axes.add_line(line)

    def control_point(self, gate_idx, wire_idx):
        """Draw a control point."""
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        radius = self.control_radius
        c = Circle(
            (x, y),
            radius*self.scale,
            ec='k',
            fc='k',
            fill=True,
            lw=self.linewidth
        )
        self._axes.add_patch(c)

    def not_point(self, gate_idx, wire_idx):
        """Draw a NOT gates as the circle with plus in the middle."""
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        radius = self.not_radius
        c = Circle(
            (x, y),
            radius,
            ec='k',
            fc='w',
            fill=False,
            lw=self.linewidth
        )
        self._axes.add_patch(c)
        l = Line2D(
            (x, x), (y - radius, y + radius),
            color='k',
            lw=self.linewidth
        )
        self._axes.add_line(l)

    def swap_point(self, gate_idx, wire_idx):
        """Draw a swap point as a cross."""
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        d = self.swap_delta
        l1 = Line2D(
            (x - d, x + d),
            (y - d, y + d),
            color='k',
            lw=self.linewidth
        )
        l2 = Line2D(
            (x - d, x + d),
            (y + d, y - d),
            color='k',
            lw=self.linewidth
        )
        self._axes.add_line(l1)
        self._axes.add_line(l2)

def circuit_plot(c, nqubits, **kwargs):
    """Draw the circuit diagram for the circuit with nqubits.

    Parameters
    ==========

    c : circuit
        The circuit to plot. Should be a product of Gate instances.
    nqubits : int
        The number of qubits to include in the circuit. Must be at least
        as big as the largest ``min_qubits`` of the gates.
    """
    return CircuitPlot(c, nqubits, **kwargs)

def render_label(label, inits={}):
    """Slightly more flexible way to render labels.

    >>> from sympy.physics.quantum.circuitplot import render_label
    >>> render_label('q0')
    '$\\\\left|q0\\\\right\\\\rangle$'
    >>> render_label('q0', {'q0':'0'})
    '$\\\\left|q0\\\\right\\\\rangle=\\\\left|0\\\\right\\\\rangle$'
    """
    init = inits.get(label)
    if init:
        return r'$\left|%s\right\rangle=\left|%s\right\rangle$' % (label, init)
    return r'$\left|%s\right\rangle$' % label

def labeller(n, symbol='q'):
    """Autogenerate labels for wires of quantum circuits.

    Parameters
    ==========

    n : int
        number of qubits in the circuit.
    symbol : string
        A character string to precede all gate labels. E.g. 'q_0', 'q_1', etc.

    >>> from sympy.physics.quantum.circuitplot import labeller
    >>> labeller(2)
    ['q_1', 'q_0']
    >>> labeller(3,'j')
    ['j_2', 'j_1', 'j_0']
    """
    return ['%s_%d' % (symbol,n-i-1) for i in range(n)]

class Mz(OneQubitGate):
    """Mock-up of a z measurement gate.

    This is in circuitplot rather than gate.py because it's not a real
    gate, it just draws one.
    """
    measurement = True
    gate_name='Mz'
    gate_name_latex='M_z'

class Mx(OneQubitGate):
    """Mock-up of an x measurement gate.

    This is in circuitplot rather than gate.py because it's not a real
    gate, it just draws one.
    """
    measurement = True
    gate_name='Mx'
    gate_name_latex='M_x'

class CreateOneQubitGate(type):
    def __new__(mcl, name, latexname=None):
        if not latexname:
            latexname = name
        return type(name + "Gate", (OneQubitGate,),
            {'gate_name': name, 'gate_name_latex': latexname})

def CreateCGate(name, latexname=None):
    """Use a lexical closure to make a controlled gate.
    """
    if not latexname:
        latexname = name
    onequbitgate = CreateOneQubitGate(name, latexname)
    def ControlledGate(ctrls,target):
        return CGate(tuple(ctrls),onequbitgate(target))
    return ControlledGate
