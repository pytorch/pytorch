from sympy.physics.units import Dimension


angle = Dimension(name="angle")  # type: Dimension

# base dimensions (MKS)
length = Dimension(name="length", symbol="L")
mass = Dimension(name="mass", symbol="M")
time = Dimension(name="time", symbol="T")

# base dimensions (MKSA not in MKS)
current = Dimension(name='current', symbol='I')  # type: Dimension

# other base dimensions:
temperature = Dimension("temperature", "T")  # type: Dimension
amount_of_substance = Dimension("amount_of_substance")  # type: Dimension
luminous_intensity = Dimension("luminous_intensity")  # type: Dimension

# derived dimensions (MKS)
velocity = Dimension(name="velocity")
acceleration = Dimension(name="acceleration")
momentum = Dimension(name="momentum")
force = Dimension(name="force", symbol="F")
energy = Dimension(name="energy", symbol="E")
power = Dimension(name="power")
pressure = Dimension(name="pressure")
frequency = Dimension(name="frequency", symbol="f")
action = Dimension(name="action", symbol="A")
area = Dimension("area")
volume = Dimension("volume")

# derived dimensions (MKSA not in MKS)
voltage = Dimension(name='voltage', symbol='U')  # type: Dimension
impedance = Dimension(name='impedance', symbol='Z')  # type: Dimension
conductance = Dimension(name='conductance', symbol='G')  # type: Dimension
capacitance = Dimension(name='capacitance')  # type: Dimension
inductance = Dimension(name='inductance')  # type: Dimension
charge = Dimension(name='charge', symbol='Q')  # type: Dimension
magnetic_density = Dimension(name='magnetic_density', symbol='B')  # type: Dimension
magnetic_flux = Dimension(name='magnetic_flux')  # type: Dimension

# Dimensions in information theory:
information = Dimension(name='information')  # type: Dimension
