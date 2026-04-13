# EOMs.py
# -> The EOM functions for each system.
# Author: Justin Bunting
# Created: 2026/04/12
# Last Modified: 2026/04/12 21:48


from math import sin, cos


# default values (can be overwritten from main file via EOMs.mass[i] = ...)
mass = [20, 20] # kg
length = [3, 3] # m
gravity = 9.81 # kg m s^-2
gamma = (1.3, 1.3) # kg s^-1
k = (400, 400) # kg s^-2


def pendulum(t, x, 
		impulseXY: tuple[float, float] = (0, 0)):
	# take in x (state)
	theta, omega = x
	# the unit vector in the theta direction
	etheta = [cos(theta), sin(theta)]
	impulseETheta = sum(i * j for i, j in zip(impulseXY, etheta))
	# solve for xDot (derivative of the state):
	dtheta = omega
	domega = -gravity / length[0] * sin(theta) + impulseETheta / (mass[0] * length[0])
	return [dtheta, domega]

def dampedPendulum(t, x, 
		impulseXY: tuple[float, float] = (0, 0), 
		gamma: tuple[float, float] | float = (0, 1.3)):
	# take in x (state)
	theta, omega = x
	# the unit vector in the theta directon
	etheta = [cos(theta), sin(theta)]
	impTheta = sum(i * j for i, j in zip(impulseXY, etheta))
	# ensure gamma is only the theta component if tuple
	if isinstance(gamma, tuple):
		gamma = gamma[1]
	# solve for xDot (derivative of the state):
	dtheta = omega
	domega = (impTheta / (mass[0] * length[0])) - (gamma / mass[0] * omega) - (gravity / length[0] * sin(theta))
	return [dtheta, domega]
	
def springPendulum(t, x, 
		impulseXY: tuple[float, float] = (0, 0), 
		gamma: tuple[float, float] = (0, 0), 
		k: tuple[float, float] | float = 1):
	# take in x
	r, rdot, theta, omega = x
	# get unit vectors for impulse decomposition
	er = [sin(theta), -cos(theta)]
	etheta = [cos(theta), sin(theta)]
	impR = sum(i * j for i, j in zip(impulseXY, er))
	impTheta = sum(i * j for i, j in zip(impulseXY, etheta))
	# ensure k is only the first spring constant if set
	if isinstance(k, tuple):
		k = k[0]
	# solve for xDot (derivative of the state):
	dr = rdot
	drdot = gravity * cos(theta) - k / mass[0] * (r - length[0]) + impR / mass[0] - gamma[0] / mass[0] * rdot + r * omega**2
	dtheta = omega
	domega = - gravity / r * sin(theta) + impTheta / (mass[0] * r) - gamma[1] / mass[0] * omega - 2 / r * rdot * omega
	return [dr, drdot, dtheta, domega]

def doublePendulum(t, x,
		impulseXY: tuple[float, float] = (0, 0),
		gamma: tuple[float, float] = (0, 0)):
	# take in x
	theta, thetaDot, alpha, alphaDot = x
	# get unit vectors for impulse decomp
	er = [sin(theta), -cos(theta)]
	etheta = [cos(theta), sin(theta)]
	impR = sum(i * j for i, j in zip(impulseXY, er))
	impTheta = sum(i * j for i, j in zip(impulseXY, etheta))
	# get useful math terms
	mu = (1 + mass[0]/mass[1])
	sinDiff = sin(theta-alpha)
	cosDiff = cos(theta-alpha)
	# solve for non-conservative contributions/forces
	QncTheta = (
		length[0] * impTheta +
		-gamma[0] * length[0]**2 * thetaDot +
		-gamma[1] * length[0] * length[1] * alphaDot * cosDiff
	)
	QncAlpha = (
		length[1] * impTheta * cosDiff +
		length[1] * impR * sinDiff +
		-gamma[0] * length[0] * length[1] * thetaDot * cosDiff +
		-gamma[1] * length[1]**2 * alphaDot
	)
	# since Cramer's rule was used, just compute each term of Cramers and solve that way
	a11 = mu
	a12 = cosDiff
	a21 = cosDiff
	a22 = 1
	b1 = (
		QncTheta / (length[0] * mass[1]) +
		-length[1] * alphaDot**2 * sinDiff +
		-mu * gravity * sin(theta)
	)
	b2 = (
		QncAlpha / (length[1] * mass[1]) +
		length[0] * thetaDot**2 * sinDiff +
		-gravity * sin(alpha)
	)
	# solve for xDot (derivative of the state):
	dTheta = thetaDot
	dThetaDot = (b1*a22 - b2*a12) / (length[0] * (a11*a22 - a12*a21))
	dAlpha = alphaDot
	dAlphaDot = (a11*b2 - a21*b1) / (length[1] * (a11*a22 - a12*a21))
	return [dTheta, dThetaDot, dAlpha, dAlphaDot]