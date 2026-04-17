# EOMs.py
# -> The EOM functions for each system.
# Author: Justin Bunting
# Created: 2026/04/12
# Last Modified: 2026/04/17 14:52


from math import sin, cos
from itertools import islice, cycle
import numpy as np


# default values (can be overwritten from main file via EOMs.mass[i] = ...)
mass = [20, 30, 400] # kg
length = [3, 3] # m
gravity = 9.81 # kg m s^-2
gamma = (1.3, 1.3) # kg s^-1
k = (400, 400) # kg s^-2
G = 6.674e-11 # m^3 kg^-1 s^-2


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

def nBodyProblem(t, x,
		massScaling: float,
		forceScaling: float,
		windowDimsMeters: tuple[float, float],
		impulseXY: tuple[float, float] = (0, 0),
		impulseBody: list[int] = [0],
		n: int = 0,
		masses: list[float] = [0]):
	n = n if not n == 0 else len(x) / 4
	xBar = x
	
	# extract x's, y's (length n)
	cols = np.array(xBar).reshape(-1, 4).T # 4xn
	x, xDot, y, yDot = cols
	m = masses if len(masses) == n else (mass[:n] if len(mass) >= n else list(islice(cycle(mass), n)))
	m = [mi * massScaling for mi in m]
	xBarDot = np.array(xBar) # create 'empty' array
	
	for i in range(n):
		# build mTilde, xTilde, yTilde (length n-1)
		mTilde = np.array([*m[:i], *m[i+1:]]).reshape(1, -1) # just exclude m[i] (1xn-1)
		xTilde = np.array([x[i] - xj for xj in [*x[:i], *x[i+1:]]]).reshape(-1, 1) # (n-1x1)
		yTilde = np.array([y[i] - yj for yj in [*y[:i], *y[i+1:]]]).reshape(-1, 1) # (n-1x1)
		
		# determine gravitational force term
		rCubed = (xTilde**2 + yTilde**2)**(3/2)
		accX = (-G * (mTilde @ (xTilde / rCubed))).item()
		accY = (-G * (mTilde @ (yTilde / rCubed))).item()
		
		# add additional force to push masses outside of screen back on, as well as impulse forces
		# the force which pushes masses back onto the screen shall be inversely proportional to v dot edge if v dot edge > 0 and out of bounds
		fExtX = (
			(impulseXY[0] * forceScaling / m[i] if i in impulseBody else 0) +
			1e2 * forceScaling / m[i] * (xDot[i] if x[i] > windowDimsMeters[0] * 0.9 and xDot[i] > 0 else (-xDot[i] if x[i] < windowDimsMeters[0] * 0.1 and xDot[i] < 0 else 0))
		)
		fExtY = (
			(impulseXY[1] * forceScaling / m[i] if i in impulseBody else 0) +
			1e2 * forceScaling / m[i] * (yDot[i] if y[i] > windowDimsMeters[1] * 0.9 and yDot[i] > 0 else (-yDot[i] if y[i] < windowDimsMeters[1] * 0.1 and xDot[i] < 0 else 0))
		)
		
		# find this portion of the solution/output
		xBarDot[i*4:i*4+4] = [
			xDot[i],
			accX + fExtX,
			yDot[i],
			accY + fExtY
		]
	
	return xBarDot

def doubleSpringPendulum(t, x,
		impulseXY: tuple[float, float] = (0, 0),
		gamma: tuple[float, float] = (0, 0), # spring friction/dashpot
		gammaD: tuple[float, float] = (0, 0), # air drag
		k: tuple[float, float] = (400, 400)):
	# unpack x
	r1, r1Dot, theta, thetaDot, r2, r2Dot, alpha, alphaDot = x
	
	# collect useful math terms
	m1, m2 = mass[0], mass[1]
	k1, k2 = k[0], k[1]
	l1, l2 = length[0], length[1]
	M = (m1 + m2)
	cosDiff = cos(alpha - theta)
	sinDiff = sin(alpha - theta)
	
	# solve for nonconservative contributions/forces
	Qnc = np.array([-gammaD[0] * r1Dot - gamma[0] * r1Dot - gammaD[1] * (r1Dot + r2Dot * cosDiff - r2 * alphaDot * sinDiff) - gamma[1] * r2Dot * cosDiff + impulseXY[0] * sin(theta) - impulseXY[1] * cos(theta),
				 	-gammaD[0] * r1**2 * thetaDot - gammaD[1] * r1 * (r1 * thetaDot + r2Dot * sinDiff + r2 * alphaDot * cosDiff) - gamma[1] * r1 * r2Dot * sinDiff + impulseXY[0] * cos(theta) + impulseXY[1] * sin(theta),
					-gammaD[1] * (r1Dot * cosDiff + r1 * thetaDot * sinDiff + r2Dot) - gamma[1] * r2Dot + impulseXY[0] * sin(alpha) - impulseXY[1] * cos(alpha),
					-gammaD[1] * r2 * (r1 * thetaDot * cosDiff - r1Dot * sinDiff + r2 * alphaDot) + impulseXY[0] * r2 * cos(alpha) + impulseXY[1] * r2 * sin(alpha)]).reshape(-1, 1) # 4 (n-1) rows, 1 col
	
	# construct Cramer's rule matrix and vector (A xDot = b', where b' = b + Qnc)
	A = np.array([	M, 					0, 						m2 * cosDiff, 		-m2 * r2 * sinDiff,
			   		0, 					M * r1**2, 				m2 * r1 * sinDiff, 	m2 * r1 * r2 * cosDiff,
					m2 * cosDiff, 		m2 * r1 * sinDiff,		m2, 				0,
					-m2 * r2 * sinDiff,	m2 * r1 * r2 * cosDiff,	0,					m2 * r2**2]).reshape(4, 4)
	
	b = np.array([	M * r1 * thetaDot**2 + 2 * m2 * r2Dot * alphaDot * sinDiff + m2 * r2 * alphaDot**2 * cosDiff + M * gravity * cos(theta) - k1*(r1 - l1),
			   		-2 * M * r1 * r1Dot * thetaDot - 2 * m2 * r1 * r2Dot * alphaDot * cosDiff + m2 * r1 * r2 * alphaDot**2 * sinDiff - M * gravity * r1 * sin(theta),
					-2 * m2 * r1Dot * thetaDot * sinDiff + m2 * r1 * thetaDot**2 * cosDiff + m2 * r2 * alphaDot**2 + m2 * gravity * cos(alpha) - k2*(r2-l2),
					-2 * m2 * r2 * r2Dot * alphaDot - 2 * m2 * r1Dot * r2 * thetaDot * cosDiff - m2 * r1 * r2 * thetaDot**2 * sinDiff - m2 * gravity * r2 * sin(alpha)]).reshape(-1, 1)
	
	bP = b + Qnc
	
	detA = np.linalg.det(A)
	xDotDot = np.zeros(4)
	for i in range(4):
		Ai = A.copy()
		Ai[:, i] = bP[:, 0] # replace column i with bP
		xDotDot[i] = np.linalg.det(Ai) / detA
	
	return [r1Dot, 		xDotDot[0],	# r1DotDot
		 	thetaDot, 	xDotDot[1],	# thetaDotDot
			r2Dot, 		xDotDot[2],	# r2DotDot
			alphaDot, 	xDotDot[3]]	# alphaDotDot