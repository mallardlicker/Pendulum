# Pendulum.py
# -> Attempt to create a pendulum simulation in pyglet.
# Author: Justin Bunting
# Created: 2026/03/30
# Last Modified: 2026/04/12 21:12


from math import sin, cos, tan, radians, degrees

import pyglet
from pyglet.window import key, mouse

from Shapes import *
from System import *


# 	initialize window

winSize = 1400 # on Mac, screen coords and render coords differ: want half-size window
window = pyglet.window.Window(
	width=winSize/2,
	height=winSize/2,
	visible=True,
	resizable=False,
	caption='Physical Systems'
)


#	diff eq's and global values

# default pendulum values
mass = [20, 20] # kilograms
length = [3, 3] # meters
gravity = 9.81 # kg m s^-2

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


# 	init system and define each scene

# set up the system (default has a pendulum running in the background, but no image)
pSys = PhysicsSystem(function=pendulum, x0=[radians(20), 0])

# scene loader based on key press
def loadScene(symbol):
	
	# render values
	origin = (winSize//2, winSize//10*5) # 10*8
	pxLength = [l * metersToPixels for l in length]
	startingAngle = [-40, 45]
	
	# load each physics system
	if symbol in (key._1, key._2):
		# set initial conditions and equation
		x0 = [radians(startingAngle[0]), 0]
		if symbol is key._1:
			pSys.reset(pendulum, x0)
		else:
			pSys.reset(dampedPendulum, x0)
		
		# generate objects (and set anchors as defined by problem)
		basePoint = Point(	origin[0], origin[1], radius=15)
		bar = 		Bar(  	origin[0], origin[1], pxLength[0], rotationAnch=180)
		endPoint1 =  Point(	origin[0], origin[1], radius=mass[0], yAnch=-pxLength[0], rotationAnch=180, color=(255, 255, 255, 255))
		endPoint2 = Point(	origin[0], origin[1], radius=mass[0]-5, yAnch=-pxLength[0], rotationAnch=180, color=(200, 30, 30, 255))
		
		# attach objects and update actions
		pSys.addObject(bar)
		pSys.addAction(bar.setAngle)
		pSys.addObject(basePoint) # add after bar to appear above
		pSys.addObject(endPoint1)
		pSys.addAction(endPoint1.setAngle)
		pSys.addObject(endPoint2)
		pSys.addAction(endPoint2.setAngle)
		
		# set simulation values
		pSys.setArgument((0, 0), 'impulseXY', tuple[float, float], False) # impulse is NOT persistent
		if symbol is key._2:
			pSys.setArgument(1.3, 'gamma', float)
		
	elif symbol is key._3:
		# set initial conditions and equation -> use length, not pxLength
		x0 = [length[0], 0, radians(startingAngle[0]), 0]
		pSys.reset(springPendulum, x0)
		
		# generate objects (and set anchors)
		basePoint = Point(	origin[0], origin[1], radius=15)
		spring = 	Spring(	origin[0], origin[1], pxLength[0], rotationAnch=180)
		endPoint1 = Point(	origin[0], origin[1], radius=mass[0], yAnch=-pxLength[0], rotationAnch=0, color=(255, 255, 255, 255))
		endPoint2 = Point(	origin[0], origin[1], radius=mass[0]-5, yAnch=-pxLength[0], rotationAnch=0, color=(200, 30, 30, 255))
		
		# attach objects and update actions
		pSys.addObject(spring)
		pSys.addAction(spring.setLength, 0, ('m2px',))
		pSys.addAction(spring.setAngle, 2)
		pSys.addObject(basePoint) # add after spring to appear above
		pSys.addObject(endPoint1)
		pSys.addAction(endPoint1.setAnchorY, 0, ('m2px',))
		pSys.addAction(endPoint1.setAngle, 2)
		pSys.addObject(endPoint2)
		pSys.addAction(endPoint2.setAnchorY, 0, ('m2px',))
		pSys.addAction(endPoint2.setAngle, 2)
		
		# set simulation values
		pSys.setArgument((0, 0), 'impulseXY', tuple[float, float], False) # impulse is NOT persistent
		pSys.setArgument((1.3, 1.3), 'gamma', tuple[float, float])
		pSys.setArgument(400, 'k', float)
	
	elif symbol is key._4:
		# set initial conditions and equation
		x0 = [radians(startingAngle[0]), 0, radians(startingAngle[1]), 0] # theta, thetaDot, alpha, alphaDot
		pSys.reset(doublePendulum, x0)
		
		# generate objects (and set anchors)
		midLoc = (origin[0] + pxLength[0] * sin(radians(startingAngle[0])), origin[1] - pxLength[0] * cos(radians(startingAngle[0])))
		basePoint = Point(	origin[0], origin[1], radius=15)
		bar1 = 		Bar(  	origin[0], origin[1], length=pxLength[0], rotationAnch=180)
		midPoint =	Point(	origin[0], origin[1], radius=mass[0], yAnch=pxLength[0], rotationAnch=0, color=(255, 255, 255, 255))
		bar2 =		Bar(	midLoc[0], midLoc[1], length=pxLength[1], rotationAnch=180)
		endPoint1 = Point(	midLoc[0], midLoc[1], radius=mass[1], yAnch=pxLength[1], rotationAnch=0, color=(255, 255, 255, 255))
		endPoint2 = Point(	midLoc[0], midLoc[1], radius=mass[1]-5, yAnch=pxLength[1], rotationAnch=0, color=(200, 30, 30, 255))
		trail = 	FadeLine(0, 0, colors=((60, 60, 220, 255), (60, 60, 60, 0)), captureRate=1, strokeWidth=4, maxPoints=300)
		
		# attach objects (bars first to render points above)
		pSys.addObject(trail)
		pSys.addObject(bar1)
		pSys.addObject(bar2)
		pSys.addObject(basePoint)
		pSys.addObject(midPoint)
		pSys.addObject(endPoint1)
		pSys.addObject(endPoint2)
		
		# add update actions
		pSys.addAction(bar1.setAngle, 0)
		pSys.addAction(midPoint.setAngle, 0)
		pSys.addAction(bar2.setPosition, 0, [('a2posVert', length[0]), 'm2px', ('offsetPos', origin)])
		pSys.addAction(endPoint1.setPosition, 0, [('a2posVert', length[0]), 'm2px', ('offsetPos', origin)])
		pSys.addAction(endPoint2.setPosition, 0, [('a2posVert', length[0]), 'm2px', ('offsetPos', origin)])
		pSys.addAction(bar2.setAngle, 2)
		pSys.addAction(endPoint1.setAngle, 2)
		pSys.addAction(endPoint2.setAngle, 2)
		pSys.addAction(trail.addPoint, [0, 2], [('a2posVert', length[0], length[1]), 'sum2', 'm2px', ('offsetPos', origin)])
		
		# set simulation values
		pSys.setArgument((0, 0), 'impulseXY', tuple[float, float], False) # impulse is NOT persistent
		pSys.setArgument((1.3, 1.3), 'gamma', tuple[float, float])
		

#	impulse setup (click impulse)
clickImpulse = False
clickStart = 0, 0
arrow = Arrow(0, 0, 5)
arrow.setVisible(False)


#	finish pyglet setup (update, event handlers, run)

def update(dt):
	pSys.update(dt) # can get time and x (state) out of this function if wanted

@window.event
def on_key_press(symbol, modifiers):
	# switch system
	if symbol in (key._1, key._2, key._3, key._4): #, key._5, key._6, key._7, key._8):
		loadScene(symbol)
	
	# control visibility for cool trails
	if symbol is key.H:
		pSys.setVisible(hideSolid=True)
	elif symbol is key.U:
		pSys.setVisible(hideSolid=False)
	
	# quick-close because clicking is slow at times..?
	if symbol is key.ESCAPE:
		pyglet.app.exit()

@window.event
def on_mouse_press(x, y, buttons, modifiers):
	global clickStart, clickImpulse, arrow
	if buttons & mouse.LEFT:
		clickStart = x, y
		clickImpulse = True
	
@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
	global arrow
	if buttons & mouse.LEFT and clickImpulse:
		arrow.setPosLen(x, y, clickStart[0] - x, clickStart[1] - y)
		arrow.setVisible(True)

@window.event
def on_mouse_release(x, y, buttons, modifiers):
	global clickImpulse, force, arrow
	if buttons & mouse.LEFT and clickImpulse:
		magBase = 50 # N/px
		dx = (clickStart[0] - x)
		dy = (clickStart[1] - y)
		force = (dx * magBase, dy * magBase)
		
		pSys.setArgument(force, 'impulseXY')
		clickImpulse = False
		arrow.setVisible(False)


@window.event
def on_draw():
	window.clear()
	
	# draw arrow first because force in background looks best
	arrow.draw()
	pSys.draw()

# 	run app
if __name__ == "__main__":
	# schedule our update function
	pyglet.clock.schedule_interval(func=update, interval=1/60)
	pyglet.app.run()