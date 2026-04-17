# System.py
# -> A class which stores any physics system that will be simulated.
# Author: Justin Bunting
# Created: 2026/04/02
# Last Modified: 2026/04/16 22:45



from typing import Any, Sequence, Iterable, get_origin
import types
from math import sin, cos

from scipy.integrate import solve_ivp
import pyglet

from Shapes import Shape, FadeLine


metersToPixels = 100 # px/m

class PhysicsSystem():
	def __init__(	self, 
					function: callable, 
			  		x0: Sequence[float] | float,
					t: float = 0,
					args: Any | None = None):
		self.reset(function, x0)
		self.hiddenGroups = set() # ! kept outside of reset for visual consistency: change if necessary
		
	def reset(	self,
		   		function: callable, 
				x0: Sequence[float] | float,
				t: float = 0,
				args: Any | None = None):
		self.function = function
		self.x0 = x0
		self.running = True
		self.isVisible = True
		
		# determine what type each argument of function is, and add defaults (zero-filled arguments)
		# to args list
		self.args = list() # values
		self.argInfo = dict() # name -> idx, type, persist
		for name, inputClass in self.function.__annotations__.items():
			if name not in ('t', 'x', 'return'):
				self.setArgument(self.zeros(inputClass), name, inputClass) # all persistent by default
		
		self.t = t
		self.x = self.x0
		
		self.objects = dict()
		self.actions = list()
	
	# attach an object
	def addObject(self, obj: Shape, group: int = 0):
		if group not in self.objects.keys():
			self.objects[group] = list()
		self.objects[group].append(obj)
	
	# attach an update action
	def addAction(self,
			updateFn: callable,
			stateInds: Sequence[int] | int = 0,
			mods: Sequence = [""]):
		self.actions.append((updateFn, stateInds, mods))
	
	# set a single argument by name: persistence only changes if type is given
	def setArgument(self, argVal, argName, argType=None, isPersistent=True):
		if argType is not None:
			if argName in self.argInfo:
				# update type and persistence
				self.argInfo[argName] = [self.argInfo[argName][0], argType, isPersistent]
				self.args[self.argInfo[argName][0]] = argVal
			else:
				# for new names, append to args
				self.argInfo[argName] = [len(self.args), argType, isPersistent]
				self.args.append(argVal)
		else:
			# check that argVal is of argType
			baseType = get_origin(self.argInfo[argName][1]) or self.argInfo[argName][1]
			if not isinstance(argVal, baseType):
				return False
			self.args[self.argInfo[argName][0]] = argVal
		return True
	
	# set the system as running or not
	def setRunning(self, isRunning: bool):
		self.running = isRunning
	
	# update the system using the solver
	def update(self, dt: float, force: bool = False):
		if not self.running and not force:
			return
		
		# solve system for this timestep
		self.t, self.x = self.solveIVPInteract(
			function=self.function,
			x=self.x, 
			dt=dt, 
			args=self.args, 
			t=self.t)
		
		# use solution to inform actions
		for act in self.actions:
			if act[0] is not None:
				# extract values based on index type
				if isinstance(act[1], int):
					vals = [float(self.x[act[1]])]
				else:
					vals = [float(self.x[i]) for i in act[1]]
				
				# print(act[0], vals)
				
				# apply mods
				for mod in act[2]:
					# print(vals)
					if mod == 'm2px': # meters to pixels
						# if isinstance(vals, list):
						vals = [v * metersToPixels for v in vals]
						# else:
						# 	vals = vals * metersToPixels
					elif mod == 'sum': # sum values
						vals = [sum(vals)]
					elif mod == 'sum2': # sum every other value together
						vals = [sum(vals[::2]), sum(vals[1::2])]
					elif mod == 'rth2xy': # TODO: this should have 'vert' check as well
						vals = [(r*sin(th), -r*cos(th)) for r, th in zip(vals[::2], vals[1::2])]
						vals = [item for subtup in vals for item in subtup]
					elif isinstance(mod, tuple):
						if 'a2pos' in mod[0]: # angle to position (mod[1] is the length)
							vert = 'Vert' in mod[0]
							if len(mod) == 2:
								l = mod[1]
								vals = [(l*sin(val), -l*cos(val)) for val in vals] if vert else [(l*cos(val), l*sin(val)) for val in vals]
								vals = [item for subtup in vals for item in subtup]
							else:
								tmp = []
								for i in range(1, len(mod)):
									l = mod[i]
									val = vals[i-1]
									tmp.extend([l*sin(val), -l*cos(val)]) if vert else tmp.extend([l*cos(val), l*sin(val)])
								vals = tmp
						elif mod[0] == 'offsetPos': # offset a position
							offset = mod[1]
							vals = [val + offset[i%2] for val, i in zip(vals, range(len(vals)))]
				
				# print(self.x)
				# print(act[0], vals)
				
				# call action/function
				if isinstance(vals, list):
					act[0](*vals)
				
		# use persistence to update arguments
		for argI in self.argInfo.values():
			if not argI[2]: # if not persistent
				self.args[argI[0]] = self.zeros(argI[1])
		
		return self.t, self.x
	
	# draw the system's attached objects
	def draw(self):
		if self.isVisible:
			for group, objList in self.objects.items():
				if group not in self.hiddenGroups:
					for obj in objList:
						obj.draw()
	
	def setVisible(self, isVisible=True, hiddenGroups: set[int] | list[int] = []):
		self.isVisible = isVisible
		self.hiddenGroups = set(hiddenGroups)
	
	# function to solve IVP given current state (x), function, and time/span
	def solveIVPInteract(
			self,
			function: callable, 
			x: Sequence[float] | float,
			dt: float,
			args: Any | None = None,
			t: float = 0.0):
		
		tCurr = t
		tMax = t + dt
		
		solution = solve_ivp(
			fun=function,
			t_span=[tCurr, tMax],
			y0=x,
			args=args
		)
		tCurr = solution.t[-1]
		x = solution.y[:, -1] # last column = final state vector
		
		return tCurr, x
	
	def zeros(self, classType):
		# if it's a union, pick the first type
		if isinstance(classType, types.UnionType):
			classType = classType.__args__[0]
		
		if classType == tuple[float]:
			return (0,)
		elif classType == tuple[float, float]:
			return (0, 0)
		elif classType == tuple[float, float, float]:
			return (0, 0, 0)
		return 0