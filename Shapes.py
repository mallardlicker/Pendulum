# Shapes.py
# -> Implementations of ShapeBase for render-able Shapes.
# Author: Justin Bunting
# Created: 2026/03/30
# Last Modified: 2026/04/12 21:12


from math import degrees, atan2
from collections import deque
from typing import Sequence

from pyglet import shapes
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.gl import (GL_LINE_STRIP, GL_TRIANGLE_STRIP, GL_BLEND, GL_SRC_ALPHA,
						GL_ONE_MINUS_SRC_ALPHA, glEnable, glBlendFunc, glLineWidth)

from ShapeBase import Shape, MultiShape

grey = (100, 100, 100, 255)
	

class Point(Shape):
	def __init__(self, x, y, 
			radius=10, 
			angle=0, xAnch=0, yAnch=0, rotationAnch=0, color=grey):
		super().__init__(x, y, angle, xAnch, yAnch, rotationAnch, color)
		self.radius = radius
		
		self.shape = shapes.Circle(self.x, self.y, self.radius)
		self._applyShapeDefaults()


class Bar(Shape):
	def __init__(self, x, y, 
			length, 
			width=6, 
			angle=0, xAnch=3, yAnch=0, rotationAnch=0, color=grey):
		super().__init__(x, y, angle, xAnch, yAnch, rotationAnch, color)
		self.length = length
		self.width = width
		
		self.shape = shapes.Rectangle(self.x, self.y, width=self.width, height=self.length)
		self._applyShapeDefaults()
		
	def setWidthLength(self, width, length):
		self.width = width
		self.length = length
		self.shape.width = self.width
		self.shape.height = self.length
	
	
class Tri(Shape):
	def __init__(self, x, y, 
			width,
			height,
			angle=0, xAnch=0, yAnch=0, rotationAnch=0, color=(80, 80, 200, 255)):
		super().__init__(x, y, angle, xAnch, yAnch, rotationAnch, color)
		self.width = width
		self.height = height
		
		points = [(self.x + x, self.y + y) for x, y in zip([0, -self.width/2, self.width/2], [0, -self.height, -self.height])]
		points = [item for sub in points for item in sub]
		self.shape = shapes.Triangle(*points)
		self._applyShapeDefaults()
		
	def setWidthHeight(self, width, height):
		self.width = width
		self.height = height
		self.updateTriCoords()
		
	def setPosition(self, x, y):
		self.x = x
		self.y = y
		self.updateTriCoords()
		
	def updateTriCoords(self):
		self.shape.x, self.shape.y = self.x, self.y
		self.shape.x2, self.shape.y2 = self.x - self.width/2, self.y - self.height
		self.shape.x3, self.shape.y3 = self.x + self.width/2, self.y - self.height
	

class Spring(Shape):
	def __init__(self, x, y,
			length, 
			width=20,
			numPts=9, 
			strokeWidth=8, 
			angle=0, xAnch=0, yAnch=0, rotationAnch=0, color=grey):
		super().__init__(x, y, angle, xAnch, yAnch, rotationAnch, color)
		self.length = length
		self.width = width
		self.numPts = numPts if numPts % 2 == 1 else numPts + 1
		self.strokeWidth = strokeWidth
		
		# each of the points along the length/radius (r) and perpendicular to the length (w)
		# r = rbar * self.length/self.numPts
		self.rbar = [i + 1 for i in range(self.numPts)]
		self.w = [self.width/2 if i % 2 == 0 else -self.width/2 for i in range(self.numPts)]
		self.rbar = [0, 0.5, *self.rbar, self.numPts+0.5, self.numPts+1]
		self.w = [0, 0, *self.w, 0, 0]
		
		# given the w/rbar coords, can easily construct multiline
		self.shape = shapes.MultiLine(
			*zip(
				[self.x + w for w in self.w],
	   			[self.y + rbar * self.length/(self.numPts+1) for rbar in self.rbar]
			),
			thickness=self.strokeWidth
		)
		self._applyShapeDefaults()
		
	def setLength(self, length):
		self.length = float(length)
		# must update the multiline shape
		self.shape.delete()
		# recreate
		self.shape = shapes.MultiLine(
			*zip(
				[self.x + w for w in self.w],
	   			[self.y + rbar * self.length/(self.numPts+1) for rbar in self.rbar]
			),
			thickness=self.strokeWidth
		)
		self._applyShapeDefaults()


class Arrow(MultiShape):
	def __init__(self, x, y,
			length,
			normBarWidth=0.03, # width of the bar section (fraction of length)
			normTriWidth=0.15, # normalized triangle width (fraction of length)
			normTriStart=0.9, # normalized trangle start/bar end (fraction of length)
			angle=0, xAnch=0, yAnch=0, rotationAnch=-90, color=(255, 0, 0, 255)):
		super().__init__(x, y, angle, xAnch, yAnch, rotationAnch)
		self.length = length
		self.normBarWidth = normBarWidth
		self.barWidth = max(self.length * self.normBarWidth, 2)
		self.normTriWidth = normTriWidth
		self.normTriStart = normTriStart
		
		self.addShape(Bar(0, 0, self.length*self.normTriStart, self.barWidth, color=(220, 70, 70, 255)), xOffset=self.barWidth/2)
		self.addShape(Tri(100, 100, 100, 50, color=(80, 80, 220, 255)), yOffset=self.length)
		self.setColor(color)
		
	def setPosLen(self, x, y, lenX, lenY):
		self.x = x
		self.y = y
		self.length = (lenX**2 + lenY**2)**0.5
		self.angle = atan2(lenY, lenX)
		
		# reconstruct arrow
		# bar length and pos/angle update
		self.barWidth = max(self.length * self.normBarWidth, 2)
		self.shapes[0].setWidthLength(self.barWidth, self.length*self.normTriStart)
		self.setShapeOffset(0, -self.barWidth/2)
		# triangle position, rotation, size update
		self.shapes[1].setWidthHeight(self.normTriWidth*self.length, (1-self.normTriStart)*self.length)
		self.setShapeOffset(1, 0, self.length)
		# self.updateMultiPos(1) # performed in offset ^

class FadeLine(Shape):
	_vertex_source = """#version 150 core
		in vec2 position;
		in vec4 colors;
		out vec4 vertex_colors;
		
		uniform WindowBlock
		{
			mat4 projection;
			mat4 view;
		} window;
		
		void main()
		{
			gl_Position = window.projection * window.view * vec4(position, 0.0, 1.0);
			vertex_colors = colors;
		}
	"""
	
	_fragment_source = """#version 150 core
		in vec4 vertex_colors;
		out vec4 final_color;
		
		void main()
		{
			final_color = vertex_colors;
		}
	"""
	
	_program = None
	
	@classmethod
	def _getProgram(cls):
		if cls._program is None:
			vert = Shader(cls._vertex_source, 'vertex')
			frag = Shader(cls._fragment_source, 'fragment')
			cls._program = ShaderProgram(vert, frag)
		return cls._program
	
	def __init__(self, x, y,
			maxPoints=100,
			strokeWidth=1,
			captureRate=1, # the rate at which points are captured: 1 is every point, 2 is every other, etc.
			colors: tuple[tuple[float, float, float, float], tuple[float, float, float, float]]=(grey, (200, 200, 200, 0)),
			angle=0, xAnch=0, yAnch=0, rotationAnch=0):
		self.maxPoints = maxPoints
		self.strokeWidth = strokeWidth
		self.captureRate = captureRate
		self.countSinceLastCapture = 1
		self._colors = colors
		self._points = deque(maxlen=self.maxPoints)
		self._vlist = None
		self._visible = True
		self.program = self._getProgram()
		
	def addPoint(self, px, py):
		self.countSinceLastCapture = self.countSinceLastCapture - 1
		if self.countSinceLastCapture == 0:
			self._points.append((float(px), float(py)))
			self._rebuild()
			self.countSinceLastCapture = self.captureRate
		
	def _computePointColors(self, n):
		ptColors = [
			tuple(
				int((r1 - r2) * i / (n-1) + r2)
				for r1, r2 in zip(self._colors[0], self._colors[1])
			)
			for i in range(n)
		]
		return ptColors
		
	def _rebuild(self):
		if self._vlist is not None:
			self._vlist.delete()
			self._vlist = None
			
		n = len(self._points)
		if n < 2:
			return
		
		ptColors = self._computePointColors(n)
		
		if self.strokeWidth <= 1:
			# thin line path: GL_LINE_STRIP
			positions = [coord for point in self._points for coord in point]
			colors = [c for rgba in ptColors for c in rgba]
			
			self._vlist = self.program.vertex_list(
				n, GL_LINE_STRIP,
				position=('f', positions),
				colors=('Bn', colors)
			)
		else:
			# thick line path: GL_TRIANGLE_STRIP with 2 vertices per point
			halfW = self.strokeWidth / 2.0
			points = list(self._points)
			
			positions = []
			colors = []
			
			for i in range(n):
				# compute tangent from adjacent points
				if i == 0:
					dx = points[1][0] - points[0][0]
					dy = points[1][1] - points[0][1]
				elif i == n - 1:
					dx = points[-1][0] - points[-2][0]
					dy = points[-1][1] - points[-2][1]
				else:
					dx = points[i+1][0] - points[i-1][0]
					dy = points[i+1][1] - points[i-1][1]
				
				# perpendicular (normal) to the tangent
				length = (dx * dx + dy * dy) ** 0.5
				if length < 1e-10:
					nx, ny = 0.0, 0.0
				else:
					nx = -dy / length
					ny = dx / length
				
				px, py = points[i]
				# two vertices offset along the normal
				positions.extend([	px + nx * halfW, py + ny * halfW,
					  				px - nx * halfW, py - ny * halfW])
				# same color for both vertices at this point
				r, g, b, a = ptColors[i]
				colors.extend([r, g, b, a, r, g, b, a])
			
			self._vlist = self.program.vertex_list(
				n * 2, GL_TRIANGLE_STRIP,
				position=('f', positions),
				colors=('Bn', colors)
			)
		
	def draw(self):
		if not self._visible or self._vlist is None:
			return
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		self.program.use()
		if self.strokeWidth <= 1:
			glLineWidth(1) # self.strokeWidth
			self._vlist.draw(GL_LINE_STRIP)
		else:
			self._vlist.draw(GL_TRIANGLE_STRIP)
		self.program.stop()
		