# ShapeBase.py
# -> The base classes Shape and MultiShape use to construct simple, renderable shapes.
# Author: Justin Bunting
# Created: 2026/04/04
# Last Modified: 2026/04/12 21:13


from abc import ABC, abstractmethod
from math import degrees, radians


class Shape(ABC):
	@abstractmethod
	def __init__(self, x, y, angle, xAnch, yAnch, rotationAnch, color):
		self.x = x
		self.y = y
		self.angle = angle
		self.baseAngle = rotationAnch
		
		# init default shape information
		self._anchor_position = xAnch, yAnch
		self._color = color
	
	def _applyShapeDefaults(self):
		self.shape.anchor_position = self._anchor_position
		self.shape.color = self._color
		self.shape.rotation = -self.baseAngle - self.angle
		
	def draw(self):
		self.shape.draw()
		
	def setPosition(self, x, y):
		self.x = x
		self.y = y
		self.shape.position = x, y
		
	def setAngle(self, angle, inRad=True):
		self.angle = degrees(angle) if inRad else angle
		self.shape.rotation = -self.baseAngle - self.angle
		
	def rotate(self, angle):
		self.angle = self.angle + angle
		self.shape.rotation = -self.baseAngle - self.angle
		
	# relative to x/y position
	def setAnchor(self, xAnch=0, yAnch=0, rotationAnch=0):
		self.shape.anchor_position = xAnch, yAnch
		self.baseAngle = rotationAnch
		self.shape.rotation = -self.baseAngle - self.angle
		
	def setAnchorY(self, yAnch):
		self.shape.anchor_y = yAnch
	
	def setColor(self, color: tuple[int, int, int, int]):
		self.shape.color = color
		
	def setVisible(self, isVisible):
		self.shape.visible = isVisible





class MultiShape(Shape):
	def __init__(self, x, y, angle, xAnch, yAnch, rotationAnch):
		self.x = x
		self.y = y
		self.angle = angle
		self._anchor_position = xAnch, yAnch # position/offset to shift/rotate etc. around (relative to x, y)
		self.baseAngle = rotationAnch
		
		self.shapes = list() # each Shape to render
		self.shapeOffsets = list() # (xOff, yOff) offsets from position (x, y)
		
	def addShape(self, shape: Shape, xOffset = 0, yOffset = 0):
		idx = len(self.shapes)
		self.shapes.append(shape)
		self.shapeOffsets.append((xOffset, yOffset))
		
		self.updateMultiPos(idx, True)
		
		# return index of shape in list
		return idx
	
	def setShapeOffset(self, index, xOffset = 0, yOffset = 0):
		self.shapeOffsets[index] = -xOffset, -yOffset
		self.updateMultiPos(index, True)
	
	def _applyShapeDefaults(self):
		pass
		
	def draw(self):
		for shape in self.shapes:
			shape.draw()
			
	def setPosition(self, x, y):
		self.x = x
		self.y = y
		self.updateMultiPos()
		
	def setAngle(self, angle, inRad=True):
		self.angle = degrees(angle) if inRad else angle
		self.updateMultiPos()
		
	def rotate(self, angle):
		self.angle = self.angle + angle
		self.updateMultiPos()
		
	# relative to x/y position
	def setAnchor(self, xAnch=0, yAnch=0, rotationAnch=0):
		self._anchor_position = xAnch, yAnch
		self.baseAngle = rotationAnch
		self.updateMultiPos(offsetUpdate=True)
		
	def setAnchorY(self, yAnch):
		self._anchor_position = self._anchor_position[0], yAnch
		self.updateMultiPos(offsetUpdate=True)
	
	def setColor(self, color: tuple[int, int, int, int], index=-1):
		if index == -1:
			for shape in self.shapes:
				shape.setColor(color)
		else:
			self.shapes[index].setColor(color)
		
	def setVisible(self, isVisible):
		for shape in self.shapes:
			shape.setVisible(isVisible)
		
	def updateMultiPos(self, index=-1, offsetUpdate=False):
		if index == -1:
			sList = self.shapes
			oList = self.shapeOffsets
		else:
			sList = [self.shapes[index]]
			oList = [self.shapeOffsets[index]]
		
		for i in range(len(sList)):
			# update position
			sList[i].setPosition(self.x, self.y)
			# update offset position
			if offsetUpdate:
				sList[i].setAnchor(self._anchor_position[0] + oList[i][0], self._anchor_position[1] + oList[i][1], self.baseAngle)
			# update angle
			sList[i].setAngle(self.angle)