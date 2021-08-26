import numpy as np;
from PIL import Image;
import argparse
from gifgenerator import *
import os
import glob
from queue import PriorityQueue
from collections import defaultdict

def argParser():
	parser = argparse.ArgumentParser()

	parser.add_argument('--image', nargs='?', default='maze.png', help="Path to maze image")

	parser.add_argument('--algorithm', nargs='?', default='astar', help="Algorithm to solve maze")
	parser.add_argument('--heruistic', nargs='?', default='manhattan', help="Heruistic for astar algo")

	parser.add_argument('--startx', type=int, default=0,help='Start x of maze')
	parser.add_argument('--starty', type=int, default=0,help='Start y of maze')
	parser.add_argument('--endx', type=int, default=0,help='End x of maze')
	parser.add_argument('--endy', type=int, default=0,help='End y of maze')

	return parser.parse_args()

class Maze:

	def __init__(self,imagePath, algorithm, heuristic ,start=(0,0), end=(0,0)):
		self.imagePath = imagePath
		self.image = Image.open(imagePath)
		self.image = self.image.convert('RGB')
		self.pixels = self.image.load()

		self.GREEN = (0,255,0)
		self.RED = (255,0,0)
		self.WHITE = (255,255,255)
		self.BLACK = (0,0,0)

		self.start = start
		self.end = end
		self.algorithm = algorithm
		self.heuristic = heuristic

		self.frequency = int(self.image.size[0]*self.image.size[1]/50)
		# self.frequency = 1000
		self.frameCount = 0

	def closestColor(self,color):
		val = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
		if(val > 200):
			return self.WHITE
		else:
			return self.BLACK

	def cleanImage(self):
		x,y = self.image.size
		for i in range(x):
			for j in range(y):
				self.pixels[i,j] = self.closestColor(self.pixels[i,j])

	def showImage(self):
		self.image.show()

	def fixWalls(self):
		x,y = self.image.size

		for i in range(x-1):
			for j in range(y-1):
				currPix = (i,j)
				nextPix = (i,j+1)
				belowPix = (i+1,j)
				diagonalPix = (i+1,j+1)

				if (self.pixels[currPix] == self.WHITE and self.pixels[diagonalPix] == self.WHITE and self.pixels[nextPix] == self.BLACK and self.pixels[belowPix] == self.BLACK) or (self.pixels[currPix] == self.BLACK and self.pixels[diagonalPix] == self.BLACK and self.pixels[nextPix] == self.WHITE and self.pixels[belowPix] == self.WHITE):
					self.pixels[currPix] = self.BLACK
					self.pixels[nextPix] = self.BLACK
					self.pixels[diagonalPix] = self.BLACK
					self.pixels[belowPix] = self.BLACK

	def isValid(self,vertex):
		x,y = self.image.size
		if vertex[0] >= 0 and vertex[0] < x and vertex[1] >= 0 and vertex[1] < y:
			return True

		return False

	def getNeighbours(self,vertex):
		x = vertex[0]
		y = vertex[1]
		return [(x-1,y-1),(x-1,y+1),(x+1,y+1),(x+1,y-1),(x-1,y),(x,y+1),(x+1,y),(x,y-1)]

	def display(self,parent):
		for i in parent:
			# for j in i;
			print(i)
			
		print("\n\n")

	def bfs(self):
		q = []
		x,y = self.image.size
		parent = []
		for i in range(x):
			temp = []
			for j in range(y):
				temp.append((-1,-1))
			parent.append(temp)


		image = self.image.copy()
		q.append(self.start)
		pixels = image.load()
		pixels[self.start] = self.GREEN
		iterations = 0

		while q:
			vertex = q.pop(0)

			if vertex == self.end:
				print("found")
				image.save('./frames/'+str(self.frameCount)+'.jpg')
				self.frameCount = self.frameCount + 1
				i = self.end[0]
				j = self.end[1]
				path = []

				while parent[i][j] != (-1,-1):
					path.append((i,j))
					i,j = parent[i][j]
				path.append((i,j))
				return path

			neighbours = self.getNeighbours(vertex)
			for neighbour in neighbours:
				if self.isValid(neighbour) and pixels[neighbour] == self.WHITE:
					pixels[neighbour] = self.GREEN
					parent[neighbour[0]][neighbour[1]] = vertex
					q.append(neighbour)

			if iterations%self.frequency == 0:
				image.save('./frames/'+str(self.frameCount)+'.jpg')
				self.frameCount = self.frameCount + 1
			
			iterations = iterations+1
		return []

	def h(self, point):
		if self.algorithm == 'djikstra':
			return 0
		
		dx = abs(point[0] - self.end[0])
		dy = abs(point[1] - self.end[1])
		D = 2
		D2 = 1
		if self.heuristic == 'euclidean':
			return D * np.sqrt(dx * dx + dy * dy)
		elif self.heuristic == 'diagonal':
			return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
		else:
			return D*(dx+dy) #Manhattan

	def popMin_fscore(self,discovered,fscore):
		min = next(iter(discovered))
		for x in discovered:
			if fscore[min] > fscore[x]:
				min = x
		discovered.remove(min)
		return min
				
	def Astar(self):
		discovered = set()
		image = self.image.copy()
		pixels = image.load()

		g_score = defaultdict(lambda: float("inf"))
		g_score[self.start] = 0

		f_score = defaultdict(lambda: float("inf"))
		f_score[self.start] = self.h(self.start)

		discovered.add(self.start)
		pixels[self.start] = self.GREEN

		came_from = dict()
		iterations = 0

		while len(discovered) != 0:
			
			current = self.popMin_fscore(discovered, f_score)
			pixels[current] = self.RED

			if current == self.end:
				path = []
				image.save('./frames/'+str(self.frameCount)+'.jpg')
				self.frameCount = self.frameCount + 1
				while current != self.start:
					path.append(current)
					current = came_from[current]
				path.append(current)
				return path

			for neighbour in self.getNeighbours(current):
				if self.isValid(neighbour) and pixels[neighbour] != self.BLACK and pixels[neighbour] != self.RED:
		
					tenative_gscore = g_score[current]+1
					if tenative_gscore < g_score[neighbour]:
						came_from[neighbour] = current
						g_score[neighbour] = tenative_gscore
						f_score[neighbour] = g_score[neighbour] + self.h(neighbour)

						if pixels[neighbour] == self.WHITE:
							pixels[neighbour] = self.GREEN
							discovered.add(neighbour)
			
			if iterations%self.frequency == 0:
				image.save('./frames/'+str(self.frameCount)+'.jpg')
				self.frameCount = self.frameCount + 1

			iterations += 1

		return []

	def solve(self):
		self.cleanImage()
		self.fixWalls()

		print(self.frequency)

		if self.algorithm == 'bfs':
			path = self.bfs()
		else:
			path = self.Astar()
		
		if path:
			for pos in path:
				self.pixels[pos] = self.GREEN
				for neighbour in self.getNeighbours(pos):
					if self.isValid(neighbour) and self.pixels[neighbour] == self.WHITE:
						self.pixels[neighbour] = self.GREEN
			
			self.showImage()
			self.image.save('./output/'+self.heuristic+'-'+self.algorithm+'-'+'out.jpg')
			createGIF(self.imagePath, self.frameCount, self.algorithm, self.heuristic)
		else:
			print("Path not found")

def main(arg):
	url = arg.image
	start = (arg.startx,arg.starty)
	end = (arg.endx,arg.endy)
	maze = Maze(url,arg.algorithm,arg.heruistic,start,end)
	files = glob.glob('./frames/*')
	for f in files:
		os.remove(f)
	maze.solve()

if __name__ == '__main__':
	arg = argParser()
	main(arg)