import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "./temp.csv"
EDGE_LENGTH = 5.0

FIG = plt.figure()


class Dijkstra():
    def __init__(self, startVertex, goalVertex):
        self.ax = FIG.add_subplot(1,1,1)
        self.vertices = np.genfromtxt(DATA_PATH, delimiter=",")
        self.graph = np.array([]) #[[vertex] [idecies of adjacent vertex]]

        self.startV = startVertex
        self.goalV = goalVertex
        self.vertices = np.insert(self.vertices, 0, self.startV, axis=0)
        self.vertices = np.vstack((self.vertices, self.goalV))

        self.buildGraph()
        self.plotGraph()

        self.preIndices = np.full(self.graph.shape[0], -1.0)
        self.values = np.full(self.graph.shape[0], 1000.0)#add goal and start vertices
        self.values[0] = 0.0
        self.path = np.array([])

        print("Start Dijkstra")
        print("use graph:")
        print(self.graph)

        print
        print("use vertices")
        print(self.vertices)

    def buildGraph(self):
        for i, vertex in enumerate(self.vertices):
            norms = np.linalg.norm(self.vertices - vertex, axis=1)
            edgeIndices = np.where((norms < EDGE_LENGTH) & (norms > 0.01))

            if i == 0:
                self.graph = np.array([vertex, edgeIndices])
            else:
                self.graph = np.vstack((self.graph, np.array([vertex, edgeIndices])))

    def plotGraph(self):
        for i, data in enumerate(self.graph):
            vertex = data[0]
            for j, index in enumerate(data[1][0]):
                self.ax.plot([vertex[0], self.vertices[index][0]], [vertex[1], self.vertices[index][1]], "k-")


        self.ax.scatter(self.vertices[:,0], self.vertices[:,1], s=100, marker="o")

    def findPath(self):
        print("The optimal path")
        print self.preIndices
        print
        self.path = self.goalV
        preIndex = self.preIndices[-1]

        while not preIndex == 0: #not start vertex
            preVertex = self.vertices[preIndex]
            print preVertex
            self.path = np.insert(self.path, 0, preVertex) # insert vertex to firt element
            preIndex = self.preIndices[preIndex]


        self.path = np.insert(self.path, 0, self.startV) # insert vertex to firt element
        self.path =  np.reshape(self.path, (self.path.shape[0]/2, 2))
        print self.path

    def plotPath(self):
        self.ax.plot(self.path[:,0], self.path[:,1], "r-", lw = 2)
        self.ax.plot(self.startV[0], self.startV[1], "go", markersize=15)
        self.ax.plot(self.goalV[0], self.goalV[1], "ro", markersize=15)

    def main(self):
        while np.min(self.values) < 1000:
            minIndex = np.argmin(self.values)
            minData = self.graph[minIndex]
            minVertex = minData[0]
            for index in minData[1][0]:# all of adjacent vertex
                edgeLength = np.linalg.norm(self.vertices[index] - minVertex)
                if self.values[index] > self.values[minIndex] + edgeLength and self.values[index] < 99999:
                    self.values[index] = self.values[minIndex] + edgeLength
                    self.preIndices[index] = minIndex

            if minIndex == self.graph.shape[0] - 1: #if find an optimal path to goal
                print("Find the goal")
                break
            self.values[minIndex] = 99999 #ignore the vertex visited onece
            #self.values = np.delete(self.values, minIndex)


        self.findPath()
        self.plotPath()

if __name__ == '__main__':
    startV = np.array([0.0, 0.0])
    goalV = np.array([10.0,10.0])
    dijkstra = Dijkstra(startV, goalV)
    dijkstra.main()
    plt.show()
