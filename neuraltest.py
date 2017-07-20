import numpy as np

class NeuralNetwork(object):
    def __init__(self):
        self.W = []
        self.dJdW = []
        self.layerSizes = []
        self.A = []
        self.Z = []

    def setData(self, X, y):
        '''Need to set this so we can build the necessary weights'''
        self.dataSize, self.inputLayerSize = X.shape
        self.dataSize2, self.outputLayerSize = y.shape
        if self.dataSize != self.dataSize2:
            return "X and y need to have the same rows!"
        self.layerSizes = [self.inputLayerSize, self.outputLayerSize]
        self.setWeights()
        self.A.append(X)

    def addHiddenLayer(self, hiddenLayerSize):
        '''Add a hidden layer of a certain size to the end of the other hidden layers'''
        self.layerSizes.insert(-1, hiddenLayerSize)
        self.setWeights()

    def setWeights(self):
        '''Generate initial random weights'''
        self.W = []
        for i in range(len(self.layerSizes)-1):
            self.W.append(np.random.randn(self.layerSizes[i], self.layerSizes[i+1]))
            
    def forward(self, X):
        '''Send data forward to get predictions'''
        self.Z = []
        self.A = [X]
        for i in range(len(self.W)):
            self.Z.append(np.dot(self.A[i], self.W[i]))
            self.A.append(self.sigmoid(self.Z[-1]))

        self.yhat = self.A[-1]
        return self.yhat

    def backProp(self, X, y):
        '''Back propogation adjusts the weights'''
        self.costPrime(X, y)
        for i in range(len(self.W)):
            self.W[i] -= self.dJdW[i]

    def cost(self, X, y):
        self.yhat = self.forward(X)
        return np.sum(0.5 * (y - self.yhat)**2, axis=0)

    def costPrime(self, X, y):
        '''Calculate the weight adjustments'''
        self.yhat = self.forward(X)
        self.Z.reverse()
        self.A.reverse()
        self.W.reverse()
        for i in range(len(self.W)):
            if i == 0:
                delta = np.multiply(self.yhat - y, self.sigmoidPrime(self.Z[i]))
            else:
                delta = np.dot(delta, self.W[i-1].T) * self.sigmoidPrime(self.Z[i])

            self.dJdW.append(np.dot(self.A[i+1].T, delta))
        self.dJdW.reverse()
        self.W.reverse()

    #Activation functions
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self, z):
        return np.exp(-z) / (1+np.exp(-z))**2

    
X = np.array([[.3,1], [.5,.2], [1,.4], [.8, .6]])
y = np.array([[.75], [.82], [.93], [.86]])

NN = NeuralNetwork()
NN.setData(X, y)
NN.addHiddenLayer(20)
#NN.addHiddenLayer(3)
#NN.addHiddenLayer(5)
#cost1 = NN.cost(y)

for i in range(1000):
    print NN.forward(X)
    NN.backProp(X, y)

    #dJdW1, dJdW2 = NN.costPrime(y)
    #NN.W[0] -= dJdW1
    #NN.W[1] -= dJdW2
    print ""

print ""
X2 = np.array([[.4, .3]])
print NN.forward(X2)


X3 = np.array([[1, .4]])
print NN.forward(X3)

