import numpy as np

class NeuralNetwork(object):
    def __init__(self):
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize) #2x3
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize) #3x1

    def forward(self, X): #3x2
        self.z1 = np.dot(X, self.W1)#3x3
        self.a = self.sigmoid(self.z1)#3x3
        self.z2 = np.dot(self.a, self.W2)#3x1
        self.yhat = self.sigmoid(self.z2)#3x1
        return self.yhat #3x1

    def cost(self, X, y):#3x2, #3x1
        self.yhat = self.forward(X)#3x1
        return np.sum(0.5 * (y - self.yhat)**2, axis=0)#1x1

    def costPrime(self, X, y):#3x2, #3x1
        self.yhat = self.forward(X) #3x1
        delta1 = np.multiply((self.yhat - y), self.sigmoidPrime(self.z2))#3x1
        dJdW2 = np.dot(self.a.T, delta1)#3x1
        delta2 = np.dot(delta1, self.W2.T) *  self.sigmoidPrime(self.z1)#3x3
        dJdW1 = np.dot(X.T, delta2)#2x3
        return dJdW1, dJdW2 #2x3, 3x1

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self, z):
        return np.exp(-z) / (1+np.exp(-z))**2

    
X = np.array([[.3,1], [.5,.2], [1,.4]])
y = np.array([[.75], [.82], [.93]])
NN = NeuralNetwork()
#cost1 = NN.cost(X, y)

for i in range(1000):
    print NN.forward(X)
    dJdW1, dJdW2 = NN.costPrime(X, y)
    NN.W1 -= dJdW1
    NN.W2 -= dJdW2
    print ""

print ""
X2 = np.array([[.8, .6]])
print NN.forward(X2)


