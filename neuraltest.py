import numpy as np
import cv2

class NeuralNetwork(object):
    def __init__(self):
        self.W = []
        self.dJdW = []
        self.layerSizes = []
        self.A = []
        self.Z = []
        self.xnorm = []
        self.rows = 0
        self.cols = 0

    def normalize(self, X, y):
        '''Normalize the input and desired output data'''
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        #xnorm = []
        for i in range(X.shape[1]):
            self.xnorm.append(max(X[:,i]))
            X[:,i] = X[:,i] / max(X[:,i])
        #for i in range(y.shape[1]):
        #    y[:,i] = y[:,i] / max(y[:,i])
        return X, y

    def setData(self, X, y):
        '''Need to set this so we can build the necessary weights'''
        try:
            self.dataSize, self.inputLayerSize = X.shape
        except ValueError:
            X = X.reshape((1, X.shape[0]))
            self.dataSize, self.inputLayerSize = X.shape


        try:
            self.dataSize2, self.outputLayerSize = y.shape
        except ValueError:
            y = y.reshape((1, y.shape[0]))
            self.dataSize2, self.outputLayerSize = y.shape

        if self.dataSize != self.dataSize2:
            return "X and y need to have the same rows!"
        self.layerSizes = [self.inputLayerSize, self.outputLayerSize]
        self.setWeights()
        self.A.append(X)
        return X, y

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
        #X, y = self.normalize(X, y)
        self.costPrime(X, y)
        for i in range(len(self.W)):
            self.W[i] -= self.dJdW[i]

    def cost(self, X, y):
        #X, y = self.normalize(X, y)
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

    def hypertan(self, z):
        return np.tanh(z)

    def hypertanPrime(self, z):
        return 1 - self.hypertan(z)**2

    def arctan(self, z):
        return np.atan(z)

    def arctanPrime(self, z):
        return 1/(z**2 + 1)

    #Feature extractors
    def relu(self, X):
        '''Any negative value turns into a zero'''
        shape = X.shape
        X = X.flatten()
        for i in range(len(X)):
            X[i] = max(X[i], 0)
        return X.reshape(shape)
    
    def maxpool(self, X, stride=2, size=2, pad=False):
        '''X is a 2-dimensional numpy array'''
        rows = int(np.ceil((float(X.shape[0])-size)/stride)) + 1
        cols = int(np.ceil((float(X.shape[1])-size)/stride)) + 1
        R = np.zeros((rows, cols))
        
        for row in range(rows):
            for col in range(cols):
                sample = X[row*stride:row*stride+size, col*stride:col*stride+size]
                value = max(sample.flatten())
                R[row, col] = value
        return R

    def avgpool(self, X, stride=2, size=2, pad=False):
        pass

    def sumpool(self, X, stride=2, size=2, pad=False):
        pass

    def convolve(self, X, kernel, stride=1, pad=False):
        size = kernel.shape[0] #kernels are square
        rows = int(np.ceil((float(X.shape[0])-size)/stride)) + 1
        cols = int(np.ceil((float(X.shape[1])-size)/stride)) + 1
        R = np.zeros((rows, cols))
        for row in range(rows):
            for col in range(cols):
                sample = X[row*stride:row*stride+size, col*stride:col*stride+size]
                conv = sample * kernel
                R[row, col] = sum(conv.flatten())
        return R

        
#X = np.array([[.3,1], [.5,.2], [1,.4], [.8, .6]])
#y = np.array([[.75], [.82], [.93], [.86]])

image = cv2.imread("/home/jonathan/Pictures/number8.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print gray.shape
#Define a number 8 using an 11x8 array
X = np.zeros((11,8))
X[2,2:6] = 1
X[5,2:6] = 1
X[8,2:6] = 1
X[2:9,2] = 1
X[2:9,5] = 1
print "Original X"
print X.shape


#X = X.flatten()
#print X
y = np.array([0,0,0,0,0,0,0,0,1,0])
#y = y.reshape((y.shape[0], 1))

#X = np.zeros((5,1,2))
#y = np.zeros((5,1,2))
#X[0,:,:] = [35,67]
#X[1,:,:] = [12,75]
#X[2,:,:] = [16,89]
#X[3,:,:] = [45,56]
#X[4,:,:] = [10,90]
#X = np.array([[35,67],[12,75],[16,89],[45,56],[10,90]]) #5x2
#y = np.array([[1,0],[0,1],[1,0],[1,0],[0,1]])#5x2

NN = NeuralNetwork()
#print X
#print "Maxpool"
kernel = np.array([[1,0,1],[0,1,0],[1,0,1]])
kernel = np.array([[1,0,-1],[0,0,0],[-1,0,1]])
X = NN.convolve(X, kernel)
X = NN.maxpool(X)
#X = X.flatten()
#X, y = NN.setData(X, y)
#NN.addHiddenLayer(200)

print "GRAY STUFF START"
print gray[34:38, 40:44]
gray = NN.convolve(gray, kernel)
gray = NN.relu(gray)
gray = NN.maxpool(gray)
gray = NN.convolve(gray, kernel)
gray = NN.relu(gray)
gray = NN.maxpool(gray)
#print gray.shape
#print gray
#cv2.imshow("gray", gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#X = gray
#print gray.shape
#print X.shape
print "GRAY STUFF END"
#NN.addHiddenLayer(50)

#NN.addHiddenLayer(3)
#NN.addHiddenLayer(5)
#cost1 = NN.cost(y)
X = X.flatten()
gray = gray.flatten()
print "X"
print X
print "GRAY"
print max(gray.flatten())
val = max(gray.flatten())
gray = gray / val

gray, y = NN.setData(gray, y)
#X, y = NN.setData(X, y)
NN.addHiddenLayer(200)

gray.shape
for i in range(1000):
    #print NN.forward(X)
    #NN.backProp(X, y)
    NN.backProp(gray, y)
    #dJdW1, dJdW2 = NN.costPrime(y)
    #NN.W[0] -= dJdW1
    #NN.W[1] -= dJdW2
    #print ""

#print ""
#X2 = np.array([[25/NN.xnorm[0], 70/NN.xnorm[1]]])
#print X2
#print NN.forward(X2)

#print ""
#X3 = np.array([[35/NN.xnorm[0], 67/NN.xnorm[1]]])
#print NN.forward(X3)

#print NN.forward(X)
print NN.forward(gray)
