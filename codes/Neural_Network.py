#inputLayer weightslayer hiddenlayer1 weightslayer hiddenlayer2 weightslayer outputlayer
import numpy as np
import time
import matplotlib.pyplot as plt
import copy
import json
import codecs
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
class NeuralNetwork(object):
    def _init(self):
        self.inputLayerSize=0
        self.outputLayerSize=0
        self.numberHiddenLayer=0
        self.hiddenLayerSizes=[]
        #Architecture parameters
        self.entireinputData=[]
        self.entireoutputData=[]
        self.sampleOut=[[]]
        self.w=[]
        self.t=[]
        self.z=[]
        self.a=[]
        self.y=[]
        self.scalar=0.1     #Learning rate
        self.momentum=0
        self.batchsize=0
        #Other parameters
        self.trainDataFile=''
        self.validataFile=''
        self.training=True
        self.verbose=''
        self.plot=False
        self.randomSeed=1
        self.checkWithNumericalGradients=False
        self.norm={}
        def readDataAndNormalize(self,inputFile,saveNormToFile,isValidating):
            inputColSize=self.inputLayerSize
            outputColsize=self.outputLayerSize
            #read data from files
            originalInput=np.genfromtxt(inputFile,delimiter='')
            if isValidating:
                self.norm=json.load(open("./output/norm_values"))
                print (self.norm)
                self.norm['maxs']=np.asarray(self.norm['maxs'])
                self.norm['mins']=np.asarray(self.norm['mins'])
            else:
                self.norm['maxs']=np.amax(originalInput, axis=0);
                self.norm['mins']=np.aamin(originalInput, axis=0);
                self.norm['limMax']=0.9
                self.norm['limMin']=0.1
                #print self.norm
                dictForFile=self.norm.copy()
                dictForFile['maxs']=dictForFile['maxs'].tolist()
                dictForFile['mins']=dictForFile['mins'].tolist()
                json.dump(dictForFile,open("./output/norm_values","w"))
                normalizedInput=(((self.norm['limMax']-self.norm['limMax'])*(originalInput - self.norm['mins'])) / (self.norm['maxs'] - self.norm['mins'])) + self.norm['limMin'];
                normalizedInputData=normalizedInput[:,:self.inputLayerSize]
                normalizedSampleOut=normalizedInput[:,:self.inputLayerSize:self.inputLayerSize+self.outputLayerSize]
                #print if verbose
                if saveNormToFile:
                    np.savetxt('./input/normalized',normalizedInput,delimeter='',fmt='%1.10f')
                if self.verbose>=2:
                    print("original input/output")
                    print(originalInput)
                    prin("normalized input/output")
                    print(normalizedInput)
                    print("demoralized input/output")
                    print(demoralizedInput)
                    return normalizedInputData, normalizedSampleOut

    def randomizeWeightsAndThresholds(self):
       
        if self.randomSeed is not None and self.randomSeed != 0:
            np.random.seed(self.randomSeed)
        
        for i in range(self.numberHiddenLayer + 1):
            if i == 0:                          # First layer
                self.w.append(np.random.randn(self.inputLayerSize, self.hiddenLayerSizes[0]))
            elif i == self.numberHiddenLayer:   # Last layer
                self.w.append(np.random.randn(self.hiddenLayerSizes[self.numberHiddenLayer-1], self.outputLayerSize))
            else:                               # Other layers
                self.w.append(np.random.randn(self.hiddenLayerSizes[i - 1], self.hiddenLayerSizes[i]))
            # Add same random values to threshold, same size than last weigh columns   # self.t.append(self.w[-1][0,:])
            self.t.append(np.random.randn(1, self.w[-1].shape[1]))
        # Print if verbose
        if self.verbose >= 2:
            print ("===BUILD ATCHITECTURE===============================")
            for i in range(self.numberHiddenLayer + 1):
                print ("weight=" + str(i))
                print (self.w[i])
       
        self.W1 = self.w[0]
        self.W2 = self.w[1]

    def forward(self, paramInputData, paramOutputData, paramBatchSize):
        
        self.a = []
        self.z = []
        # If receive paramInputData then always do it batch because it is for verification
        if paramBatchSize == 0 or paramBatchSize >= len(paramInputData):
            x = paramInputData
            y = paramOutputData
        else:
            rand = np.random.randint(len(paramInputData) - paramBatchSize)
            x = paramInputData[rand:rand + paramBatchSize,:]
            y = paramOutputData[rand:rand + paramBatchSize,:]

        self.z.append(x)    # First z must be the same input
        self.a.append(x)    # First a must be the same input

        # Calculate next layers
        for i in range(self.numberHiddenLayer + 1):
            self.z.append(np.dot(self.a[-1], self.w[i]) - self.t[i])
            self.a.append(self.sigmoid(self.z[-1]))    # [-1] returns the last element from array
        # Print if verbose
        if self.verbose >= 2:
            print ("===FORWARD PROP===============================")
            for i in range(len(self.z)):
                print ("z=" + str(i))
                print (self.z[i])
                print ("a=" + str(i))
                print (self.a[i])
        # Get y and return it
        yh = self.a[-1]
        return yh,y

    def sigmoid(self, z):
        #Apply sigmoid activation function
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self, z):
        # Derivative of sigmoid function
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costFunction(self, y, yh):
        yDiff = np.abs(y-yh)
        J = 0.5*sum((yDiff)**2)
        # Compute results for final table
        results = np.column_stack((y, yh))
        percError = np.sum(yDiff) / np.sum(results[:,0]) * 100
        # Print if verbose
        if self.verbose >= 2:
            print ("===COST FUNCTION===============================")
            print ("===Y===")
            print (y)
            print ("===Yh===")
            print (yHat)
            print ("===Y-Yh===")
            print (y - yh)
            print ("===J===")
            print (J)
        return J, results, percError

    def costFunctionPrime(self, y, yh):
        
        self.deltas = []
        
       
        self.dJdWs = []
        # Cycle through network levels backwards
        for i in reversed(range(1,self.numberHiddenLayer + 2)):
            if i == self.numberHiddenLayer + 1:     # Last position
                self.deltas.append(np.multiply(-(y-yh), self.sigmoidPrime(self.z[i])))
            else:
                self.deltas.append(np.dot(self.deltas[-1], self.w[i].T) * self.sigmoidPrime(self.z[i]))
            self.dJdWs.append(np.dot(self.a[i-1].T, self.deltas[-1]))
        # Reverse arrays
        self.deltas = self.deltas[::-1]
        self.dJdWs = self.dJdWs[::-1]
        # Print if verbose
        if self.verbose >= 2:
            print ("===COST FUNCTION PRIME======")
            for i in reversed(range(len(self.deltas))):
              print ("delta[" + str(i) + "]="); print (self.deltas[i])
              print ("dJdW[" + str(i) + "]="); print (self.dJdWs[i])
        # Return cost derivatives with respect to w's
        return self.dJdWs

    def fixWeights(self, scalar, momentum):
        deltasW = []
        for i in range(len(self.w)):
            deltaW = -(scalar * self.dJdWs[i])
            deltasW.append(deltaW)
            try:
                deltaW = deltaW + (momentum * self.deltaW_old[i])
            except AttributeError:
                pass    # "self.deltaW_old doesn't exist"
            self.w[i] = self.w[i] + deltaW
        # Save deltaW for next epoch
        self.deltaW_old = copy.deepcopy(deltasW)
        # Print if verbose
        if self.verbose >= 2:
            print ("===FIX WEIGHTS=============")
            for i in range(len(self.w)):
             print ("weight[" + str(i) + "]="); print (self.w[i])

    def fixThresholds(self, scalar, momentum):
        for i in range(len(self.t)):
            self.deltas[i] = np.sum(self.deltas[i])
            deltaT = +(scalar * self.deltas[i]) # the + means (- * -) because of derivatives
            try:
                deltaT = deltaT + (momentum * self.deltaT_old[i])
            except AttributeError:
                pass    # "self.deltaT_old doesn't exist"
            self.t[i] = self.t[i] + deltaT
        # Save deltaW for next epoch
        self.deltaT_old = copy.deepcopy(self.deltas)

    # This method is just for checking the derivative calculations. Here we try
    # to simulate the derivatives using slopes and small deltas = 1e-4
    def calculateNumericalGradient(self):
        e = 1e-4
        initialW = self.w
        numgrad = []
        # Iterate over every weight and slightly change its value
        for w in range(len(self.w)):
            for i in range(self.w[w].shape[0]):
                for j in range(self.w[w].shape[1]):
                    self.w[w][i][j] = self.w[w][i][j] + e
                    self.forward()
                    loss2 = self.costFunction()
                    self.w[w][i][j] = self.w[w][i][j] - (2*e)
                    self.forward()
                    loss1 = self.costFunction()
                    slope = (loss2 - loss1) / (2*e)
                    numgrad.append(slope[0])
                    #Return the value we changed
                    self.w[w][i][j] = self.w[w][i][j] + e

        # Transform to dJdWs format
        start = 0
        num_dJdWs = []
        # print numgrad
        for i in range(len(self.dJdWs)):
            shape = self.dJdWs[i].shape
            size = shape[0]*shape[1]
            data = numgrad[start: start + size]
            start = start + size
            
            num_dJdWs.append(np.reshape(data,(shape[0], shape[1])))

        #Return Params to original value:
        self.w = initialW
        # Return numeral gradients
        return num_dJdWs

    def compareWithNumericalCalculation(self):
        grad = self.dJdWs
        numgrad = self.calculateNumericalGradient()
        grad_flat = np.concatenate([elem.ravel() for elem in grad])
        numgrad_flat = np.concatenate([elem.ravel() for elem in numgrad])
        print ("Numerical gradient check:")
        print (np.linalg.norm(grad_flat - numgrad_flat)/np.linalg.norm(grad_flat + numgrad_flat))

    def plotValidate(self, results, showPlot, savePlot):
        fig = plt.figure()
        ax2 = fig.add_subplot(111)
        ax2.scatter(results[:,0], results[:,1])
        percOfError = np.sum(np.abs(results[:,0] - results[:,1])) / np.sum(results[:,0]) * 100
        legendStr = self.getNetworkSummary()
        legendStr = legendStr + "percError[%.3f] "%percOfError
        ax2.legend([legendStr]); plt.grid(1);
        if savePlot:
            filename = self.getNetworkSummary().replace('.',',')
            plt.savefig('./plots/' + filename + '_val.png')
        if showPlot:
            plt.show();

    def plotTraining(self, costs, percsError, showPlot, savePlot):
        fig = plt.figure()
        plt.plot(np.arange(0,self.epochs,1), costs)
        plt.plot(np.arange(0,self.epochs,1), percsError)
        legends = []
        legends.append("costs - min[%.3f]" % np.amin(costs))
        legends.append("percsError - min[%.3f]" % np.amin(percsError))
        # legendStr = self.getNetworkSummary()
        # legendStr = legendStr + "min[%.3f] "%np.amin(costs)
        plt.legend(legends); plt.grid(1);
        if savePlot:
            filename = self.getNetworkSummary().replace('.',',')
            plt.savefig('./plots/' + filename + '_train.png')
        if showPlot:
            plt.show();

    def getNetworkSummary(self):
        summary = "lay[%d:%d," % ((self.numberHiddenLayer + 2), self.inputLayerSize)
        for i in self.hiddenLayerSizes:
            summary = summary + "%d," % i
        summary = summary + "%d] " % self.outputLayerSize
        summary = summary + "sca[%.2f] " % self.scalar
        summary = summary + "mom[%.1f] " % self.momentum
        summary = summary + "bat[%d] " % self.batchSize
        summary = summary + "epo[%d] " % self.epochs
        return summary

    def printToFile(self, results):
        for i in range(len(self.w)):
            np.savetxt('./output/weight_' + str(i), self.w[i], delimiter=' ', fmt='%1.20f')
            np.savetxt('./output/threshold_' + str(i), self.t[i], delimiter=' ', fmt='%1.20f')
        np.savetxt('./output/results_normalized', results, delimiter=' ', fmt='%1.10f')
        
    def trainNetwork(self, epochs, trainDataFile, useInteractivePlot):
        # Enable interactive mode
        plt.ion()
        # plt.hold(True)
        #
        self.epochs = epochs
        # Build weights data and save values separatedly
        self.randomizeWeightsAndThresholds()
       

        inputs, outputs = self.readDataAndNormalize(trainDataFile, True, False)
        rowsTrain = 351
        rowsValidate = 50
        trainInput, trainOutput = inputs[0:rowsTrain,:], outputs[0:rowsTrain,:]
        validateInput, validateOutput = inputs[rowsTrain:rowsTrain + rowsValidate,:], outputs[rowsTrain:rowsTrain + rowsValidate,:]

        # Mark start training time
        startTime = time.clock()

        costs = []
        percsError = []
        for i in range(self.epochs):
            yh, yTrain = self.forward(trainInput, trainOutput, self.batchSize)
            cost, results, percError = self.costFunction(yTrain, yhTrain)
            self.costFunctionPrime(yTrain, yhTrain)

            # Check gradient calculations using numerical gradients
            # if self.checkWithNumericalGradients:
            #     self.compareWithNumericalCalculation

            # Fix weights after gradient calculations
            self.fixWeights(self.scalar, self.momentum)
            self.fixThresholds(self.scalar, self.momentum)

            # Compute cost again using validation data
            yhVal, yVal = self.forward(validateInput, validateOutput, 0)
            cost, results, percError = self.costFunction(yVal, yhVal)

            # Save results
            costs.append(cost)
            percsError.append(percError)

            # Print if verbose
            if self.verbose >= 1 and (i%10000 == 0 or i == (self.epochs-1)):
                print ("epoch=%d, cost=%f, percError=%f, minPE=%f" % (i, cost, percError, np.amin(percsError)))
                if useInteractivePlot and i > 0:
                    # y = np.random.random()
                    # plt.scatter(i, y)
                    plt.plot(np.arange(0,len(percsError),1), percsError)
                    plt.pause(0.001)
                if self.verbose >= 2:
                    print (results)

            # If error too low go output
            if percError < 0.5 and i > (epochs * 0.8):
                print ("epoch=%d, cost=%f, percError=%f" % (i, cost, percError))
                self.epochs = i + 1
                print (len(costs))
                print (len(percsError))
                print (self.epochs)
                break

            # Update learning rate after certains epochs

        # Disable interactive mode
        plt.ioff()
        # Print results to file
        self.printToFile(results)
        # Print final info
        print ("--------------------")
        print (self.getNetworkSummary())
        print ("Training time: %f" % (time.clock() - startTime))
        # Return results
        return costs, results, percsError

    def validateLearning(self, validateFile):
        # Mark start training time
        startTime = time.clock()
        # Get validation set from file
        inputs, outputs = self.readDataAndNormalize(validateFile, True, True)
        # Get weights and thresholds from training resulting files
        self.w = []; self.t = []
        for i in range(self.numberHiddenLayer + 1):
            # shape = self.w[i].shape
            self.w.append(np.loadtxt('./output/weight_' + str(i)))
            self.t.append(np.loadtxt('./output/threshold_' + str(i)))
            if len(self.w[-1].shape) == 1:
                self.w[-1] = np.array([self.w[-1]]).T
        # Make forward and calculate cost
        yh, y = self.forward(inputs, outputs, 0)      # paramBatchSize
        cost, results, percError = self.costFunction(y, yHat)
        # Denormalize results
        denormalizedResult = (((results - self.norm['limMin']) * (self.norm['maxs'][-1] - self.norm['mins'][-1])) / (self.norm['limMax']-self.norm['limMin'])) + self.norm['mins'][-1];
        # print denormalizedResult
        np.savetxt('./output/results', denormalizedResult, delimiter=' ', fmt='%1.3f')
        # Check time and return
        print ("Validating time: %f" % (time.clock() - startTime))
        return denormalizedResult


