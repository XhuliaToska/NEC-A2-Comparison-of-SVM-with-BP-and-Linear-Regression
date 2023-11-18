from Neural_Network import*
NN=NeuralNetwork()
#Network Architecture
NN.inputlayerSize=4
NN.outputaLayerSize=1
NN.numberHiddenLayer=2
NN.hiddenLayerSize=[9,3]
NN.epochs=1000000
NN.scalar=0.3
NN.momentum=0.9
NN.batchsize=1
#other settings
NN.verbose=1 #cost for every 1000 epochs
NN.plot=True
NN.checkwithNUmericalGradients=False
NN.randomSeed=1 #seed random numbers
