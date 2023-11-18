#We should create a method for training ,another validate and call the independly
from Neural_Network import *
from Neuronal_Network_settings import *
startTime=time.perf_counter()
iniNN=copy.deepcopy(NN)
for i in range (1): #(15,18)
    for j in range(1):#(2,21)
        if False:
            time.sleep(5)
            NN.hiddenLayerSizes[0]=i
            NN.hiddenLayerSizes[1]=j
            inculdeTraining=True
            IncludeValidate=True
            if inculdeTraining:
                costs, results, percsError=NN.trainNetwork(NN.epochs,'./input/turbine -input-train',False)
                NN.plotTraining(costs, percError, True,False)
            if IncludeValidate:
                   print("-------------")
                   print("Now validating")
                   results=NN.validateLearning("./input/turbine-input-validate")
                   NN.plotValidate(results,True,True)
                   NN.copy.deepcopy(iniNN)
                   print("--------------")
                   print((i,j(time.clock()-startTime)))
                   print("-----------------")
                   endTime=time.clock()
                   print("--------------")
                   print("total time"+str(endTime-startTime))








