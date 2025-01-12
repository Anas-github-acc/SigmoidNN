from network import Network, CrossEntropyCost, QuadraticCost
import mnist_loader

import numpy as np

if(__name__ == "__main__"):
  training_data, validation_data, test_data = mnist_loader.minload_data_wrapper()

  testing_data = test_data # @param ["test_data", "validation_data"] {type: "raw"}

  Cost = CrossEntropyCost #@param ["QuadraticCost", "CrossEntropyCost"] {type:"raw"}
  Hidden_Layer = "155" # @param {"type":"string"}
  neurons_layer = [784, 100, 10]

  net = Network(neurons_layer, cost=Cost)
  net.default_weight_init()

  Epochs = 30 # @param {type: "number"}
  Mini_Batch_Size = 10 # @param {"type":"slider","min":0,"max":100,"step":1}
  Learning_Rate = 0.5 # @param {"type":"slider","min":0,"max":1,"step":0.01}
  Regularization = 0.5 # @param {"type":"slider","min":0,"max":1,"step":0.05}

  Monitor_Evaluation_Accuracy = True # @param {type:"boolean"}
  Monitor_Evaluation_Cost = False # @param {type:"boolean"}
  Monitor_Training_Accuracy = False # @param {type:"boolean"}
  Monitor_Training_Cost = False # @param {type:"boolean"}


  net.SGD(training_data=training_data,
          epochs=Epochs,
          mini_batch_size=Mini_Batch_Size,
          eta=Learning_Rate,
          lmbda = Regularization,
          test_data=testing_data,
          monitor_evaluation_accuracy=Monitor_Evaluation_Accuracy,
          monitor_evaluation_cost=Monitor_Evaluation_Cost,
          monitor_training_accuracy=Monitor_Training_Accuracy,
          monitor_training_cost=Monitor_Training_Cost
        )
  
  Save = True # @param {type:"boolean"}
  if Save:
    net.save("net.json")


