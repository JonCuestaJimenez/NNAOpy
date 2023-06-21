from scipy.optimize import minimize
import tensorflow as tf
from pyDOE import *
import numpy as np
import time

class Simulator():
    def __init__(self):
        pass

    def ModifyParameters(self, new_parameters):
            pass
    
    def RunSimulation(self):
        pass

    def GetErrorNorm(self):
        pass

    def ShowResults(self):
        pass

    def GetInitialAproximationToParameters(self):
        pass

    def SetBoundaries(self):
        pass

class InputParametersCalibrator():
    def __init__(self, Simulator, Options):
        self.Simulator=Simulator
        self.Options=Options
        self.CheckOptions()
        
    def CheckOptions(self):
        if self.Options.type_Optimize == 'Normal':
            pass
        elif self.Options.type_Optimize == 'NeuralNetwork':
            if isinstance(self.Options.number_samples, int):
                pass
            else:
                raise Exception('Wrong input at number_samples, please set an int value')
            if isinstance(self.Options.number_epochs, int):
                pass
            else:
                raise Exception('Wrong input at number_epochs, please set an int value')
            if self.Options.data_type == 'Values' or self.Options.data_type == 'Images':
                pass
            else:
                raise Exception('Wrong input at data_type, please set the option to Values or Images')
        else:
            raise Exception('Wrong input at typeOptimize, please set the option to Normal or NeuralNetwork')
        if isinstance(self.Options.tolerance, float) or isinstance(self.Options.tolerance, int):
            pass
        else:
            raise Exception('Wrong input at tolerance, please set a float or int value')


    def FunctionToMinimize(self, new_parameters):
        self.Simulator.ModifyParameters(new_parameters)
        self.Simulator.RunSimulation()
        error=self.Simulator.GetErrorNorm()
        return error

    def Optimize(self):
        self.initial_parameters=self.Simulator.GetInitialAproximationToParameters()
        self.Simulator.SetBoundaries()
        if self.Options.type_Optimize == 'Normal':
            self.start_time = time.time()
            result = minimize(self.FunctionToMinimize, self.initial_parameters, bounds=self.Simulator.bounds, method=self.Options.method, tol=self.Options.tolerance)
            self.end_time = time.time()
            self.result=result
        elif self.Options.type_Optimize == 'NeuralNetwork':
            self.start_time = time.time()
            self.NeuralNetwork()            
            result = minimize(self.NeuralNetworkToMinimize, self.initial_parameters, bounds=self.Simulator.bounds, method=self.Options.method, tol=self.Options.tolerance)
            self.end_time = time.time()        
            self.result=result
        return result
    
    def DesignOfExperiments(self):
        parameters_cloud=lhs(self.Simulator.num_parameters, samples=self.Options.number_samples)
        if self.Options.data_type == 'Values':
            means=self.Simulator.GetInitialAproximationToParameters()
            self.means=means
            deviation=parameters_cloud-0.5
            for i in range(self.Simulator.num_parameters):
                deviation[:,i]=means[i]*self.Simulator.deviations[i]*deviation[:,i]*2
            parameters_cloud=deviation+means
        self.parameters_cloud=parameters_cloud
        return parameters_cloud
    
    def ObtainErrorsFromExperiments(self):
        errors_cloud=[0]*len(self.parameters_cloud)
        for i in range(len(self.parameters_cloud)):
            errors_cloud[i]=self.FunctionToMinimize(self.parameters_cloud[i,:])
        return errors_cloud
    
    def NeuralNetwork(self):
        x_train=self.DesignOfExperiments()
        y_train=self.ObtainErrorsFromExperiments()
        if self.Options.data_type == 'Values':
            for i in range(self.Options.number_samples):
                x_train[i,:]=x_train[i,:]/self.means
            x_train=x_train-0.9
            x_train=x_train*5    
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.Simulator.num_parameters,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation=tf.math.abs)
        ])
        model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_absolute_error'])
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        model.fit(x_train, y_train, epochs=self.Options.number_epochs, shuffle=True, batch_size=32)
        self.model=model

    def NeuralNetworkToMinimize(self,new_parameters):
        if self.Options.data_type == 'Values':
            for i in range(self.Simulator.num_parameters):
                new_parameters[i]=new_parameters[i]/self.initial_parameters[i]
            new_parameters = np.array(new_parameters)
            new_parameters = new_parameters-0.9
            new_parameters = new_parameters*5
        new_parameters = tf.convert_to_tensor(new_parameters, dtype=tf.float32)
        new_parameters = tf.reshape(new_parameters,(1,self.Simulator.num_parameters))
        neural_error=self.model.predict(new_parameters)
        return neural_error
    
    def GetOptimizedParameters(self):
        self.Simulator.optimized_parameters=self.result.x
        return self.Simulator.optimized_parameters
    
    def PrintOptimizedParameters(self):
        self.GetOptimizedParameters()
        self.Simulator.ShowResults()

    def GetExecutionTime(self):
        execution_time = self.end_time - self.start_time
        return execution_time
    
    def PrintExecutionTime(self):
        print("Execution time:", self.GetExecutionTime(), "seconds")

    def GetNumOfIterations(self):
        return self.result.nfev
    
    def PrintNumOfIterations(self):
        print("Number of iterations: ", self.GetNumOfIterations())   

    def GetErrorAchieved(self):
        return self.result.fun

    def PrintErrorAchieved(self):
        print("The error achieved is: ", self.GetErrorAchieved())