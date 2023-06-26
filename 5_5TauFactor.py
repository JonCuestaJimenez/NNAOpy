import taufactor as tau
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from NNAO import Simulator
from NNAO import InputParametersCalibrator

class TauFactorSimulator(Simulator):
    def __init__(self):
        self.file_name='SampleData_2Phase.tif'
        self.desired_results=[7]

    def ModifyParameters(self, new_parameters):
        int_img=np.round(new_parameters).astype(int)
        img_new=int_img.reshape(self.shape)
        img_new = np.where(img_new == 1, True, img_new)
        img_new = np.where(img_new == 0, False, img_new) 
        self.img=img_new
    
    def RunSimulation(self):
        simulation = tau.Solver(self.img)
        simulation.solve()
        self.simulation=simulation

    def GetErrorNorm(self):
        tau=self.simulation.tau.item()
        error=abs(tau-self.desired_results[0])/self.desired_results[0]
        return error

    def ShowResults(self):
        fig1 = plt.figure()
        ax = fig1.add_subplot(projection='3d')
        x, y, z = np.indices(self.initial_img.shape)
        x, y, z = x.flatten(), y.flatten(), z.flatten()
        initial_img_flat = self.initial_img.flatten()
        ax.scatter(x, y, z, c=initial_img_flat)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        optimized_parameters=np.round(self.optimized_parameters).astype(int)
        optimized_parameters=optimized_parameters.reshape(self.shape)
        fig2 = plt.figure()
        ax = fig2.add_subplot(projection='3d')
        x, y, z = np.indices(optimized_parameters.shape)
        x, y, z = x.flatten(), y.flatten(), z.flatten()
        optimized_parameters = optimized_parameters.flatten()
        ax.scatter(x, y, z, c=optimized_parameters)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    def GetInitialAproximationToParameters(self):
        img_big = tifffile.imread(self.file_name)
        self.img=img_big[:10,:10,:10]
        self.initial_img=self.img
        self.shape=self.img.shape
        self.img_flat=self.img.flatten()
        self.initial_parameters=self.img_flat
        return self.img_flat
    
    def SetBoundaries(self):
        bound0=np.zeros(self.shape)
        bound1=np.ones(self.shape)
        bounds = tuple(zip(bound0.flatten(), bound1.flatten()))
        self.bounds=bounds

class OptionsContainer():
    def __init__(self):
        self.type_Optimize='Normal'
        self.method='Powell'
        self.tolerance=1e-7
        self.data_type='Images'
        self.number_samples=50
        self.number_epochs=1000

Options=OptionsContainer()

TauFactor=TauFactorSimulator()

Object=InputParametersCalibrator(TauFactor, Options)

result=Object.Optimize()
Object.PrintOptimizedParameters()
Object.PrintErrorAchieved()
Object.PrintExecutionTime()
Object.PrintNumOfIterations()