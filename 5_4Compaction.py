from compaction import compact
import numpy as np
from NNAO import Simulator
from NNAO import InputParametersCalibrator

class CompactionSimulator(Simulator):
    def __init__(self):
        self.desired_results=[0.5,0.497,0.492,10,19.9,9.84]

    def ModifyParameters(self, new_parameters):
        self.c = new_parameters[0]
        self.rho_grain = new_parameters[1]
        self.excess_pressure = new_parameters[2]
        self.rho_void = new_parameters[3]
    
    def RunSimulation(self):
        return_dz= np.array([0, 0, 0], dtype=float)
        result=compact(self.dz,self.porosity,self.c,self.rho_grain,self.excess_pressure,self.porosity_min,self.porosity_max,self.rho_void,self.gravity,return_dz)
        self.result=result
        self.return_dz=return_dz

    def GetErrorNorm(self):
        error=abs((self.result[0]-self.desired_results[0])/self.desired_results[0])+abs((self.result[1]-self.desired_results[1])/self.desired_results[1])+abs((self.result[2]-self.desired_results[2])/self.desired_results[2])+abs((self.return_dz[0]-self.desired_results[3])/self.desired_results[3])+abs((self.return_dz[1]-self.desired_results[4])/self.desired_results[4])+abs((self.return_dz[2]-self.desired_results[5])/self.desired_results[5])
        return error

    def ShowResults(self):
        print("The new value of c is: ", self.optimized_parameters[0])
        print("The new value of rho_grain is: ", self.optimized_parameters[1])
        print("The new value of excess_pressure is: ", self.optimized_parameters[2])
        print("The new value of rho_void is: ", self.optimized_parameters[3])

    def GetInitialAproximationToParameters(self):
        self.dz = np.array([10, 20, 10], dtype=float)
        self.porosity = np.array([0.5, 0.5, 0.5], dtype=float)
        c = 5e-8
        rho_grain = 2650.0
        excess_pressure = 10000
        self.porosity_min = 0
        self.porosity_max = 0.5
        rho_void = 1000.0
        self.gravity = 9.81
        parameters=[c,rho_grain,excess_pressure,rho_void]
        self.num_parameters=len(parameters)
        self.initial_parameters=parameters
        return parameters
    
    def SetBoundaries(self):
        self.deviations=[0.2,0.1,1,0.05]
        bmax=[0]*self.num_parameters
        bmin=[0]*self.num_parameters
        bmax[0] = (1+self.deviations[0])*self.initial_parameters[0]
        bmin[0] = (1-self.deviations[0])*self.initial_parameters[0]
        bmax[1] = (1+self.deviations[1])*self.initial_parameters[1]
        bmin[1] = (1-self.deviations[1])*self.initial_parameters[1]
        bmax[2] = (1+self.deviations[2])*self.initial_parameters[2]
        bmin[2] = (1-self.deviations[2])*self.initial_parameters[2]
        bmax[3] = (1+self.deviations[3])*self.initial_parameters[3]
        bmin[3] = (1-self.deviations[3])*self.initial_parameters[3]
        bounds = [(bmin[b], bmax[b]) for b in range(self.num_parameters)]
        self.bounds=bounds

class OptionsContainer():
    def __init__(self):
        self.type_Optimize='Normal'
        self.method='Nelder-Mead'
        self.tolerance=1e-7
        self.data_type='Values'
        self.number_samples=50
        self.number_epochs=1000

Options=OptionsContainer()

Compaction=CompactionSimulator()

Object=InputParametersCalibrator(Compaction, Options)

result=Object.Optimize()
Object.PrintOptimizedParameters()
Object.PrintErrorAchieved()
Object.PrintExecutionTime()
Object.PrintNumOfIterations()