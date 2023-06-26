import json
import importlib
import vtk
import KratosMultiphysics
import numpy as np
from NestedDictManager import DictionaryNavigator
from NNAO import Simulator
from NNAO import InputParametersCalibrator

class KratosSimulator(Simulator):
    def __init__(self):
        self.desired_results=[3.6e-04, 100000]

    def ModifyParameters(self, new_parameters):
        Navigator.SetNestedValue(self.data1, 'properties.0.Material.Variables.YOUNG_MODULUS', new_parameters[0])
        Navigator.SetNestedValue(self.data1, 'properties.0.Material.Variables.THICKNESS', new_parameters[1])
        with open('StructuralMaterials.json', 'w') as f:
            json.dump(self.data1, f, indent=4)        
        Navigator.SetNestedValue(self.data2, 'processes.loads_process_list.0.Parameters.modulus', new_parameters[2])
        with open('ProjectParameters.json', 'w') as f:
            json.dump(self.data2, f, indent=4)
    
    def RunSimulation(self):
        with open("ProjectParameters.json", 'r') as parameter_file:
            parameters = KratosMultiphysics.Parameters(parameter_file.read())

        analysis_stage_module_name = parameters["analysis_stage"].GetString()
        analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
        analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

        analysis_stage_module = importlib.import_module(analysis_stage_module_name)
        analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

        simulation = analysis_stage_class(KratosMultiphysics.Model(), parameters)
        simulation.Run()

    def GetErrorNorm(self):
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName('C:/Users/joncj/Desktop/PTYTHON/EX02.gid/vtk_output/Structure_0_5.vtk')
        reader.Update()
        displacement = reader.GetOutput().GetPointData().GetArray('DISPLACEMENT')
        second_components = []
        for i in range(displacement.GetNumberOfTuples()):
            point_displacement = displacement.GetTuple(i)
            second_component = point_displacement[0]
            second_components.append(second_component)
        max_displacement = max(second_components)
        stress = reader.GetOutput().GetPointData().GetArray('VON_MISES_STRESS')
        components = []
        for i in range(stress.GetNumberOfTuples()):
            point_stress0 = stress.GetTuple(i)
            point_stress = point_stress0[0]
            components.append(point_stress)
        max_stress = max(components)
        error_new=((abs(max_displacement-self.desired_results[0]))/self.desired_results[0])+((abs(max_stress-self.desired_results[1]))/self.desired_results[1])
        return error_new

    def ShowResults(self):
        print("The new value of YOUNG_MODULUS is: ", self.optimized_parameters[0])
        print("The new value of THICKNESS is: ", self.optimized_parameters[1])
        print("The new value of LOAD is: ", self.optimized_parameters[2])

    def GetInitialAproximationToParameters(self):
        parameters=[0]*3
        with open('StructuralMaterials.json') as f:
            self.data1 = json.load(f)                
        parameters[0]=Navigator.GetNestedValue(self.data1, 'properties.0.Material.Variables.YOUNG_MODULUS') 
        parameters[1]=Navigator.GetNestedValue(self.data1, 'properties.0.Material.Variables.THICKNESS') 
        with open('ProjectParameters.json') as f:
            self.data2 = json.load(f)          
        parameters[2]=Navigator.GetNestedValue(self.data2, 'processes.loads_process_list.0.Parameters.modulus')   
        self.num_parameters=len(parameters)
        self.initial_parameters=parameters
        return parameters
    
    def SetBoundaries(self):
        self.deviations=[0.1,0.1,0.1]
        bmax=[0]*self.num_parameters
        bmin=[0]*self.num_parameters
        bmax[0] = (1+self.deviations[0])*self.initial_parameters[0]
        bmin[0] = (1-self.deviations[0])*self.initial_parameters[0]
        bmax[1] = (1+self.deviations[1])*self.initial_parameters[1]
        bmin[1] = (1-self.deviations[1])*self.initial_parameters[1]
        bmax[2] = (1+self.deviations[2])*self.initial_parameters[2]
        bmin[2] = (1-self.deviations[2])*self.initial_parameters[2]
        bounds = [(bmin[b], bmax[b]) for b in range(self.num_parameters)]
        self.bounds=bounds

class OptionsContainer():
    def __init__(self):
        self.type_Optimize='NeuralNetwork'
        self.method='Nelder-Mead'
        self.tolerance=1e-7
        self.data_type='Values'
        self.number_samples=300
        self.number_epochs=1000

Options=OptionsContainer()

Navigator=DictionaryNavigator()

Kratos=KratosSimulator()

Object=InputParametersCalibrator(Kratos, Options)

parameters0=Kratos.GetInitialAproximationToParameters()
result=Object.Optimize()
optimized_parameters=Object.GetOptimizedParameters()
initial_error=Object.FunctionToMinimize(parameters0)
real_error=Object.FunctionToMinimize(optimized_parameters)
Object.PrintOptimizedParameters()
Object.PrintErrorAchieved()
Object.PrintExecutionTime()
Object.PrintNumOfIterations()

print('The real error with the initial parameters is: ', initial_error)
print('The real error with the optimized parameters is: ', real_error)