import json
import importlib
import vtk
import KratosMultiphysics
from NestedDictManager import DictionaryNavigator
from NNAO import Simulator
from NNAO import InputParametersCalibrator

class KratosSimulator(Simulator):
    def __init__(self):
        self.desired_results=[6.5e-3]

    def ModifyParameters(self, new_parameters):
        Navigator.SetNestedValue(self.data1, 'properties.0.Material.Variables.DENSITY', new_parameters[0])
        Navigator.SetNestedValue(self.data1, 'properties.0.Material.Variables.YOUNG_MODULUS', new_parameters[1])
        with open('StructuralMaterials.json', 'w') as f:
            json.dump(self.data1, f, indent=4)
    
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
        reader.SetFileName('C:/Users/joncj/Desktop/PTYTHON/EX01.gid/vtk_output/Structure_0_1.vtk')
        reader.Update()
        displacement = reader.GetOutput().GetPointData().GetArray('DISPLACEMENT')
        second_components = []
        for i in range(displacement.GetNumberOfTuples()):
            point_displacement = displacement.GetTuple(i)
            second_component = abs(point_displacement[1])
            second_components.append(second_component)
        max_displacement = max(second_components)
        error_new=(abs(max_displacement-self.desired_results[0]))/self.desired_results[0]
        return error_new

    def ShowResults(self):
        print("The new value of DENSITY is: ", self.optimized_parameters[0])
        print("The new value of YOUNG_MODULUS is: ", self.optimized_parameters[1])

    def GetInitialAproximationToParameters(self):
        parameters=[0]*2
        with open('StructuralMaterials.json') as f:
            self.data1 = json.load(f)                
        parameters[0]=Navigator.GetNestedValue(self.data1, 'properties.0.Material.Variables.DENSITY')
        parameters[1]=Navigator.GetNestedValue(self.data1, 'properties.0.Material.Variables.YOUNG_MODULUS')  
        self.num_parameters=len(parameters)
        self.initial_parameters=parameters
        return parameters
    
    def SetBoundaries(self):
        self.deviations=[0.1,0.1]
        bmax=[0]*self.num_parameters
        bmin=[0]*self.num_parameters
        bmax[0] = (1+self.deviations[0])*self.initial_parameters[0]
        bmin[0] = (1-self.deviations[0])*self.initial_parameters[0]
        bmax[1] = (1+self.deviations[1])*self.initial_parameters[1]
        bmin[1] = (1-self.deviations[1])*self.initial_parameters[1]
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

Navigator=DictionaryNavigator()

Options=OptionsContainer()

Kratos=KratosSimulator()

Object=InputParametersCalibrator(Kratos, Options)

result=Object.Optimize()
Object.PrintOptimizedParameters()
Object.PrintErrorAchieved()
Object.PrintExecutionTime()
Object.PrintNumOfIterations()