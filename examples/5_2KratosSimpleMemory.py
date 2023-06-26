import sys
import time
import importlib
import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as StructuralMechanicsApplication
from NNAO import Simulator
from NNAO import InputParametersCalibrator

def CreateAnalysisStageWithFlushInstance(cls, global_model, parameters, material_parameters):
    class AnalysisStageWithFlush(cls):
        def __init__(self, model,project_parameters, material_parameters, flush_frequency=10.0):
            super().__init__(model,project_parameters)
            self.material_parameters_unused=material_parameters
            self.flush_frequency = flush_frequency
            self.last_flush = time.time()
            sys.stdout.flush()

        def Initialize(self):
            super().Initialize()
            props=self.model.GetModelPart("Structure").GetProperties()[1]
            props.SetValue(KratosMultiphysics.DENSITY, self.material_parameters_unused["properties"][0]["Material"]["Variables"]["DENSITY"].GetDouble())
            props.SetValue(KratosMultiphysics.YOUNG_MODULUS, self.material_parameters_unused["properties"][0]["Material"]["Variables"]["YOUNG_MODULUS"].GetDouble())
            props.SetValue(StructuralMechanicsApplication.CROSS_AREA, self.material_parameters_unused["properties"][0]["Material"]["Variables"]["CROSS_AREA"].GetDouble())
            props.SetValue(StructuralMechanicsApplication.I33, self.material_parameters_unused["properties"][0]["Material"]["Variables"]["I33"].GetDouble())
            sys.stdout.flush()

        def FinalizeSolutionStep(self):
            super().FinalizeSolutionStep()
            if self.parallel_type == "OpenMP":
                now = time.time()
                if now - self.last_flush > self.flush_frequency:
                    sys.stdout.flush()
                    self.last_flush = now

        def Finalize(self):
            currentmax=0
            for node in self.model.GetModelPart("Structure").Nodes:
                disp=node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT)                
                currentmax=max(abs(disp[1]), currentmax)
            self.max_displacement_unused=currentmax
            super().Finalize()

    return AnalysisStageWithFlush(global_model, parameters, material_parameters)

class KratosSimulator(Simulator):
    def __init__(self):
        self.desired_results=[7.5e-3]
        with open("ProjectParameters.json", 'r') as parameter_file:
            self.original_parameters = KratosMultiphysics.Parameters(parameter_file.read())
        filename=self.original_parameters["solver_settings"]["material_import_settings"]["materials_filename"].GetString()
        with open(filename) as parameter_file:
            self.original_material_parameters = KratosMultiphysics.Parameters(parameter_file.read())

        analysis_stage_module_name = self.original_parameters["analysis_stage"].GetString()
        analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
        analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

        analysis_stage_module = importlib.import_module(analysis_stage_module_name)
        self.analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

    def ModifyParameters(self, new_parameters):
        area=new_parameters[2]*new_parameters[3]
        inercia=(new_parameters[2]*new_parameters[3]**3)/12
        self.original_material_parameters["properties"][0]["Material"]["Variables"]["DENSITY"].SetDouble(new_parameters[0])
        self.original_material_parameters["properties"][0]["Material"]["Variables"]["YOUNG_MODULUS"].SetDouble(new_parameters[1])
        self.original_material_parameters["properties"][0]["Material"]["Variables"]["CROSS_AREA"].SetDouble(area)
        self.original_material_parameters["properties"][0]["Material"]["Variables"]["I33"].SetDouble(inercia)
        self.original_parameters["processes"]["loads_process_list"][1]["Parameters"]["modulus"].SetDouble(new_parameters[4])
        self.modified_parameters=self.original_parameters
        self.modified_material_parameters=self.original_material_parameters

    def RunSimulation(self):
        simulation = CreateAnalysisStageWithFlushInstance(self.analysis_stage_class, KratosMultiphysics.Model(), self.modified_parameters, self.modified_material_parameters)
        simulation.Run()
        self.max_displacement=simulation.max_displacement_unused

    def GetErrorNorm(self):
        error_new=(abs(self.max_displacement-self.desired_results[0]))/self.desired_results[0]
        return error_new

    def ShowResults(self):
        print("The new value of DENSITY is: ", self.optimized_parameters[0])
        print("The new value of YOUNG_MODULUS is: ", self.optimized_parameters[1])
        print("The new value of width is: ", self.optimized_parameters[2])
        print("The new value of height is: ", self.optimized_parameters[3])
        print("The new value of LOAD is: ", self.optimized_parameters[4])

    def GetInitialAproximationToParameters(self):
        parameters=[0]*5
        parameters[0]=self.original_material_parameters["properties"][0]["Material"]["Variables"]["DENSITY"].GetDouble()
        parameters[1]=self.original_material_parameters["properties"][0]["Material"]["Variables"]["YOUNG_MODULUS"].GetDouble()
        parameters[2]=0.1
        parameters[3]=0.2
        parameters[4]=self.original_parameters["processes"]["loads_process_list"][1]["Parameters"]["modulus"].GetDouble()
        self.num_parameters=len(parameters)
        self.initial_parameters=parameters
        return parameters
    
    def SetBoundaries(self):
        self.deviations=[0.1,0.1,0.05,0.05,0.15]
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
        bmax[4] = (1+self.deviations[4])*self.initial_parameters[4]
        bmin[4] = (1-self.deviations[4])*self.initial_parameters[4]
        bounds = [(bmin[b], bmax[b]) for b in range(self.num_parameters)]
        self.bounds=bounds

class OptionsContainer():
    def __init__(self):
        self.type_Optimize='Normal'
        self.method='Nelder-Mead'
        self.tolerance=1e-5
        self.number_samples=50
        self.data_type='Values'
        self.number_epochs=1000
        
Options=OptionsContainer()

Kratos=KratosSimulator()

Object=InputParametersCalibrator(Kratos, Options)

result=Object.Optimize()
Object.PrintOptimizedParameters()
Object.PrintErrorAchieved()
Object.PrintExecutionTime()
Object.PrintNumOfIterations()