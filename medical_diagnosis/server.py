from medical_diagnosis.Model import MedicalModel
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import TextElement
from mesa.visualization.UserParam import UserSettableParameter


class HelloWorldElement(TextElement):
    """
    Display a text count of how many happy agents there are.
    """

    def __init__(self):
        pass

    def render(self, model):
        return "Hello World!"


happy_element = HelloWorldElement()

model_params = {
    "N": UserSettableParameter('slider', "Number of agents", 3, 2, 10, 1,
                               description="Choose how many agents to include in the model")
}

server = ModularServer(MedicalModel, [happy_element], "Evacuation model", model_params)
server.port = 8521
