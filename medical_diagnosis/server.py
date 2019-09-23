import numpy as np

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter

from medical_diagnosis.Model import MedicalModel, ARGUMENT_NAMES, COLORS


class PrintedArgumentationElement(TextElement):
    """
    Displays the argumentation process
    """

    def __init__(self):
        pass

    def render(self, model):
        text = ""
        for agent in model.schedule.agents:
            text += "Doctor {} believes in {}<br>".format(agent._doctor_id, np.round(agent.belief_array, 3))

        return text


class ServerClass:
    def __init__(self, n_init_arg=5):
        self.n_init_arg = n_init_arg

        model_params = {
            "N": UserSettableParameter('slider', "Number of agents", 3, 2, 10, 1,
                                       description="Choose how many agents to include in the model")
        }
        # Create a line chart tracking avg_belief for all the initial arguments
        list = []
        for i in range(self.n_init_arg):
            dict = {"Label": ARGUMENT_NAMES[i], "Color": COLORS[i]}
            list.append(dict)
        line_chart = ChartModule(list)

        # Here we can display text, now is just displaying agent's beliefs
        printed_arguments = PrintedArgumentationElement()
        # Create server
        self.server = ModularServer(MedicalModel, [line_chart, printed_arguments], "Evacuation model", model_params)
        self.server.port = 8521
