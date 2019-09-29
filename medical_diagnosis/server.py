import numpy as np

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule, TextElement, BarChartModule
from mesa.visualization.UserParam import UserSettableParameter

from medical_diagnosis.Model import MedicalModel, ARGUMENT_NAMES, COLORS


class PrintedArgumentationElement(TextElement):
    """
    Displays the argumentation process
    """

    def __init__(self):
        pass

    def render(self, model):
        text = model.argumentation_text
        return text


class PrintedBeliefArray(TextElement):
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
    def __init__(self, n_doctors=3, n_init_arg=5, default_case=True):
        self.n_init_arg = n_init_arg

        model_params = {
            "N": UserSettableParameter('slider', "Number of agents", n_doctors, 2, 10, 1,
                                       description="Choose how many agents to include in the model"),
            "n_init_arg": n_init_arg,
            "default_case": default_case
        }
        # Create a line chart tracking avg_belief for all the initial arguments
        list = []
        for i in range(self.n_init_arg):
            dict = {"Label": ARGUMENT_NAMES[i], "Color": COLORS[i]}
            list.append(dict)
        line_chart = ChartModule(list)

        list = []
        for i in range(self.n_init_arg):
            dict = {"Label": ARGUMENT_NAMES[i], "Color": COLORS[i]}
            list.append(dict)
        bar_chart = BarChartModule(list, scope="agent")

        printed_arguments = PrintedArgumentationElement()

        list_of_visualizations = [line_chart, bar_chart, printed_arguments]

        # Create server
        self.server = ModularServer(MedicalModel, list_of_visualizations, "Evacuation model", model_params)
        self.server.port = 8521
