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


class PrintedDiagnosis(TextElement):
    """
    Displays the argumentation process
    """

    def __init__(self):
        pass

    def render(self, model):
        text = model.diagnosis_text
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


class LegendElement(TextElement):
    """
    Displays a legend for a visualization
    """

    def __init__(self, text):
        self.text = text

    def render(self, model):
        return self.text


class ServerClass:
    def __init__(self, n_doctors=3, n_init_arg=5, default_case=True):
        self.n_init_arg = n_init_arg

        model_params = {
            # "N": UserSettableParameter('slider', "Number of agents", n_doctors, 2, 10, 1,
            #                            description="Choose how many agents to include in the model"),
            "N": n_doctors,
            "n_init_arg": n_init_arg,
            "default_case": default_case
        }
        # Create a line chart tracking avg_belief for all the initial arguments
        list_var = []
        for i in range(self.n_init_arg):
            dict = {"Label": ARGUMENT_NAMES[i], "Color": COLORS[i]}
            list_var.append(dict)
        avg_belief_line_chart = ChartModule(list_var)

        list_var = []
        for i, disease in enumerate(MedicalModel.LIST_OF_DISEASES.values()):
            dict = {"Label": disease, "Color": COLORS[i]}
            list_var.append(dict)
        disease_line_chart = ChartModule(list_var)

        list_var = []
        for i in range(self.n_init_arg):
            dict = {"Label": ARGUMENT_NAMES[i], "Color": COLORS[i]}
            list_var.append(dict)
        bar_chart = BarChartModule(list_var, scope="agent")

        printed_arguments = PrintedArgumentationElement()
        diagnosis = PrintedDiagnosis()

        title = LegendElement("<h1>Welcome to our simulation</h1>")
        legend_avg_belief = LegendElement("The graph below represents the average belief between all doctors for each "
                                          "of the possible arguments")
        legend_belief_array = LegendElement("The graph below displays the belief array for each of the doctors (e.g. "
                                            "Doctor 0, Doctor 1..)")
        # TODO: explain this legend more accurately
        legend_conclusion = LegendElement("The graph below displays the probability in the conclusion.")
        list_of_visualizations = [title, legend_belief_array, bar_chart, legend_conclusion, disease_line_chart,
                                  legend_avg_belief, avg_belief_line_chart, printed_arguments, diagnosis]

        # Create server
        self.server = ModularServer(MedicalModel, list_of_visualizations, "Evacuation model", model_params)
        self.server.port = 8521
