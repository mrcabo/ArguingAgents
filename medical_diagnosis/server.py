import numpy as np

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule, TextElement, BarChartModule
from mesa.visualization.UserParam import UserSettableParameter

from medical_diagnosis.Model import MedicalModel, ARGUMENT_NAMES, COLORS


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
    def __init__(self, n_doctors=3, n_init_arg=5, experiment_case=1):
        self.n_init_arg = n_init_arg

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

        diagnosis = PrintedDiagnosis()

        # title = LegendElement("<h1>Welcome to our simulation</h1>")
        legend_belief_array = LegendElement('<font size="5"><b>1. </b></font> The graph below displays the belief '
                                            'array for each of the doctors (e.g. Doctor 0, Doctor 1..)')
        # TODO: explain this legend more accurately
        legend_conclusion = LegendElement('<font size="5"><b>2. </b></font> The graph below displays the probability '
                                          'of the committee in each conclusion.')
        legend_avg_belief = LegendElement('The graph below represents the average belief between all doctors '
                                          'for each of the possible arguments')

        list_of_visualizations = [legend_belief_array, bar_chart, legend_conclusion, disease_line_chart, diagnosis,
                                  legend_avg_belief, avg_belief_line_chart]
        model_legend = ""
        if experiment_case == 1:  # Default case
            model_legend = """<h1>Default Case Scenario.</h1><br><h3>The initial set of arguments is the 
            following:</h3><br>
            """
            for arg_name, arg_idx in MedicalModel.LIST_OF_ARGUMENTS.items():
                model_legend += "<b>" + arg_name + "</b>" + ": " + arg_idx + "<br>"

        model_params = {
            "Legend": UserSettableParameter('static_text', value=model_legend),
            "N": n_doctors,
            "n_init_arg": n_init_arg,
            "experiment_case": experiment_case
        }
        # Create server
        self.server = ModularServer(MedicalModel, list_of_visualizations, "Evacuation model", model_params)
        self.server.port = 8521
