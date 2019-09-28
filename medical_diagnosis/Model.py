from functools import partial
import numpy

from mesa import Model
from mesa.time import RandomActivation, BaseScheduler
from mesa.datacollection import DataCollector

from medical_diagnosis.DoctorAgent import DoctorAgent

ARGUMENT_NAMES = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J')
COLORS = ('#00FF00', '#FF0000', '#0000FF', '#383B38', '#FF00FF',
          '#8000FF', '#FF7F00', '#F6F90E', '#6E1122', '#3B541F')


def calculate_avg_belief(idx, model):
    """
    Calculates the mean of the belief for a certain argument between all agents
    Args:
        idx (int): The index of the argument to be calculated
        model (MedicalModel): Model object of our simulation

    Returns:
        Mean value for the belief in an argument between agents
    """
    avg = 0
    for agent in model.schedule.agents:
        avg += agent.belief_array[idx]
    avg = avg / len(model.schedule.agents)
    return avg


class MedicalModel(Model):
    """
        A model with a set of medical agents
        In principle arguing over a final decision
        Through multiple time steps, updating each other's belief array
    """
    # https://www.sciencedirect.com/science/article/pii/S0185106316301135
    LIST_OF_ARGUMENTS = {"A": "The patient has high fevers, which is one of the main symptoms of Zika",
                         "B": "The patient has high fevers, which is one of the main symptoms of Chikungunya",
                         "C": "The patient recently traveled to Brazil. To date, Brazil is the country with the largest "
                              "number of reported cases of Zika; this number is estimated to be between 500,000 and 1,500,000",
                         "D": "The patient presents acute joint pain, which is a common symptom of Chikungunya",
                         "E": "RT-PCR test results came positive. The sensitivity of this test for CHIKV (Chikungunya) "
                              "in the early stages of infection is 88.3%."}

    LIST_OF_DISEASES = {"X": "He has Zika",
                        "Y": "He has Chikungunya",
                        "Z": "He has Dengue"}

    def __init__(self, N=3, n_init_arg=5, default_case=True):
        self.num_agents = N
        self.n_initial_arguments = n_init_arg  # Number of initial arguments that doctors will consider
        self.ground_truth = "Y"  # hardcoded for now..
        self.default_case = default_case
        self.argumentation_text = ""
        if self.default_case:
            self.schedule = BaseScheduler(self)  # For now so they speak in order..
        else:
            self.schedule = RandomActivation(self)  # Every tick, agents move in a different random order

        if self.default_case:
            if (self.num_agents != 3) or (self.n_initial_arguments != 5):
                print("Sorry, the default case only works with 3 doctors and 5 initial arguments")
                exit()
            ground_truth = "X"  # The ground truth for this particular diagnosis (real disease)
            # belief array value of zero means agent didn't have knowledge of that argument yet.
            belief_array = [[0.75, 0.30, 0.80, 0.50, -1.0],
                            [0.80, 0.50, 0.70, 0.40, -1.0],
                            [0.40, 0.90, 0.60, 0.75, 0.98]]
            for i in range(self.num_agents):
                doctor = DoctorAgent(i, self, belief_array[i])
                self.schedule.add(doctor)
            self.argumentation_text += "<h1>Starting simulation of the default case.</h1><br><br>The initial set of " \
                                       "arguments is the following:<br>"
            for arg_name, arg in self.LIST_OF_ARGUMENTS.items():
                self.argumentation_text += "<b>" + arg_name + "</b>" + ": " + arg + "<br>"

        else:
            atoms = [numpy.random.choice(numpy.arange(0, 1, 0.01)) for x in range(5)]
            possible_decisions = [0, 1, 2]

            # actual ground truth of the diagnosis
            # in this particular case, we assume it to be decision 2
            ground_truth = 2

            # create agents

            for agent in range(self.num_agents):
                # belief array is randomly generated for each each
                # represents likeliness of an agent to believe a given atom
                belief_array = [numpy.random.choice(numpy.arange(0, 1, 0.01)) for x in range(5)]
                doctor = DoctorAgent(agent, self, belief_array, possible_decisions, atoms, ground_truth)
                self.schedule.add(doctor)

        # Create dictionary where avg_belief will be tracked for each argument
        dict_model_collector = {}
        for i in range(self.n_initial_arguments):
            avg_belief = partial(calculate_avg_belief, i)
            dict_model_collector[ARGUMENT_NAMES[i]] = avg_belief
        # Collects data that will be collected in every step of the simulation
        self.datacollector = DataCollector(
            model_reporters=dict_model_collector,
            agent_reporters={"Belief Array": "belief_array"})

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """
            Advance the model by one step.
            Randomly initialize doctors and print out ensemble decision,
            based on initial belief vectors and atom probabilities
        """
        print("Beginning of argumentation round..")
        print("Doctor belief arrays before argumentation \n\n")
        self.argumentation_text += "Beginning of argumentation round..<br>Doctor belief arrays before argumentation<br>"
        for doctor in self.schedule.agents:
            text = "Doctor {}: {}".format(doctor._doctor_id, doctor.belief_array)
            print(text)
            self.argumentation_text += text + "<br>"

        self.schedule.step()
        # print("\n\n")
        # print("Doctor belief arrays after argumentation \n\n")
        # doctors = self.schedule.agents
        # for doctor in doctors:
        #     print(doctor.belief_array)

        self.datacollector.collect(self)
