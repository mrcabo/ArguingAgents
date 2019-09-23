from functools import partial
import numpy

from mesa import Model
from mesa.time import RandomActivation
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

    def __init__(self, N=3, n_init_arg=5):
        self.num_agents = N
        self.n_initial_arguments = n_init_arg  # Number of initial arguments that doctors will consider
        self.schedule = RandomActivation(self)

        # initialize atoms with respective probabilities
        # that is probability of each symptom being right

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
        dict_avg_belief_arr = {}
        for i in range(self.n_initial_arguments):
            avg_belief = partial(calculate_avg_belief, i)
            dict_avg_belief_arr[ARGUMENT_NAMES[i]] = avg_belief
        # Collects data that will be collected in every step of the simulation
        self.datacollector = DataCollector(
            model_reporters=dict_avg_belief_arr,
            agent_reporters={"Belief Array": "belief_array"})

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """
            Advance the model by one step.
            Randomly initialize doctors and print out ensemble decision,
            based on initial belief vectors and atom probabilities
        """
        doctors = self.schedule.agents
        print("Doctor belief arrays before argumentation \n\n")

        for doctor in doctors:
            print(doctor.belief_array)

        self.schedule.step()
        print("\n\n")
        print("Doctor belief arrays after argumentation \n\n")
        doctors = self.schedule.agents
        for doctor in doctors:
            print(doctor.belief_array)

        self.datacollector.collect(self)
