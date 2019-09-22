from mesa import Model
from mesa.time import RandomActivation
from DoctorAgent import DoctorAgent
import numpy


class MedicalModel(Model):
    """
        A model with a set of medical agents
        In principle arguing over a final decision
        Through multiple time steps, updating each other's belief array
    """

    def __init__(self, N=3):
        self.num_agents = N
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
