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

    def __init__(self, n):
        self.num_agents = n
        self.schedule = RandomActivation(self)

        # initialize atoms with respective probabilities
        # that is probability of each symptom being right

        atoms = [numpy.random.choice(numpy.arange(0, 1, 0.01)) for x in range(5)]
        possible_decisions = [0, 1, 2]
        
        # create agents

        for agent in range(self.num_agents):
            # belief array is randomly generated for each each
            # represents likeliness of an agent to believe a given atom
            belief_array = [numpy.random.choice(numpy.arange(0, 1, 0.01)) for x in range(5)]
            doctor = DoctorAgent(agent, self, belief_array, possible_decisions, atoms)
            self.schedule.add(doctor)

    def step(self):

        """
            Advance the model by one step.
            Randomly initialize doctors and print out ensemble decision,
            based on initial belief vectors and atom probabilities
        """

        self.schedule.step()
