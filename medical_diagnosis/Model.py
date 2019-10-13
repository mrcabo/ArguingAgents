from functools import partial
import logging

import numpy

from mesa import Model
from mesa.time import RandomActivation, BaseScheduler
from mesa.datacollection import DataCollector

from medical_diagnosis.DoctorAgent import DoctorAgent

ARGUMENT_NAMES = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J')
COLORS = ('#00FF00', '#FF0000', '#0000FF', '#383B38', '#FF00FF',
          '#8000FF', '#FF7F00', '#F6F90E', '#6E1122', '#3B541F')

logger = logging.getLogger('medical_diagnosis')


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


def random_belief_array():
    # TODO the range is fixed. number of arguments is a parameter that can be changed (will be useful for the batch
    #  runs)
    return [numpy.random.choice(numpy.arange(0.1, 1, 0.1)) for x in range(5)]

def random_influence():
    return numpy.random.choice(numpy.arange(0, 1, 0.01))

def softmax(x):
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum(axis=0)


def get_belief_val(idx, agent):
    return agent.belief_array[idx]


def get_diagnosis_probabilities(idx, model):
    return model.diagnosis_probabilities[idx]


def log_belief_arrays(model):
    for doctor in model.schedule.agents:
        text = "Doctor {}: {}".format(doctor._doctor_id, numpy.round(doctor.belief_array, 2))
        logger.info(text)


class MedicalModel(Model):
    """
        A model with a set of medical agents
        In principle arguing over a final decision
        Through multiple time steps, updating each other's belief array
    """
    # https://www.sciencedirect.com/science/article/pii/S0185106316301135
    LIST_OF_ARGUMENTS = {"A": "The patient has high fevers, which is one of the main symptoms of Zika",
                         "B": "The patient has high fevers, which is one of the main symptoms of Chikungunya",
                         "C": "The patient recently traveled to Brazil. To date, Brazil is the country with the "
                              "largest number of reported cases of Zika; this number is estimated to be between "
                              "500,000 and 1,500,000",
                         "D": "The patient presents acute joint pain, which is a common symptom of Chikungunya",
                         "E": "RT-PCR test results came positive. The sensitivity of this test for CHIKV (Chikungunya) "
                              "in the early stages of infection is 88.3%."}

    LIST_OF_DISEASES = {"X": "Zika",
                        "Y": "Chikungunya"}

    # TODO: this should be initialized inside the model. For batch they could be randomized with different weights,
    #  for default case they should be hard coded, also with different weights e.g. arg E bigger weight
    ZIKA_ARRAY = [1., 0., 1., 0., 0.]
    CHIKV_ARRAY = [0., 1., 0., 1., 1.]

    def __init__(self, N=3, n_init_arg=5, experiment_case=1):
        self.num_agents = N
        self.n_initial_arguments = n_init_arg  # Number of initial arguments that doctors will consider
        self.ground_truth = "Y"  # hardcoded for now..
        self.experiment_case = experiment_case
        self.argumentation_text = ""
        self.diagnosis_text = ""
        # TODO:maybe calculate the diagnosis_probabilities before sending the to the collector, os it doesn't start at 0
        self.diagnosis_probabilities = numpy.zeros(len(self.LIST_OF_DISEASES)).tolist()
      
        self.schedule = RandomActivation(self)  # Every tick, agents move in a different random order

        if self.experiment_case == 1: #default case
            if (self.num_agents != 3) or (self.n_initial_arguments != 5):
                print("Sorry, the default case only works with 3 doctors and 5 initial arguments")
                exit()
            ground_truth = "X"  # The ground truth for this particular diagnosis (real disease)
            # TODO: this should be fixed for the default case. For the batch is when it should be randomized
            belief_array = [[0.75, 0.30, 0.80, 0.50, 0.50],
                            [0.80, 0.50, 0.70, 0.40, 0.50],
                            [0.40, 0.90, 0.55, 0.75, 0.98]]

            for i in range(self.num_agents):
                doctor = DoctorAgent(i, self, belief_array[i])
                # TODO: again, hardcoded for default case, random for batch runs..
                doctor.influence = 0.5
                doctor.stubbornness = 0.5
                if i == 2:
                    doctor.influence = 0.7
                    doctor.stubbornness = 0.6
                self.schedule.add(doctor)

            logger.info("Starting simulation for the default case. The initial set of arguments is the following:")
            self.argumentation_text += "<h1>Starting simulation for the default case.</h1><br>The initial set of " \
                                       "arguments is the following:<br>"
            for arg_name, arg_idx in self.LIST_OF_ARGUMENTS.items():
                self.argumentation_text += "<b>" + arg_name + "</b>" + ": " + arg_idx + "<br>"

        elif self.experiment_case == 2: #Batch run case
            
            if (self.num_agents != 3) or (self.n_initial_arguments != 5):
                print("Sorry, the default case only works with 3 doctors and 5 initial arguments")
                exit()
            ground_truth = "X"  # The ground truth for this particular diagnosis (real disease)
            belief_array = [random_belief_array() for i in range(self.num_agents)]

            for i in range(self.num_agents):
                doctor = DoctorAgent(i, self, belief_array[i])
                # Random influence values
                doctor.influence = random_influence()
                doctor.stubbornness = random_influence()
                print(str(i)+" "+str(doctor.influence)+ " " + str(doctor.stubbornness))
                if i == 2:
                    doctor.influence = random_influence()
                    doctor.stubbornness = random_influence()
                self.schedule.add(doctor)



        # Create dictionary where avg_belief will be tracked for each argument
        dict_model_collector = {}
        for i in range(self.n_initial_arguments):
            avg_belief = partial(calculate_avg_belief, i)
            dict_model_collector[ARGUMENT_NAMES[i]] = avg_belief
        # Create dictionary where the diagnosis probabilities will be tracked
        for i, disease in enumerate(self.LIST_OF_DISEASES.values()):
            disease_prob = partial(get_diagnosis_probabilities, i)
            dict_model_collector[disease] = disease_prob
        # Create dictionary where the belief array of each agent will be tracked
        belief_array_collector = {}
        for arg_idx in range(self.n_initial_arguments):
            belief_value = partial(get_belief_val, arg_idx)
            belief_array_collector[ARGUMENT_NAMES[arg_idx]] = belief_value

        # Collects data that will be collected in every step of the simulation
        self.datacollector = DataCollector(
            model_reporters=dict_model_collector,
            agent_reporters=belief_array_collector)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """
            Advance the model by one step.
            Randomly initialize doctors and print out ensemble decision,
            based on initial belief vectors and atom probabilities
        """
        logger.info('-' * 40)
        logger.info("Beginning of a new argumentation round. Doctor belief arrays before argumentation are:")
        log_belief_arrays(self)

        self.schedule.step()
        logger.info("Doctor belief arrays after argumentation:")
        log_belief_arrays(self)

        # placeholder = [1. for x in range(5)]
        committee_summary = [0. for x in range(5)]
        for doctor in self.schedule.agents:
            # placeholder *= numpy.multiply(placeholder, doctor.belief_array)
            # taking in consideration both certainties and uncertainties
            # we consider values under 0.5 as a doctor's uncertainty in an arguments
            # hence it will negatively affect the committee belief array summary
            # ref: a more detailed discussion has been added in the report
            placeholder = [-x if x < 0.5 else x for x in doctor.belief_array]
            # print('placeholder: ', placeholder)
            committee_summary = [sum(x) for x in zip(committee_summary, placeholder)]

        probabilities = softmax(committee_summary)
        print('softmax: ', probabilities)
        # print('zika array: ', self.ZIKA_ARRAY)
        # print('chikv array: ', self.CHIKV_ARRAY)

        probability_zika = sum(numpy.multiply(self.ZIKA_ARRAY, probabilities)) / sum(probabilities)
        probability_chikv = sum(numpy.multiply(self.CHIKV_ARRAY, probabilities)) / sum(probabilities)

        # TODO: this is a bit hard coded. it should be done better so it accepts different number of conclusions
        self.diagnosis_probabilities[0] = probability_zika
        self.diagnosis_probabilities[1] = probability_chikv

        logger.info("Probability for the diagnosis being {} is: {}".format(self.LIST_OF_DISEASES['X'],
                                                                           round(probability_zika, 2)))
        logger.info("Probability for the diagnosis being {} is: {}".format(self.LIST_OF_DISEASES['Y'],
                                                                           round(probability_chikv, 2)))

        disease = self.LIST_OF_DISEASES['X'] if probability_zika > probability_chikv else self.LIST_OF_DISEASES['Y']
        self.diagnosis_text = "The diagnosis for the patient is: {}.".format(disease)
        logger.info(self.diagnosis_text)

        self.datacollector.collect(self)
