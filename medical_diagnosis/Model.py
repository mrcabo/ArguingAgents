from functools import partial
import logging
import numpy

from mesa import Model
from mesa.time import RandomActivation, BaseScheduler
from mesa.datacollection import DataCollector

from medical_diagnosis.DoctorAgent import DoctorAgent, transform_convincing_value
from medical_diagnosis.initialisations import initialisations

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


def random_belief_array(lenght, mu=0.5, sigma=0.25):
    return numpy.random.normal(mu, sigma, lenght).tolist()


def random_influence(mu=0.5, sigma=0.25):
    return numpy.random.normal(mu, sigma)


def softmax(x):
    return numpy.exp(x) / sum(numpy.exp(x))


def get_belief_val(idx, agent):
    return agent.belief_array[idx]


def get_diagnosis_probabilities(idx, model):
    return model.diagnosis_probabilities[idx]


def get_final_decision(model):
    return model.final_decision


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

    def __init__(self, N=3, n_init_arg=5, experiment_case="default", sigma=0.25, arg_weight_vector=None):
        self.num_agents = N
        self.n_initial_arguments = n_init_arg  # Number of initial arguments that doctors will consider
        self.experiment_case = experiment_case
        self.diagnosis_text = ""
        self.diagnosis_probabilities = numpy.zeros(len(self.LIST_OF_DISEASES)).tolist()
        self.final_decision = None
        # How relevant is said argument to reach conclusion X or Y
        if arg_weight_vector is not None:
            self.arg_weight_vector = arg_weight_vector
        else:
            self.arg_weight_vector = {"Zika": numpy.zeros(self.n_initial_arguments, dtype=float),
                                      "Chikungunya": numpy.zeros(self.n_initial_arguments, dtype=float)}
        self.schedule = RandomActivation(self)  # Every tick, agents move in a different random order

        if self.experiment_case == "batch":  # Batch run case
            for i in range(self.num_agents):
                belief_array = random_belief_array(lenght=self.n_initial_arguments, sigma=sigma)
                doctor = DoctorAgent(i, self, belief_array)
                # Random influence values
                doctor.influence = random_influence(0.5, 0.25)
                doctor.stubbornness = random_influence(0.5, 0.25)
                self.schedule.add(doctor)

        else:
            if (self.num_agents != 3) or (self.n_initial_arguments != 5):
                print("Sorry, the default case only works with 3 doctors and 5 initial arguments")
                exit()
            # Hard coding the weight vectors for the default case, as we feel like they should be..
            self.arg_weight_vector["Zika"] = numpy.asarray([0.4, 0., 0.6, 0., 0.])
            self.arg_weight_vector["Chikungunya"] = numpy.asarray([0., 0.25, 0., 0.25, 0.5])

            # call intialisation with number of doctors, case number and number of arguments
            placeholder = initialisations(nb_doctors=self.num_agents, case=self.experiment_case, n_args=5)
            belief_array = placeholder[0]
            influence_stubborn_list = placeholder[1]

            # For testing different cases
            for i in range(self.num_agents):
                doctor = DoctorAgent(i, self, belief_array[i])
                doctor.influence, doctor.stubbornness = influence_stubborn_list[i]
                self.schedule.add(doctor)

            logger.info("Starting simulation for the default case. The initial set of arguments is the following:")

        self.calculate_committee()  # Calculates initial point for the committee decision.

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
        self.calculate_committee()
        logger.info("Probability for the diagnosis being {} is: {}".format(self.LIST_OF_DISEASES['X'],
                                                                           round(self.diagnosis_probabilities[0], 2)))
        logger.info("Probability for the diagnosis being {} is: {}".format(self.LIST_OF_DISEASES['Y'],
                                                                           round(self.diagnosis_probabilities[1], 2)))
        if self.diagnosis_probabilities[0] > self.diagnosis_probabilities[1]:
            disease = self.LIST_OF_DISEASES['X']
        else:
            disease = self.LIST_OF_DISEASES['Y']
        self.final_decision = disease
        self.diagnosis_text = "The diagnosis for the patient is: {}.".format(disease)
        logger.info(self.diagnosis_text)

        self.datacollector.collect(self)

    def calculate_committee(self):
        # Calculate the sum over all the belief arrays of all the doctors. Convincing values are used, so values <
        # 0.5 will have a negative value, being beliefs closer to 0 convincing values closer to -1
        committee_sum = numpy.zeros(self.n_initial_arguments)
        for doctor in self.schedule.agents:
            conv_val = transform_convincing_value(doctor.belief_array)
            committee_sum = numpy.add(committee_sum, conv_val)
        committee_sum = transform_convincing_value(committee_sum, inv=True)
        # Convert it to probabilities
        probabilities_committee = softmax(committee_sum)
        probability_zika = sum(numpy.multiply(self.arg_weight_vector["Zika"], probabilities_committee)) / sum(
            probabilities_committee)
        probability_chikv = sum(numpy.multiply(self.arg_weight_vector["Chikungunya"], probabilities_committee)) / sum(
            probabilities_committee)

        probabilities = softmax([probability_zika, probability_chikv])

        print(probabilities)
        self.diagnosis_probabilities[0] = probabilities[0]
        self.diagnosis_probabilities[1] = probabilities[1]

        logger.info("Probability for the diagnosis being {} is: {}".format(self.LIST_OF_DISEASES['X'],
                                                                           round(probabilities[0], 2)))
        logger.info("Probability for the diagnosis being {} is: {}".format(self.LIST_OF_DISEASES['Y'],
                                                                           round(probabilities[1], 2)))

        disease = self.LIST_OF_DISEASES['X'] if probability_zika > probability_chikv else self.LIST_OF_DISEASES['Y']
        self.final_decision = disease
        self.diagnosis_text = "The diagnosis for the patient is: {}.".format(disease)
        logger.info(self.diagnosis_text)
