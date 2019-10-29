import argparse
import logging
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from medical_diagnosis.Model import MedicalModel
from medical_diagnosis.Model import get_diagnosis_probabilities, get_final_decision
from medical_diagnosis.server import ServerClass
from mesa.batchrunner import BatchRunner


def parse_arguments():
    parser = argparse.ArgumentParser(description='Simulates argumentation between several doctors')
    parser.add_argument('--n_doctors', type=int, default=3,
                        help='Number of doctors. For the batch run, it will run experiments from 1 doctors to '
                             'n_doctors')
    parser.add_argument('--n_init_arg', type=int, default=5,
                        help='Number of initial arguments.')
    parser.add_argument('--experiment_case', type=int, default=1,
                        help='Use the default case instead of random beliefs in arguments. (1-default case, '
                             '2-batch run)')
    parser.add_argument('--n_batch_iter', type=int, default=5,
                        help='Number of iterations in the batch run.')
    args = parser.parse_args()
    return args.n_doctors, args.n_init_arg, args.experiment_case, args.n_batch_iter


if __name__ == '__main__':
    # create logger with 'medical_diagnosis'
    logger = logging.getLogger('medical_diagnosis')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('output.log', mode='w')
    fh.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%I:%M:%S %p')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.debug("Initiating Simulation")

    arguments = parse_arguments()
    n_doctors, n_init_arg, experiment_case, n_batch_iter = arguments
    if experiment_case == 2:  # Batch run
        # Let's do that experiment_case is a batch run of the default case, so diseases are the same. Also,
        # ground truth remains Chikunguya.
        # Hard coding the weight vectors for the default case, as we feel like they should be..
        arg_weight_vector = {"Zika": np.asarray([0.4, 0., 0.6, 0., 0.]),
                             "Chikungunya": np.asarray([0., 0.25, 0., 0.25, 0.5])}
        fixed_params = {
            "n_init_arg": n_init_arg,
            "experiment_case": 2,
            "sigma": 0.25,
            "arg_weight_vector": arg_weight_vector
        }
        variable_params = {
            "N": range(1, n_doctors, 1)
        }

        # Create dictionary where the diagnosis probabilities will be tracked
        # dict_batch_collector = {}
        dict_batch_collector = {"Final_decision": get_final_decision}
        for i, disease in enumerate(MedicalModel.LIST_OF_DISEASES.values()):
            disease_prob = partial(get_diagnosis_probabilities, i)
            dict_batch_collector[disease] = disease_prob

        batch_run = BatchRunner(
            MedicalModel,
            variable_params,
            fixed_params,
            iterations=n_batch_iter,
            max_steps=50,
            model_reporters=dict_batch_collector
        )

        batch_run.run_all()

        run_data = batch_run.get_model_vars_dataframe()
        # All this dictionary approach is in case we want to show something else than just counting the correct
        # answers..

        # Create dict that contains the correct_diagnosis dictionaries keyed by the number of agents
        dict_correct_diagnosis = {}
        for i in range(1, n_doctors, 1):
            # Get the data frame relevant for N = i
            df = run_data[run_data.N == i]  # for i number of doctors
            # Create dict that contains the tuples of probabilities for the diseases where the diagnosis was correct,
            # keyed by the run number it was made.
            correct_diagnosis = {}
            for row in df.iterrows():
                if row[1].Final_decision == "Chikungunya":
                    correct_diagnosis[row[0]] = (row[1].Zika, row[1].Chikungunya)
            dict_correct_diagnosis[i] = correct_diagnosis

        n_total_correct_diagnosis = []
        for num_doct in dict_correct_diagnosis.keys():
            n_total_correct_diagnosis.append(len(dict_correct_diagnosis[num_doct]))

        ind = np.arange(len(n_total_correct_diagnosis))
        bar_chart = plt.bar(ind, n_total_correct_diagnosis)
        plt.xlabel('Number of doctors during argumentation')
        plt.ylabel('Number of correct diagnosis')
        plt.title('Number of correct diagnosis per {} cases'.format(n_batch_iter))
        plt.xticks(ind, tuple(np.arange(1, n_doctors, 1).astype(str)))
        plt.show()
    else:
        server = ServerClass(n_doctors, n_init_arg, experiment_case)
        server.server.launch()
