import argparse
import logging

from medical_diagnosis.Model import MedicalModel
from medical_diagnosis.server import ServerClass
from mesa.batchrunner import BatchRunner

def parse_arguments():
    parser = argparse.ArgumentParser(description='Simulates argumentation between several doctors')
    parser.add_argument('--batch', type=bool, default=False,
                        help='Run a batch of examples to get statistics.')
    parser.add_argument('--n_doctors', type=int, default=3,
                        help='Number of doctors.')
    parser.add_argument('--n_init_arg', type=int, default=5,
                        help='Number of initial arguments.')
    parser.add_argument('--experiment_case', type=int, default=1,
                        help='Use the default case instead of random beliefs in arguments.')
    args = parser.parse_args()
    return args.batch, args.n_doctors, args.n_init_arg, args.experiment_case


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
    n_doctors, n_init_arg, experiment_case = arguments
    if experiment_case == 2: #Batch run
        fixed_params = {
            "n_init_arg": n_init_arg
            "N": n_doctors,
            "experiment_case": 2
        }
        batch_run = BatchRunner(
                MedicalModel,
                variable_params,
                fixed_params,
                iterations=5,
                max_steps=100,
                model_reporters={"Gini": compute_gini}
            )

            batch_run.run_all()
    else:
        server = ServerClass(n_doctors, n_init_arg, experiment_case)
        server.server.launch()
