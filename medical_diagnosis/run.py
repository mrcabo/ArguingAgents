import argparse

from medical_diagnosis.Model import MedicalModel
from medical_diagnosis.server import server


def parse_arguments():
    parser = argparse.ArgumentParser(description='Simulates argumentation between several doctors')
    parser.add_argument('--batch', type=bool, default=False,
                        help='Run a batch of examples to get statistics.')
    parser.add_argument('--n_doctors', type=int, default=3,
                        help='Number of doctors.')
    args = parser.parse_args()
    return args.batch, args.n_doctors


if __name__ == '__main__':
    arguments = parse_arguments()
    batch, n_doctors = arguments
    if batch:
        model = MedicalModel(n_doctors)

        step = 1

        print('Initiation of round ' + str(step) + " of argumentation:\n\n")
        model.step()
        print("\n\n")
        print('Conclusion of round ' + str(step) + " of argumentation\n\n")
    else:
        server.launch()
