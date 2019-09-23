import argparse

from medical_diagnosis.Model import MedicalModel
from medical_diagnosis.server import ServerClass


def parse_arguments():
    parser = argparse.ArgumentParser(description='Simulates argumentation between several doctors')
    parser.add_argument('--batch', type=bool, default=False,
                        help='Run a batch of examples to get statistics.')
    parser.add_argument('--n_doctors', type=int, default=3,
                        help='Number of doctors.')
    parser.add_argument('--n_init_arg', type=int, default=5,
                        help='Number of initial arguments.')
    args = parser.parse_args()
    return args.batch, args.n_doctors, args.n_init_arg


if __name__ == '__main__':
    arguments = parse_arguments()
    batch, n_doctors, n_init_arg = arguments
    if batch:
        model = MedicalModel(n_doctors)

        step = 1

        print('Initiation of round ' + str(step) + " of argumentation:\n\n")
        model.step()
        print("\n\n")
        print('Conclusion of round ' + str(step) + " of argumentation\n\n")
    else:
        server = ServerClass(n_init_arg)
        server.server.launch()
