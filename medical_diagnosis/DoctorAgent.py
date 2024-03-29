import logging

from mesa import Agent
import numpy

logger = logging.getLogger('medical_diagnosis')


def dot(x, y):
    """
        Dot product as sum of list comprehension doing element-wise multiplication
    """
    return (sum(x_i * y_i for x_i, y_i in zip(x, y)))


def transform_convincing_value(array, inv=False):
    """
    The mapping from the belief array to the convincing value is :math:`f(x) = 2x-1`, so that for 0.5 (uncertainty)
    the convincing value equals zero, and for 0 (complete disbelief) the convincing value equals minus one.

    Args:
        array: List of values to be transformed
        inv (bool): If true, do the inverse operation.

    Returns:
        The transformed array
    """

    if inv:
        return numpy.divide(numpy.add(array, 1), 2)
    else:
        return numpy.multiply(array, 2) - 1


def default_case_influencing(agent):
    """
    Simulates how an agent influence others. We define a convincing value, which reflects how far from 0.5 the belief
    value is. We map the beliefs values to the convincing values following :math:`f(x) = 2x-1`.

    For example, if we have the argument belief A, the convincing value A' is :math:`f(A)=A'` (e.g. if A=0.4 then
    A'=-0.2).

    Then, when appropriate, we update the other doctors beliefs with :math:`f(B'+\eta A')^{-1}`

    Args:
        agent (DoctorAgent): Agent that will influence the others

    Returns:
        None
    """
    logger.info("Doctor {} speaks now, trying to influence the other doctors".format(agent.unique_id))

    agent_conv_array = transform_convincing_value(agent.belief_array)  # A'
    for colleague in agent.model.schedule.agents:
        if colleague != agent:
            colleague_conv_array = transform_convincing_value(colleague.belief_array)  # B'
            signs_agent = numpy.sign(agent_conv_array)
            signs_vector = numpy.sign(agent_conv_array - colleague_conv_array)  # (A' - B')
            for arg_idx, _ in enumerate(agent.belief_array):  # Loop over every argument
                # Can't influence others with higher beliefs in that argument
                if signs_vector[arg_idx] == signs_agent[arg_idx]:
                    alpha = 0.25  # constant parameter to better simulate a real speed for convincing other people
                    eta = agent.influence * (1 - colleague.stubbornness) * alpha  # Regulates the influence
                    delta_belief = eta * agent_conv_array[arg_idx]
                    # An agent can only influence up to the same level of uncertainty that he has. So we limit it in
                    # case the update step becomes too big
                    if (signs_agent[arg_idx] * agent_conv_array[arg_idx]) > (
                            signs_agent[arg_idx] * (colleague_conv_array[arg_idx] + delta_belief)):
                        new_val = colleague_conv_array[arg_idx] + delta_belief
                    else:
                        new_val = agent_conv_array[arg_idx]

                    colleague_conv_array[arg_idx] = new_val
            # Convert B' back to B
            colleague.belief_array = transform_convincing_value(colleague_conv_array, inv=True).tolist()


class DoctorAgent(Agent):
    """
        The agent model class for the doctors.

        Attributes:
            _doctor_id: agent's unique ID
            belief_array (list): Represents how the doctor believes in each of the arguments presented. A 1
                represents absolute certainty while 0 represents complete disbelief. -1 means the agent doesn't have
                knowledge for that argument yet
            stubbornness (float): How difficult it is to change this doctor's mind. From 0 to 1, 1 being impossible
                to change his mind
            influence (float): How good is the agent at convincing people. From 0 to 1, 1 being the highest chances
                of convincing
     """

    def __init__(self, unique_id, model, belief_array, influence=0.5, stubbornness=0.5):
        super().__init__(unique_id, model)

        self._doctor_id = unique_id
        self.belief_array = belief_array
        self.influence = influence
        self.stubbornness = stubbornness

    def step(self):
        default_case_influencing(self)
