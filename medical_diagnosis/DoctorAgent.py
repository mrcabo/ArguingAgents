from mesa import Agent
import numpy


def dot(x, y):
     """
         Dot product as sum of list comprehension doing element-wise multiplication

     """
     return(sum(x_i*y_i for x_i, y_i in zip(x, y)))

class DoctorAgent(Agent):
     """
        A doctor agent
        
        Activated at each time step
        
        Attributes

        * Likeliness to believe a given fact[for all symptoms](randomly drawn from a normal distribution)
        * Model with five symptom atoms(statements stating symptons)
        * Expertise in a given medical field(randomly drawn from a normal distribution)
        * Influential(pursuasiveness of beliefs) during an argumentation session(randomly drawn)
    
      """

     def __init__(self, unique_id, model, belief_array, possible_decisions, atoms, ground_truth):
         super().__init__(unique_id, model)

         self._doctor_id = unique_id
         self.belief_array = belief_array
         self.resultant_belief = dot(belief_array, atoms)
         self.decision_class = numpy.random.choice(possible_decisions)
         self.expertise_class = numpy.random.choice(possible_decisions)
         self.expertise = numpy.random.choice(numpy.arange(0, 1, 0.01))
         self.influence = numpy.random.choice(numpy.arange(0, 1, 0.01))
         self.ground_truth = ground_truth
         # print("Doctor agent initialized")

     def step(self):
        """
        A doctor agent set of rules for each step of argumentation

        Fetch the ensemble decision from a committe of doctors
        Compare with resultant belief in the decision
        Access majority decision and resultant belief of other agents in the committee
        If expertise higher for that particular decision class randomly update other agents belief array

        """
     
        # print("inside the step function")
        ensemble_decision_before = dict()
        doctors = self.model.schedule.agents
        


        for doctor in doctors:
             # print('Resultant Belief: ' + str(doctor.resultant_belief))
             if doctor.decision_class in ensemble_decision_before:
                  ensemble_decision_before[doctor.decision_class] += 1
             else:
                  ensemble_decision_before[doctor.decision_class] = 1

        # print ensemble decision before belief array update
        # print('Final decision from ensemble before belief array update: ', ensemble_decision_before)


        """
        # loop through decision of all the doctors in the committee
        # if expertise class of a doctor is same as ground_truth
        # in this case, this particular doctor can randomly update belief_array of
        # all the doctors whose expertise class is not ground_truth or whose expertise
        # is less than this doctor is ground_truth and expertise classes match
        """

        if self.expertise_class == self.ground_truth:
             for doctor in doctors:
                  if doctor.expertise_class != self.ground_truth:
                       doctor.belief_array = [numpy.random.choice(numpy.arange(0, 1, 0.01)) for x in range(5)]
                  if doctor.expertise_class == self.ground_truth and self.expertise > doctor.expertise:
                       doctor.belief_array = [numpy.random.choice(numpy.arange(0, 1, 0.01)) for x in range(5)]

        ensemble_decision_after = dict()
        
        for doctor in doctors:
             # print('Resultant Belief: ' + str(doctor.resultant_belief))
             if doctor.decision_class in ensemble_decision_after:
                  ensemble_decision_after[doctor.decision_class] += 1
             else:
                  ensemble_decision_after[doctor.decision_class] = 1

        # print ensemble decision after belief array update
        # print('Final decision from ensemble after belief array update: ', ensemble_decision_after)
            



