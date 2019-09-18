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

     def __init__(self, unique_id, model, belief_array, possible_decisions, atoms):
         super().__init__(unique_id, model)

         self.belief_array = belief_array
         self.resultant_belief = dot(belief_array, atoms)
         self.decision_class = numpy.random.choice(possible_decisions)
         self.expertise_class = numpy.random.choice(possible_decisions)
         self.expertise = numpy.random.choice(numpy.arange(0, 1, 0.01))
         self.influence = numpy.random.choice(numpy.arange(0, 1, 0.01))
         print("Doctor agent initialized")

     def step(self):
        """
        A doctor agent set of rules for each step of argumentation

        Fetch the ensemble decision from a committe of doctors
        Compare with resultant belief in the decision
        Access majority decision and resultant belief of other agents in the committee
        If expertise higher for that particular decision class randomly update other agents belief array

        """
     
        print("inside the step function")
        ensemble_decision = dict()
        doctors = self.model.schedule.agents

        # loop through decision of all the doctors in the committee
        for doctor in doctors:
             if doctor.decision_class in ensemble_decision:
                  ensemble_decision[doctor.decision_class] += 1
             else:
                  ensemble_decision[doctor.decision_class] = 1

        # print ensemble decision
        print('Final decision from ensemble: ', ensemble_decision)
            



