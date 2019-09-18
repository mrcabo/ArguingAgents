

from Model import MedicalModel


model = MedicalModel(3)

step = 1

print('Initiation of round ' + str(step) + " of argumentation:\n\n" )
model.step()
print("\n\n")
print('Conclusion of round ' + str(step) + " of argumentation\n\n" )
