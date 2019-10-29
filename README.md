# ArguingAgents

 In this study, we explore probabilistic argumentation where a degree of (un)certainty is associated with a given 
 argument.  This quantifies the strength of an argument in the discourse and how it eventually influences the final 
 conclusive decision.

## How to run it

This project was designed as a python package, so it could be uploaded to a python package repository in the future,
such as PyPi for example. All the import paths that are used in the code are declared with this in mind, which is 
why we need to indicate the python interpreter where to find our python package. That is done with the PYTHONPATH 
environment variable.

A simple example on how to run this code
```bash
#!/bin/bash
cd /home/<username>/Downloads
git clone https://github.com/mrcabo/ArguingAgents.git
cd ArguingAgents
python3 -m venv venv
source venv/bin/activate
export PYTHONPATH="$PYTHONPATH:/home/<username>/Downloads/ArguingAgents"
pip install -r requirements.txt
cd medical_diagnosis
python run.py
```

Creating a virtual environment is not necessary but recommended. If you don't have it installed, try:

```bash
sudo apt update && apt install -y python3-venv
```
## Frameworks

For this project we will use the Mesa framework. It is a modular framework for building, analyzing and visualizing agent-based models.

Its goal is to be the Python 3-based alternative to NetLogo, Repast, or MASON

[Mesa - github](https://github.com/projectmesa/mesa)
[Mesa - docs](https://mesa.readthedocs.io/en/master/overview.html)

## Modules

* ``run.py``: Launches the simulation. Run `python run.py --help` for a list of all the possible parameters.  
* ``Model.py``: Contains the overall model class. This _step()_ function is the one being called in every timestep of
 a simulation.
* ``DoctorAgent.py``: Contains the doctor class. It's where the doctors behaviour is defined.
* `initialisations.py`: Is where the initializations for the beliefs arrays are defined, based on different scenarios.

## Troubleshooting

If using vscode, follow this [instructions](https://code.visualstudio.com/docs/python/environments#_use-of-the-pythonpath-variable)

Your settings.json should look something like 

```json
{
    "python.pythonPath": "venv/bin/python3.7",
    "terminal.integrated.env.linux": {
        "PYTHONPATH": "${env:PYTHONPATH}:/home/diego/Documents/RUG/DesignMultiAgentSystems/repo"
    }
}
```



