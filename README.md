# ArguingAgents

Brief description <- **Insert here :)**

## How to run it

```bash
#!/bin/bash
cd <path_to_base_dir>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PYTHONPATH:<path_to_base_dir>"
python run.py
```

where <path_to_base_dir> is the path to the directory where the repository was downloaded.

## Frameworks

For this project we will use the Mesa framework. It is a modular framework for building, analyzing and visualizing agent-based models.

Its goal is to be the Python 3-based alternative to NetLogo, Repast, or MASON

[Mesa - github](https://github.com/projectmesa/mesa)
[Mesa - docs](https://mesa.readthedocs.io/en/master/overview.html)

## Useful Links and Papers

[Modeling Influence In Group Decision Making](http://www.cse.dmu.ac.uk/~chiclana/publications/SOCO-DEC-2015.pdf)
[A probabilistic approach to modelling uncertain logical arguments](https://www.sciencedirect.com/science/article/pii/S0888613X12001442)

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



