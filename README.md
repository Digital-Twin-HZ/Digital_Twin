# Digital_Twin
Create a digital twin which dynamically models the processes within the Waste Water Treatment Plant, with the purpose of being able to run the WWTP smoothly and more efficiently, but more importantly anticipate unforeseen circumstances.

## Dev Info
* any file placed inside a folder named 'data' at root will never be pushed
* a python virtual environment is advised to be created as: ``python -m venv venv``, since folder 'venv' is gitignored
* any external library (for instance PIP) should be added to the requirements.txt

## Dev Setup
1. ``python -m venv venv``
2. Activating venv
* On Unix or MacOS, using the bash shell: source /path/to/venv/bin/activate
* On Unix or MacOS, using the csh shell: source /path/to/venv/bin/activate.csh
* On Unix or MacOS, using the fish shell: source /path/to/venv/bin/activate.fish
* On Windows using the Command Prompt: path\to\venv\Scripts\activate.bat
* On Windows using PowerShell: path\to\venv\Scripts\Activate.ps1
3. ``python -m pip install -r requirements.txt``