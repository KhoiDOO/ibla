# IBLA - Imbalance Learning Archive

# Setup
## Environment
Create a virtual environment and upgrade the version of ```pip```
```
python3 -m venv .env
source .env/bin/activate
python -m pip install -U pip
```
Install the required packages
```
cd /path/to/cloned/repo/directory
pip install -r requirements.txt
```
Install Pytorch packages, the version of the Nvidia driver is ```535``` and the CUDA version is ```12.1``` on a Ubuntu-based system
```
pip3 install torch torchvision torchaudio
```
## Dataset

# Contribute to this project
## Folder Structure 
- dataset
  - [ds].py where ```ds``` is any dataset name
  - getds.py used as a mapping from args to a dataset
- loss
  - [loss].py where ```loss``` is any loss name
- metrics
  - [metric].py where ```metric``` is any metric name
- model
  - [model_name]
    - *.py
- utils
- main.py
- mapping.py
- requirements.txt
- trainer.py

## Dataset Contribution
- Data sources and related annotations must be put into ```dataset/source``` folder, which is an ignored folder. A custom dataset class inheriting ```torch.utils.data.Dataset``` of Pytorch and read all data from ```dataset/source```.
- :warning::warning:: Do not change the content of any file inside ```dataset/source```, just create a new ```.py``` file to contribute your custom dataset. 

## Model/Loss Contribution
- Create a new folder/new file ```.py``` to store your model architecture/loss function and its sub-components
- :warning::warning:: Do not change the content of any file inside ```model/``` and ```loss/```

## Workflow
- For collaborators, create a new branch with the name as follows: ```<task>-<name>``` (i.e. model_deeplabv3) and then pull a request to merge to branch ```main```.
- For outer collaborators, fork this repo, and then pull a request also.
- For review the pull request, you can tag the owner of this repo, the owner will merge and complete the final task to emerge your contribution thoroughly.

# Citation
```
```
