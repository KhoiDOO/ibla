import os
import wandb
from glob import glob
import shutil
import random

if __name__ == "__main__":
    api = wandb.Api()

    runs = api.runs("scalemind/IBLA")

    for run in runs:
        run.tags.append("system_check")
        run.update()