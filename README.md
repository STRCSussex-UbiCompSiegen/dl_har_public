# Increasing Replicability, Comparability and Collaboration in HAR Through a Common Code Base (PerCom 2022 WiP Submission)

This is the official GitHub page of the workshop paper publication "Increasing Replicability, Comparability and Collaboration in HAR Through a Common Code Base" presented at the IEEE International Confernce on Pervasive Computing and Communications (PerCom) Work in Progress (WiP) Session. [[cite our work]](#cite)

## Abstract

## Repository structure

The repository is structured into one main repository (this one) and three submodules (dataloader, model and analysis). To commence experiments, run the ```main.py``` file within this directory. 

The experiments require that you have preprocessed the datasets and have them locally stored on your working machine. To do so, check the ```dataloader``` submodule for further instructions.

### Weights and biases integration
In order to use weights and biases for logging results, one first needs to set up a free account on [their website](https://wandb.ai). Once registered run the following code within a python environment of your choice.

```
pip install wandb
wandb login
```

You can switch logging with weights and biases on and off by setting the ```--wandb``` flag when running the main script. Make sure that you define the ```wandb_project``` and ```wandb_entity``` variable according to your settings within weights and biases. How to create and entity and project can be looked up within the [documentation of weights and biases](https://docs.wandb.ai).

## Work in Progress

