# A Public Repository to Improve Replicability, Comparability and Collaboration in Deep Learning for Human Activity Recognition (PerCom 2022 WiP Submission)

This is the official GitHub page of the workshop paper publication "Increasing Replicability, Comparability and Collaboration in HAR Through a Common Code Base" presented at the IEEE International Confernce on Pervasive Computing and Communications (PerCom) Work in Progress (WiP) Session. [[cite our work]](#cite)

## Abstract

## Repository structure

The repository is structured into one main ```public``` repository (this one) and three submodules (```dataloader```, ```model``` and ```analysis```). More information on each submodule can be found in the ```ReadMe``` of each submodule.

### Work in Progress
All files associated with the experiments mentioned in the _Work in Progress_ section of the publication can be found in the ```work_in_progress``` directory. This directory contains the job scripts, console log files, train and test results as well as an excel sheet containing the plots shown in the paper.

## Rerunning experiments

### Setup

### Preprocessing

In order to run any experiments the datasets need to be downloaded locally on your working machine. To do so, run the ```preprocessing.py``` python script within the ```dataloader``` submodule, specifiying the dataset you want to download by passing the according YAML file via the ```-d``` argument. More on this can be read up within the ```dataloader``` submodule. 

### Training & Predicition
To commence experiments, run the ```main.py``` file within this directory. The script requires to be passed only the ```-d``` (dataset) argument. Currently we support the dataset options: ```opportunity``` [[4]](#4), ```rwhar``` [[6]](#6), ```skoda``` [[8]](#8), ```pamap2``` [[9]](#9), ```hhar``` [[5]](#5) and ```shl``` [[7]](#7).

Using the ```-v``` argument one can define to either run ```loso``` (Leave-One-Subject-Out) or ```split``` (Train-Valid-Test) (cross-)validation. Note that you need to have previously run the corresponding preprocessing for the dataset. More on this can be read up within the ```dataloader``` submodule. 

### Analysis
The ```main.py``` script automatically performs an analysis on the obtained training, validation and (if applicable) testing results. If wished to rerun this analysis, one needs to run the ```analysis.py``` script passing it the ```-d``` (directory) of the log directory of the corresponding experiment. Note that this requires that you saved the predictions results using the ```--save_results``` flag of the ```main.py``` script.


### Weights and biases integration
In order to use weights and biases for logging results, one first needs to set up a free account on [their website](https://wandb.ai). Once registered run the following code within a python environment of your choice.

```
pip install wandb
wandb login
```

You can switch logging with weights and biases on and off by setting the ```--wandb``` flag when running the main script. Make sure that you define the ```wandb_project``` and ```wandb_entity``` variable according to your settings within weights and biases. How to create and entity and project can be looked up within the [documentation of weights and biases](https://docs.wandb.ai).

## Cite

Coming soon

## References
<a id="1">[1]</a> 
Francisco Javier Ordóñez and Daniel Roggen. 2016. 
Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition. 2016.  https://doi.org/10.3390/s16010115

<a id="2">[2]</a> 
Marius Bock, Alexander Hölzemann, Michael Moeller, and Kristof Van Laerhoven. 2021. Improving Deep Learning for HAR with Shallow LSTMs,” in
International Symposium on Wearable Computers. https://doi.org/10.1145/3460421.3480419

<a id="3">[3]</a> 
Alireza Abedin, Mahsa Ehsanpour, Qinfeng Shi, Hamid Rezatofighi, Damith C. Ranasinghe. 2021. Attend and Discriminate: Beyond the State-of-the-Art for
Human Activity Recognition Using Wearable Sensors. https://doi.org/10.1145/3448083

<a id="4">[4]</a> 
Daniel Roggen, Alberto Calatroni, Mirco Rossi, Thomas Holleczek, Kilian Förster,Gerhard Tröster, Paul Lukowicz, David Bannach, Gerald Pirkl, Alois Ferscha, Jakob Doppler, Clemens Holzmann, Marc Kurz, Gerald Holl, Ricardo Chavarriaga, Hesam Sagha, Hamidreza Bayati, Marco Creatura, and José del R. Millàn. 2010. Collecting Complex Activity Datasets in Highly Rich Networked Sensor Environments. https://doi.org/10.1109/INSS.2010.5573462

<a id="5">[5]</a> 
Allan Stisen, Henrik Blunck, Sourav Bhattacharya, Thor S. Prentow, Mikkel B.Kjærgaard, Anind Dey, Tobias Sonne, and Mads M. Jensen. 2015. Smart Devices are Different: Assessing and Mitigating Mobile Sensing Heterogeneities for Activity Recognition. https://doi.org/10.1145/2809695.2809718

<a id="6">[6]</a> 
Timo Sztyler and Heiner Stuckenschmidt. 2016. On-Body Localization of Wearable Devices: An Investigation of Position-Aware Activity Recognition. https://doi.org/10.1109/PERCOM.2016.7456521

<a id="7">[7]</a> 
Hristijan Gjoreski, Mathias Ciliberto, Lin Wang, Francisco Javier Ordóñez, Sami Mekki, Stefan Valentin, and Daniel Roggen. 2018. The University of Sussex-Huawei Locomotion and Transportation Dataset for Multimodal Analytics with Mobile Devices. https://doi.org/10.1109/ACCESS.2018.2858933

<a id="8">[8]</a> 
Piero Zappi, Thomas Stiefmeier, Elisabetta Farella, Daniel Roggen, Luca Benini, Gerhard Troster. 2007. Activity Recognition From On-Body Sensors by Classifier Fusion: Sensor Scalability and Robustness. https://doi.org/10.1109/ISSNIP.2007.4496857

<a id="9">[9]</a> 
Attila Reiss and Didier Stricker. 2012. Introducing a New Benchmarked Dataset for Activity Monitoring. https://doi.org/10.1109/ISWC.2012.13
