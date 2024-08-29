# **Course project:** Visual World Paradigm
**Authors:** *Elif Dilara Aygün*, *Karl Jorge Cleres Andreo* & *Yanhong Xu*

**Course:** *Acquisition and analysis of eye-tracking data*

**Semester:** *Summer semester 2024*

## Project Description
> Investigating predictive eye movements for static and moving stimuli.

## Instruction for a new student
> To gain a comprehensive understanding of the project, we recommend you to review the presentation PDF, located at _~/presentation_, which provides an overview of the project's objectives and methodology. Afterwards we recommend you to read th report under _~/report_ which additionally presents the findings and final conclusions of this project.  
>
> If you want to run the OpenSesame experiment, please navigate into the _~/experiment_ folder and read the README.md file, which contains detailed instructions on how to proceed to run the experiment.

## Overview of Folder Structure 

```
│sub-team-cat (~)    <- Project's main folder.
│
├── report           <- Report HTML
│   ├── report.pdf
│
├── presentation     <- Final presentation slides (PDF and PPTX format).
│
├── _research        <- Literatue research, important papers, 
│   					WIP scripts, code, notes, comments,
│                       to-dos and anything in a preliminary state.
│
├── plots            <- All plots can be recreated using the plotting scripts in the scripts folder.
│   ├── global       <- Global plots (for all subjects).
│   ├── subject-i 	 <- Subject specific plots.
│
├── scripts          <- Preprocessing and analysis scripts as Jupyter Notebooks.
│
├── experiment       <- OpenSesame file to run the experiment with audio & visual stimuli. 
│
├── data             
│   ├── raw          <- Raw eye-tracking data.
│   ├── preprocessed <- Data resulting from preprocessing.
│
├── stimuli          <- All possible stimuli during the experiment, used for plotting.
│                       Stimuli table.
│
├── study_materials  <- Lab notes, consent and data privacy forms;   
│   ├── lab_notes    <- Obeservations during the experiments.
│
├── README.md        <- Top-level README for reproducability.
│
└── requirements.txt <- List of modules and packages that are used for this project.                     
```