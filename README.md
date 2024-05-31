# Active Learning to Guide Labeling Efforts for Question Difficulty Estimation
This repository contains the code to reproduce the results of the paper _Active Learning to Guide Labeling Efforts for Question Difficulty Estimation_. 

[//]: # (The paper is available here as an arXiv preprint.)

## Workflow

Download the RACE ([here](https://huggingface.co/datasets/ehovy/race)) and RACE-c ([here](https://huggingface.co/datasets/tasksource/race-c)) dataset. Together, these datasets form the RACE++ dataset. Save the datasets in the `data/raw/` folder.

Prepare the datasets by running:
```
python data_preparation.py
python create_small_dev.py race_pp --size 1000
```

Create the environment from the `environment.yml` file and activate it:
```
conda env create -f environment.yml
conda activate qdet_active
```

The configuration files are located in the folder `src/config/race_pp/`. Run the experiments with:
```
python main.py race_pp
```

Finally, inspect the results by running the `analysis.ipynb` notebook.

[//]: # (## Cite as)

[//]: # (If you use this code in your workflow or scientific publication, please consider citing the corresponding paper:)
[//]: # (```)
[//]: # (TODO: add bibtex entry)
[//]: # (```)

