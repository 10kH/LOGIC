# LOGIC: LLM-Originated Guidance for Internal Cognitive Improvement of Small Language Models in Stance Detection

## Project Structure

The essential files for our experiments are organized within a folder named **LOGIC**. This folder contains the following structure:

LOGIC/  


├── data/    


│ └── raw_data/  


├── dataset.py  


├── main_sem16.py  


├── main.py  


├── models.py  


├── run_sem16.sh  
  

└── run.sh  


### Data

- **data/raw_data/**: This subfolder includes:
  - The original VAST dataset from [Emily Allaway's zero-shot stance repository](https://github.com/emilyallaway/zero-shot-stance).
  - The original SemEval2016 Task 6 dataset from [SemEval 2016](https://alt.qcri.org/semeval2016/).
  - Additional datasets created by us for experimental purposes.

### Source Files

- **dataset.py**: Script for handling and processing datasets.
- **main_sem16.py**: Main script for running experiments on the SemEval2016 dataset.
- **main.py**: Main script for running general experiments.
- **models.py**: Script containing model definitions and related functions.
- **run_sem16.sh**: Shell script for executing experiments on the SemEval2016 dataset.
- **run.sh**: Shell script for executing experiments on the VAST dataset.

## Reproducing the Experiments

To reproduce our experiments, follow these steps:

1. **Modify Parser Variables**:
   - Refer to the parser variables defined in **main.py**.
   - Adjust these variables according to your experimental needs.

2. **Adjust Shell Scripts**:
   - Edit the variables in **run.sh** and **run_sem16.sh** to match your setup.

3. **Execute Shell Scripts**:
   - Run the shell scripts to start the experiments. The execution results will be logged and stored in a new folder named **logs**, created during execution.

## Further Information

For a more comprehensive theoretical background and detailed methodologies, please refer to the body of our accompanying paper.

