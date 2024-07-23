# LOGIC: LLM-Originated Guidance for Internal Cognitive Improvement of Small Language Models in Stance Detection

## Project Structure

The essential files for our experiments are organized within a folder named `LOGIC`. This folder contains the following structure:


## Data

### `data/raw_data/`:
This subfolder includes:
- The original VAST dataset from Emily Allaway's zero-shot stance repository.

## Source Files

- `dataset.py`: Script for handling and processing datasets.
- `main.py`: Main script for running general experiments.
- `models.py`: Script containing model definitions and related functions.
- `run.sh`: Shell script for executing experiments on the VAST dataset.

## Reproducing the Experiments

To reproduce our experiments, follow these steps:

### Modify Parser Variables:

- Refer to the parser variables defined in `main.py`.
- Adjust these variables according to your experimental needs.

### Adjust Shell Scripts:

- Edit the variables in `run.sh` to match your setup.

### Execute Shell Scripts:

- Run the shell script to start the experiments. The execution results will be logged and stored in a new folder named `logs`, created during execution.

## Further Information

For a more comprehensive theoretical background and detailed methodologies, please refer to the body of our accompanying paper.
