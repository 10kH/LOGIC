# LOGIC: LLM-Originated Guidance for Internal Cognitive Improvement of Small Language Models in Stance Detection

**Status**: Accepted  
**Type**: AI Application  
**Journal**: PeerJ Computer Science (SCIE, Q1)    

## Abstract

Stance detection is a critical task in natural language processing that determines an author's viewpoint toward a specific target, playing a pivotal role in social science research and various applications. Traditional approaches incorporating Wikipedia-sourced data into small language models (SLMs) to compensate for limited target knowledge often suffer from inconsistencies in article quality and length due to the diverse pool of Wikipedia contributors.

To address these limitations, we utilize large language models (LLMs) pretrained on expansive datasets to generate accurate and contextually relevant target knowledge. By providing concise, real-world insights tailored to the stance detection task, this approach surpasses the limitations of Wikipedia-based information.

Despite their superior reasoning capabilities, LLMs are computationally intensive and challenging to deploy on smaller devices. To mitigate these drawbacks, we introduce a reasoning distillation methodology that transfers the reasoning capabilities of LLMs to more compact SLMs, enhancing their efficiency while maintaining robust performance.

Our stance detection model, **LOGIC**, is built on BART and fine-tuned with auxiliary learning tasks, including reasoning distillation. By incorporating LLM-generated target knowledge into the inference process, LOGIC achieves state-of-the-art performance on the VAST dataset, outperforming advanced models like GPT-3.5 Turbo and GPT-4 Turbo in stance detection tasks.

---

## Project Structure

The essential files for our experiments are organized within a folder named `LOGIC`. This folder contains the following structure:

### Data

#### `data/raw_data/`:
This subfolder includes:
- The original VAST dataset from https://github.com/emilyallaway/zero-shot-stance. This dataset is divided into train, dev, and test splits. The file names are as follows: vast_dev.csv, vast_test.csv, vast_train.csv.

#### `data/`:
This folder includes:
- Additional datasets created by us for experimental purposes (LLM target knowledge, LLM reasoning).
- LLM target knowledge data files: new_topic_chatgpt.json, new_topic_chatgpt.pkl, topic_str_chatgpt.json, topic_str_chatgpt.pkl. The file actually used for the performance comparison is new_topic_chatgpt.pkl. The two types of files were used to find the better one by utilizing the columns in the original data CSV. The JSON files are the original files, while the PKL files are converted versions used for experiments to ensure compatibility with Wikipedia data. The contents of each JSON and PKL file are identical.
- Wikipedia target knowledge file: wiki_dict.pkl. This file is in PKL format because it follows the data format first presented in the repository of the paper "Infusing Knowledge from Wikipedia to Enhance Stance Detection" (https://github.com/zihaohe123/wiki-enhanced-stance-detection) and used in the paper "Zero-Shot and Few-Shot Stance Detection on Varied Topics via Conditional Generation" (https://github.com/wenhycs/ACL2023-Zero-Shot-and-Few-Shot-Stance-Detection-on-Varied-Topics-via-Conditional-Generation).
- VAST_reasoing_long_and_short.csv: This file contains LLM reasoning and is used for model training. Detailed information is provided in the paper.
- For ease of use, the raw_data files are also included in the `data` folder.

### Source Files

- `dataset.py`: Script for handling and processing datasets.
- `main.py`: Main script for running experiments on the VAST dataset.
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

For a more comprehensive theoretical background, detailed methodologies, and experimental settings, please refer to the body of our accompanying paper.
