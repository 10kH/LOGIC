# LOGIC: LLM-Originated Guidance for Internal Cognitive Improvement of Small Language Models in Stance Detection

**Status**: Accepted  
**Type**: AI Application  
**Journal**: PeerJ Computer Science (SCIE, Q1)    

## Authors

- **Woojin Lee\***  
  Department of Artificial Intelligence, Konkuk University, Seoul, Republic of Korea  

- **Jaewook Lee\***  
  Department of Artificial Intelligence, Konkuk University, Seoul, Republic of Korea  

- **Harksoo Kim†**  
  Department of Computer Science and Engineering, Konkuk University, Seoul, Republic of Korea  

\* Co-first authors  
† Corresponding author  

---

## Abstract

Stance detection is a critical task in natural language processing that determines an author's viewpoint toward a specific target, playing a pivotal role in social science research and a wide range of applications. Traditional approaches that rely on Wikipedia-sourced data to compensate for limited target knowledge in small language models (SLMs) often face inconsistencies in article quality and length due to the diverse pool of Wikipedia contributors.

To address these limitations, we leverage large language models (LLMs) pretrained on expansive datasets to generate accurate and contextually relevant target knowledge. By providing concise, real-world insights tailored to the stance detection task, our approach surpasses the constraints of Wikipedia-based information.

Despite their superior reasoning capabilities, LLMs are computationally intensive and challenging to deploy on smaller devices. To mitigate these drawbacks, we introduce a reasoning distillation methodology that transfers the reasoning capabilities of LLMs to more compact SLMs, improving efficiency while maintaining robust performance.

Our stance detection model, **LOGIC**, is built on BART and fine-tuned with auxiliary learning tasks, including reasoning distillation. By incorporating LLM-generated target knowledge into the inference process, LOGIC achieves state-of-the-art performance on the VAST dataset, outperforming advanced models such as GPT-3.5 Turbo and GPT-4 Turbo in stance detection tasks.

---

## Project Structure

All essential files for our experiments are organized under a folder named `LOGIC`, which is structured as follows:

### Data

#### `data/raw_data/`
- This subfolder contains the original VAST dataset from [this repository](https://github.com/emilyallaway/zero-shot-stance), divided into train, dev, and test splits:
  - `vast_dev.csv`
  - `vast_test.csv`
  - `vast_train.csv`

#### `data/`
- **Additional Datasets**: Includes custom datasets created for experimental purposes (e.g., LLM target knowledge, LLM reasoning).
- **LLM Target Knowledge Files**: 
  - `new_topic_chatgpt.json`
  - `new_topic_chatgpt.pkl`
  - `topic_str_chatgpt.json`
  - `topic_str_chatgpt.pkl`
  
  Among these, `new_topic_chatgpt.pkl` was primarily used for performance comparisons. We generated both JSON and PKL versions to test various data integration approaches. The JSON files are the original outputs, while the PKL files are converted versions to ensure compatibility with Wikipedia-based data. Both formats contain identical content.
- **Wikipedia Target Knowledge File**: 
  - `wiki_dict.pkl`  
    This PKL file follows the data format from the repository of the paper *"Infusing Knowledge from Wikipedia to Enhance Stance Detection"* ([GitHub link](https://github.com/zihaohe123/wiki-enhanced-stance-detection)) and was also used in the paper *"Zero-Shot and Few-Shot Stance Detection on Varied Topics via Conditional Generation"* ([GitHub link](https://github.com/wenhycs/ACL2023-Zero-Shot-and-Few-Shot-Stance-Detection-on-Varied-Topics-via-Conditional-Generation)).
- **LLM Reasoning File**: 
  - `VAST_reasoing_long_and_short.csv`  
    Contains LLM-generated reasoning for training. See the paper for details.
- For convenience, the `raw_data` files also reside in the `data` folder.

### Source Files

- **`dataset.py`**: Handles and processes the datasets.  
- **`main.py`**: Main script for running experiments on the VAST dataset.  
- **`models.py`**: Contains model definitions and related functions.  
- **`run.sh`**: Shell script for executing experiments on the VAST dataset.

---

## Reproducing the Experiments

1. **Modify Parser Variables**  
   - Refer to the parser variables in `main.py` and adjust them as needed for your experiments.

2. **Adjust Shell Scripts**  
   - Update the variable settings in `run.sh` according to your environment.

3. **Execute Shell Scripts**  
   - Run the shell script. The execution results will be automatically logged and saved in a newly created `logs` folder.

---

## Further Information

For more details on theoretical background, methodology, and experimental settings, please refer to our paper. Additional clarifications or updates may be provided in the project repository as needed.
