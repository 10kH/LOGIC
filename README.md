We have compressed the files necessary for our experiments into a folder named LOGIC, our project name. Inside this folder, there are subfolders named data and source files dataset.py, main_sem16.py, main.py, models.py, run_sem16.sh, and run.sh.

    
Within the data folder, there is a subfolder named raw_data which contains the original VAST dataset from https://github.com/emilyallaway/zero-shot-stance and the original sem16t6 dataset from https://alt.qcri.org/semeval2016/. Other datas are made by us for experiments.


To reproduce the experiments, you can refer to the parser variables in main.py and modify the variables in the run.sh and run_sem16t6 shell scripts accordingly. The execution results will be recorded in the logs folder which will be made by the execution of shell files. 


More theoretical details are fully described in the body of the paper.
