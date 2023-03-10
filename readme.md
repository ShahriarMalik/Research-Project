# Conflict-based Path Planning for Multiple Autonomous Agent
This is the official repository for the research project of **Shahriar Malik**.

## About the Research Project
| Title | Conflict-based Path Planning for Multiple Autonomous Agent
| --- | --- |
| Student | **Shahriar Malik**
| University | Hamburg University of Technology (TUHH)
| Insitute | Institute for Software Systems
| Supervised by | Prof. Dr. Sibylle Schupp, M.Sc Lars Beckers


## About this Python Project
### General
| What was used | How it was used
| --- | --- |
| Python  | version 3.8.2
| VS code | version 1.74.
| Windows | Windows 10 Pro version 21H1
| Packages| See _requirements.txt_

### Project folders
The folder _UPPAAL_ contains the UPPAAL version  4.1.24 downloaded from [uppaal.org](https://uppaal.org/downloads/).
It contains _uppaal.jar_, which will run the created UPPAAL Model automatically. It also contains an executable called _verifyta_,
which is used in order to generate simulatuin trace.

The folder _UPPAALPY_ contains the module uppaalpy, which is a UPPAAL wrapper for python by Deniz Koluaçık.
The used version 0.0.4 was committed on June 8th, 2021 downloaded from his [github repository](https://github.com/koluacik/uppaal-py).

The folder _MAPF_CBS_ contains the CBS library by Haoran Peng.
The used version 0.0.4 was committed on June 8th, 2021 downloaded from his [github repository](https://github.com/GavinPHR/Multi-Agent-Path-Finding).
**WARNING: The module has been adjusted in order to work properly and to make things easier**

The folder _xml-files_ contains all Uppaal Timed Automata given as .xml files used throughout the project.

The folder _RL_ contains all files related to implementation of Q-learning algorithm


The folder _MilesStone_path_ contains image of path generated by CBS library

The folder _Scenario_ contains YAML file which is the input of **main.py** 

### Program flow for TAMAA based method
1. Running **main.py** will run **modelGeneration.py** . In line number 230 of **main.py**, you may change the YAML file. It contains all the information regarding environment. YAML should be formatted as it is. All START coordiante and GOAL coordiante should be same. TASKWCET and TASKBECT should be same length. TASKSMILETSONE means where task can be performed. It should be from start coordiante. Robot number can be changed. The grid size is (1920, 1080). So, all values should be within the range of grid.Figure 3.3 of the report is encoded in Experiment1.yaml file. 
2. **modelGeneration.py** uses the library **uppaalpy** in order to create an UPPAAL system and later write it to CBS.xml_. It will start UPPAAL automatically
3. **main.py** uses the **CBS library** in order to calculate travelling time. It saves the travelling time and milestone in travellingTime.json file and image in Milestone_path folder. **main.py** also generate information.json file which contains information about the environment which will be used by **modelGeneration.py**.


### Program flow for Q-learning alogrithm
1. Running **makeQtable.ipynb** will make a q-table from the goodTraces.txt file.
2. Traces can be generated from sample.xml file using verifyta.exe command line verifyer.  **verifyta.exe sample.xml 1>goodTraces.txt** command should be used to generate .txt file. It is necessary to clean the first few lines of newly generated goodTraces.txt file. 
3. **makeQtable.ipynb** file will run **Environment_Up.py**  and **DQN.py** file. A q-table named qTable.csv will be generated on current directory.


