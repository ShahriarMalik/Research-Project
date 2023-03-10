from turtle import pos
from numpy import source
from soupsieve import select

from UPPAALPY.classes.nodes import BranchPoint, Location, Node
from UPPAALPY.classes.nta import NTA
from UPPAALPY.classes.simplethings import (
    ConstraintLabel,
    Declaration,
    Label,
    Name,
    Parameter,
    Query,
    SystemDeclaration,
    UpdateLabel,
)
from UPPAALPY.classes.tagraph import TAGraph
from UPPAALPY.classes.templates import Template
from UPPAALPY.classes.transitions import Nail, Transition
from UPPAALPY.path_analysis import *

import copy
import sys
import lxml.etree as ET
import subprocess
import os
import json
import yaml
import ast



# ###################### input from YAML file given by user #########################
# Load travelling time and number of milestones from JOSN file generated by path planning algorithm (CBS.py)
with open('travellingTime.json') as f:
    d = json.load(f)
# milestoneArray ## 
listOfMilestone = []
for x in d.keys():
      listOfMilestone += list(ast.literal_eval(x)) 
listOfMilestone = list(dict.fromkeys(listOfMilestone))
# Get Maximum maximum travlling time 
maximumTime = []

totalTimeForTaskCoverage = []

for x in d.keys():
    maximumTime.append(list(d[x].values())[-1])
    totalTimeForTaskCoverage.append(list(d[x].values())[-1])
maximumTime = max(maximumTime)


# Load task and agent data from JSON file generated by path planning algorithm (CBS.py)
with open('information.json') as f:
    taskData = json.load(f)
# All tasks
tasks = {}

# Best case execution time of task given by user
TASKBCET = {} 

# Worst case execution time of task given by user
TASKWCET = {}   
# A dictioray containing which task needs to be performed at particular location
# Some task can be performed at multiple location
TASKSMILETSONE = {}   
for x in taskData.keys():
   if x == 'TASKBCET':
       TASKBCET.update(taskData[x])
   elif x == 'TASKWCET':
       TASKWCET.update(taskData[x])
       tasks.update(taskData[x])
   elif x == 'TASKSMILETSONE':
       TASKSMILETSONE.update(taskData[x])
     
   elif x == 'ROBOT_NUMBER':
        AgentNum = taskData[x]
        AgentNum = 1
# Calculate total task coverage time for TCTL query
for i in tasks:
    totalTimeForTaskCoverage.append(TASKWCET[i])
    
      
def findTaskPosition(TaskID):
    """If agent is in particular location, then flag needs to be raised so that agent make a transition to location where it can perform tasks 


    Args:
        TaskID (string): Task ID (1, 2 , 3 ....) as a string

    Returns:
        string: Returns guard for task automaton
    """
    counter = 0
    position = ""
    for i in range(len(listOfMilestone)):
        for cooridante, task in TASKSMILETSONE.items():
            
            if str(listOfMilestone[i]) == cooridante:
                if str(TaskID) == task[-1]:
                    counter += 1
                    if counter == 1:
                        position = "position[id]["+str(i)+"] == true"
                    elif counter == 2:
                        position = position + "&& position[id]["+str(i)+"] == true"
                    elif counter == 3:
                        position = position + "&& position[id]["+str(i)+"] == true"
                    elif counter == 4:
                        position = position + "&& position[id]["+str(i)+"] == true"
                    elif counter == 5:
                        position = position + "&& position[id]["+str(i)+"] == true"
                    elif counter == 6:
                        position = position + "&& position[id]["+str(i)+"] == true"
    return position


def rulesOfTaskExecution(TaskID):
    """In this research project, we consider task 
        T1 must be completed before task T2 and task T2 must be completed before task T3. If there are more tasks then it will be like this


    Args:
        TaskID (string): Task ID (1, 2 , 3 ....) as a string

    Returns:
        string: Task needs to be done one after another. This function returns guard value for task automaton.
    """
    taskStatus = ""
    if TaskID == "2":
      taskStatus = "!tf[id][1]&&"
    elif TaskID == "3":
      taskStatus = "!tf[id][1] && !tf[id][2]&&"    
    elif TaskID == "4":
        taskStatus = "!tf[id][1] && !tf[id][2]&& !tf[id][3]"
    elif TaskID == "5":
        taskStatus = "!tf[id][1] && !tf[id][2]&& !tf[id][3]&& !tf[id][4]"
    return taskStatus  
            
                          
#Lenght of all locations
mileStoneLength = len(listOfMilestone)

def globalDiclatration(numberMilestone, numAgent, numTask):
    """Generate global declaration 

    Args:
        numberMilestone (int): number of milestone
        numAgent (int): number of agent
        numTask (int): length of task dictionray

    Returns:
        string: return a string for gloabal declaration
    """
    # Main task from YAML file and task T0 to control the movement of Agent
    numTask = numTask + 1
    declarationBody = ""
    declarationBody += "// Place global declarations here.\n"
    declarationBody += "clock t;\n "
    declarationBody += "// Toatal Agent Number.\n"
    declarationBody += "const int AgentNum =" + str(numAgent) + ";\n"
    declarationBody += "typedef int[0,AgentNum-1] AgentScale;\n"
    declarationBody += "// Total milestone Number.\n"
    declarationBody += "const int MilestoneNum ="+ str(numberMilestone) +";\n" 
    declarationBody += "// Main task from YAML file and task T0 to control the movement of Agent \n"
    declarationBody += "const int TaskNum ="+ str(numTask) +";\n" 
    arrayAgentNumMilestoneNum = ""
    for i in range(numAgent):
        if i == range(numAgent)[0]:
            arrayAgentNumMilestoneNum += "{"
        for j in range(numberMilestone):
            if j == range(numberMilestone)[0]:
                arrayAgentNumMilestoneNum += "{"
            arrayAgentNumMilestoneNum += "false"
            if j != range(numberMilestone)[-1]:
                arrayAgentNumMilestoneNum += ","
            if j == range(numberMilestone)[-1]:
                arrayAgentNumMilestoneNum += "}"
        if i != range(numAgent)[-1]:
            arrayAgentNumMilestoneNum += ","
        if i == range(numAgent)[-1]:
            arrayAgentNumMilestoneNum += "}"
            
    declarationBody += "// Position Array.\n"
    declarationBody += "bool position[AgentNum][MilestoneNum]= " + arrayAgentNumMilestoneNum + ";\n"
    
    tfAgentNumTaskNum = ""
    
    for i in range(numAgent):
        if i == range(numAgent)[0]:
            tfAgentNumTaskNum += "{"
        for j in range(numTask):
            if j == range(numTask)[0]:
                tfAgentNumTaskNum += "{"
            if j == range(numTask)[0]:
                tfAgentNumTaskNum += "true"
            else:
                tfAgentNumTaskNum += "false"
            if j != range(numTask)[-1]:
                tfAgentNumTaskNum += ","
            if j == range(numTask)[-1]:
                tfAgentNumTaskNum += "}"
        if i != range(numAgent)[-1]:
            tfAgentNumTaskNum += ","
        if i == range(numAgent)[-1]:
            tfAgentNumTaskNum += "}"
    declarationBody += "// Task finish array.\n"
    declarationBody += "bool tf[AgentNum][TaskNum]= " + tfAgentNumTaskNum + ";\n"
    
    tsAgentNumTaskNum = ""
    
    for i in range(numAgent):
        if i == range(numAgent)[0]:
            tsAgentNumTaskNum += "{"
        for j in range(numTask):
            if j == range(numTask)[0]:
                tsAgentNumTaskNum += "{"
            if j == range(numTask)[0]:
                tsAgentNumTaskNum += "true"
            else:
                tsAgentNumTaskNum += "false"
            if j != range(numTask)[-1]:
                tsAgentNumTaskNum += ","
            if j == range(numTask)[-1]:
                tsAgentNumTaskNum += "}"
        if i != range(numAgent)[-1]:
            tsAgentNumTaskNum += ","
        if i == range(numAgent)[-1]:
            tsAgentNumTaskNum += "}"
    declarationBody += "// Task start array.\n"
    declarationBody += "bool ts[AgentNum][TaskNum]= " + tsAgentNumTaskNum + ";\n"
    declarationBody += "chan move[AgentNum]; \n"
    declarationBody += "int stone = 100; \n"
    declarationBody += "const int load = 10; \n"
    
    return declarationBody;

    
# os.system('cls')
# print(globalDiclatration(3,2,3))

# Load the blank system from UPPAAL 
sys = NTA.from_xml(path='xml_files/blank_system.xml')
sys.templates[0].graph.remove_node(('Template', 'id0'))
blank_template = copy.deepcopy(sys.templates[0])
sys.templates.pop()

# System declaration
sys.declaration = Declaration(globalDiclatration(mileStoneLength,AgentNum, len(tasks.keys())))

# ##### movement TA
movement = copy.deepcopy(blank_template)
movement.name = Name(name="Movement", pos=[16, -8])
movement.parameter = Parameter("const AgentScale id")
movement.declaration = Declaration("clock t;\r\n")


createdMainLocation = []
createdAux1Location = []
createdAux2Location = []
P =""
F =""
T =""
xx = -400
yy = -200
         

################### Creation of Movement Timed Automata ################################
# add location
loc_id0 = Location(id="initial", pos=(216, 176),name=Name("initial", (216, 192)) )
# loc_id0.committed = True
movement.graph.add_location(loc_id0)
movement.graph.initial_location="initial"

for i in range(len(listOfMilestone)):
    main_location = "P"+str(i)
    main_location = Location(id="P"+str(i), pos=(xx + i * 150, yy + i - 100), name= Name( "P"+str(i), (xx + i * 150, yy + i - 100)))
    # main_location = Location(id="P"+str(i),  pos=(236, 136), name= Name( "P"+str(i),(216, 176)))
    if main_location not in createdMainLocation:
        createdMainLocation.append(main_location)
        movement.graph.add_location(main_location)
        main_location =""
    for j in range(len(listOfMilestone)):
        if i !=j:
            allPairOfCoordinates = "("+str(listOfMilestone[i])+", "+ str(listOfMilestone[j]) + ")"
            for x in d.keys():
                if allPairOfCoordinates == x:
                    if  list(d[x].values())[-1] > 1:
                        if  list(d[x].values())[-1] < maximumTime:
                            if str(list(d[x].values())[-1]) != "inf":
                                # print(list(d[x].values())[-1])
                                # print( "inside: ", i, j)
                            
                                # Auxiliary location 1
                                auxililiary_location_1= "F"+str(i)+"T"+str(j)
                                if auxililiary_location_1 not in createdAux1Location:
                                    # print("auxililiary_location_1: ",auxililiary_location_1)
                                    createdAux1Location.append(createdAux1Location)
                                    auxililiary_location_1 = Location(id="F"+str(i)+"T"+str(j), pos=(xx + 100 + j * 150, yy + 100 + i * 150), name= Name("F" + str(i) + "T"+str(j),(xx + 100 + j * 150, yy + 100 + i * 150)))
                                    
                                    auxililiary_location_1.invariant = Label(kind="invariant", value="t<=" + str(list(d[x].values())[-1]), pos=(240, 32),)
                                    movement.graph.add_location(auxililiary_location_1)
                                
                                # Auxiliary location 2
                                auxililiary_location_2 = "F"+str(j)+"T"+str(i)
                                if auxililiary_location_2 not in createdAux2Location:
                                    # print("auxililiary_location_2: ",auxililiary_location_2)
                                    createdAux2Location.append(createdAux2Location)
                                    auxililiary_location_2 = Location(id="F"+str(j)+"T"+str(i), pos=(xx + 120 + j * 170, yy + 100 + i * 110), name= Name("F" + str(j) + "T"+str(i),(xx + 120 + j * 170, yy + 100 + i * 110)))
                                
                                    auxililiary_location_2.invariant = Label(kind="invariant", value="t<=" + str(list(d[x].values())[-1]), pos=(240, 32),)
                                    movement.graph.add_location(auxililiary_location_2)
                            
# add transitions
transition_p0 = Transition(source = "initial",target ="P" + str(0))
transition_p0.assignment = Label(kind="assignment", value="position[id][0]=true", pos=(160, 24),)
movement.graph.add_transition(transition_p0)
transition_p1_list =[]

# It is direct translation of CreateConnection function of movement TA algorithm
for i in range(len(listOfMilestone)):
    for j in range(len(listOfMilestone)):
        if i !=j:
            allPairOfCoordinates = "("+str(listOfMilestone[i])+", "+ str(listOfMilestone[j]) + ")"
            for x in d.keys():
                if allPairOfCoordinates == x:
                    if  list(d[x].values())[-1] > 1:
                        if  list(d[x].values())[-1] < maximumTime:
                            if str(list(d[x].values())[-1]) != "inf":
                                transition_p1 = Transition(source = "P" + str(i), target ="F" + str(i) + "T" + str(j))
                
                                transition_p1.assignment = Label(kind="assignment", value="t=0,position[id][" + str(i) + "]=false", pos=(160, 24),)
                                transition_p1.synchronisation = Label(kind="synchronisation", value="move[id]?", pos=(160, 24),)
                                movement.graph.add_transition(transition_p1)
                                # print(type(transition_p1))
                                # print("transition_p1: ", transition_p1)
                                transition_p1 = ""
                                
                                transition_p2 = Transition(source ="F" + str(i) + "T" + str(j) , target ="P" + str(j))
                                transition_p2.assignment = Label(kind="assignment", value="t=0,position[id][" + str(j) + "]=true", pos=(190, 44),)
                                
                                transition_p2.guard = Label(kind="guard", value="t>=" + str(list(d[x].values())[-1]), pos=(160, 24),)
                                movement.graph.add_transition(transition_p2)
                                
                                transition_p1 = Transition(source = "P" + str(j), target ="F" + str(j) + "T" + str(i))
                
                                transition_p1.assignment = Label(kind="assignment", value="t=0,position[id][" + str(j) + "]=false", pos=(160, 24),)
                                transition_p1.synchronisation = Label(kind="synchronisation", value="move[id]?", pos=(160, 24),)
                                movement.graph.add_transition(transition_p1)
                                # print(type(transition_p1))
                                # print("transition_p1: ", transition_p1)
                                transition_p1 = ""
                                
                                transition_p2 = Transition(source ="F" + str(j) + "T" + str(i) , target ="P" + str(i))
                                transition_p2.assignment = Label(kind="assignment", value="t=0,position[id][" + str(i) + "]=true", pos=(190, 44),)
                                
                                transition_p2.guard = Label(kind="guard", value="t>=" + str(list(d[x].values())[-1]), pos=(160, 24),)
                                movement.graph.add_transition(transition_p2)
                                
        


# #####################******* Task Excution TA    *******###############################

taskExecution = copy.deepcopy(blank_template)
taskExecution.name = Name(name="taskExecution", pos=[16, -8])
taskExecution.parameter = Parameter("const AgentScale id")
    
    
isBusyFunction = """  \n clock t;
 \n \n                                         
bool isBusy(int taskID)
{
    bool busy = false;
    int other_id = 0, other_position = -1, position_id = 0;

    for(other_id = 0; other_id < AgentNum; other_id++)
    {
        if(other_id != id && ts[other_id][taskID])
        {
            for(position_id = 0; position_id < MilestoneNum; position_id++)
            {
                if(position[other_id][position_id])
                {
                    other_position = position_id;
                }
            }
            if(position[id][other_position])
            {
                busy = true;
            }
            else
            {
                busy = false;
            }

            return busy;
        }
    }

    return busy;
     \n \n
}"""

   
def lastTask(taskID):
    assignLastTaskFunction =""
    for task in tasks:
        lastElementofTask = task[-1]
    if taskID == lastElementofTask:
        assignLastTaskFunction = ", oneRoundTaskCompleted()"
    return assignLastTaskFunction    
        
        
# last task indicates one round task complete
oneRoundTaskCompleted = ""    
def oneRoundTask(allTask):
    """
    """
    for taskID in allTask:
       oneRoundTaskCompleted = """  \n \n
       void oneRoundTaskCompleted()
        {
            if(tf[id]["""+ taskID[-1] +"""]) { 
                stone = stone - load;
                if(stone <= 0)
                {
                    stone = 0;
                }
            }
            else
            {
                //
            }
        } \n \n"""
        
        
    return isBusyFunction + oneRoundTaskCompleted 

taskExecution.declaration = Declaration(oneRoundTask(tasks))


# add initial location
inital = Location(id="initial", pos=(216, 176),name=Name("T0", (216, 192)) )
# inital_location.initial = True
taskExecution.graph.add_location(inital)
taskExecution.graph.initial_location="initial"

# add transition for the initial location

transition_initial = Transition(source = "initial",target ="initial")
transition_initial.assignment = Label(kind="assignment", value="t = 0", pos=(160, 24),)
transition_initial.assignment = Label(kind="synchronisation", value="move[id]!", pos=(160, 24),)

taskExecution.graph.add_transition(transition_initial)

x = 10 
y = 20
taskNumber = 0
for i in tasks:
    task = i
    task_location = Location(id = i, pos=(x+ 240, y + 190), name = Name( i, (x + 250, y + 200)))
    task_location.invariant = Label(kind="invariant", value="t<=" + str(TASKWCET[i]), pos = (x+ 220, y + 170),)
    taskExecution.graph.add_location(task_location)

    task_location =""
    
    
    transition_From_T0 = Transition(source = "initial", target = str(i))
    transition_From_T0.guard = Label(kind="guard", value=str(rulesOfTaskExecution(i[-1])) + ""+ str(findTaskPosition(i[-1])), pos=(x + 180, y + 44),)
    
    transition_From_T0.assignment = Label(kind="assignment", value="t=0, ts[id][" + str(i[-1]) + "]=true, tf[id][" + str(i[-1]) + "]=false", pos=(x + 160, y + 24),)
    taskExecution.graph.add_transition(transition_From_T0)
    
    transition_To_T0 = Transition(source = str(i), target = "initial" )
    transition_To_T0.guard = Label(kind="guard", value="t>=" + str(TASKBCET[i]), pos=(x + 200, y + 66),)
    
    transition_To_T0.assignment = Label(kind="assignment", value="t=0, ts[id][" + str(i[-1]) + "]=false, tf[id][" + str(i[-1]) + "]=true"+ str(lastTask(i[-1])), pos=(x + 160, y + 24),)
    transition_To_T0.nails = [Nail(251, 146)]
    taskExecution.graph.add_transition(transition_To_T0)
    
    
    
    x += 40
    y += 70
    taskNumber += 1
    
    


# append the template to the system
sys.templates.append(movement)
sys.templates.append(taskExecution)


# System Declaration
def systemDeclation(AgentNum) -> "int":
    mainSystemDeclaration = ""
    mainSystemDeclarationInstance = ""
    mainSystemDeclarationinstantiation = ""
    for i in range(AgentNum):
        mainSystemDeclarationInstance += "movement" + str(i) + " = Movement("+ str(i) +");\n"
        mainSystemDeclarationInstance += "taskExecution" + str(i) + " = taskExecution("+ str(i) +");\n"
        if i == range(AgentNum)[0]:
            mainSystemDeclarationinstantiation += "system "
            mainSystemDeclarationinstantiation += "movement" + str(i) + ", taskExecution" + str(i) + ", "
        elif i == range(AgentNum)[-1]:
            mainSystemDeclarationinstantiation += "movement" + str(i) + ", taskExecution" + str(i) +";\n"
        else:
            mainSystemDeclarationinstantiation += " movement" + str(i) + ", taskExecution" + str(i) + ", "
    mainSystemDeclaration += mainSystemDeclarationInstance +"\n "+ mainSystemDeclarationinstantiation  
    return mainSystemDeclaration


sys.system = SystemDeclaration(systemDeclation(AgentNum))
# Queries
taskCoverageQuery = Query("E<> stone == 0", "Task Coverage")
# 100 unit stone. In each round, 13 unit minus. So, 100 / 10 = 10 
taskCoverageQueryWithTimingRequirement= Query("E<> (stone == 0 && t<=" +str(sum(totalTimeForTaskCoverage) * 10)+")", "Task Coverage with timing reqirement")
sys.queries.append(taskCoverageQuery)
sys.queries.append(taskCoverageQueryWithTimingRequirement)

# save the system to xml
sys.to_file(path='xml_files/CBS.xml')

# run UPPAAL 
running_uppaal = subprocess.Popen(['java', '-jar', 'UPPAAL/uppaal-4.1.24/uppaal.jar', 'xml_files/CBS.xml'])



