
import numpy as np
import pandas as pd
import copy
from Vehicle import Vehicle


class Environment(object):
    def __init__(self, txt_file, n_veh):
        self.txt_file = txt_file
        self.n_veh = n_veh
        self.n_tasks = 5

    def parsing_file(self):
        # List contain All possible locations
        # Extract number of lines
        self.locations = []
        self.str_tasks = []
        with open(self.txt_file, encoding='utf-16') as f:
            lines_data = f.readlines()
            for i, line in enumerate(lines_data):
                if ("movement" in line):
                    starting_line = i
                    break
                if ("taskExe" in line):
                    starting_line = i
                    break
            self.n_sim = 0
            for i, line in enumerate(lines_data[starting_line+1:]):
                if ("movement" in line):
                    break
                if ("taskExe" in line):
                    break
                if ("(0,0)" in line):
                    self.n_sim +=1 
        # Extract Locations and Tasks
        self.locations = []
        self.str_tasks = []
        lines_data = lines_data[starting_line:]
        for line in lines_data:
            if ("movement" in line):
                s_split = line.strip().split(".")
                if not(s_split[1][:-1] in self.locations):
                    self.locations.append(s_split[1][:-1])
            if ("taskExe" in line):
                s_split = line.strip().split(".")
                if not(s_split[1][:-1] in self.str_tasks):
                    self.str_tasks.append(s_split[1][:-1])
        # Store each simulation data in sim_data_raw dictionary
        sim_data_raw = dict()
        # Read each line from .txt file 
        
        # Extract data of each simulation one by one
        # n_sim --> Total number of Simulation
        for n in range(self.n_sim):
            # Store each location and task data in dict
            sim_data_raw[n]=dict()
            # Iterate for each task and location
            for l in range(0, len(lines_data), self.n_sim+1):
                data_raw=[]
                # Extract each time, status pair (675.36, 0) in a task or location and store it in data_raw list
                # Raw data: String "(0,0) (0,1) (5.03,0) (5.03,1) (10.15,0) ..." --> Processed Data: List[(float, float),...] [(0,0),(0,1),(5.03,0),(5.03,1),(10.15,0),...] 
                for d in lines_data[l+n+1].split(" ")[1:]:
                    d=d.strip()
                    data_raw.append((float(d[1:-1].split(",")[0]), float(d[1:-1].split(",")[1])))
                sim_data_raw[n][lines_data[l][:-2]] = data_raw
        dataset = []
        # Convert above raw data of each simulation into the formate of (q_state, q_action, reward)
        for i in range(len(sim_data_raw)):
            print(i)
            print("Parsing Data")
            # Extract data of each vehicle from a simulation data
            # Because number of vehicles were 3, therefore we have v0,v1 and v3
            vehicles_data=[]
            for v in range(0, self.n_veh):
                vehicles_data.append(self.parse_per_vehicle(sim_data_raw[i], v))

            # Merge all vehicles data into a squence based on time
            print("Merge Data")
            merged_data = self.merge_data((vehicles_data, list(set(sim_data_raw[i]["stone"]))))
            # Convert merged data into formate which Q-Learning accept by extracting [q_state, q_action, reward] for each step
            print("State Data")
            # Find maximum time
            max_val=0 
            for s in sim_data_raw[i].keys():
                if(sim_data_raw[i][s][-1][0]>max_val):
                    max_val = sim_data_raw[i][s][-1][0]

            state_action_data = self.convert_to_state_data(merged_data, max_val)
            dataset.append(state_action_data)
        # return processed final data
        return dataset
        
    def parse_per_vehicle(self, data, vid):
        # initial state 
        task_state=[1, 0, 0, 0, 0]
        # Tasks Updates for each Agent
        data_cp = copy.deepcopy(data)
        out_data = [("movement"+str(vid)+".P0",(0,0))]
        init_data=(0, 0)
        # Remove (0,0) from each location data for vehicle vid
        for loc in self.locations:
            while (0.0, 0.0) in data_cp["movement"+str(vid)+"."+loc]: data_cp["movement"+str(vid)+"."+loc].remove((0.0,0.0))
            #data_cp["movement"+str(vid)+"."+loc]= list(set(data_cp["movement"+str(vid)+"."+loc]))
        # Remove (0,0) from each task data for vehicle vid
        for loc in self.str_tasks:
            while (0.0, 0.0) in data_cp["taskExe"+str(vid)+"."+loc]: data_cp["taskExe"+str(vid)+"."+loc].remove((0.0,0.0))
            #data_cp["taskExe"+str(vid)+"."+loc]= list(set(data_cp["taskExe"+str(vid)+"."+loc]))
        # Extract 
        while True:
            # Set upper limit of time
            temp_data= (100000, 0)
            fl=False
            # Extract MOV Action data from each Location
            for loc in self.locations:
                # Iterate over all (time, status) pair in each location
                for l in data_cp["movement"+str(vid)+"."+loc]:
                    # if time of current pair is equal to init_data time and that pair not added into list then add this pair into list with following formate
                    # (movement{vid}.{loc},(time, status)) and break inner loop
                    if(l[0]==init_data[0] and l[1]!=init_data[1] and not(("movement"+str(vid)+"."+loc,l) in out_data)):
                        temp_data=copy.deepcopy(l)
                        out_data.append(("movement"+str(vid)+"."+loc, temp_data))
                        fl=True
                        break
                    # else if current pair time is less than temp_time and greater than previous added pair time, then set it as temp_time and save it location 
                    elif(l[0]<temp_data[0] and l[0]>init_data[0]):
                        temp_data = copy.deepcopy(l)
                        loct = copy.deepcopy(loc)
            # Repeate above for EXE Action data
            if(not fl):
                tm = False
                fl = True
                for loc in self.str_tasks:
                    for l in data_cp["taskExe"+str(vid)+"."+loc]:
                        if(l[0]==init_data[0] and l[1]!=init_data[1] and not(("taskExe"+str(vid)+"."+loc,l) in out_data)):
                            temp_data=copy.deepcopy(l)
                            out_data.append(("taskExe"+str(vid)+"."+loc, temp_data))
                            fl=False
                            break
                        elif(l[0]<temp_data[0] and l[0]>init_data[0]):
                            temp_data = copy.deepcopy(l)
                            tm= True
                            loct = copy.deepcopy(loc)
                # Save data into list
                if(tm and fl):
                    out_data.append(("taskExe"+str(vid)+"."+loct,temp_data))
                elif(fl):
                    out_data.append(("movement"+str(vid)+"."+loct,temp_data))
            if(temp_data==init_data):
                break
            else:
                init_data=temp_data
        return out_data
    # Merge all Vehicles data
    def merge_data(self, data):
        vehicles_data, stone = data
        vehicles_data_cp = copy.deepcopy(vehicles_data)
        data = []
        t = 0
        # Iterate over all data until all vehicles data list are got empty
        while True:
            # Save all pairs have time equal to t into dtb
            # Initialize dtb with empty list 
            dtb = []
            # select temp time value
            # Check if v0 list empty or not if not then select first pair time as starting time
            break_flag = True 
            for v in range(self.n_veh):
                if(len(vehicles_data_cp[v])!=0):
                    t_temp = vehicles_data_cp[v][0][1][0]
                    break_flag = False

            if break_flag: # Means all lists are empty therefore break outer loop
                break

            # Search for pairs in v0 data have time equal to t_temp or less than t_temp
            for v in range(self.n_veh):
                for d in vehicles_data_cp[v]:
                    # If current pair time is equal to t_temp then append this pair into dtb list
                    if(d[1][0]==t_temp):
                        dtb.append((v, d))
                    # else If current pair time is less than t_temp then make dtb empty and add current pair into it and update t_temp to current pair_time
                    elif(d[1][0]<t_temp):
                        dtb=[]
                        dtb.append((v, d))
                        t_temp=d[1][0]
            # Search for pairs in stone data have time equal to t_temp or less than t_temp
            for d in stone[2:]:
                # If current pair time is equal to t_temp then append this pair into dtb list
                if(d[0]==t_temp):
                    dtb.append((self.n_veh, d))
                # else If current pair time is less than t_temp then make dtb empty and add current pair into it and update t_temp to current pair_time
                elif(d[0]<t_temp):
                    dtb=[]
                    dtb.append((self.n_veh, d))
                    t_temp=d[0]
            # append dtb list items into data list
            data = data + dtb
            # Remove pairs of each vehicle from their list present in dtb list 
            for d in dtb:
                i, datab = d
                for v in range(self.n_veh):
                    if(i==v and datab in vehicles_data_cp[v]):
                        vehicles_data_cp[v].remove(datab)
                if(i==self.n_veh and datab in stone):
                    stone.remove(datab)
        return data
    # Convert Raw Data to State, Action and Reward
    def convert_to_state_data(self, data, ft):
        # Create Vehicle Objects for each vehicle 
        dataset = []
        veh_obj = dict()
        for v in range(self.n_veh):
            veh_obj[v] = Vehicle()
        i=0
        # Iterate over all items of data
        for d in data:
            n, dat = d
            # n --> Vehicle number
            # dat --> Respective data point like (taskExe0.T0, (745.25, 0))
            # Check either first item type is str or not
            if(isinstance(dat[0], str)):
                # Calculate Reward = Maximum_Time - Current_Step_Time 
                if(i>0):
                    try:
                        reward= dat[1][0]-ft
                    except:
                        reward= dat[1]-ft
                    #print("Reward: ", reward)
                    dataset.append((q_state, q_action, reward))
                
                # Get current q_state from status of each vehicle
                veh_data = [veh_obj[v].get_info() for v in range(self.n_veh)]
                q_state = self.prepare_state(veh_data)
                
                # Update status of Vehicles
                for v in range(self.n_veh):
                    t_type = dat[0].split(".")[1]
                    if(n==v):
                        # If action is Task Execution 
                        if("taskExe" in dat[0]):
                            # Q_ACTION = (VEHICLE_ID, ACTION_TYPE, TASK/LOCATION)
                            q_action = (v, "EXE", t_type)
                            # Reset Task Status
                            if(veh_obj[v].get_task_status()[4]==1):
                                veh_obj[v].reset_task()
                            for t in self.str_tasks:
                                # If task is T, set goal to T --> T_index
                                if(t_type==t):
                                    # If STATUS is 1, then set task T done
                                    if(dat[1][1]): veh_obj[v].task_done(self.str_tasks.index(t_type))
                                    # Else set vehicle goal to task T
                                    else: veh_obj[v].initiate_task(self.str_tasks.index(t_type))
                        # If action is Movement
                        elif("movement" in dat[0]):
                            # Q_ACTION = (VEHICLE_ID, ACTION_TYPE, TASK/LOCATION)
                            q_action = (v, "MOV", dat[0].split(".")[1])
                            # Update the position of vehicle 
                            veh_obj[v].update_pos(dat[0].split(".")[1])
                i+=1
        return dataset
    
    # Prepare Q-state from current status of each vehicle
    def prepare_state(self, veh_state_data):
        # Get all vehicles Task_done lists and concatenate them
        # For example VehX --> [is_T0_done, is_T1_done, is_T2_done, is_T3_done, is_T4_done]
        task_status = []
        round_status = []
        pos_status = []
        c_task_status = []
        for s in veh_state_data:
            # Task Status list 
            task_status = task_status+s["tasks"]
            # Get all vehicles round status and add into a list
            round_status.append(s["round_status"])
            # Get all vehicles positions and add into a list called pos_list
            pos_status.append(s["pos"])
            # Get all vehicles current goal/task
            c_task_status.append(s["c_task"])

        # Concatenate all statuses
        # return ("{" +task_status+"}"+"{" +round_status+"}"+"{" +pos_status+"}"+"{" +c_task_status + "}")
        
        return (task_status+round_status+pos_status+c_task_status)
