
import numpy as np
import copy
import random
import matplotlib.pyplot as plt

class QLearning:
    def __init__(self, data, n_veh):
        self.data = data
        self.episodes = 100
        self.alpha=0.9
        self.n_veh = n_veh
        self.discount_rate=0.5
        # Initialize List of Q_tables  
        self.q_s_a=[dict() for i in range(n_veh)]
    # Get Q value for state action pair from Vehivle vid Q_table
    def get_q_value(self, state, action, vid):
        # Check Either state present in Q_table or not
        # If Yes then return the value of State Action value
        # Else add state into dict and set the value of state, action value to 0 
        if(tuple(state) in self.q_s_a[vid].keys()):
            # Check Either action present in dict of state or not
            # If Yes then resturn Q(S,A)
            # Else set Q(S,A) = 0 
            if(tuple(action) in self.q_s_a[vid][tuple(state)].keys()):
                return copy.deepcopy(self.q_s_a[vid][tuple(state)][tuple(action)])
            else:
                self.q_s_a[vid][tuple(state)][tuple(action)]=0
        else:
            self.q_s_a[vid][tuple(state)]=dict()
            self.q_s_a[vid][tuple(state)][tuple(action)]=0
        return copy.deepcopy(self.q_s_a[vid][tuple(state)][tuple(action)])

    def set_q_value(self, state, action, value, vid):
        # Update the Q value of a STATE, ACTION pair in Q_TABLE of Vehicle VID
        self.q_s_a[vid][tuple(state)][tuple(action)]=value

    def get_max_act_val(self, state, vid):
        # Get maximum Q_Value in all pairs of state 
        # Check Either state present in Q_table or not
        # If present than return maximum Q_value from list of all possible pairs of state S [Q(S, A1), Q(S, A2), ...]
        # Else return 0 
        if(tuple(state) in self.q_s_a[vid].keys()):
            v_max = max(self.q_s_a[vid][tuple(state)].values())
        else:
            self.q_s_a[vid][tuple(state)]=dict()
            v_max = 0
        return v_max

    def train(self):
        # Store each episode loss value
        ep_loss = []
        # Train for N episodes
        for e in range(self.episodes):
            # Sample simulation trial from dataset
            ep = self.get_batch_episode()
            # Store loss value of each step
            v_l = []
            # Update each vehicle Q_table
            for v in range(0, self.n_veh):
                # initialize loss l = 0 
                l = 0
                # Iterate over whole trajectory
                for i, d in enumerate(ep[v]):
                    # s --> State
                    # a --> Action
                    # r --> Reward
                    s, a, r = d
                    # Check either currect state is terminal state
                    # If not, then get the maximum Q-value of next state 
                    if(i!=len(ep[v])-1):
                        # Get Next state
                        s_next, a_next, _= ep[v][i+1]
                        # Get Q_value of current state, action
                        c_q = self.get_q_value(s, a, v)
                        # Get maximum Q_value of next state and action
                        c_n_q = self.get_max_act_val(s_next, v)
                        # Calculate Error R + disc * max Q (St+1,a) - Q(St, At)
                        loss = r+self.discount_rate*c_n_q-c_q
                        # Calculate updated Q_value
                        u_q = c_q + self.alpha*(loss)
                        # Update Q_table
                        self.set_q_value(s, a, u_q, v)
                    else:
                        # Get Q_value of current state
                        c_q = self.get_q_value(s, a, v)
                        # Calculate Error R - Q(St, At) for terminal state
                        loss = r-c_q
                        # Update Q_value of state St
                        u_q = c_q + self.alpha*(loss)
                        # Update Q_Table
                        self.set_q_value(s, a, u_q, v)
                    l += abs(loss)
                v_l.append(l/len(ep[v]))
            
            ep_loss.append(sum(v_l)/len(v_l))
            print(f"Episode {e}/{self.episodes}: Vehicle_Losses: {v_l}")
        self.plotting(ep_loss)

    def get_batch_episode(self):
        # Sample Episode from dataset contain simulations data
        ep_data = random.sample(self.data, 1)[0]
        ep_data1 = self.extract_veh_data(ep_data, 0)
        ep_data2 = self.extract_veh_data(ep_data, 1)
        ep_data3 = self.extract_veh_data(ep_data, 2)
        return [ep_data1, ep_data2, ep_data3]

    def extract_veh_data(self, epdata, veh):
        # Extract specific vehicle vid data from episode data
        data_veh = []
        for i, d in enumerate(epdata):
                s, a, r = d
                if (a[0]==veh):
                    data_veh.append((s, a, r))
        return data_veh

    def plotting(self, loss_data):
        plt.figure()
        plt.plot(range(1, len(loss_data)+1),loss_data)
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.title("Q Learning Loss Function")
        plt.show()

    