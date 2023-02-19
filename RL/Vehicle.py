import copy

class Vehicle:
    def __init__(self) -> None:
        self.task_status = [0, 0, 0, 0, 0]
        self.pos = "P0"
        self.round_status = 0
        self.c_task = 0
        
    def update_pos(self, new_pos):
        self.pos = new_pos
        self.c_task = 0

    def task_done(self, id):
        self.task_status[id]=1
        self.c_task = 0

    def initiate_task(self, id):
        self.c_task = id
    
    def reset_task(self):
        self.task_status = [1, 0, 0, 0, 0]
        self.round_status = 1

    def get_task_status(self):
        return self.task_status

    def get_info(self):
        state=dict()
        state["tasks"]=self.task_status
        state["round_status"]=self.round_status
        state["pos"]=self.pos
        state["c_task"]=self.c_task
        return state