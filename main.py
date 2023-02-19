#!/usr/bin/env python3
'''
Author: Haoran Peng
Email: gavinsweden@gmail.com
'''
import datetime
import json
import os
import random
import time
import sys
from copy import deepcopy
import cv2
import numpy as np
import yaml
from shutil import make_archive
import os, shutil



from MAPF_CBS.planner import Planner
from MAPF_CBS.agent import *


class Simulator:

    def __init__(self):
        # Set up a white 1080p canvas
        self.canvas = np.ones((1080, 1920, 3), np.uint8) * 255
        # Draw the rectangluar obstacles on canvas
        self.draw_rect(np.array([np.array(v) for v in RECT_OBSTACLES.values()]))

        # Transform the vertices to be border-filled rectangles
        static_obstacles = self.vertices_to_obsts(RECT_OBSTACLES)

        # Call cbs-mapf to plan
        self.planner = Planner(GRID_SIZE, ROBOT_RADIUS, static_obstacles)
        self.path = self.planner.plan(START, GOAL)

        # Assign each agent a colour
        self.colours = self.assign_colour(len(self.path))

        # Put the path into dictionaries for easier access
        d = dict()
        for i, path in enumerate(self.path):
            self.draw_path(self.canvas, path, i)  # Draw the path on canvas
            d[i] = path
        self.path = d

    '''
    Transform opposite vertices of rectangular obstacles into obstacles
    '''

    @staticmethod
    def vertices_to_obsts(obsts):
        def drawRect(v0, v1):
            o = []
            base = abs(v0[0] - v1[0])
            side = abs(v0[1] - v1[1])
            for xx in range(0, base, 30):
                o.append((v0[0] + xx, v0[1]))
                o.append((v0[0] + xx, v0[1] + side - 1))
            o.append((v0[0] + base, v0[1]))
            o.append((v0[0] + base, v0[1] + side - 1))
            for yy in range(0, side, 30):
                o.append((v0[0], v0[1] + yy))
                o.append((v0[0] + base - 1, v0[1] + yy))
            o.append((v0[0], v0[1] + side))
            o.append((v0[0] + base - 1, v0[1] + side))
            return o

        static_obstacles = []
        for vs in obsts.values():
            static_obstacles.extend(drawRect(vs[0], vs[1]))
        return static_obstacles

    '''
    Randomly generate colours
    '''

    @staticmethod
    def assign_colour(num):
        def colour(x):
            x = hash(str(x + 42))
            return ((x & 0xFF, (x >> 8) & 0xFF, (x >> 16) & 0xFF))

        colours = dict()
        for i in range(num):
            colours[i] = colour(i)
        return colours

    def draw_rect(self, pts_arr: np.ndarray) -> None:
        for pts in pts_arr:
            cv2.rectangle(self.canvas, tuple(pts[0]), tuple(pts[1]), (0, 0, 255), thickness=3)

    def draw_path(self, frame, xys, i):
        for x, y in xys:
            cv2.circle(frame, (int(x), int(y)), 10, self.colours[i], -1)

    '''
    Press any key to start.
    Press 'q' to exit.
    '''

    def start(self):
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', (1280, 720))
        wait = True
        try:
            i = 0
            while True:
                frame = deepcopy(self.canvas)
                for id_ in self.path:
                    x, y = tuple(self.path[id_][i])
                    cv2.circle(frame, (x, y), ROBOT_RADIUS - 5, self.colours[id_], 5)
                cv2.imshow('frame', frame)
                if wait:
                    cv2.waitKey(0)
                    wait = False
                k = cv2.waitKey(100) & 0xFF
                if k == ord('q'):
                    break
                i += 1
        except Exception:
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def load_scenario(fd):
    with open(fd, 'r') as f:
        global GRID_SIZE, ROBOT_RADIUS, RECT_OBSTACLES, START, GOAL,TASKBCET, TASKWCET, TASKSMILETSONE, ROBOT_NUMBER
        data = yaml.load(f, Loader=yaml.FullLoader)
        GRID_SIZE = data['GRID_SIZE']
        ROBOT_RADIUS = data['ROBOT_RADIUS']
        RECT_OBSTACLES = data['RECT_OBSTACLES']
        START = data['START']
        GOAL = data['GOAL']
        TASKBCET = data['TASKBCET']
        TASKWCET = data['TASKWCET'] 
        TASKSMILETSONE = data['TASKSMILETSONE']
        ROBOT_NUMBER = data['ROBOT_NUMBER']



'''
Use this function to show your START/GOAL configurations
'''


# def show_pos(pos):
#     cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('frame', (1280, 720))
#     frame = np.ones((1080, 1920, 3), np.uint8) * 255
#     for x, y in pos:
#         cv2.circle(frame, (x, y), ROBOT_RADIUS - 5, (0, 0, 0), 5)
#     cv2.imshow('frame', frame)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


def run_custom():
    # Transform the vertices to be border-filled rectangles
    static_obstacles = Simulator.vertices_to_obsts(RECT_OBSTACLES)

    # Call cbs-mapf to plan
    planner = Planner(GRID_SIZE, ROBOT_RADIUS, static_obstacles)
    path = planner.plan(START, GOAL, debug=True, max_iter=10000, low_level_max_iter=10000)
    return path


class Draw:
    def __init__(self, paths: list, radius: int = 30, square_width: int = 50, square_height: int = 50):
        # Setup a canvas
        self.rect = np.array([np.array(v) for v in RECT_OBSTACLES.values()])
        self.square_width = square_width
        self.square_height = square_height
        self.start_radius = radius

        # Group paths by starting point
        self.group_paths = dict()

        for i, path in enumerate(paths):
            self.group_paths.setdefault(tuple(path[0]), []).append(path)

    def draw(self):

        # Draw image for each group
        for idx, paths in enumerate(self.group_paths.values()):
            self.canvas = np.ones((1080, 1920, 3), dtype=np.uint8) * 255

            # Draw grid layout
            for i in range(0, self.canvas.shape[0], self.square_height):
                cv2.line(self.canvas, (0, i), (self.canvas.shape[1], i), (0, 0, 0, 32), 1)
            for i in range(0, self.canvas.shape[1], self.square_width):
                cv2.line(self.canvas, (i, 0), (i, self.canvas.shape[0]), (0, 0, 0, 32), 1)

            for d_idx, path in enumerate(paths):
                # Draw robot at starting point of each path
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                # Draw milestone and put text.
                cv2.circle(self.canvas, tuple(path[0]), self.start_radius, color, 5)
                cv2.putText(self.canvas, f"S", (path[0][0] + 30, path[0][1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

                cv2.circle(self.canvas, tuple(path[-1]), int(self.start_radius / 2), color, 5)
                cv2.putText(self.canvas, f"G{d_idx + 1}", (path[-1][0] + 30, path[-1][1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

                for x, y in path:
                    cv2.circle(self.canvas, (x, y), 10, color, -1)

            # Draw the obstacles
            for pts in self.rect[1:]:
                # set color to grey
                color = (128, 128, 128, 128)
                cv2.rectangle(self.canvas, tuple(pts[0]), tuple(pts[1]), color, thickness=-1)

            # Save the image
            # abs_path = os.path.abspath(os.path.dirname(__file__)) + f'{idx}.png'
            abs_path = os.path.join('./MilesStone_path' , f'Path_From_Milestone_{idx}.png')
            success = cv2.imwrite(abs_path, self.canvas)
            if not success:
                print('Error saving image')


if __name__ == '__main__':
    # From command line, call:
    # python3 visualizer.py scenario1.yaml
    global START, GOAL, GRID_SIZE, ROBOT_RADIUS,TASKBCET, TASKWCET, TASKSMILETSONE, ROBOT_NUMBER
    # load_scenario(sys.argv[1])
    load_scenario('Scenario/Experiment1.yaml')
    # To delete old image files 
    folder = './MilesStone_path'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    

    data = {}
    desired_data ={}
    desired_data['TASKBCET'] = TASKBCET
    desired_data['TASKWCET'] = TASKWCET
    desired_data['TASKSMILETSONE'] = TASKSMILETSONE
    desired_data["ROBOT_NUMBER"] = ROBOT_NUMBER
    orig_start = START[:]
    orig_goal = GOAL[:]
    paths = []

    # Fallback to single agent pathfinding. Try to find a path for each agent to every possible destination
    for i in range(len(orig_start)):
        for j in range(len(orig_goal)):
            START = [orig_start[i]]
            GOAL = [orig_goal[j]]
            p = run_custom()
            if len(p) > 0:
                p = p[0]
                paths.append(p)
                data[str((START[0], GOAL[0]))] = {
                    'path': paths[-1].tolist(),
                    'travelling-time': len(paths[-1])
                }
            else:
                paths.append([START[0], GOAL[0]])
                data[str((START[0], GOAL[0]))] = {
                    'path': [],
                    'travelling-time': float('inf')
                }

    # Write data to file
    with open('travellingTime.json', 'w') as f:
        json.dump(data, f)
    with open('information.json', 'w') as f:
        json.dump(desired_data, f)
    d = Draw(paths)
    d.start_radius = ROBOT_RADIUS
    d.square_width = GRID_SIZE
    d.square_height = GRID_SIZE
    d.draw()
    os.system("python modelGeneration.py")
    

