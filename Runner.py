import copy

import matplotlib.pyplot as plt

from agent.rand.RandAgent import RandAgent
from agent.sscusm.CsmAgent import CsmAgent
from agent.usm.UsmAgent import UsmAgent
from maze.Maze import Maze

import numpy as np

from maze.resources.hallway import hallway
from maze.resources.hallway_2 import hallway_2
from maze.resources.hallway_3 import hallway_3
from maze.resources.prim_maze import prim_maze


class Runner:
    def __init__(self, maze: Maze, iters=1024, trial_times=128, steps_per_trial=48):
        self.agent_names = ['USM', 'CSM', 'Rand']
        # self.agent_names = ['CSM']
        self.iters = iters
        self.trial_times = trial_times
        self.steps_per_trial = steps_per_trial
        self.maze = maze

    def run(self):
        check_points = []
        for i in range(self.iters):
            if i % 32 == 0:
                check_points.append(i)
        check_points.append(self.iters - 1)

        results = []

        for agent_name in self.agent_names:

            check_point_values_list = []
            check_point_reach_time_list = []
            iteration_durations_list = []
            result = {'name': agent_name}

            for _ in range(1):

                _agent = None
                if agent_name == 'USM':
                    _agent = UsmAgent(self.maze)
                if agent_name == 'CSM':
                    _agent = CsmAgent(self.maze)
                if agent_name == 'Rand':
                    _agent = RandAgent(self.maze)
                assert _agent is not None

                check_point_values, check_point_reach_time, iteration_durations = _agent.iterate(self.iters,
                                                                                                 check_points,
                                                                                                 self.trial_times,
                                                                                                 self.steps_per_trial)

                check_point_values_list.append(check_point_values)
                check_point_reach_time_list.append(check_point_reach_time)
                iteration_durations_list.append(iteration_durations)

            result['check_point_values'] = np.mean(check_point_values_list, axis=0)
            result['check_point_reach_time'] = np.mean(check_point_reach_time_list, axis=0)
            result['iteration_durations'] = np.mean(iteration_durations_list, axis=0)

            results.append(result)

        plt.figure(figsize=(6, 24))
        ax = plt.subplot(4, 1, 1)
        ax.set_title("Average reward through time")
        for result in results:
            plt.plot(result['check_point_reach_time'], result['check_point_values'], label=result['name'])
        plt.legend()

        ax = plt.subplot(4, 1, 2)
        ax.set_title("Value at check point")
        for result in results:
            plt.plot(check_points, result['check_point_values'], label=result['name'])
        plt.legend()

        ax = plt.subplot(4, 1, 3)
        ax.set_title("Time to reach check point ")
        for result in results:
            plt.plot(check_points, result['check_point_reach_time'], label=result['name'])
        plt.legend()

        ax = plt.subplot(4, 1, 4)
        ax.set_title("Iteration durations")
        for result in results:
            plt.plot(np.arange(self.iters), result['iteration_durations'], label=result['name'])
        plt.legend()

        plt.show()

        return results


# hallway_runner = Runner(hallway)
# hallway_runner.run()

# hallway_2_runner = Runner(hallway_2)
# hallway_2_runner.run()

# hallway_3_runner = Runner(hallway_3)
# hallway_3_runner.run()

prim_runner = Runner(prim_maze)
prim_runner.run()
