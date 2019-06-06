import copy

import matplotlib.pyplot as plt

from agent.rand.RandAgent import RandAgent
from agent.sscusm.SSCUsmAgent import SSCUsmAgent
from agent.sscusm.CsmAgent import CsmAgent
from agent.usm.UsmAgent import UsmAgent
from maze.Maze import Maze

import numpy as np

from maze.resources.hallway import hallway
from maze.resources.hallway_2 import hallway_2
from maze.resources.hallway_3 import hallway_3


class Runner:
    def __init__(self, maze: Maze, iters=1280, trial_times=256, steps_per_trial=48):
        # self.agents = [UsmAgent(maze), SSCUsmAgent(maze), SSCBoltzmannUsmAgent(maze), RandAgent(maze)]
        self.agents = [UsmAgent(maze), CsmAgent(maze), RandAgent(maze)]
        # self.agents = [SSCUsmAgent(maze), SSCBoltzmannUsmAgent(maze)]
        # self.agents = [SSCBoltzmannUsmAgent(maze)]
        self.iters = iters
        self.trial_times = trial_times
        self.steps_per_trial = steps_per_trial

    def run(self):
        check_points = []
        for i in range(self.iters):
            if i % 32 == 0:
                check_points.append(i)
        check_points.append(self.iters - 1)

        results = []

        for agent in self.agents:

            check_point_values_list = []
            check_point_durations_list = []
            iteration_durations_list = []
            result = {'name': agent.name}

            for _ in range(4):
                _agent = copy.deepcopy(agent)

                check_point_values, check_point_durations, iteration_durations = _agent.iterate(self.iters,
                                                                                                check_points,
                                                                                                self.trial_times,
                                                                                                self.steps_per_trial)

                check_point_values_list.append(check_point_values)
                check_point_durations_list.append(check_point_durations)
                iteration_durations_list.append(iteration_durations)

            result['check_point_values'] = np.mean(check_point_values_list, axis=0)
            result['check_point_durations'] = np.mean(check_point_durations_list, axis=0)
            result['iteration_durations'] = np.mean(iteration_durations_list, axis=0)

            results.append(result)

        plt.figure(figsize=(6, 18))
        ax = plt.subplot(3, 1, 1)
        ax.set_title("check point values")
        for result in results:
            plt.plot(check_points, result['check_point_values'], label=result['name'])
        plt.legend()

        ax = plt.subplot(3, 1, 2)
        ax.set_title("check point durations")
        for result in results:
            plt.plot(check_points, result['check_point_durations'], label=result['name'])
        plt.legend()

        ax = plt.subplot(3, 1, 3)
        ax.set_title("iteration durations")
        for result in results:
            plt.plot(np.arange(self.iters), result['iteration_durations'], label=result['name'])
        plt.legend()

        plt.show()

        return results


# hallway_2_runner = Runner(hallway_2)
# hallway_2_runner.run()

hallway_runner = Runner(hallway)
hallway_runner.run()

# hallway_3_runner = Runner(hallway_3)
# hallway_3_runner.run()
