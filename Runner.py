import matplotlib.pyplot as plt

from agent.rand.RandAgent import RandAgent
from agent.sscusm.SSCUsmAgent import SSCUsmAgent
from agent.usm.UsmAgent import UsmAgent
from maze.Maze import Maze
from maze.resources.hallway_2 import hallway_2

import numpy as np


class Runner:
    def __init__(self, maze: Maze, iters=100, trial_times=100, steps_per_trial=30):
        self.agents = [UsmAgent(maze), RandAgent(maze), SSCUsmAgent(maze)]
        self.iters = iters
        self.trial_times = trial_times
        self.steps_per_trial = steps_per_trial

    def run(self):
        check_points = []
        for i in range(self.iters):
            if i % 50 == 0:
                check_points.append(i)
        check_points.append(self.iters - 1)

        results = []

        for agent in self.agents:
            print(agent.name)
            result = {'name': agent.name}
            check_point_values, check_point_durations, iteration_durations = agent.iterate(self.iters, check_points,
                                                                                           self.trial_times,
                                                                                           self.steps_per_trial)
            result['check_point_values'] = check_point_values
            result['check_point_durations'] = check_point_durations
            result['iteration_durations'] = iteration_durations

            results.append(result)

        plt.figure(figsize=(6, 12))
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


demo_runner = Runner(hallway_2)
# demo_runner = Runner(tiger)
demo_runner.run()
