import copy

import matplotlib.pyplot as plt

from agent.rand.RandAgent import RandAgent
from agent.sscusm.CsmAgent import CsmAgent
from agent.usm.UsmAgent import UsmAgent
from maze.Maze import Maze

import numpy as np
import pandas as pd

from maze.resources.hallway import hallway
from maze.resources.hallway_2 import hallway_2
from maze.resources.hallway_3 import hallway_3
from maze.resources.prim_maze import prim_maze
from maze.resources.tiger import tiger


class Runner:
    def __init__(self, maze: Maze, maze_name: str, iters=1536, trial_times=128, steps_per_trial=48):
        self.agent_names = ['USM', 'CSM', 'Rand']
        # self.agent_names = ['CSM']
        self.iters = iters
        self.trial_times = trial_times
        self.steps_per_trial = steps_per_trial
        self.maze = maze
        self.maze_name = maze_name
        self.fixed_start = True

    def run(self):
        check_points = []
        for i in range(self.iters):
            if i % 48 == 0:
                check_points.append(i)
        check_points.append(self.iters - 1)

        results = []

        for agent_name in self.agent_names:

            check_point_values_list = []
            check_point_reach_time_list = []
            iteration_durations_list = []
            result = {'name': agent_name}

            for _ in range(12):

                _agent = None
                if agent_name == 'USM':
                    _agent = UsmAgent(self.maze, self.fixed_start)
                if agent_name == 'CSM':
                    _agent = CsmAgent(self.maze, self.fixed_start)
                if agent_name == 'Rand':
                    _agent = RandAgent(self.maze, self.fixed_start)
                assert _agent is not None

                check_point_values, check_point_reach_time, iteration_durations = _agent.iterate(self.iters,
                                                                                                 check_points,
                                                                                                 self.trial_times,
                                                                                                 self.steps_per_trial)
                del _agent

                check_point_values_list.append(check_point_values)
                check_point_reach_time_list.append(check_point_reach_time)
                iteration_durations_list.append(iteration_durations)

            result['check_point_values'] = np.mean(check_point_values_list, axis=0)
            result['check_point_reach_time'] = np.mean(check_point_reach_time_list, axis=0)
            result['iteration_durations'] = np.mean(iteration_durations_list, axis=0)

            results.append(result)

        line_styles_dict = {'CSM': '-', 'USM': '--', 'Rand': ':'}
        plt.figure(figsize=(6, 30))
        ax = plt.subplot(5, 1, 1)
        ax.tick_params(labelsize=14)
        ax.set_title("Average reward through time", fontsize=20)
        ax.set_xlabel('sec', fontsize=18)
        ax.set_ylabel('value', fontsize=18)
        for result in results:
            plt.plot(result['check_point_reach_time'], result['check_point_values'], label=result['name'], color='k',
                     linestyle=line_styles_dict[result['name']])
        plt.legend(fontsize=16)

        ax = plt.subplot(5, 1, 2)
        ax.tick_params(labelsize=14)
        ax.set_title("Average reward through time", fontsize=20)
        ax.set_xlabel('sec', fontsize=18)
        ax.set_ylabel('value', fontsize=18)
        usm_result = None
        csm_result = None
        for result in results:
            if result['name'] == 'USM':
                usm_result = result
            elif result['name'] == 'CSM':
                csm_result = result
        usm_longest_time = usm_result['check_point_reach_time'][-1]
        csm_id = 0
        for csm_id in range(len(csm_result['check_point_reach_time'])):
            if csm_result['check_point_reach_time'][csm_id] > usm_longest_time:
                break
        for result in results:
            if result['name'] == 'CSM':
                plt.plot(result['check_point_reach_time'][:csm_id + 1], result['check_point_values'][:csm_id + 1],
                         label=result['name'], color='k', linestyle=line_styles_dict[result['name']])
            elif result['name'] == 'USM':
                plt.plot(result['check_point_reach_time'], result['check_point_values'], label=result['name'],
                         color='k', linestyle=line_styles_dict[result['name']])
            else:
                mean_rand = float(np.mean(result['check_point_values']))
                plt.axhline(mean_rand, label='avg(Rand)', color='k', linestyle=line_styles_dict[result['name']])
        plt.legend(fontsize=16)

        ax = plt.subplot(5, 1, 3)
        ax.tick_params(labelsize=14)
        ax.set_title("Value at check point", fontsize=20)
        ax.set_xlabel('step', fontsize=18)
        ax.set_ylabel('value', fontsize=18)
        for result in results:
            plt.plot(check_points, result['check_point_values'], label=result['name'], color='k',
                     linestyle=line_styles_dict[result['name']])
        plt.legend(fontsize=16)

        ax = plt.subplot(5, 1, 4)
        ax.tick_params(labelsize=14)
        ax.set_title('Time to reach check point', fontsize=20)
        ax.set_xlabel('step', fontsize=18)
        ax.set_ylabel('sec', fontsize=18)
        for result in results:
            plt.plot(check_points, result['check_point_reach_time'], label=result['name'], color='k',
                     linestyle=line_styles_dict[result['name']])
        plt.legend(fontsize=16)

        ax = plt.subplot(5, 1, 5)
        ax.tick_params(labelsize=14)
        ax.set_title('Iteration durations', fontsize=20)
        ax.set_xlabel('step', fontsize=18)
        ax.set_ylabel('sec', fontsize=18)
        for result in results:
            plt.plot(np.arange(self.iters), result['iteration_durations'], label=result['name'], color='k',
                     linestyle=line_styles_dict[result['name']])
        plt.legend(fontsize=16)

        plt.show()

        columns = []
        for result in results:
            column_check_point_values = pd.Series(result['check_point_values'],
                                                  name=result['name'] + '_check_point_values')
            column_check_point_reach_time = pd.Series(result['check_point_reach_time'],
                                                      name=result['name'] + '_check_point_reach_time')
            columns.append(column_check_point_values)
            columns.append(column_check_point_reach_time)

        df = pd.concat(columns, axis=1)
        df.to_csv(self.maze_name + '.csv', index=False, sep=',')


tiger_runner = Runner(tiger, 'Tiger')
tiger_runner.run()
del tiger_runner

hallway_runner = Runner(hallway, 'Hallway')
hallway_runner.run()
del hallway_runner

hallway_2_runner = Runner(hallway_2, 'McCallum')
hallway_2_runner.run()
del hallway_2_runner

# prim_runner = Runner(prim_maze, 'prim')
# prim_runner.run()
