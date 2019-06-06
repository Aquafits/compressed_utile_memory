import copy
import random
import time

import numpy as np

from maze.Maze import Maze


class RandAgent(object):
    def __init__(self, maze: Maze, bumped_penalty=0):

        self.gamma = 0.8
        self.maze = maze
        self.name = 'Rand'

        self.pos = self.get_random_pos()  # 执行这一句之前一定要保证maze正确加载
        self.reward = 0
        self.bumped_penalty = bumped_penalty

        self.observations = maze.observations
        self.actions = maze.actions

        self.cached_action = None

    def new_round(self):
        self.pos = self.get_random_pos()
        self.reward = 0

    def get_random_pos(self):
        [y, x] = self.maze.walls[0]
        while [y, x] in self.maze.walls or [y, x] in self.maze.snake_pits or [y, x] in self.maze.treasures:
            y = random.randint(0, self.maze.y_size - 1)
            x = random.randint(0, self.maze.x_size - 1)

        return [y, x]

    # 用随机探索/选择下一步的动作
    def select_randomly(self):
        from agent.utils import opposite_pairs

        action = np.random.choice(self.actions)
        # if self.cached_action is not None:
        #     while action == opposite_pairs.get(self.cached_action):
        #         action = np.random.choice(self.actions)

        return action

    # 在迷宫中移动, 返回是否移动成功
    def move(self, action, i) -> bool:
        if action not in self.actions:
            return False

        new_pos = copy.copy(self.pos)

        if action == 'north':
            new_pos[0] += -1
        elif action == 'south':
            new_pos[0] += 1
        elif action == 'east':
            new_pos[1] += 1
        elif action == 'west':
            new_pos[1] += -1

        # 检测是否碰撞到边界或墙壁
        bumped = self.bumped(new_pos)

        if bumped:
            self.reward += (self.gamma ** i) * self.bumped_penalty
            return False
        else:
            self.pos = new_pos
            self.instant_reward(i=i)
            self.cached_action = action
            return True

    # 获得代理的观察，这里和模型的数据有耦合，更改模型中的observations的顺序会出错
    def observe(self):
        north_pos = copy.copy(self.pos)
        north_pos[0] += -1
        south_pos = copy.copy(self.pos)
        south_pos[0] += 1
        east_pos = copy.copy(self.pos)
        east_pos[1] += 1
        west_pos = copy.copy(self.pos)
        west_pos[1] += -1

        observation_id = 0
        if self.bumped(west_pos):
            observation_id += 8
        if self.bumped(east_pos):
            observation_id += 4
        if self.bumped(north_pos):
            observation_id += 2
        if self.bumped(south_pos):
            observation_id += 1
        return self.observations[observation_id]

    def bumped(self, pos):
        bumped = pos in self.maze.walls or \
                 pos[0] < 0 or pos[1] < 0 or \
                 pos[0] >= self.maze.y_size or pos[1] >= self.maze.x_size
        return bumped

    # 获得立即回报
    def instant_reward(self, i):
        if self.pos in self.maze.snake_pits:
            r = self.maze.snake_penalty
        elif self.pos in self.maze.treasures:
            r = self.maze.treasure_reward
        else:
            r = self.maze.default_reward
        self.reward += (self.gamma ** i) * r
        return r

    # 迭代
    def iterate(self, iters, check_points=None, trial_times=20, steps_per_trial=20):

        if check_points is None:
            check_points = []
        check_point_values = []
        iteration_durations = []
        check_point_durations = []

        self.new_round()

        for i in range(iters):

            start_time = time.clock()
            test_start_time = start_time
            test_end_time = start_time

            if i in check_points:
                test_start_time = time.clock()
                print("    Checkpoint at {}:".format(i))
                val = self.generate_average_discounted_return(trial_times, steps_per_trial)
                check_point_values.append(val)
                test_end_time = time.clock()
                check_point_durations.append(test_end_time - test_start_time)

            end_time = time.clock()
            iteration_durations.append((end_time - test_end_time) - (test_start_time - start_time))

        return check_point_values, check_point_durations, iteration_durations

    def test_iterate(self, iters):
        self.pos = self.get_random_pos()
        self.reward = 0
        print("starts at {},".format(self.pos), end=' ')

        get_treasure = False
        for i in range(iters):
            action = self.select_randomly()
            while True:
                moved = self.move(action, i)
                if not moved:
                    action = self.select_randomly()
                else:
                    break

            if self.pos in self.maze.treasures:
                print("gets reward {}, goal after {} steps.".format(self.reward, i + 1))
                get_treasure = True
                break

        if not get_treasure:
            print("gets reward {}, no goal after {} steps.".format(self.reward, iters))
        return self.reward

    def generate_average_discounted_return(self, trial_times, steps_per_trial):
        rewards = []
        for _ in range(trial_times):
            pos = self.pos
            reward = self.reward

            print("    Trail {:<3}:".format(_), end=' ')
            rewards.append(self.test_iterate(steps_per_trial))

            self.pos = pos
            self.reward = reward
        return np.mean(rewards)
