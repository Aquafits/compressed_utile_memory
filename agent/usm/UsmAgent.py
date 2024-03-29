import copy
import random
import time

import numpy as np

from typing import Tuple

from agent.usm.Usm import Usm, Instance
from maze.Maze import Maze


class UsmAgent(object):
    def __init__(self, maze: Maze, fixed_start: bool, bumped_penalty=0):

        self.maze = maze
        self.name = 'USM'

        self.pos = self.get_start_pos(fixed_start)  # 执行这一句之前一定要保证maze正确加载
        self.reward = 0
        self.bumped_penalty = bumped_penalty
        self.fixed_start = fixed_start

        self.observations = maze.observations
        self.actions = maze.actions

        self.cached_action = ''
        self.cached_observation = ''
        self.cached_state = None
        self.cached_reward = 0

        self.usm: Usm = Usm(maze.observations, maze.actions, gamma=0.9)

    def new_round(self):

        self.pos = self.get_start_pos(self.fixed_start)
        self.reward = 0

        self.cached_action = ''
        self.cached_observation = ''
        self.cached_state = None
        self.cached_reward = 0

        self.usm.new_round(self.observe())

    def get_start_pos(self, provided=True):
        if provided:
            i = np.random.choice([_ for _ in range(len(self.maze.start_positions))])
            return self.maze.start_positions[i]
        else:
            return self.get_random_pos()

    def get_random_pos(self):
        [y, x] = self.maze.walls[0]
        while [y, x] in self.maze.walls or [y, x] in self.maze.snake_pits or [y, x] in self.maze.treasures:
            y = random.randint(0, self.maze.y_size - 1)
            x = random.randint(0, self.maze.x_size - 1)

        return [y, x]

    # 用e-greedy方式探索/选择下一步的动作
    def select_e_greedily(self, e=0.2, learning=True):
        from agent.utils import opposite_pairs

        # # 当观察相同，有概率选择与上次一样的动作
        # if self.usm.is_last_two_instances_observe_the_same() and np.random.uniform(0, 1) < e:
        #     return self.usm.get_last_instance().action

        if self.cached_state is None or np.random.uniform(0, 1) < e:
            # 如果当前的状态未知，则进行随机探索，直至move方法更新缓存的状态
            action = np.random.choice(self.actions)
        else:
            # 选择Q值最高的一个动作：move函数有副作用，会使得self.cached_state更新，在上一次迭代更新过的这个leaf下找获得最大Q值的动作
            action = self.usm.get_action_with_max_q(self.cached_state)

        # 选择和之前instance动作的相反值不一样的随机的动作，即不走回头路，返回值是一个str，这个先验设定高于已学的经验
        # if learning:
        #     if self.usm.get_last_instance() is not None:
        #         while action == opposite_pairs.get(self.usm.get_last_instance().action):
        #             action = np.random.choice(self.actions)
        # else:
        #     if self.usm.get_last_test_instance() is not None:
        #         while action == opposite_pairs.get(self.usm.get_last_test_instance().action):
        #             action = np.random.choice(self.actions)

        return action

    # 在迷宫中移动, 返回(是否移动成功，是否需要检查边缘节点）
    def move(self, action, i, learning=True) -> Tuple[bool, bool]:
        if action not in self.actions:
            return False, False

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
            self.reward += (self.usm.gamma ** i) * self.bumped_penalty
            return False, False
        else:
            # 如果没有撞墙，缓存这次的动作、观察、回报和状态，更新所有有关节点的实例
            if learning:
                do_check: bool = False

                old_pos = self.pos
                self.pos = new_pos

                self.cached_action = action
                self.cached_observation = self.observe()
                self.cached_reward = self.instant_reward(i=i)

                previous = self.usm.get_last_instance()
                new_instance = Instance(previous, self.cached_action, self.cached_observation, self.cached_reward)
                new_state = self.usm.add_instance(new_instance)

                if self.cached_state is not new_state:
                    self.cached_state = new_state

                if i % 18 == 0:
                    do_check = True

                if self.cached_state is None:
                    name = 'unknown'
                else:
                    name = self.cached_state.name
                print("{:<3}: {} -> {}, acted {}, now observing {}, at state {}".format(
                    i, old_pos, new_pos, action, self.cached_observation, name))

                return True, do_check
            else:
                self.pos = new_pos

                self.cached_action = action
                self.cached_observation = self.observe()
                self.cached_reward = self.instant_reward(i=i)

                previous = self.usm.get_last_test_instance()
                new_instance = Instance(previous, self.cached_action, self.cached_observation, self.cached_reward)
                new_state = self.usm.add_test_instance(new_instance)
                if self.cached_state is not new_state:
                    self.cached_state = new_state

                return True, False

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
        self.reward += (self.usm.gamma ** i) * r
        return r

    # 迭代
    def iterate(self, iters, check_points=None, trial_times=20, steps_per_trial=20):

        if check_points is None:
            check_points = []
        check_point_values = []
        iteration_durations = []
        check_point_reach_time = []

        alg_start_time = time.clock()

        self.new_round()

        for i in range(iters):

            start_time = time.clock()
            test_start_time = start_time
            test_end_time = start_time

            if i < 0.25 * iters:
                e = 0.8
            else:
                e = 0.2 * (1 - (i / iters))
            action = self.select_e_greedily(e)
            while True:
                moved, do_check = self.move(action, i)
                if moved:
                    if do_check and self.cached_state is not None:
                        assert self.cached_state.is_leaf is True
                        self.cached_state = self.usm.check_fringe(self.cached_state)
                    break
                else:
                    action = self.select_e_greedily(e)

            if i in check_points:
                test_start_time = time.clock()
                check_point_reach_time.append(test_start_time - alg_start_time)

                print("    Checkpoint at {}:".format(i))

                val = self.generate_average_discounted_return(trial_times, steps_per_trial)
                check_point_values.append(val)
                test_end_time = time.clock()

            if self.pos in self.maze.treasures:
                print("{} is goal. New round".format(self.pos))
                self.new_round()

            # if self.pos in self.maze.snake_pits:
            #     print("{} is snake pit. New round".format(self.pos))
            #     self.new_round()

            end_time = time.clock()
            iteration_durations.append((end_time - test_end_time) + (test_start_time - start_time))

        return check_point_values, check_point_reach_time, iteration_durations

    def test_iterate(self, iters):
        self.pos = self.get_start_pos(self.fixed_start)
        self.usm.clear_test_instance()
        self.usm.add_test_instance(Instance(None, "", self.observe(), 0))
        self.reward = 0
        self.cached_state = None

        print("starts at {},".format(self.pos), end=' ')

        get_treasure = False
        for i in range(iters):
            e = 0.1
            action = self.select_e_greedily(e, learning=False)
            while True:
                moved, do_check = self.move(action, i, learning=False)
                if not moved:
                    action = self.select_e_greedily(e, learning=False)
                else:
                    break

            if self.pos in self.maze.treasures:
                print("gets reward {}, goal after {} steps.".format(self.reward, i + 1))
                get_treasure = True
                break
        self.usm.clear_test_instance()

        if not get_treasure:
            print("gets reward {}, no goal after {} steps.".format(self.reward, iters))
        return self.reward

    def generate_average_discounted_return(self, trial_times, steps_per_trial):
        rewards = []
        for _ in range(trial_times):
            pos = self.pos
            reward = self.reward
            state = self.cached_state
            observation = self.cached_observation
            action = self.cached_action

            print("    Trail {:<3}:".format(_), end=' ')
            rewards.append(self.test_iterate(steps_per_trial))

            self.pos = pos
            self.reward = reward
            self.cached_state = state
            self.cached_observation = observation
            self.cached_action = action
        return np.mean(rewards)
