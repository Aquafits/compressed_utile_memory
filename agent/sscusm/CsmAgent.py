import copy
import random
import time

import numpy as np

from typing import Tuple

from agent.sscusm.Csm import Csm, Instance
from maze.Maze import Maze


class CsmAgent(object):
    def __init__(self, maze: Maze, bumped_penalty=0):

        self.maze = maze
        self.name = 'CSM'

        self.pos = self.get_start_pos()  # 执行这一句之前一定要保证maze正确加载
        self.reward = 0
        self.bumped_penalty = bumped_penalty

        self.observations = maze.observations
        self.actions = maze.actions

        self.cached_action = ''
        self.cached_observation = ''
        self.cached_state = None
        self.cached_reward = 0
        self.cached_check_point = 0

        self.usm: Csm = Csm(maze.observations, maze.actions, gamma=0.8)
        self.usm.longest_edge = self.blind_exploration()

    def new_round(self):

        self.pos = self.get_start_pos()
        self.reward = 0

        self.cached_action = ''
        self.cached_observation = ''
        self.cached_state = None
        self.cached_reward = 0

        self.usm.new_round(self.observe(), self.cached_check_point)

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

    def blind_exploration(self, iters=256):
        self.usm.clear_test_instance()
        self.usm.add_test_instance(Instance(None, "", self.observe(), 0, cached_check_point=-1))

        goal_count = 0
        for i in range(iters):
            e = 1.0
            p = 1
            action = self.select_e_greedily(e, force_ahead=p)
            while True:
                moved, do_check = self.move(action, i, learning=False)
                if not moved:
                    p /= 2
                    action = self.select_e_greedily(e, force_ahead=p)
                else:
                    break

            if self.pos in self.maze.treasures:
                self.pos = self.get_random_pos()
                goal_count += 1

        self.cached_action = ''
        self.cached_observation = ''
        self.cached_state = None
        self.cached_reward = 0

        actions = [_.action for _ in self.usm.test_instances]

        east_most = 0
        west_most = 0
        north_most = 0
        south_most = 0

        x = 0
        y = 0
        for a in actions:
            if a == 'south':
                y += 1
                if y > south_most:
                    south_most = y
            elif a == 'north':
                y -= 1
                if y < north_most:
                    north_most = y
            elif a == 'east':
                x += 1
                if x > east_most:
                    east_most = x
            elif a == 'west':
                x -= 1
                if x < west_most:
                    west_most = x
            elif a == '':
                x = 0
                y = 0

        return (south_most - north_most) + (east_most - west_most)

    # 用boltzmann方式探索/选择下一步的动作
    def select_by_boltzmann_sampling(self, temperature=10., learning=True, force_ahead=0.25):
        from agent.utils import opposite_pairs

        if self.cached_state is None:
            # 如果当前的状态未知，则进行随机探索，直至move方法更新缓存的状态
            action = np.random.choice(self.actions)
        else:
            actions_q = self.usm.q_mat.get_actions_q_by_leaf(self.cached_state)
            actions = [action_q['action'] for action_q in actions_q]
            q = [action_q['q'] for action_q in actions_q]
            exponent = np.true_divide(q - np.max(q), temperature)
            boltzmann_distribution = np.exp(exponent) / np.sum(np.exp(exponent))
            action = np.random.choice(actions, p=boltzmann_distribution)

        # 选择和之前instance动作的相反值不一样的随机的动作，即不走回头路，返回值是一个str，这个先验设定高于已学的经验
        if np.random.uniform(0, 1) < force_ahead:
            if learning:
                if self.usm.get_last_instance() is not None:
                    while action == opposite_pairs.get(self.usm.get_last_instance().action):
                        action = np.random.choice(self.actions)
            else:
                if self.usm.get_last_test_instance() is not None:
                    while action == opposite_pairs.get(self.usm.get_last_test_instance().action):
                        action = np.random.choice(self.actions)

        return action

    # 用e-greedy方式探索/选择下一步的动作
    def select_e_greedily(self, e=0.2, learning=True, force_ahead=0.25):
        from agent.utils import opposite_pairs

        if self.cached_state is None or np.random.uniform(0, 1) < e:
            # 如果当前的状态未知，则进行随机探索，直至move方法更新缓存的状态
            action = np.random.choice(self.actions)
        else:
            # 选择Q值最高的一个动作：move函数有副作用，会使得self.cached_state更新，在上一次迭代更新过的这个leaf下找获得最大Q值的动作
            action = self.usm.get_action_with_max_q(self.cached_state)

        # 选择和之前instance动作的相反值不一样的随机的动作，即不走回头路，返回值是一个str，这个先验设定高于已学的经验
        if np.random.uniform(0, 1) < force_ahead:
            if learning:
                if self.usm.get_last_instance() is not None:
                    while action == opposite_pairs.get(self.usm.get_last_instance().action):
                        action = np.random.choice(self.actions)
            else:
                if self.usm.get_last_test_instance() is not None:
                    while action == opposite_pairs.get(self.usm.get_last_test_instance().action):
                        action = np.random.choice(self.actions)

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
                new_instance = Instance(previous, self.cached_action, self.cached_observation, self.cached_reward,
                                        self.cached_check_point)
                new_state = self.usm.add_instance(new_instance)
                if self.cached_state is not new_state:
                    self.cached_state = new_state
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
                new_instance = Instance(previous, self.cached_action, self.cached_observation, self.cached_reward,
                                        cached_check_point=-1)
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

        self.cached_check_point = check_points[0]
        self.new_round()

        for i in range(iters):

            start_time = time.clock()
            test_start_time = start_time
            test_end_time = start_time

            if i < 128:
                temperature = float('inf')
            else:
                temperature = 16 * np.cos((np.pi / 3) * (i / iters) + (np.pi / 6))

            # action = self.select_e_greedily(e)
            p = 1.0
            action = self.select_by_boltzmann_sampling(temperature, force_ahead=p)
            while True:

                moved, do_check = self.move(action, i)
                if moved:
                    if do_check and self.cached_state is not None:
                        assert self.cached_state.is_leaf is True
                        self.cached_state = self.usm.check_fringe(self.cached_state)
                    break
                else:
                    # action = self.select_e_greedily(e)
                    temperature *= 1.25
                    p /= 2
                    action = self.select_by_boltzmann_sampling(temperature, force_ahead=p)

            if i in check_points:
                self.cached_check_point = i  # 更新缓存的检查点标签

                index = check_points.index(i)
                test_start_time = time.clock()
                check_point_reach_time.append(test_start_time - alg_start_time)

                print("    Checkpoint at {}:".format(index))

                val = self.generate_average_discounted_return(trial_times, steps_per_trial)
                if index > 3 and (val - min(check_point_values)) < 0.875 * (
                        max(check_point_values) - min(check_point_values)):
                    self.ensure_learning_quality(check_points[index - 1], i)
                    self.usm.update_q_mat()
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
        self.pos = self.get_start_pos()
        self.usm.clear_test_instance()
        self.usm.add_test_instance(Instance(None, "", self.observe(), 0, cached_check_point=-1))
        self.reward = 0
        self.cached_state = None

        print("starts at {},".format(self.pos), end=' ')

        get_treasure = False
        for i in range(iters):
            temperature = 1 / (1 + i)
            p = 1
            action = self.select_by_boltzmann_sampling(temperature, learning=False, force_ahead=p)
            while True:
                moved, do_check = self.move(action, i, learning=False)
                if not moved:
                    temperature *= 2
                    p /= 2
                    action = self.select_by_boltzmann_sampling(temperature, learning=False, force_ahead=p)
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

    def ensure_learning_quality(self, left, right):
        for instance in self.usm.instances:
            instance.check_point = right

        for leaf in self.usm.leaves:
            leaf.instances = [_ for _ in leaf.instances if _.check_point != left]
            for child in leaf.children:
                child.instances = [_ for _ in child.instances if _.check_point != left]
