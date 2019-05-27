import copy
import random
import numpy as np

from Usm import Usm, Instance, Maze


class UsmAgent(object):
    def __init__(self, maze: Maze):

        self.maze = maze

        self.pos = self.get_random_pos()  # 执行这一句之前一定要保证maze正确加载
        self.reward = 0

        self.observations = maze.observations
        self.actions = maze.actions

        self.cached_action = ''
        self.cached_observation = ''
        self.cached_state = None
        self.cached_reward = 0

        self.usm: Usm = Usm(maze.observations, maze.actions, gamma=0.9)

    def new_round(self):

        self.pos = self.get_random_pos()
        self.reward = 0

        self.cached_action = ''
        self.cached_observation = ''
        self.cached_state = None
        self.cached_reward = 0

        self.usm.new_round(self.observe())

    def get_random_pos(self):
        [y, x] = self.maze.walls[0]
        while [y, x] in self.maze.walls or [y, x] in self.maze.snake_pits or [y, x] in self.maze.treasures:
            y = random.randint(0, self.maze.y_size - 1)
            x = random.randint(0, self.maze.x_size - 1)

        return [y, x]

    # 用e-greedy方式探索/选择下一步的动作
    def select_e_greedily(self, e=0.2):
        from resources.utils import opposite_pairs

        # 当观察相同，有概率选择与上次一样的动作
        if self.usm.is_last_two_instances_observe_the_same() and np.random.uniform(0, 1) < e:
            return self.usm.get_last_instance().action

        if self.cached_state is None or np.random.uniform(0, 1) < e:
            # 如果当前的状态未知，则进行随机探索，直至move方法更新缓存的状态
            action = np.random.choice(self.actions)
        else:
            # 选择Q值最高的一个动作
            action = self.usm.get_action_with_max_q()

        # 选择和之前instance动作的相反值不一样的随机的动作，即不走回头路，返回值是一个str，这个先验设定高于已学的经验
        if self.usm.get_last_instance() is not None:
            while action == opposite_pairs.get(self.usm.get_last_instance().action):
                action = np.random.choice(self.actions)

        return action

    # 在迷宫中移动, 返回是否移动成功
    def move(self, action):
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

        # 如果没有撞墙，缓存这次的动作、观察、回报和状态，更新所有有关节点的实例
        if not bumped:
            old_pos = self.pos
            self.pos = new_pos
            self.cached_action = action
            self.cached_observation = self.observe()
            self.cached_reward = self.instant_reward()

            previous = self.usm.get_last_instance()
            new_instance = Instance(previous, self.cached_action, self.cached_observation, self.cached_reward)
            self.cached_state = self.usm.add_instance(new_instance)

            print("{} -> {}, acted {}, now observing {}, at state {}".format(
                    old_pos, new_pos, action, self.cached_observation, self.cached_state.name))
            return True
        else:
            return False

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
    def instant_reward(self):
        if self.pos in self.maze.snake_pits:
            r = self.maze.snake_penalty
        elif self.pos in self.maze.treasures:
            r = self.maze.treasure_reward
        else:
            r = self.maze.default_reward
        self.reward += r
        return r

    # 迭代
    def iterate(self, iters):
        self.new_round()
        for i in range(iters):
            if i < 200:
                e = 0.9
            else:
                e = 0.2
            action = self.select_e_greedily(e)
            while self.move(action) is False:
                action = self.select_e_greedily(e)

            if self.pos in self.maze.treasures:
                print("{} is goal. New round".format(self.pos))
                self.new_round()

        pass
