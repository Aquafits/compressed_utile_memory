from enum import Enum
from typing import List


class Maze:
    def __init__(self,
                 walls,
                 treasures,
                 snake_pits,
                 y_size,
                 x_size,
                 observations,
                 actions,
                 snake_penalty=-40,
                 treasure_reward=60,
                 default_reward=0):
        # 墙壁、宝藏（奖励点）、蛇坑（惩罚点）和迷宫大小
        self.walls = walls
        self.treasures = treasures
        self.snake_pits = snake_pits
        self.y_size = y_size
        self.x_size = x_size

        # 观察列表和动作列表
        self.observations: List[str] = observations
        self.actions: List[str] = actions

        # 惩罚值，奖励值和折扣因子
        self.snake_penalty = snake_penalty
        self.treasure_reward = treasure_reward
        self.default_reward = default_reward


class Instance:
    def __init__(self, previous, action: str, observation: str, reward):
        self.previous: Instance = previous
        self.action = action
        self.observation = observation
        self.reward = reward

        # 这个字段仅记录绝对访问顺序，一个在UsmNode局部的instances列表里的instance，其follow不一定还在该列表中
        self.follow: Instance = None


class TreeNodeType(Enum):
    root = 0
    action = 1
    observation = 2


class TreeNode:
    def __init__(self, name, parent=None, children=None):
        # 记录节点在树中的深度
        if parent is None:
            self.__depth = 0
        else:
            self.__depth = parent.depth + 1

        # 这个节点的名称，类型是action的是动作名称，类型是observation的是观察名称，类型是root的是"root"
        self.__name = name

        # 指定参数中的亲节点，和子节点
        if children is None:
            children = []
        self.__parent: TreeNode = parent
        self.__children: List[TreeNode] = children

        # 实例列表
        self.__instances: List[Instance] = []

        # 一些标记
        self.__is_leaf = False  # 标记这是一个视作状态的节点

        self.__type = None  # 标记这个节点的类型
        if self.__parent is None:
            self.__type = TreeNodeType.root
        elif self.__parent.type is TreeNodeType.action or self.__parent.type is TreeNodeType.root:
            self.__type = TreeNodeType.observation
        elif self.__parent.type is TreeNodeType.observation:
            self.__type = TreeNodeType.action

    @property
    def type(self):
        return self.__type

    @property
    def name(self):
        return self.__name

    @property
    def depth(self):
        return self.__depth

    @property
    def instances(self):
        return self.__instances

    @property
    def is_leaf(self):
        return self.__is_leaf

    @is_leaf.setter
    def is_leaf(self, value):
        self.__is_leaf = value

    @property
    def children(self):
        return self.__children

    @children.setter
    def children(self, value):
        self.__children = value


class QMat:
    def __init__(self, leaves, actions) -> None:
        self.q_values = [{'leaf': _, 'actions_q': [{'action': _, 'q': 0.} for _ in actions]} for _ in leaves]

    def get_q_by_leaf_and_action(self, leaf: TreeNode, action: str):
        for leaf_actions_q in self.q_values:
            if leaf_actions_q['leaf'] == leaf:
                for action_q in leaf_actions_q['actions_q']:
                    if action_q['action'] == action:
                        return action_q['q']

    def set_q_by_leaf_and_action(self, leaf: TreeNode, action: str, q: float):
        for leaf_actions_q in self.q_values:
            if leaf_actions_q['leaf'] == leaf:
                for action_q in leaf_actions_q['actions_q']:
                    if action_q['action'] == action:
                        action_q['q'] = q

    def get_actions_q_by_leaf(self, leaf: TreeNode):
        for leaf_actions_q in self.q_values:
            if leaf_actions_q['leaf'] == leaf:
                return leaf_actions_q['actions_q']


class Usm:

    def __init__(self, observations: List[str], actions: List[str], gamma=0.9):
        self.__root: TreeNode = TreeNode("root")
        self.__instances: List[Instance] = []

        self.__leaves: List[TreeNode] = []
        self.__gamma: float = gamma

        # observations 和 actions 是为了构建usm树
        self.__actions: List[str] = actions
        self.__observations: List[str] = observations

        # 初始化Usm树，将初始观察节点加入到树中，设置叶节点属性并且向下扩展出边缘节点
        self.__root.children = [TreeNode(_, parent=self.__root) for _ in observations]

        for _ in self.__root.children:
            _.is_leaf = True
            self.__leaves.append(_)

        for _ in self.__root.children:
            self.build_fringe(_)

        # 初始化Q表
        self.__q_mat = QMat(self.__leaves, self.__actions)

    @property
    def observations(self):
        return self.__observations

    @property
    def actions(self):
        return self.__actions

    @property
    def q_mat(self):
        return self.__q_mat

    @property
    def gamma(self):
        return self.__gamma

    @property
    def instances(self):
        return self.__instances

    @instances.setter
    def instances(self, value):
        self.__instances = value

    def new_round(self, initial_observation):
        self.clear_instance()
        self.add_instance(Instance(None, "", initial_observation, 0))

    def build_fringe(self, node: TreeNode):
        # 这里我们只做一层边缘节点
        if node.type is TreeNodeType.observation:
            node.children = [TreeNode(_, parent=node) for _ in self.__actions]
        elif node.type is TreeNodeType.action:
            node.children = [TreeNode(_, parent=node) for _ in self.__observations]

    def get_last_instance(self):
        if len(self.instances) == 0:
            return None
        else:
            return self.instances[-1]

    def is_last_two_instances_observe_the_same(self):
        if len(self.instances) >= 2:
            return self.instances[-1].observation == self.instances[-2].observation
        else:
            return False

    def clear_instance(self):
        self.instances = []

    def add_instance(self, instance: Instance):
        # 加入绝对实例历史
        if self.get_last_instance() is not None:
            self.instances[-1].follow = instance
        self.instances.append(instance)

        current_state: TreeNode = self.get_state(instance)
        if current_state is not None:
            if current_state.type is TreeNodeType.action:
                # 如果状态是一个动作节点，那么回溯一个实例，找到与回溯的实例 observation 字段相同的边缘节点，并置入实例
                previous = instance.previous
                for _ in current_state.children:
                    if _.name == previous.observation:
                        _.instances.append(previous)

                # 然后将该实例加入该状态
                current_state.instances.append(instance)
            elif current_state.type is TreeNodeType.observation:
                # 如果状态是一个观察节点，找到与该实例 action 字段相同的边缘节点，并置入实例
                for _ in current_state.children:
                    if _.name == instance.action:
                        _.instances.append(instance)

                # 然后将该实例加入该状态
                current_state.instances.append(instance)

            self.update_q_mat()

            return current_state
        else:
            return None

    def get_state(self, instance: Instance):
        nodes: List[TreeNode] = self.__root.children
        depth = 1  # nodes的深度

        # 进行后缀匹配
        while depth < 20:
            # 在多轮实验的时候，有的时候instance序列不够长，但是被标记成状态的节点已经很深了，就匹配不到，匹配不到就返回None
            if instance is None:
                return None

            if nodes[0].type is TreeNodeType.observation:
                # 子节点的类型是 observation，那么应该搜索 name 字段与 instance 的 observation 字段相同的子节点
                hit = False
                for n in nodes:
                    if n.name == instance.observation:
                        hit = True
                        if n.is_leaf:
                            return n
                        else:
                            # 不用追溯上一个实例，用当前实例的 action 字段搜索这个节点 n 的子节点
                            nodes = n.children
                            depth += 1
                assert hit is True

            elif nodes[0].type is TreeNodeType.action:
                # 子节点的类型是 action，那么应该搜索 name 字段与 instance 的 action 字段相同的子节点
                hit = False
                for n in nodes:
                    if n.name == instance.action:
                        hit = True
                        if n.is_leaf:
                            return n
                        else:
                            # 追溯上一个实例，用上一个实例的 observation 字段搜索这个节点 n 的子节点
                            instance = instance.previous
                            nodes = n.children
                            depth += 1

                assert hit is True

    def update_q_mat(self):
        for leaf in self.__leaves:
            for action in self.actions:
                updated_q = self.get_r(leaf, action) + self.gamma * self.get_pr_u(leaf, action)
                self.q_mat.set_q_by_leaf_and_action(leaf, action, updated_q)

    def get_u(self, leaf: TreeNode):
        if leaf is None:
            return 0.
        else:
            actions_q: List[dict] = self.q_mat.get_actions_q_by_leaf(leaf)
            u = float('-inf')
            for action_q in actions_q:
                if action_q['q'] > u:
                    u = action_q['q']
            return u

    def get_r(self, leaf: TreeNode, action: str) -> float:
        instances_s_a: List[Instance] = [_ for _ in leaf.instances if _.action == action]
        r = 0.
        for instance in instances_s_a:
            r += instance.reward

        if len(instances_s_a) == 0:
            return 0
        else:
            return r / len(instances_s_a)

    def get_pr_u(self, leaf: TreeNode, action: str) -> float:
        instances_s_a: List[Instance] = [_ for _ in leaf.instances if _.action == action]
        pr_u = 0.
        for instance in instances_s_a:
            next_leaf = self.get_state(instance.follow)
            pr_u += self.get_u(next_leaf)

        if len(instances_s_a) == 0:
            return 0.
        else:
            return pr_u / len(instances_s_a)

    def get_action_with_max_q(self):
        leaf: TreeNode = self.get_state(self.instances[-1])
        if leaf is None:
            return ''
        else:
            actions_q: List[dict] = self.q_mat.get_actions_q_by_leaf(leaf)
            u = float('-inf')
            action = ''
            for action_q in actions_q:
                if action_q['q'] > u:
                    u = action_q['q']
                    action = action_q['action']
            return action
