from enum import Enum
from typing import List
import numpy as np
from scipy import stats


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

    @instances.setter
    def instances(self, value):
        self.__instances = value

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
        self.actions = actions
        self.q_values = [{'leaf': _, 'actions_q': [{'action': _, 'q': 0.} for _ in self.actions]} for _ in leaves]

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

    def split_at_leaf(self, leaf: TreeNode):
        for child in leaf.children:
            self.q_values.append({'leaf': child, 'actions_q': [{'action': _, 'q': 0.} for _ in self.actions]})

        # 只删除一个用for..in应该没问题
        for leaf_actions_q in self.q_values:
            if leaf_actions_q['leaf'] is leaf:
                self.q_values.remove(leaf_actions_q)
                break


class Usm:

    def __init__(self, observations: List[str], actions: List[str], gamma=0.9):
        self.__root: TreeNode = TreeNode("root")
        self.__instances: List[Instance] = []
        self.__test_instances: List[Instance] = []  # 测试的时候使用

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

    def new_round(self, initial_observation):
        self.clear_instance()
        self.add_instance(Instance(None, "", initial_observation, 0))

    def build_fringe(self, node: TreeNode):
        # 这里我们只做一层边缘节点
        if node.type is TreeNodeType.observation:
            node.children = [TreeNode(_, parent=node) for _ in self.__actions]
        elif node.type is TreeNodeType.action:
            node.children = [TreeNode(_, parent=node) for _ in self.__observations]

    def add_test_instance(self, instance):
        if self.get_last_test_instance() is not None:
            self.__test_instances[-1].follow = instance
        self.__test_instances.append(instance)

        current_state: TreeNode = self.get_state(instance)
        return current_state

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
                if previous is not None:
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

    # 论文中的L(T_i)
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

                if instance.action == '':
                    # 根节点的action字段匹配不到任何状态，返回None
                    return None

                assert hit is True

    # 论文中的 Q(s,a) = R(s,a) + \gamma * Pr(s'|s,a) * U(s')
    def update_q_mat(self):
        for leaf in self.__leaves:
            for action in self.actions:
                updated_q = self.get_r(leaf, action) + self.gamma * self.get_pr_u(leaf, action)
                self.q_mat.set_q_by_leaf_and_action(leaf, action, updated_q)

    # 论文中的 a_{t+1} = \argmax_{a \in A} Q(L(T_t), a)
    def get_action_with_max_q(self, leaf: TreeNode):
        if leaf is None:
            return ''
        else:
            actions_q: List[dict] = self.q_mat.get_actions_q_by_leaf(leaf)
            u = float('-inf')
            action = ''

            first_action_q = actions_q[0]['q']  # 第一个动作的q值
            all_same = True  # 判断大家q值是不是一样

            for action_q in actions_q:
                if action_q['q'] > u:
                    u = action_q['q']
                    if u != first_action_q:
                        all_same = False
                    action = action_q['action']

            # 如果大家q值都是一样，应该随便选，总是选最先出现的就走不出去了
            if all_same:
                action = np.random.choice(self.actions)
            return action

    # 论文中使用ks校验分裂状态
    def check_fringe(self, current_state: TreeNode) -> TreeNode:
        # 这个父节点要有一定数量的实例才比较好
        if len(current_state.instances) < 1:
            return current_state

        # 先生成父节点包含的实例集合的预期收益集合，封装成一维的ndarray
        parent_qs = np.array([self.get_expected_discounted_reward_of_instance(_) for _ in current_state.instances])

        # 对于每一个子节点，生成子节点包含的实例集合的预期收益集合，封装成一维的ndarray，与父节点的分布进行ks校验
        for child in current_state.children:
            # 这个子节点要有一定数量的实例才比较好
            if len(child.instances) > 0:
                child_qs = np.array([self.get_expected_discounted_reward_of_instance(_) for _ in child.instances])
                d, p_value = stats.ks_2samp(parent_qs, child_qs)

                # 当p_value太低，推翻他们两个来自同一个分布的零假设，这个节点需要分裂
                if p_value < 0.1:
                    self.split_state(current_state)
                    break

        # 更新现在的agent状态，并返回
        return self.get_state(self.get_last_instance())

    def split_state(self, current_state: TreeNode):

        # 更新叶子节点列表，删除已经不是叶节点的节点，并加入新晋的叶节点
        self.__leaves.remove(current_state)
        for child in current_state.children:
            self.build_fringe(child)
            self.__leaves.append(child)
            child.is_leaf = True

            # 从新的叶节点的实例列表中，挑选实例到该节点的相应边缘节点中
            if child.type == TreeNodeType.action:
                # 如果是一个action节点成为了新的leaf，那么需要回溯这个节点的实例列表中的每一个实例的前一个非空实例，根据observation值放到不同的子观察节点
                previous_instances = [_.previous for _ in child.instances if _.previous is not None]
                for grand_child in child.children:
                    assert grand_child.type == TreeNodeType.observation
                    grand_child.instances = [_ for _ in previous_instances if _.observation == grand_child.name]
            elif child.type == TreeNodeType.observation:
                # 如果是一个observation节点成为了新的leaf，那么需要根据这个节点的实例列表中的每一个实例，根据action值放到不同的子动作节点
                for grand_child in child.children:
                    assert grand_child.type == TreeNodeType.action
                    grand_child.instances = [_ for _ in child.instances if _.action == grand_child.name]

        # 清除原叶子节点的标记和缓存的实例
        current_state.is_leaf = False
        current_state.instances = []

        # Q表也需要分裂，分裂后执行一次跟新
        self.q_mat.split_at_leaf(current_state)
        self.update_q_mat()

        pass

    """
    不是特别重要的在中间调用的方法
    """

    def get_last_instance(self):
        if len(self.instances) == 0:
            return None
        else:
            return self.instances[-1]

    def get_last_test_instance(self):
        if len(self.__test_instances) == 0:
            return None
        else:
            return self.__test_instances[-1]
        pass

    def clear_instance(self):
        self.__instances = []

    def clear_test_instance(self):
        self.__test_instances = []

    # 论文中的 Q(T_i) = r_i + \gamma * U(L(T_{i+1}))
    def get_expected_discounted_reward_of_instance(self, instance: Instance):
        r = instance.reward
        next_leaf = self.get_state(instance.follow)
        u = self.get_u(next_leaf)
        return r + self.gamma * u

    # 论文中的 U(s) = max_{a \in A} Q(s,a)
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

    # 论文中的 R(s,a) = (\sum_{T_i \in Instances(s).follow(a)} r_i)/|Instances(s,a)|
    def get_r(self, leaf: TreeNode, action: str) -> float:
        instances_s_follow_a: List[Instance] = [_.follow for _ in leaf.instances
                                                if (_.follow is not None and _.follow.action == action)]
        r = 0.
        for instance in instances_s_follow_a:
            r += instance.reward

        if len(instances_s_follow_a) == 0:
            return 0
        else:
            return r / len(instances_s_follow_a)

    # 使用了论文中的 Pr(s'|s,a) = (\forall T_i \in Instances(s).follow(a) s.t. L(T_{i+1} = s'))/|Instances(s,a)|
    # 实现了Pr_U(s,a) = \sum{s'} Pr(s'|s,a)U(s')
    def get_pr_u(self, leaf: TreeNode, action: str) -> float:
        instances_s_follow_a: List[Instance] = [_.follow for _ in leaf.instances
                                                if (_.follow is not None and _.follow.action == action)]
        pr_u = 0.
        for instance in instances_s_follow_a:
            next_leaf = self.get_state(instance)
            pr_u += self.get_u(next_leaf)

        if len(instances_s_follow_a) == 0:
            return 0
        else:
            return pr_u / len(instances_s_follow_a)
