from typing import List


class Maze:
    def __init__(self,
                 walls,
                 treasures,
                 snake_pits,
                 start_positions,
                 y_size,
                 x_size,
                 observations,
                 actions,
                 snake_penalty=-4,
                 treasure_reward=32,
                 default_reward=0):
        # 墙壁、宝藏（奖励点）、蛇坑（惩罚点）和迷宫大小
        self.walls = walls
        self.treasures = treasures
        self.snake_pits = snake_pits
        self.start_positions = start_positions
        self.y_size = y_size
        self.x_size = x_size

        # 观察列表和动作列表
        self.observations: List[str] = observations
        self.actions: List[str] = actions

        # 惩罚值，奖励值和折扣因子
        self.snake_penalty = snake_penalty
        self.treasure_reward = treasure_reward
        self.default_reward = default_reward

