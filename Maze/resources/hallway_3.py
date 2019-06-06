from maze.Maze import Maze

walls = [
    [2, 2], [2, 3], [2, 7], [2, 8],
    [3, 2], [3, 3], [3, 7], [3, 8],
    [4, 2], [4, 3], [4, 7], [4, 8],
]

treasures = [[3, 5]]
snake_pits = []

observations = [
    '____',  # 0
    '___S',  # 1
    '__N_',  # 2
    '__NS',  # 3
    '_E__',  # 4
    '_E_S',  # 5
    '_EN_',  # 6
    '_ENS',  # 7
    'W___',  # 8
    'W__S',  # 9
    'W_N_',  # 10
    'W_NS',  # 11
    'WE__',  # 12
    'WE_S',  # 13
    'WEN_',  # 14
    'WENS',  # 15
]
actions = [
    'west',
    'east',
    'north',
    'south'
]

hallway_3 = Maze(walls, treasures, snake_pits, 7, 11, observations, actions, default_reward=-1, treasure_reward=32)
