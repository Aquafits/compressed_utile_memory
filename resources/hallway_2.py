from Usm import Maze

walls = [
    [1, 1], [1, 2], [1, 4], [1, 5],
    [2, 1], [2, 2], [2, 4], [2, 5],
    [3, 1], [3, 2], [3, 4], [3, 5]
]

treasures = [[2, 3]]
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

hallway_2 = Maze(walls, treasures, snake_pits, 5, 7, observations, actions)
