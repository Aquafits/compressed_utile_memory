from maze.Maze import Maze

walls = [
    [1, 2]
]

treasures = [[0, 2]]
snake_pits = [[0, 0], [0, 4]]
start_positions = [[1, 1], [1, 3]]

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

tiger = Maze(walls, treasures, snake_pits, start_positions, 2, 5, observations, actions, default_reward=-0.1,
             snake_penalty=-4, treasure_reward=16)
