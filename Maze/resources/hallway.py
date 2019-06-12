from maze.Maze import Maze

walls = [
    [1, 0], [1, 1], [1, 3], [1, 5], [1, 7], [1, 9], [1, 10]
]

treasures = [[1, 8]]
# snake_pits = [[1, 2], [1, 4], [1, 6]]
snake_pits = []
start_positions = [[0, 0],
                   [0, 1],
                   [0, 2],
                   [0, 3],
                   [0, 4],
                   [0, 5],
                   [0, 6],
                   [0, 7],
                   [0, 8],
                   [0, 9],
                   [0, 10],
                   [0, 11],
                   [1, 2],
                   [1, 4],
                   [1, 6]]

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

hallway = Maze(walls, treasures, snake_pits, start_positions, 2, 11, observations, actions, default_reward=-0.1,
               treasure_reward=32,
               snake_penalty=-3)
