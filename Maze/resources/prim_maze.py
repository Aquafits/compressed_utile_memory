from maze.Maze import Maze
import numpy as np
import random

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


def generate_prim_maze(map_y_size=None, map_x_size=None, crowding_index=None):
    if map_x_size is None:
        map_x_size = random.choice(range(17, 25))  # 默認情況下，寬度在17～25
    if map_y_size is None:
        map_y_size = random.choice(range(17, 25))  # 默認情況下，高度在17～25
    if crowding_index is None:
        crowding_index = random.random() * 0.35 + 0.65  # 默認情況下，擁擠程度在0.65～1.0

    if map_x_size % 2 == 0:
        map_x_size -= 1

    if map_y_size % 2 == 0:
        map_y_size -= 1

    grid_x_size = int((map_x_size + 1) / 2)
    grid_y_size = int((map_y_size + 1) / 2)

    # 上右下左
    # 1 2 4 8

    grids = np.full((grid_y_size, grid_x_size), -1)

    grid_y = random.choice(range(grid_y_size))
    grid_x = random.choice(range(grid_x_size))

    goal_map_x = grid_x * 2
    goal_map_y = grid_y * 2
    treasures = [[goal_map_y, goal_map_x]]

    grids[grid_y, grid_x] = 15

    edge_grids = set()

    while True:
        if (grid_y > 0) and (grids[grid_y - 1, grid_x] == -1):
            edge_grids.add((grid_y - 1, grid_x))

        if (grid_x > 0) and (grids[grid_y, grid_x - 1] == -1):
            edge_grids.add((grid_y, grid_x - 1))

        if (grid_x < grid_x_size - 1) and (grids[grid_y, grid_x + 1] == -1):
            edge_grids.add((grid_y, grid_x + 1))

        if (grid_y < grid_y_size - 1) and (grids[grid_y + 1, grid_x] == -1):
            edge_grids.add((grid_y + 1, grid_x))

        (grid_y, grid_x) = edge_grids.pop()
        grids[grid_y, grid_x] = 15

        temp_neighbors = set()  # 當前的格子的可能的4個鄰居。

        if (grid_y > 0) and (grids[grid_y - 1, grid_x] != -1):
            temp_neighbors.add((grid_y - 1, grid_x))

        if (grid_x > 0) and (grids[grid_y, grid_x - 1] != -1):
            temp_neighbors.add((grid_y, grid_x - 1))

        if (grid_x < grid_x_size - 1) and (grids[grid_y, grid_x + 1] != -1):
            temp_neighbors.add((grid_y, grid_x + 1))

        if (grid_y < grid_y_size - 1) and (grids[grid_y + 1, grid_x] != -1):
            temp_neighbors.add((grid_y + 1, grid_x))

        while True:
            (neighbor_y, neighbor_x) = temp_neighbors.pop()

            if neighbor_y == grid_y - 1:  # 樓上鄰居
                grids[neighbor_y, neighbor_x] -= 4
                grids[grid_y, grid_x] -= 1

            if neighbor_x == grid_x + 1:  # 右邊鄰居
                grids[neighbor_y, neighbor_x] -= 8
                grids[grid_y, grid_x] -= 2

            if neighbor_y == grid_y + 1:  # 樓下鄰居
                grids[neighbor_y, neighbor_x] -= 1
                grids[grid_y, grid_x] -= 4

            if neighbor_x == grid_x - 1:  # 左邊鄰居
                grids[neighbor_y, neighbor_x] -= 2
                grids[grid_y, grid_x] -= 8

            if len(temp_neighbors) == 0:
                break
            elif random.random() < crowding_index:
                break

        # print(grids)
        if len(edge_grids) == 0:
            break

    map = np.zeros((map_y_size, map_x_size), dtype=int)

    for grid_y in range(grid_y_size):
        for grid_x in range(grid_x_size):
            value = grids[grid_y, grid_x]
            if value >= 8:  # 有左邊的牆
                value -= 8
                if grid_x > 0:  # 除去最左邊一列
                    map[grid_y * 2, grid_x * 2 - 1] = 1  # 給左邊建立一道牆壁
                if grid_y > 0 and grid_x > 0:  # 給左上角建立一道
                    map[grid_y * 2 - 1, grid_x * 2 - 1] = 1
                if grid_x > 0 and grid_y * 2 + 1 < map_y_size:  # 給左下角建立一道
                    map[grid_y * 2 + 1, grid_x * 2 - 1] = 1

            if value >= 4:  # 有下邊的牆
                value -= 4
                if grid_x > 0 and grid_y * 2 + 1 < map_y_size:  # 給左下角建立一道
                    map[grid_y * 2 + 1, grid_x * 2 - 1] = 1
                if grid_x * 2 + 1 < map_x_size and grid_y * 2 + 1 < map_y_size:  # 給右下角建立一道
                    map[grid_y * 2 + 1, grid_x * 2 + 1] = 1

            if value >= 2:  # 有右邊的牆
                value -= 2
                if grid_y > 0 and grid_x * 2 + 1 < map_x_size:  # 給右上角建立一道
                    map[grid_y * 2 - 1, grid_x * 2 + 1] = 1
                if grid_x * 2 + 1 < map_x_size and grid_y * 2 + 1 < map_y_size:  # 給右下角建立一道
                    map[grid_y * 2 + 1, grid_x * 2 + 1] = 1

            if value >= 1:  # 有上邊的牆
                value -= 1
                if grid_y > 0:  # 除去最上邊一列
                    map[grid_y * 2 - 1, grid_x * 2] = 1  # 給上邊建立一道牆壁
                if grid_y > 0 and grid_x > 0:  # 給左上角也建立一道
                    map[grid_y * 2 - 1, grid_x * 2 - 1] = 1
                if grid_y > 0 and grid_x * 2 + 1 < map_x_size:  # 給右上角建立一道
                    map[grid_y * 2 - 1, grid_x * 2 + 1] = 1

    # print(map)

    # 打印并且紀錄 walls
    walls = []
    for map_y in range(map_y_size):
        for map_x in range(map_x_size):
            if map[map_y, map_x] == 0:  # 道路
                if (map_y, map_x) == (goal_map_y, goal_map_x):  # 如果是目標終點
                    print('\033[42mGL', end="")
                    print('\033[0m', end="")
                else:
                    print('\033[47m  ', end="")
                    print('\033[0m', end="")
            else:  # 牆壁
                walls.append([map_y, map_x])
                print('\033[41m  ', end="")
                print('\033[0m', end="")
        print()
    print()

    [y, x] = walls[0]
    while [y, x] in walls or [y, x] in snake_pits or [y, x] in treasures:
        y = random.randint(0, map_y_size - 1)
        x = random.randint(0, map_x_size - 1)

    start_positions = [[y, x]]

    return Maze(walls, treasures, snake_pits, start_positions, map_y_size, map_x_size, observations, actions, default_reward=-0.1,
                treasure_reward=32)


prim_maze = generate_prim_maze(13, 20, 0.8)
