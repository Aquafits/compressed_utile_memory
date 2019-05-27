from Usm import Maze
from UsmAgent import UsmAgent
from resources.demo_maze import usm_demo_maze
from resources.hallway_2 import hallway_2


class Runner:
    def __init__(self, maze: Maze, iters=2000):
        self.agent = UsmAgent(maze)
        self.iters = iters

    def run(self):
        self.agent.iterate(self.iters)


demo_runner = Runner(hallway_2)
demo_runner.run()
