import matplotlib.pyplot as plt
import random
import networkx as nx
import os
import uuid
from typing import Union

def drawGraph(G: Union[list, nx.classes.graph.Graph], node_color: list =None):

    if isinstance(G, nx.classes.graph.Graph):
        nx.draw(G, node_color=node_color)
        plt.show()
    elif isinstance(G, list):
        for i in range(len(G)):
            plt.figure(i)
            nx.draw(G[i], node_color=node_color)
    plt.show()


""" create a grid of graphs with rows increasing node count """
def makeGraphGrid(data: dict, rows: int = 4, cols: int = 4, useGdb13: bool = True, save: bool = True):
    fig, all_axes = plt.subplots(rows, cols)
    ax = all_axes.flat
    print(data.keys())

    if useGdb13:
        nodeA = [random.choice(data[5]) for _ in range(cols)]
        nodeB = [random.choice(data[6]) for _ in range(cols)]
        nodeC = [random.choice(data[7]) for _ in range(cols)]
        nodeD = [random.choice(data[11]) for _ in range(cols)]
    else:
        nodeA = [random.choice(data[14]) for _ in range(cols)]
        nodeB = [random.choice(data[15]) for _ in range(cols)]
        nodeC = [random.choice(data[16]) for _ in range(cols)]
        nodeD = [random.choice(data[17]) for _ in range(cols)]

    axi = 0
    for x in [nodeA, nodeB, nodeC, nodeD]:
        for y in range(cols):
            nx.draw(x[y], ax=ax[axi], node_size=10)
            axi += 1

    for a in ax:
        a.margins(0.40)
    fig.tight_layout()
    if save:
        print("saving")
        plt.savefig(os.path.join("img", uuid.uuid4().hex[:10] + ".png"))
    plt.show()
