import os
from platform import node
import random
import uuid
from typing import List, Union

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import networkx as nx
import numpy as np


def getColorMap(G: nx.classes.graph.Graph):
    coloring = nx.coloring.greedy_color(G)
    return [coloring[x] for x in sorted(coloring.keys())]


def drawGraph(G: Union[List, nx.classes.graph.Graph], nodepos: Union[List, np.ndarray], labels: List = None, title=None):

    if isinstance(G, nx.classes.graph.Graph):
        if nodepos is not None:
            #npos = {k: v for k, v in zip(list(G.nodes), nodepos)}
            nx.draw(G, node_color=getColorMap(G), pos=nodepos)
        if title is not None:
            plt.suptitle(title)
    elif isinstance(G, list):
        if labels is not None:
            assert len(G) == len(labels)
        if nodepos is not None:
            assert len(G) == len(nodepos)
            # positions = []
            # for i in range(len(G)):
            #     nodes = list(G[i].nodes)
            #     nodepositions = nodepos[i]
            #     npos = {k: v for k, v in zip(nodes, nodepositions)}  # keys are nodes, values are positions
            #     positions.append(npos)
            fig, all_axes = plt.subplots(1, len(G))
            ax = all_axes.flat
            for i in range(len(G)):
                nx.draw(G[i], ax=ax[i], node_color=getColorMap(G[i]), pos=nodepos[i])
                if labels:
                    ax[i].set_title(round(labels[i], 4))
        if title is not None:
            plt.suptitle(title)
    plt.show()


def makeGrid(data: List[nx.classes.graph.Graph], nodepos: List,  rows: int = 4, cols: int = 4, save: bool = True):
    assert len(data) == rows * cols
    assert len(nodepos) == len(data)
    fig, all_axes = plt.subplots(rows, cols)
    ax = all_axes.flat

    idx = 0
    axi = 0
    for _ in range(rows):
        for _ in range(cols):
            nx.draw(data[idx], pos=nodepos[idx], ax=ax[axi], node_size=20, node_color=getColorMap(data[idx]))
            axi += 1
            idx += 1
    fig.tight_layout()
    if save:
        print("saving")
        plt.savefig(os.path.join("img", uuid.uuid4().hex[:10] + ".png"))
    plt.show()


def vizMotorPropellerPairings():
	pass 

def vizMotorPropertiesVsPropDiameter():
	pass 



def showFitness(fits: List[Union[float, List]]):
    assert isinstance(fits, list)
    for f in fits:
        plt.plot(f)
    plt.legend([f"{i+1}" for i in range(len(fits))], loc='upper left')
    plt.suptitle(f"Top {len(fits[0])} fitness over {len(fits)} generations")
    plt.show()


def hist(data: List, title: str = None):
    if title is not None:
        plt.suptitle(f"{title}, n={len(data)}")
    plt.hist(data)
    plt.ylabel("count")
    plt.show()

# def viewSmile(smile: str):
#     template = Chem.MolFromSmiles('c1nccc2n1ccc2')
#     AllChem.Compute2DCoords(template)
#     ms = [Chem.MolFromSmiles(smi) for smi in (
#         'OCCc1ccn2cnccc12', 'C1CC1Oc1cc2ccncn2c1', 'CNC(=O)c1nccc2cccn12')]
#     for m in ms:
#         _ = AllChem.GenerateDepictionMatching2DStructure(m, template)
