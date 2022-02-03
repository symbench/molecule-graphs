import os
import random
import uuid
from typing import List, Union

import matplotlib.pyplot as plt
import networkx as nx

def getColorMap(G: nx.classes.graph.Graph):
    coloring = nx.coloring.greedy_color(G)
    return [coloring[x] for x in sorted(coloring.keys())]



def drawGraph(G: Union[List, nx.classes.graph.Graph], labels: List = None,
              node_color: List = None):

    if isinstance(G, nx.classes.graph.Graph):
        nx.draw(G, node_color=node_color)
        plt.show()
    elif isinstance(G, list):
        if labels is not None:
            assert len(G) == len(labels)
        fig, all_axes = plt.subplots(1, len(G))
        ax = all_axes.flat
        for i in range(len(G)):
            # nx.draw_shell(G[i], ax=ax[i], node_size=20, with_labels=True)
            # nx.draw_planar(G[i], ax=ax[i], node_size=20, node_color=getColorMap(G[i]))
            # nx.draw_circular(G[i], ax=ax[i], node_size=20, with_labels=False) 
            nx.draw_circular(G[i], ax=ax[i], node_size=50, node_color=getColorMap(G[i]))
            if labels:
                ax[i].set_title(labels[i])
    plt.show()


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


def showFitness(fits: List[Union[float, List]]):
	assert isinstance(fits, list)
	if isinstance(fits[0], list):
		for f in fits:
			plt.plot(f)
	plt.legend([f"gen {i}" for i in range(len(fits))], loc='upper left')
	plt.show()

# def viewSmile(smile: str):
#     template = Chem.MolFromSmiles('c1nccc2n1ccc2')
#     AllChem.Compute2DCoords(template)
#     ms = [Chem.MolFromSmiles(smi) for smi in (
#         'OCCc1ccn2cnccc12', 'C1CC1Oc1cc2ccncn2c1', 'CNC(=O)c1nccc2cccn12')]
#     for m in ms:
#         _ = AllChem.GenerateDepictionMatching2DStructure(m, template)
