import networkx as nx
import os
import numpy as np
from pysmiles import read_smiles
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
import viz as vz


""" utils """
def getPairs(n: int):
    return list(combinations(range(n), 2))


""" filter dictionary of graphs of int nodeCount : [graph] by specified lower bound lim """
def filterGraphsByNodeCount(Gs: dict, lim: int = 5):
    return {k : v for k, v in Gs.items() if k >= lim}


""" get a dictionary of keys with node counts and values with lists of graph """
def sortGraphsByNodeCount(Gs: dict):
    nodeCounts = set([len(g) for g in Gs])
    counts = {n: [] for n in nodeCounts}
    for g in Gs:
        counts[len(g)].append(g)
    return counts


def getRandomIdx(size: int):
    return np.random.randint(0, size)

def getRandomIdxs(size: int, count: int = 10):
    return [getRandomIdx(size) for _ in range(count)]


""" return False if the two graphs are definitely not isomorphic """
def isIsomorphic(G1: nx.classes.graph.Graph, G2: nx.classes.graph.Graph):
    return nx.is_isomorphic(G1, G2)


class MoleculeDataset():
    def __init__(self, dname: str, numFilesToRead=None):
        self.dname = dname
        self.numFilesToRead = numFilesToRead
        self.loadData()
        self.readSmileFiles()
        self.size = len(self.smiles)
        print(f"done initializing, size: {self.size}")

    def loadData(self):
        if self.numFilesToRead is not None:
            self.smifiles = os.listdir(os.path.join(os.getcwd(), "data", self.dname))[:self.numFilesToRead]
        else:
            self.smifiles = os.listdir(os.path.join(os.getcwd(), "data", self.dname))

    def readSmileFiles(self):
        cwd = os.getcwd()
        osj = os.path.join
        count = 0
        smiles = []
        print("reading SMILE files...")
        for file in tqdm(self.smifiles):
            with open(osj(cwd, "data", self.dname, file), 'r') as f:
                smls = f.readlines()
                smls = [s.strip() for s in smls]
                count += len(smls)
                smiles.extend(smls)
        print(f"done loading {count}={len(smiles)} smiles")
        self.smiles = smiles

    """ query utils """
    def getRandomSmile(self):
        return self.smiles[self.getRandomIdx()]




""" read SMILE strings and return the graph adjacency matrix as a numpy matrix"""
def getAdjMat(smile: str):
    return nx.to_numpy_matrix(read_smiles(smile))

""" takes a numpy adjacency matrix and returns a networkX graph """
def getMoleculeGraph(adjmat: np.matrix):
    return nx.from_numpy_matrix(adjmat)

""" color the given graph according to the specified strategy. Return a list
representing the color map of the graph, i.e. the color of each node when drawn. """
def colorGraph(G: nx.classes.graph.Graph, strategy: str ="random_sequential"):
    c = nx.coloring.greedy_color(G, strategy=strategy)
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    map = []
    ks = sorted(c.keys())
    for k in ks:
        map.append(colors[c[k]])
    return map


def run(useGdb13: bool, numberDesignsToLoad: int = 100):

    if useGdb13: # load gdb13
        molecule_dir = "gdb13"
        md = MoleculeDataset(molecule_dir)
    else: # load gdb17
        molecule_dir = "gdb17"
        numFilesToRead = 20
        md = MoleculeDataset(molecule_dir, numFilesToRead=numFilesToRead)

    idxs = getRandomIdxs(md.size, count=numberDesignsToLoad)
    print("getting adjacency matrices...")
    adjacencyMats = [getAdjMat(md.smiles[idx]) for idx in tqdm(range(len(idxs)))]
    print("building graphs...")
    graphs = [getMoleculeGraph(a) for a in tqdm(adjacencyMats)]

    nodeCounts = [nx.number_of_nodes(g) for g in graphs]
    edgeCounts = [nx.number_of_edges(g) for g in graphs]

    print(f"nodeCounts[:10] {nodeCounts[:10]}")
    print(f"edgeCounts[:10] {edgeCounts[:10]}")

    nodeCounts = sortGraphsByNodeCount(graphs)
    print(nodeCounts.keys())
    filteredNodeCounts = filterGraphsByNodeCount(nodeCounts)
    print(f"node counts after filtering {filteredNodeCounts.keys()}")

    print("filtered node counts summary node count : number of graphs")
    for k, v in filteredNodeCounts.items():
        print(k, len(v))

    print(f"min nodes {min(nodeCounts)} - max nodes {max(nodeCounts)}")
    print(f"min edges {min(edgeCounts)} - max edges {max(edgeCounts)}")

    # draw 1 graph
    vz.drawGraph(graphs[getRandomIdx(numberDesignsToLoad)])
    # vz.makeGraphGrid(nodeCountsDict, cols=8, useGdb13=useGdb13)

    # draw multiple graphs
    #num = 3
    #vz.drawGraph([graphs[idx] for idx in getRandomIdxs(numberDesignsToLoad, num)])

    # draw grid
    vz.makeGraphGrid(filteredNodeCounts, useGdb13=useGdb13, save=False)



# driver
if __name__ == "__main__":
    useGdb13 = True # if False, loads 20/500 files from GDB17 by default
    numberDesignsToLoad = 500

    run(useGdb13)
