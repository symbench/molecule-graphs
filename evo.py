import networkx as nx
import numpy as np
from abc import ABC, abstractmethod
from typing import List
import itertools
from pysmiles import read_smiles
import random
from tqdm import tqdm
import os
import viz as vz


class MoleculeDataset():
    def __init__(self, dname: str, numFilesToRead=None, batchSize: int = 512):
        self.dname = dname
        self.numFilesToRead = numFilesToRead
        self.loadData()
        self.readSmileFiles()
        self.totalSize = len(self.smiles)
        self.batchSize = batchSize
        self.loadBatch()
        print(f"done initializing, size: {self.totalSize}")

    def loadData(self):
        if self.numFilesToRead is not None:
            self.smifiles = os.listdir(os.path.join(
                os.getcwd(), "data", self.dname))[:self.numFilesToRead]
        else:
            self.smifiles = os.listdir(os.path.join(
                os.getcwd(), "data", self.dname))

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
        self.smiles = smiles

    def _randomIdx(self):
        return np.random.randint(0, self.totalSize)

    def _randomIdxs(self):
        return [self._randomIdx() for _ in range(self.batchSize)]

    def loadBatch(self):
        idxs = self._randomIdxs()
        batch = []
        print("loading batch...")
        for i in tqdm(idxs):
            batch.append(MGraph(self.smiles[i]))
        return batch


class Individual(ABC):

    @abstractmethod
    def pair(self, other):
        pass

    @abstractmethod
    def mutate(self):
        pass


class MGraph(Individual):

    newid = itertools.count()

    def __init__(self, smile: str = None, adjmat: np.ndarray = None):
        self.id = next(self.newid)
        if smile is not None:
            self.smile = smile
            self.loadAdjmat()
        elif adjmat is not None:
            self.adjmat = adjmat

    def loadAdjmat(self):
        self.adjmat = nx.to_numpy_matrix(read_smiles(self.smile))

    def _getGraph(self):
        return nx.from_numpy_matrix(self.adjmat)

    def _getNodes(self):
        return list(self._getGraph.nodes)

    def _getEdges(self):
        return list(self._getGraph.edges)

    def connectedComponents(self):
        return nx.number_connected_components(self._getGraph)

    def removeNode(self, count: int = 1):
        G = self._getGraph()
        nodesToRemove = [random.randint(0, len(G.nodes)) for _ in range(count)]
        print(f"removing nodes {nodesToRemove}")
        G.remove_nodes_from(nodesToRemove)
        self.adjmat = nx.to_numpy_array(G)
        self.removeIsolates()

    def removeIsolates(self):
        G = self._getGraph()
        G.remove_nodes_from(list(nx.isolates(G)))
        self.adjmat = nx.to_numpy_array(G)

    def edgeSwap(self, count: int = 1):
        self.adjmat = nx.to_numpy_array(
            nx.double_edge_swap(self._getGraph(), nswap=count))

    def mutate(self, p=0.5):
        r = random.random()
        if r > p:
            self.removeNode()
        else:
            self.edgeSwap()

    def pair(self, other: np.ndarray):
        return MGraph(adjmat=nx.to_numpy_array(
            nx.disjoint_union(self._getGraph(), other._getGraph())))


class Population:

    def __init__(self, individuals: List, n_offspring):
        self.individuals = individuals
        self.size = len(individuals)
        self.n_offspring = n_offspring
        self.calculateFitness()

    def calculateFitness(self):

        def getFitness(G: nx.classes.graph.Graph):
            avDeg = sum([v for (n, v) in G.degree()]) / len(G.nodes)
            diam = nx.diameter(G)
            conc = nx.number_connected_components(G)
            shorp = nx.average_shortest_path_length(G)
            # add max node centrality, find more

            return avDeg + conc + (1 / diam) + (1 / shorp)

        fits = []
        for i in self.individuals:
            g = i._getGraph()
            fits.append(getFitness(g))

        self.fitness = fits

    def getFittest(self, topk: int = 64):
        idx = np.argsort(self.fitness)[::-1][:topk]
        return [self.individuals[i] for i in idx]

    def replace(self):
        pass

    def getParents(self):
        mothers = self.individuals[-2 * self.n_offspring::2]
        fathers = self.individuals[-2 * self.n_offspring+1::2]

        return mothers, fathers


class Evolution:

    def __init__(self, individuals: List, n_offspring: int = 2,
                    epochs: int = 10):  # noqa: E127
        self.pool = Population(individuals=individuals,
                               n_offspring=n_offspring)
        self.n_offspring = n_offspring
        self.epochs = epochs
        self.generations = []

    def step(self):
        mothers, fathers = self.pool.getParents()
        offspring = []

        for mother, father in zip(mothers, fathers):
            child = mother.pair(father)
            offspring.append(child)
        #  todo implement this
        self.pool.replace(offspring)


# driver
if __name__ == "__main__":
    useGdb13 = True
    # if False, loads 20/500 files from GDB17 by default
    if useGdb13:
        molecule_dir = "gdb13"
        md = MoleculeDataset(molecule_dir, batchSize=256)
    else:
        molecule_dir = "gdb17"
        numFilesToRead = 20
        md = MoleculeDataset(
            molecule_dir, numFilesToRead=numFilesToRead, batchSize=256)

    b1 = md.loadBatch()
    print(type(b1), type(b1[0]))
    print(b1[0].id)

    ev = Evolution(b1)
    ev.step()
    # g1 = b1[0]._getGraph()
    # b1[0].removeNode(count=3)
    # g1node = b1[0]._getGraph()
    # b1[0].edgeSwap(count=2)
    # g1edge = b1[0]._getGraph()

    # b2 = md.loadBatch()
    # g2 = b2[0]._getGraph()
    # # b2[0].removeNode(count=3)
    # # g2node = b2[0]._getGraph()
    # # b2[0].edgeSwap(count=2)
    # # g2edge = b2[0]._getGraph()
    # rep = b1[0].pair(b2[0])._getGraph()

    # vz.drawGraph([g1, g2, rep], labels=["g1", "g2", "rep"])
