import itertools
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Callable, List

import networkx as nx
import numpy as np
from pysmiles import read_smiles
from tqdm import tqdm

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
        #  print( f"rdkit call from smile to mol returns {type(Chem.MolFromSmiles(self.smile))}")
        self.adjmat = nx.to_numpy_matrix(read_smiles(self.smile))

    def _getGraph(self):
        return nx.from_numpy_matrix(self.adjmat)

    def _getNodes(self):
        return list(self._getGraph().nodes)

    def _getEdges(self):
        return list(self._getGraph().edges)

    def connectedComponents(self):
        return nx.number_connected_components(self._getGraph())

    def removeNode(self):
        G = self._getGraph()
        count = random.randint(0, len(G.nodes) // 3)
        nodesToRemove = [random.randint(0, len(G.nodes)) for _ in range(count)]
        print(f"removing nodes {nodesToRemove}")
        G.remove_nodes_from(nodesToRemove)
        self.adjmat = nx.to_numpy_array(G)

    def removeIsolates(self):
        G = self._getGraph()
        iso = list(nx.isolates(G))
        if len(iso) > 0:
            G.remove_nodes_from(iso)
            self.adjmat = nx.to_numpy_array(G)

    def removeEdge(self):
        G = self._getGraph()
        count = random.randint(0, len(G.edges) // 4)
        edgesToRemove = set([random.choice(list(G.edges)) for _ in range(count)])
        if len(edgesToRemove) > 0:
            G.remove_edges_from(edgesToRemove)
            self.adjmat = nx.to_numpy_array(G)

    def edgeSwap(self, count: int = 4):
        self.adjmat = nx.to_numpy_array(
            nx.double_edge_swap(self._getGraph(), nswap=count))

    def mutate(self, p=0.5):
        r = random.random()
        if r > p:
            self.removeNode()
        else:
            # self.edgeSwap()
            self.removeEdge()
        self.removeIsolates()

    def pair(self, other: np.ndarray):
        return MGraph(adjmat=nx.to_numpy_array(
            nx.disjoint_union(self._getGraph(), other._getGraph())))

class Population:

    def __init__(self, fitness: Callable, individuals: List, n_offspring: int):
        self.fitness = fitness
        self.individuals = individuals
        self.individuals.sort(key=lambda x: self.fitness(x._getGraph()))
        self.size = len(individuals)
        self.n_offspring = n_offspring

    def replace(self, newIndividuals: List):
        self.individuals.extend(newIndividuals)
        self.individuals.sort(key=lambda x: self.fitness(x._getGraph()))
        self.individuals = self.individuals[-self.size:]

    def getParents(self):
        mothers = self.individuals[-2 * self.n_offspring::2]
        fathers = self.individuals[-2 * self.n_offspring+1::2]

        return mothers, fathers


class Evolution:

    def __init__(self, fitness: Callable, individuals: List, n_offspring: int = 2):
        self.pool = Population(fitness, individuals,
                               n_offspring)
        self.n_offspring = n_offspring

    def step(self):
        mothers, fathers = self.pool.getParents()
        offspring = []

        for mother, father in zip(mothers, fathers):
            child = mother.pair(father)
            child.mutate()
            offspring.append(child)
        #  todo implement this
        self.pool.replace(offspring)

    def getTopK(self, k: int):
        g = [self.pool.individuals[i]._getGraph() for i in range(k)]
        f = [round(self.pool.fitness(g[i]), 4) for i in range(k)]
        return g, f



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

    b = md.loadBatch()

    def fitness(G: nx.classes.graph.Graph):
        connected = nx.is_connected(G)
        avDeg = sum([v for (_, v) in G.degree()]) / len(G.nodes)
        diam = nx.diameter(G) if connected else -5
        conc = nx.number_connected_components(G)
        shorp = nx.average_shortest_path_length(G) if connected else -5
        # add max node centrality, find more

        return avDeg + conc + (1 / diam) + (1 / shorp)

    print("initial")
    vz.drawGraph([b[i]._getGraph() for i in range(5)])

    ev = Evolution(fitness, b) # can do this in parallel?
    numEpochs = 5
    for i in range(numEpochs):
        s = time.time()
        ev.step()
        e = time.time()
        g, f = ev.getTopK(5)
        vz.drawGraph(g, labels=f)
        print(f"after {i+1} generation (pool size: {ev.pool.size})")
        print(f"step {i+1} complete in: {round(e-s, 4)}s")
    
