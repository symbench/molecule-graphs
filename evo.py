import itertools
import os
import random
from re import S
import time
from abc import ABC, abstractmethod
from typing import Callable, List

import networkx as nx
import numpy as np
from pysmiles import read_smiles
from tqdm import tqdm
import matplotlib.pyplot as plt

import rdkprocessing as rdkp

import viz as vz


class MoleculeDataset():
    def __init__(self, dname: str, numFilesToRead=None, batchSize: int = 512):
        self.dname = dname
        self.numFilesToRead = numFilesToRead
        self.loadData()
        self.readSmileFiles()
        self.totalSize = len(self.smiles)
        self.batchSize = batchSize
        # self.loadBatch()
        print(f"done initializing, size: {self.totalSize}")

    def loadData(self):
        if self.numFilesToRead is not None:
            self.smifiles = os.listdir(os.path.join(
                os.getcwd(), "data", self.dname))[:self.numFilesToRead]
        else:
            self.smifiles = os.listdir(os.path.join(
                os.getcwd(), "data", self.dname))
        self.smifiles.remove(
            ".DS_Store") if ".DS_Store" in self.smifiles else self.smifiles

    def readSmileFiles(self):
        print(f"reading from: {self.smifiles}")
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
            batch.append(self.smiles[i])
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

    # refactor to only include adjacency matrix and node positions
    def __init__(self, adjmat: np.ndarray, nodepos: List):
        self.id = next(self.newid)
        # assign class variables of adjmat and node positions
        self.adjmat = adjmat
        self.origpos = nodepos
        self.curpos = nodepos

    # def loadAdjmat(self):
    #     #  todo - adjust this to use the representation from rdkprocessing
    #     self.adjmat = nx.to_numpy_matrix(read_smiles(self.smile))

    def _getGraph(self):
        return nx.from_numpy_matrix(self.adjmat)

    def _getNodes(self):
        return list(self._getGraph().nodes)

    def _getEdges(self):
        return list(self._getGraph().edges)

    def _getNodePos(self):
        return self.curpos
    
    def _getOrigNodePos(self):
        return self.origpos

    def connectedComponents(self):
        return nx.number_connected_components(self._getGraph())

    def removeNode(self):
        #  remove nodes
        G = self._getGraph()
        pos = self.curpos
        count = random.randint(0, len(G.nodes) // 3)
        nodesToRemove = set([random.randint(0, len(G.nodes)) for _ in range(count)])
        print(f"removing nodes {nodesToRemove}")
        G.remove_nodes_from(nodesToRemove)


        remainingNodes = set(list(G.nodes)) - nodesToRemove
        print(f"remaining nodes {remainingNodes}")
        self.adjmat = nx.to_numpy_array(G)
        newpos = {i: None for i in remainingNodes}

        # todo - reassign node ids, how to deal with positions of added nodes?
        # for i, node in enumerate(remainingNodes):
        #     print(f"i: {i}, node: {node}")
        #     newpos[node] = pos[i]
        # print(f"newpos {newpos}")
        # self.curpos = newpos


    def removeIsolates(self):
        G = self._getGraph()
        iso = list(nx.isolates(G))
        if len(iso) > 0:
            G.remove_nodes_from(iso)
            self.adjmat = nx.to_numpy_array(G)

    def removeEdge(self):
        G = self._getGraph()
        count = random.randint(0, len(G.edges) // 3)
        edgesToRemove = set([random.choice(list(G.edges))
                            for _ in range(count)])
        if len(edgesToRemove) > 0:
            G.remove_edges_from(edgesToRemove)
            self.adjmat = nx.to_numpy_array(G)

    def edgeSwap(self, count: int = 2):
        self.adjmat = nx.to_numpy_array(
            nx.double_edge_swap(self._getGraph(), nswap=count))

    def nudgeNodes(self, count: int = 4, sigma: float = 0.1):
        orig = self.curpos
        idx = random.sample(range(len(orig)), count)
        mu = 0
        xd = np.random.normal(mu, sigma, count)
        yd = np.random.normal(mu, sigma, count)
        print(orig)
        print(idx)
        print(xd)
        print(yd)
        for i, j in enumerate(idx):
            print(i, j)
            orig[j][0] -= xd[i]
            orig[j][1] -= yd[i]
        self.curpos = orig


    #  todo provide larger set of mutation parameters
    def mutate(self, p=0.5):
        r = random.random()
        if r > p:
            self.removeNode()
        else:
            self.removeEdge()
            #self.edgeSwap()
        #self.nudgeNodes()
        self.removeIsolates()

    def pair(self, other: np.ndarray):

        # update nodepositions
        minSize = min(len(self.curpos), len(other.curpos))
        selfpos = self.curpos[:minSize]
        otherpos = other.curpos[:minSize]
        childpos = np.mean([selfpos, otherpos], axis=0)
        print(f"par1 pos: {len(selfpos)}, par2 pos: {len(otherpos)}, child pos: {len(childpos)}")
        print(childpos)

        return MGraph(adjmat=nx.to_numpy_array(
            nx.disjoint_union(self._getGraph(), other._getGraph())), nodepos=childpos)


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
        # todo
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

    def getAllIndividuals(self):
        ind = [self.pool.individuals[i]._getGraph() for i in range(self.pool.size)]
        pos = [self.pool.individuals[i]._getNodePos() for i in range(self.pool.size)]
        return ind, pos

    def getAllFitnesses(self, graphs):
        return [self.pool.fitness(g) for g in graphs]

    def getTopK(self, k: int):
        g = self.getAllIndividuals()
        f = self.getAllFitnesses(g)
        return g[:k], f[:k]


def loadMoleculeDataset(mdir: str, batchSize: int = 256) -> MoleculeDataset:
    return MoleculeDataset(mdir, batchSize)


# driver
if __name__ == "__main__":
    batchSize = 256
    molset = loadMoleculeDataset("gdb13rand1M", batchSize=batchSize)
    smiles = molset.loadBatch()
    smiles = [smiles[random.choice(range(len(smiles)))] for _ in range(batchSize)]
    molecules = list(map(rdkp.getMolFromSmileRDK, smiles))
    adjmats = list(map(rdkp.getAdjMatFromMol, molecules))
    graphs = list(map(rdkp.getNetXgraphFromMol, molecules))

    # nodeCounts = [len(g.nodes) for g in graphs]
    # edgeCounts = [len(g.edges) for g in graphs]
    #vz.hist(edgeCounts, title="edge counts")
    
    coords = map(rdkp.get2DCoordsFromMol, molecules)
    coords = list(map(rdkp.normCoords, coords))
    #vz.makeGrid(graphs[:16], coords[:16], save=False)

    def fitness(G: nx.classes.graph.Graph):
        connected = nx.is_connected(G)
        avDeg = sum([v for (_, v) in G.degree()]) / len(G.nodes)
        diam = nx.diameter(G) if connected else -5
        conc = nx.number_connected_components(G)
        shorp = nx.average_shortest_path_length(G) if connected else -5
        # add max node centrality, find more
        # todo span area
        # symmetry

        return avDeg + conc + (1 / diam) + (1 / shorp)

    # print("initial")
    # vz.drawGraph([b[i]._getGraph() for i in range(5)])
        
    # sample = evolution.pool.individuals[:16]
     
    # example = sample[-4:]
    # exg = [e._getGraph() for e in example]
    # pos = [e.curpos for e in example]
    # print(pos)
    # for e in example:
    #     e.mutate()
    # mut1 = [e._getGraph() for e in example]
    # m1pos = [e.curpos for e in example]
    # print(m1pos)
    # for e in example:
    #     e.mutate()
    # mut2 = [e._getGraph() for e in example]
    # m2pos = [e.curpos for e in example]
    # for e in example:
    #     e.mutate()
    # mut3 = [e._getGraph() for e in example]
    # m3pos = [e.curpos for e in example]
    
    # vz.makeGrid(exg + mut1 + mut2 + mut3, pos + m1pos + m2pos + m3pos, save=True)

    
    # do evolution
    evolution = Evolution(fitness, [MGraph(adj, nodep) for (adj,nodep) in zip(adjmats, coords)])  # can do this in parallel?
    numEpochs = 3
    fits = []
    sampleSize = 5
    for i in range(numEpochs):
        start = time.time()
        evolution.step()
        end = time.time()
        individuals, positions = evolution.getAllIndividuals()
        fitnesses = sorted(evolution.getAllFitnesses(individuals))  # max at higher idx
        print(f"FITS {fitnesses[-sampleSize:]}")
        fits.append(fitnesses[-sampleSize:])
        samples = individuals[-sampleSize:]
        positions = positions[-sampleSize:]
        vz.drawGraph(samples, positions, labels=fitnesses[-sampleSize:])
        print(f"after {i+1} generation (pool size: {evolution.pool.size})")
        print(f"step {i+1} complete in: {round(end-start, 4)}s")
    vz.showFitness(fits)
