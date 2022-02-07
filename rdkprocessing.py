import networkx as nx
import numpy as np
from rdkit import Chem, Avalon
from rdkit.Chem import AllChem, rdmolops
import matplotlib.pyplot as plt

import evo
import random
import hovercalc2 as hc


def getNetXgraphFromMol(mol: Chem.Mol):
	return nx.from_numpy_matrix(getAdjMatFromMol(mol))


#  todo mutations in evolution now include change node positioning
def drawGraphWithCoords(G, coords):
	nx.draw(G, pos=coords)


def get2DCoordsFromMol(mol):
	idx = AllChem.Compute2DCoords(mol)
	return mol.GetConformer(idx).GetPositions()
	

def getMolFromSmileRDK(smile):
	return Chem.MolFromSmiles(smile)


def normCoords(coords):
	# consider returning a list of tuples of (x, y)
	return (coords/np.linalg.norm(coords))[:, 0:2]  # x, y


def getAdjMatFromMol(mol):
	return rdmolops.GetAdjacencyMatrix(mol)


def getAtomsFromMol(mol):
	return mol.GetAtoms()


if __name__ == "__main__":
	moldata = evo.loadMoleculeDataset("gdb13rand1M")
	batch = moldata.loadBatch()
	smile = batch[random.choice(range(len(batch)))].smile
	molecule = getMolFromSmileRDK(smile)
	G = getNetXgraphFromMol(molecule)
	coords = normCoords(get2DCoordsFromMol(molecule))
	drawGraphWithCoords(G, coords)
	plt.show()