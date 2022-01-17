# Molecule Graph Exploration

This repo contains molecule data from [GDB-13](https://pubs.acs.org/doi/abs/10.1021/ja902302h) and [GDB-17](https://pubs.acs.org/doi/abs/10.1021/ci300415d) in efforts to design interesting topologies primarily for the UAV problem.

Each molecule is represented as a string, one per line, in the [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) format. These molecules are loaded into a `MoleculeDataset` then interpreted as NetworkX graphs for visualization and manipulation.

For convenience, this repo also contains the (updated?) BEMP component data used in the current SwRI toolchain version.


## Usage

1. from a python 3 environment, run `pip install -r requirements.txt` (note: conda does not work here since it does not support a required package, [pysmiles](https://pypi.org/project/pysmiles/), in its index)
2. run `python molecules.py` to read the specified dataset and visualize sample graphs from the resulting graphs

## Todo

- brainstorm graph descriptors that admit "feasible" UAV designs (e.g. connectivity, centrality, rewiring, dropout, symmetry)
- dissect SwRI design graph for existing blueprints to understand if/how vehicle geometry is represented in the expected design format
- formulate evolutionary approaches revolving around structured manipulation to molecule graphs to encourage "good" UAV designs
- develop glue code from graph generation method to take a vehicle shape, scale selection, and candidate BMP components as input and produce a SwRI design graph as output.
- maybe: revisit hovercalc python implementation and parallelize, investigate degree to which vehicle geometry is considered in this static simulation and how to accommodate multiple propellers of differing orientation.
- later: CAD tool issue where component orientation (joint location) limits vehicle design diversity, develop surrogate or workaround for this (since the SwRI toolchain requires a valid CAD assembly of the vehicle to test).
