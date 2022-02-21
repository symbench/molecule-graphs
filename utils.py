import os
import networkx as nx
import graph_tool.all as gt
import time
from pygraphml import GraphMLParser
from pygraphml import Graph


DATA_DIR = os.path.join(os.getcwd(), "data")
GDB17_DIR = os.path.join(DATA_DIR, "gdb17")
COMP_DIR = os.path.join(DATA_DIR, "components")


def splitFile(fname, lpf=100000):
    fpath = os.path.join(GDB17_DIR, fname)
    small = None
    with open(fpath) as large:
        for num, line in enumerate(large):
            if num % lpf == 0:
                if small:
                    small.close()
                small_fname = os.path.join(GDB17_DIR, f'{num + lpf}.smi')
                small = open(small_fname, "w")
            small.write(line)
        if small:
            small.close()


def readGraphmlNx(fpath):
    sgt = time.time()
    g = gt.load_graph(fpath)
    egt = time.time()
    print(f"loaded graph with graph-tool in {egt-sgt}s")
    vx = []
    ed = []
    for v in g.vertices():
        vx.append(v)
    for e in g.edges():
        ed.append(e)
    print(f"(gt) nodes: {len(vx)} edges: {len(ed)}")

    snx = time.time()
    G = nx.read_graphml(fpath)
    enx = time.time()
    print(f"loaded graph with networkx in {enx-snx}s")
    print(f"(nx) nodes: {len(G.nodes)} edges: {len(G.edges)}")

    pos = gt.sfdp_layout(g)
    gt.graph_draw(g, pos, output_size=(1000, 1000), vertex_color=[1, 1, 1, 0])

def readPygraphml(fname):
	fname = os.path.join(DATA_DIR, fname)
	
	with open(fname) as f:
		print(f.read())
	print("file data above")

	parser = GraphMLParser()
	print("reading graph...")
	g = parser.parse(fname)
	g.show()

def readGraphmlGraphTool(fname):
	fname = os.path.join(DATA_DIR, fname)
	print("loading graph...")
	with open(fname) as f: 
		fls = f.readlines()
	print(fls[:1000])

	g = gt.load_graph(fname)
	oe = g.get_out_edges(g.vertex(0))
	ie = g.get_in_edges(g.vertex(0))
	print("loaded graph")
	for v in list(g.vertices())[:10]:
		print(v)

if __name__ == "__main__":
	readGraphmlGraphTool("all_schema_uam_comps.graphml")
