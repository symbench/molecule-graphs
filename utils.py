import os
import networkx as nx
import graph_tool.all as gt
import time


DATA_DIR = os.path.join(os.getcwd(), "data")
GDB17_DIR = os.path.join(DATA_DIR, "gdb17")


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


def readGraphml(fpath):
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


gp = os.path.join(DATA_DIR, "quadcopter-1.graphml")
readGraphml(gp)

#gdb17 = "GDB17.50000000.smi"
# splitFile(gdb17)
