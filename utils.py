import os
import networkx as nx

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
    print("reading graph")
    G = nx.read_graphml(fpath)
    print(nx.info(G))
    print(f"number_connected_components: {nx.number_connected_components(G)}")


"}"    # print("drawing graph")
    # n = nx.draw_networkx_nodes(G, pos=nx.spring_layout(G))
    # this above command is taking a long time


gp = os.path.join(DATA_DIR, "all_schema_UAV.graphml")
readGraphml(gp)

#gdb17 = "GDB17.50000000.smi"
#splitFile(gdb17)
