import os


GDB17_DIR = os.path.join(os.getcwd(), "data", "gdb17")

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

splitFile("GDB17.50000000.smi")
