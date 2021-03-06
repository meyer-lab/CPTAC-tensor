#!/usr/bin/env python3
#from cptactensor.figures.common import overlayCartoon
import sys
import logging
import time
import matplotlib

matplotlib.use("AGG")

fdir = "./"
cartoon_dir = r"./cpactensor/figures"
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if __name__ == "__main__":
    nameOut = "figure" + sys.argv[1]

    start = time.time()

    exec("from cpactensor.figures." + nameOut + " import makeFigure")
    ff = makeFigure()
    ff.savefig(fdir + nameOut + ".svg", dpi=ff.dpi, bbox_inches="tight", pad_inches=0)

    logging.info("%s is done after %s seconds.", nameOut, time.time() - start)
