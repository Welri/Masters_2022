import pathlib
import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt

# DIRECTORY MANAGEMENT FOR DARP IMPORT
path = pathlib.Path(__file__).parent.absolute()
path = pathlib.Path(path).parent.absolute()
os.chdir(path)
sys.path.append(str(path))

import DARP_Python_Main as DPM

# Ensures it prints entire arrays when logging instead of going [1 1 1 ... 2 2 2]
np.set_printoptions(threshold=np.inf)

DPM.PRINTS = False
DPM.FIGSIZE=12
DPM.TREE_COLOR = 'w'
DPM.PATH_COLOR = 'k'
DPM.PRINT_DARP = True
DPM.PRINT_COLOURS = True
DPM.PRINT_TREE = False
DPM.PRINT_PATH = True
DPM.PRINT_RIP = True
DPM.PRINT_TARGET = True

DPM.PRINT_HALF_SHIFTS = True # CAREFUL when changing this
DPM.PRINT_DYNAMIC_CONSTRAINTS = True # CAREFUL when changing this
DPM.PRINT_CIRCLE_CENTRES = False # Only valid with dynamic constraints active

DPM.LINEWIDTH = 1
DPM.S_MARKERIZE = DPM.LINEWIDTH*4
DPM.MARKERSIZE= DPM.LINEWIDTH*12
DPM.TICK_SPACING = 1
DPM.DARP_FIGURE_TITLE = "DARP Results"
DPM.FIGURE_TITLE = "Champaigne Castle with Removed Obstacles to Remove Enclosed Spaces"
DPM.TARGET_FINDING = True # Does path truncation and target printing
DPM.JOIN_REGIONS_FOR_REFUEL = False

file_log = "MAIN_LOGGING.txt"
target_log = "TARGET_LOG.txt"

# Other parameters
Imp = False
maxIter = 10000
distance_measure = 0 # 0,1,2 - Euclidean, Manhattan, GeodisicManhattan
print_graphs = False

size = np.arange(10,105,5)
robots = np.array([2]) #np.arange(2,11,1)
obstacles = np.array([0]) #np.arange(0,100,5)
counter = 0
for s1 in size:
    for s2 in size:
        for r in robots:
            for obs in obstacles:
                counter = counter + 1
                print("Run: ", counter, "Size: ", s1, "x", s2, "Robots: ", r, "Obstacles:", obs)
                try:
                    # Variables
                    horizontal = DPM.DISC_H*s1*2
                    vertical = DPM.DISC_V*s2*2
                    n_r = r
                    obs_perc = obs

                    # Generate environment grid
                    GG = DPM.generate_grid(horizontal,vertical)
                    GG.randomise_obs(obs_perc)
                    GG.randomise_robots(n_r)
                    GG.randomise_target()

                    rows = GG.rows
                    cols = GG.cols
                    dcells = math.ceil(rows*cols/10)

                    EnvironmentGrid = GG.GRID

                    #  Call this to do directory management and recompile Java files - better to keep separate for when running multple sims
                    DPM.algorithm_start(recompile=False)

                    # Call this to run DARP and MST
                    RA = DPM.Run_Algorithm(EnvironmentGrid, GG.rip, dcells, Imp, print_graphs,dist_meas=distance_measure,log_active=True,log_filename=file_log,target_filename=target_log,target_active=True)
                    RA.set_continuous(GG.rip_sml,GG.rip_cont, GG.tp_cont)
                    RA.main()
                except:
                    print("Failure")
if print_graphs == True:
    plt.show()

