import numpy as np
import math
import matplotlib.pyplot as plt
import DARP_Python_Main as DPM
import Refuelling_Protocol02 as RP

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
DPM.FIGURE_TITLE = "STC"
DPM.TARGET_FINDING = True # Does path truncation and target printing
DPM.JOIN_REGIONS_FOR_REFUEL = False
DPM.EXAMPLE = False

main_logfile = "MAIN_LOGGING.txt"
target_logfile = "TARGET_LOG.txt"

# Other parameters
Imp = False
maxIter = 10000
distance_measure = 0 # 0,1,2 - Euclidean, Manhattan, GeodisicManhattan
print_graphs = False

titles = ["abort","es_flag","maxIter","obs","DARP_success","discr_achieved","iterations","etime_DARP_total","total_STC_etime","etime_tree","etime_wpnt","etime_schedule","AOE","AOEperc","maxDiscr","conBool","runs","total_iterations","total_time","total_energy","rotations","distances","schedules","tp_detect_time","max_time","max energy","start_cont","refuels","take-off","landing","refuel_time","TO_height","flight_time","vel","height","r_min","DISC_H","DISC_V","r_max","ARC_L","GSD_h","V_max","H_max","rows","cols","n_r","cc","rl","dcells","imp","target_finding","tp_cont","GRID","rip","rip_sml","rip_cont_temp","A","Ilabel"]

f = open(main_logfile, "a")
f.write("\n")
for title in titles:
    f.write(str(title))
    f.write(",")
f.write("\n")
f.close()

DPM.RUN_PRIM = True
GROUND_STATION_RUNS = False

size1 = np.arange(10,105,5) # 10 to 50, 55 to 70
size2 = np.arange(10,105,5) # 10 to 50, 55 to 70
rl_vals = np.array([0.001])
cc_vals = np.array([0.1])
robots = np.array([2,4,6,8]) #np.arange(2,11,1)
obstacles = np.array([0]) #np.arange(0,100,5)
counter = 0
runs = 5
print("Predicted runs: ", len(size1)*len(size2)*len(rl_vals)*len(cc_vals)*len(robots)*len(obstacles)*runs)
for run in range(runs):
    for s1 in size1:
        for s2 in size2:
            for r in robots:
                for obs in obstacles:
                    for cc in cc_vals:
                        for rl in rl_vals:
                            counter = counter + 1
                            print("Run: ", counter, "Size: ", s1, "x", s2, "Robots: ", r, "Obstacles:", obs)
                            rerun = True
                            while(rerun == True):
                                rerun = False
                                try:
                                    # Variables
                                    horizontal = DPM.DISC_H*s1*2
                                    vertical = DPM.DISC_V*s2*2
                                    n_r = r
                                    obs_perc = obs
                                    if (GROUND_STATION_RUNS):
                                        GG = RP.generate_grid(horizontal, vertical)
                                        GG.randomise_obs(obs_perc)
                                        RR = RP.refuelling(GG.rows, GG.cols, GG.GRID)
                                        RR.spacing = 2
                                        success = RR.refuel(n_r)
                                        GG.GRID = RR.GRID
                                        GG.randomise_target()
                                        tp_cont = GG.tp_cont
                                        EnvironmentGrid = GG.GRID

                                        rows = RR.rows
                                        cols = RR.cols
                                        obs = RR.obs # Note this is before removal of enclosed space, which can increase the number of robots
                                        n_r_equivalent = RR.n_r
                                        dcells = math.ceil(rows*cols/10) # discrepancy of X% allowed
                                        
                                        #  Call this to do directory management and recompile Java files - better to keep separate for when running multiple sims
                                        DPM.algorithm_start(recompile=True)

                                        # Call this to run DARP and MST
                                        RA = DPM.Run_Algorithm(EnvironmentGrid, RR.rip, dcells, Imp, print_graphs, dist_meas=distance_measure, log_active=True, log_filename=main_logfile, target_filename=target_logfile, target_active=False, refuels = RR.refuels, ground_station=True, cc_vals=np.array([cc]), rl_vals=np.array([rl]))
                                        RA.set_continuous(RR.rip_sml,RR.rip_cont,tp_cont=tp_cont,start_cont = RR.start_cont)
                                        RA.main()
                                    else:
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
                                        RA = DPM.Run_Algorithm(EnvironmentGrid, GG.rip, dcells, Imp, print_graphs,dist_meas=distance_measure,log_active=True,log_filename=main_logfile,target_filename=target_logfile,target_active=True, rl_vals=[rl], cc_vals=[cc])
                                        RA.set_continuous(GG.rip_sml,GG.rip_cont, GG.tp_cont)
                                        RA.main()
                                        if (RA.n_r != r):
                                            print("Robot Failure...")
                                            rerun = True
                                    if (RA.DARP_success == False)or(RA.abort == True):
                                        print("Abort or DARP Failure...")
                                        rerun = True
                                    if (RA.obs*100/(RA.rows*RA.cols) > obs + 1) or (RA.obs*100/(RA.rows*RA.cols) < obs - 1):
                                        print("Obs Failure...")
                                        rerun = True
                                    if (rerun == True):
                                        print("RERUN")
                                except Exception as e:
                                    print("Failure: ", e)
                                    rerun = True
                                    print("RERUN")
if print_graphs == True:
    plt.show()
