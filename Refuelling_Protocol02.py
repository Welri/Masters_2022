import DARP_Python_Main as DPM
import matplotlib.pyplot as plt
import numpy as np
import math
import random as rand

class target_case_checker:
    def get_data(self,file_name,refuels=False):
        FILE = open(file_name,"r")
        self.refuels = refuels
        if (self.refuels == True):
            start_string = FILE.readline()
            self.start_cont = self.string_to_grid(start_string,1,2,data_type="float")
            self.start_cont = self.start_cont[0]
            self.no_refuels = int(FILE.readline())
        # else:
        #     FILE.readline()
        #     FILE.readline()
        self.take_off = float(FILE.readline())
        self.landing = float(FILE.readline())
        self.refuel_time = int(FILE.readline()) # Seconds
        self.take_off_height = float(FILE.readline()) # Metres
        self.flight_time = int(FILE.readline()) # Seconds
        self.vel = float(FILE.readline())
        self.height = float(FILE.readline())
        self.r_min = float(FILE.readline())
        self.disc_h = float(FILE.readline())
        self.disc_v = float(FILE.readline())
        self.r_max = float(FILE.readline())
        self.arc_l = float(FILE.readline())
        self.gsd_h = float(FILE.readline())
        self.v_max = float(FILE.readline())
        self.h_max = float(FILE.readline())
        # Read in parameters and variables
        self.rows = int(FILE.readline())
        self.cols = int(FILE.readline())
        self.n_r = int(FILE.readline())
        self.cc = float(FILE.readline())
        self.rl = float(FILE.readline())
        self.dcells = int(FILE.readline())
        self.Imp = self.import_bool(FILE.readline())
        self.TARGET_FINDING = self.import_bool(FILE.readline())
        target_string = FILE.readline()
        Grid_string = FILE.readline()
        rip_string = FILE.readline()
        ripsml_string = FILE.readline()
        ripcont_string = FILE.readline()
        Astring = FILE.readline()
        ILabelString = FILE.readline()

        # Target values
        self.tp_cont = self.string_to_grid(target_string,1,2,data_type="float")
        self.tp_cont = self.tp_cont[0]

        # Environment Grid
        self.Grid = self.string_to_grid(Grid_string,self.rows,self.cols,data_type="int")

        # Large cell RIP coordinates
        self.rip = self.string_to_grid(rip_string,self.n_r,2,data_type="int")

        # Small cell RIP coordinates
        self.rip_sml = self.string_to_grid(ripsml_string,self.n_r,2,data_type="int")

        # Continuous RIP coordinates
        self.rip_cont = self.string_to_grid(ripcont_string,self.n_r,2,data_type="float")
        
        # Cell assignments
        self.A = self.string_to_grid(Astring,self.rows,self.cols,data_type="int")

        self.Ilabel = self.string_to_ilabel_grid(ILabelString,self.n_r,self.rows,self.cols)
        FILE.close()
                # Set UAV and Camera dependent constants
        DPM.REFUEL_TIME = self.refuel_time
        DPM.TAKE_OFF_HEIGHT = self.take_off_height
        DPM.FLIGHT_TIME = self.flight_time
        DPM.Take_off = self.take_off
        DPM.Landing = self.landing
        DPM.VEL = self.vel
        DPM.Height = self.height
        DPM.r_min = self.r_min
        DPM.DISC_H = self.disc_h
        DPM.DISC_V = self.disc_v
        DPM.r_max = self.r_max
        DPM.ARC_L = self.arc_l
        DPM.GSD_h = self.gsd_h
        DPM.V_max = self.v_max
        DPM.H_max = self.h_max
    def import_bool(self, string):
        # Extract boolean variables
        if string[0] == "1" or string[0] == "t" or string[0] == "T":
            return(True)
        elif string[0] == "0" or string[0] == "f" or string[0] == "F":
            return(False)
        else:
            if string[0] == " ":
                for c in string:
                    if c == " ":
                        continue
                    else:
                        return(self.import_bool(c))
            print("ERROR: failed to import boolean value from -> ", string)
            return(-1)       
    def string_to_grid(self,string,rows,cols,data_type="int"):
        if (data_type=="int"):
            Grid = np.zeros(rows*cols,dtype=int)
        elif (data_type=="float"):
            Grid = np.zeros(rows*cols,dtype=float)
        e = 0
        c = 0
        while( (c<len(string)) and (e<len(Grid))):
            if (string[c] == ' ') or (string[c] == '\n') or (string[c] == '\t'):
                c+=1
                continue
            else:
                st = ""
                while((string[c] != " ") and (string[c] != "\n") and (string[c] != "\t")):
                    st = st + string[c]
                    c+=1
                    if (c>=len(string)):
                        break
                if (data_type == "int"):
                    Grid[e] = int(st)
                elif (data_type == "float"):
                    Grid[e] = float(st)
                e += 1 
        Grid = Grid.reshape(rows, cols)
        return(Grid)
    def string_to_ilabel_grid(self,string,r,rows,cols):
        Grid = np.zeros(r*rows*cols,dtype=int)
        e = 0
        c = 0
        while( (c<len(string)) and (e<len(Grid))):
            if (string[c] == ' ') or (string[c] == '\n') or (string[c] == '\t'):
                c+=1
                continue
            else:
                st = ""
                while((string[c] != " ") and (string[c] != "\n") and (string[c] != "\t")):
                    st = st + string[c]
                    c+=1
                    if (c>=len(string)):
                        break
                Grid[e] = int(st)

                e += 1 
        Grid = Grid.reshape(r, rows, cols)
        return(Grid)

class generate_grid:
    def __init__(self,hor,vert):
        # Divide Environment Into Large Nodes
        self.rows = math.ceil(vert/(DPM.DISC_V*2))
        self.cols = math.ceil(hor/(DPM.DISC_H*2))
        self.GRID = np.zeros([self.rows, self.cols], dtype=int)
        self.horizontal = 2*DPM.DISC_H*self.cols 
        self.vertical = 2*DPM.DISC_V*self.rows
    def set_robots(self, n_r, coords):
        # Function not recently tested - might not work
        self.n_r =  n_r # number of robots
        self.rip_cont = coords 
        self.rip_sml = np.zeros([len(coords),2],dtype=int)
        self.rip = np.zeros([len(coords),2],dtype=int)
        for r in range(self.n_r):
            # rip = self.rip_cont[r]
            # small cell position and large cell position
            self.rip_sml[r][0] = math.floor(self.rip_cont[r][0]/DPM.DISC_V) # row
            self.rip_sml[r][1] = math.floor(self.rip_cont[r][1]/DPM.DISC_H) # col
            self.rip[r][0] = math.floor(self.rip_cont[r][0]/(DPM.DISC_V*2)) # row
            self.rip[r][1] = math.floor(self.rip_cont[r][1]/(DPM.DISC_H*2)) # col
            self.GRID[self.rip[r][0]][self.rip[r][1]] = 2
            # Adjust rip_cont to centre cell
            self.rip_cont[r][0] = (self.rip_sml[r][0]+0.5)*DPM.DISC_V # vertical
            self.rip_cont[r][1] = (self.rip_sml[r][1]+0.5)*DPM.DISC_H # horizontal
    def set_obs(self, obs_coords):
        for obs in obs_coords:
            if np.shape(obs) == (2,):
                self.GRID[self.rows - obs[0] - 1][obs[1]] = 1
            elif np.shape(obs) == (2,2):
                row_max = np.max([self.rows - obs[0][0] - 1,self.rows - obs[1][0] - 1])
                row_min = np.min([self.rows - obs[0][0] - 1,self.rows - obs[1][0] - 1])
                col_max = np.max([obs[0][1],obs[1][1]])
                col_min = np.min([obs[0][1],obs[1][1]])
                for row in range(row_min,row_max+1):
                    for col in range(col_min,col_max+1):
                        self.GRID[row][col] = 1
    def set_target(self, targ_coord):
        self.tp_cont = np.array(targ_coord,dtype=float)
    def randomise_robots(self, n_r):
        possible_indexes = np.argwhere(self.GRID == 0)
        np.random.shuffle(possible_indexes)
        self.n_r = n_r
        self.rip_sml = np.zeros([n_r,2],dtype=int)
        self.rip_cont = np.zeros([n_r,2],dtype=float)
        # self.rip = np.zeros([n_r,2],dtype=int)
        if self.n_r < self.rows*self.cols:
            self.rip = possible_indexes[0:self.n_r]
            possible_indexes = np.delete(possible_indexes,np.arange(0,self.n_r,1),0)
            val1 = self.rip[:, 0]
            val2 = self.rip[:, 1]
            self.GRID[val1, val2] = 2
        else:
            print("MADNESS! Why do you have so many robots?")

        for r in range(self.n_r):
            self.rip_sml[r][0] = self.rip[r][0]*2
            self.rip_sml[r][1] = self.rip[r][1]*2
            self.rip_cont[r][0] = (self.rip_sml[r][0]+0.5)*DPM.DISC_V # vertical
            self.rip_cont[r][1] = (self.rip_sml[r][1]+0.5)*DPM.DISC_H # horizontal
    def randomise_obs(self, obs_perc):
        possible_indexes = np.argwhere(self.GRID == 0)
        np.random.shuffle(possible_indexes)
        self.obs = math.floor(self.rows*self.cols*obs_perc/100)
        if self.obs < (self.rows*self.cols*0.75):
            indices = possible_indexes[0:self.obs]
            possible_indexes = np.delete(possible_indexes,np.arange(0,self.obs,1),0)
            val1 = indices[:, 0]
            val2 = indices[:, 1]
            self.GRID[val1, val2] = 1
        else:
            print("MADNESS! Why so many obstacles? More than 75%% seems a bit crazy.")
    def randomise_target(self):
        # Target position from top left corner (gets converted to bottom left)
        possible_indexes = np.argwhere(self.GRID == 0)
        np.random.shuffle(possible_indexes)
        self.tp_cont = possible_indexes[0]
        self.tp_cont[0] = (self.tp_cont[0]+rand.random())*2*DPM.DISC_V  # vertical
        self.tp_cont[1] = (self.tp_cont[1]+rand.random())*2*DPM.DISC_H # horizontal

class refuelling:
    def __init__(self,rows,cols,grid):
        self.rows = rows
        self.cols = cols
        self.GRID = grid
        self.obs = len(np.argwhere(self.GRID == 1))
        self.ground_station = np.array([])
        self.centre_obstacles = True
        self.spacing = 2 # 1 or 2
    def set_start(self,cont_coord):
        self.ground_station = np.zeros(2, dtype=int)
        self.ground_station[0] = np.round(cont_coord[0]/(DPM.DISC_V*2) - 0.5) # y
        self.ground_station[1] = np.round(cont_coord[1]/(DPM.DISC_H*2) - 0.5) # x
    def set_robots_rip(self,n_r,coords):
        # input large cell coordinates
        self.n_r = n_r
        self.rip = coords
        self.rip_sml = np.zeros([len(coords),2],dtype=int)
        self.rip_cont = np.zeros([len(coords),2],dtype=float)
        for r in range(self.n_r):
            self.rip_sml[r][0] = self.rip[r][0]*2
            self.rip_sml[r][1] = self.rip[r][1]*2
            self.rip_cont[r][0] = (self.rip_sml[r][0]+0.5)*DPM.DISC_V
            self.rip_cont[r][1] = (self.rip_sml[r][1]+0.5)*DPM.DISC_H
            self.GRID[self.rip[r][0]][self.rip[r][1]] = 2
    def set_obs_rip(self,obs_coords):
        # large cell coordinates
        for obs in obs_coords:
            self.GRID[obs[0]][obs[1]] = 1
    def determine_refuels(self,n_r):
        max_cell_time = 4*(np.max([DPM.ARC_L*1.3,DPM.DISC_H,DPM.DISC_V])/DPM.VEL)
        size = 0
        size_new = 3
        while(size_new != size):
            size = size_new
            start_obs = size - 2
            m_time = 5*(2*np.pi*DPM.r_min)/DPM.VEL + (2*np.sqrt(2*(size/2)**2)*np.sqrt(DPM.DISC_H**2+DPM.DISC_V**2))/DPM.VEL
            m_time = m_time + DPM.Take_off + DPM.Landing
            cells_max = (DPM.FLIGHT_TIME-1.3*m_time)/(max_cell_time)
            cells_free = self.rows*self.cols - self.obs - (start_obs*start_obs)
            n_required = cells_free / (cells_max)
            n_available = n_r
            n_eq = math.ceil(n_required/n_available) * n_available
            refuels = n_eq/n_available
            size_new = 3
            for i in range(1, math.ceil(n_eq/(8/self.spacing))):
                size_new = size_new + 2
        if(DPM.PRINTS):
            print("Estimated T_m: ", m_time/60)
        
        refuels = math.ceil(n_required/n_available)-1
        return(refuels)
    def possible_robots(self):
        # Note: self.n_r (equivalent robots) needs to be determined before this function can be called
        self.possible_robots_list = list()
        self.cardinal_dir = [[1,0],[-1,0],[0,1],[0,-1]]
        okay = False
        self.robot_moves()
        for r in range(self.overall_size-1, self.rows-(self.overall_size)):
            for c in range(self.overall_size-1, self.cols-(self.overall_size)):
                if self.GRID[r][c] != 1:
                    # Check relevant robot positions
                    okay = True
                    for n in range(self.n_r):
                        # Check if robot position have obstacles in them
                        r_n = r + self.moves[n][0]
                        c_n = c + self.moves[n][1]
                        if(r_n<self.rows)and(r_n>=0)and(c_n<self.cols)and(c_n>=0):
                            if self.GRID[r_n][c_n] == 1:
                                okay = False
                        else:
                            okay = False
                        # Check for obstacles in vicinity of valid robot location - check for obstacles in cardinal locations
                        ob_count = 4
                        if okay == True:
                            for m in self.cardinal_dir:
                                r_o = r_n + m[0]
                                c_o = c_n + m[1]
                                if(r_o<self.rows)and(r_o>=0)and(c_o<self.cols)and(c_o>=0):
                                    if self.GRID[r_o][c_o] == 1:
                                        ob_count-=1
                                else:
                                    ob_count-=1
                                if ob_count <= 2: # All the cardinal directions should be free (note that the central ground station isn't an obstacle yet.)
                                    okay = False   
                if okay == True:
                    self.possible_robots_list.append([r,c])
    def robot_moves(self):
        size = 3
        self.moves = np.zeros([math.ceil(self.n_r/(8/self.spacing))*int(8/self.spacing),2], dtype=int)
        self.pos = np.zeros([math.ceil(self.n_r/(8/self.spacing))*int(8/self.spacing),2], dtype=int)
        mi = 0
        for i in range(1, math.ceil(self.n_r/(8/self.spacing))):
            size = size + 2
        self.overall_size = size # Saving the initial position region size
        rows = np.arange(-math.floor(size/2),math.ceil(size/2),self.spacing)
        cols = np.arange(-math.floor(size/2),math.ceil(size/2),self.spacing)
        # First row
        r = rows[0]
        for c in cols:
            self.moves[mi] = [r,c]
            self.pos[mi] = [0,1]
            mi += 1

        # Inbetween rows
        if (size > 3) or (self.spacing == 1):
            for r in rows[1:-1]:   
                self.moves[mi] = [r,cols[-1]]
                self.pos[mi] = [1,1]
                mi += 1
            for r in rows[1:-1]:
                self.moves[mi] = [r,cols[0]]
                self.pos[mi] = [0,0]
                mi += 1 

        # Last row
        r = rows[-1]
        cols_rev = np.arange(math.floor(size/2),-math.ceil(size/2),-self.spacing)
        for c in cols_rev:
            self.moves[mi] = [r,c]
            self.pos[mi] = [1,0]
            mi += 1
    def refuel(self,n_r):
        # Conservatively calculate the number of refuels needed
        self.refuels = self.determine_refuels(n_r)
        self.n_r = n_r * (self.refuels+1) # equivalent number of robots given the number of refuels
        # Print equivalent robots and exit if more than 16 (possible robots can only do up to 16 for now)
        print("Robots: ", n_r," Refuels: ", self.refuels," Equivalent Robots: ", self.n_r)
       
        # Calculate possible starting positions
        if(len(self.ground_station) == 0):
            self.possible_robots()
            if len(self.possible_robots_list) == 0:
                print("WARING: No valid starting location found...")
            # Randomly choose a valid starting position and set rip
            start = self.possible_robots_list[ rand.randint(0,len(self.possible_robots_list)-1) ]
        else:
            start = self.ground_station
            self.robot_moves()
            # TODO: Check if robot locations are obstacles - give warning and terminate

        self.start_cont = np.zeros(2)
        self.start_cont[0] = (start[0]+0.5)*DPM.DISC_V*2
        self.start_cont[1] = (start[1]+0.5)*DPM.DISC_H*2
        self.rip = np.zeros([self.n_r,2],dtype=int)
        self.rip_sml = np.zeros([self.n_r,2],dtype=int)
        self.rip_cont = np.zeros([self.n_r,2],dtype=float) 
        for r in range(self.n_r):
            move = self.moves[r]
            self.rip[r][0] = move[0] + start[0]
            self.rip[r][1] = move[1] + start[1]
            pos = self.pos[r]
            self.rip_sml[r][0] = self.rip[r][0]*2 + pos[0]
            self.rip_sml[r][1] = self.rip[r][1]*2 + pos[1]
            self.rip_cont[r][0] = (self.rip_sml[r][0]+0.5)*DPM.DISC_V
            self.rip_cont[r][1] = (self.rip_sml[r][1]+0.5)*DPM.DISC_H
            self.GRID[self.rip[r][0]][self.rip[r][1]] = 2
        # Make centre of robot formation (the landing and take off zone) an obstacle
        if (self.centre_obstacles == True):
            size = 1
            for i in range(1, math.ceil(self.n_r/(8/self.spacing))):
                size = size + 2
            for row in np.arange(-math.floor(size/2),math.ceil(size/2),1,dtype=int):
                for col in np.arange(-math.floor(size/2),math.ceil(size/2),1,dtype=int):
                    self.set_obs_rip([[start[0]+row,start[1]+col]])
        return(True)

if __name__ == "__main__":           
    # Ensures it prints entire arrays when logging instead of going [1 1 1 ... 2 2 2]
    np.set_printoptions(threshold=np.inf)

    # What graphs should it print
    DPM.PRINT_DARP = False
    DPM.PRINT_TREE = False
    DPM.PRINT_PATH = True
    DPM.PRINT_CIRCLE_CENTRES = False
    DPM.JOIN_REGIONS_FOR_REFUEL = True
    DPM.PRINT_TARGET = True
    DPM.TARGET_FINDING = False
    DPM.PRINT_LANDING = False
    DPM.PRINT_TAKE_OFF = False
    DPM.PRINT_SCHEDULE = True
    # '''
    # Establish Environment Size - Chooses max horizontal and vertical dimensions and create rectangle
    horizontal = 23*DPM.DISC_H*2 # m
    vertical = 23*DPM.DISC_V*2 # m
    n_r = 2
    obs_perc = 5

    # Establish Small Node size
    GG = generate_grid(horizontal, vertical)
    GG.randomise_obs(obs_perc)
    RR = refuelling(GG.rows, GG.cols, GG.GRID)
    RR.spacing = 2
    # RR.set_start([RR.rows*DPM.DISC_V*2 - 1000,500])
    success = RR.refuel(n_r)
    GG.GRID = RR.GRID
    GG.randomise_target()
    tp_cont = GG.tp_cont
    EnvironmentGrid = GG.GRID
    '''
    TC = target_case_checker()
    TC.get_data("DARP_data/TARGET_CASES/DARP00_A.txt")
    tp_cont = TC.tp_cont
    # DPM.FLIGHT_TIME = 6*60*60 # Battery
    # DPM.VEL = 14
    # DPM.r_min = DPM.VEL**2 / ( DPM.g_acc * math.tan(DPM.phi_max*math.pi/180.0) ) # m - Minimum turning radius
    # DPM.ARC_L = (DPM.DISC_V/2.0 - DPM.r_min) + DPM.r_min*np.pi/2.0 + (DPM.DISC_H/2.0 - DPM.r_min) # two straight segments, if there are straight segments, and the arc
    
    for r in range(TC.n_r):
        TC.Grid[TC.rip[r][0]][TC.rip[r][1]] = 0
    
    # DPM.FLIGHT_TIME = 1*60*60
    RR = refuelling(TC.rows, TC.cols, TC.Grid)
    RR.spacing = 1
    start = [1000,12000] # DARP00_A
    # start = [2500,5500] # DARP02_A
    # start = [800,800] # DARP03_A
    # start = [500,1500] # Toy
    RR.set_start([RR.rows*DPM.DISC_V*2 - start[0],start[1]]) 
    n_r = 2
    success = RR.refuel(n_r)
    EnvironmentGrid = RR.GRID
    '''
    if(success):  
        # Other parameters
        distance_measure = 0 # 0, 1, 2 - Euclidean, Manhattan, GeodisicManhattan
        Imp = False
        maxIter = 10000

        rows = RR.rows
        cols = RR.cols
        obs = RR.obs # Note this is before removal of enclosed space, which can increase the number of robots
        n_r_equivalent = RR.n_r
        dcells = math.ceil(rows*cols/10) # discrepancy of X% allowed

        print_graphs = True

        # RUNNING SIMULATION #
        file_log = "MAIN_LOGGING.txt"
        target_log = "TARGET_LOG.txt"
        
        #  Call this to do directory management and recompile Java files - better to keep separate for when running multiple sims
        DPM.algorithm_start(recompile=True)

        # Call this to run DARP and MST
        RA = DPM.Run_Algorithm(EnvironmentGrid, RR.rip, dcells, Imp, print_graphs,dist_meas=distance_measure,log_active=False,log_filename=file_log,target_filename=target_log,target_active=True,refuels = RR.refuels,ground_station=True,cc_vals=np.array([0.1,0.01,0.001,0.0001,0.00001]),rl_vals=np.array([0.001,0.0001,0.00001,0.000001]))
        RA.set_continuous(RR.rip_sml,RR.rip_cont,tp_cont=tp_cont,start_cont = RR.start_cont)
        RA.main()

        if print_graphs == True:
            plt.show()
