'''
* This file reruns the DARP algorithm Version 2 with the same inputs, including target position and robot position
* Due to the random nature of DARP, the solution would likely be different every time
* This file also has the capacity to rerun simply the MST portion of the algorithm, without a changing DARP output
'''

# TODO: Add capacity to rerun DARP as well - then this becomes final target checker

import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
import sys
import random
import time


# # DIRECTORY MANAGEMENT FOR DARP IMPORT
path = pathlib.Path(__file__).parent.absolute()
path = pathlib.Path(path).parent.absolute()
os.chdir(path)
sys.path.append(str(path))

import DARP_Python_Main as DPM  # pylint: disable=import-error

# Ensures it prints entire arrays when logging instead of going [1 1 1 ... 2 2 2]
np.set_printoptions(threshold=np.inf)

class target_case_checker_DARP:
    def __init__(self):
        # DIRECTORY MANAGEMENT
        path = pathlib.Path(__file__).parent.absolute()
        # Changing current working directory to be able to find Case**.txt files
        os.chdir(path)
    def get_data(self,file_name):
        FILE = open(file_name,"r")
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

        FILE.close()
    def rerun_DARP(self, file_log = "MAIN_LOGGING.txt", show_grid=False,distance_measure = 0,recompile=True):
        DPM.PRINT_DARP = True
        DPM.PRINT_PATH = True
        DPM.PRINT_TREE = False
        DPM.PATH_COLOR = 'k'
        DPM.TARGET_FINDING = self.TARGET_FINDING
        
        DPM.algorithm_start(recompile=recompile)
        
        RA = DPM.Run_Algorithm(self.Grid, self.rip, self.dcells, self.Imp, show_grid, dist_meas=distance_measure,log_active=True,log_filename=file_log,target_active=False)
        RA.set_continuous(self.rip_sml,self.rip_cont,self.tp_cont)
        RA.main()
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

class target_case_checker_MST:
    def __init__(self):
        # DIRECTORY MANAGEMENT
        path = pathlib.Path(__file__).parent.absolute()
        # Changing current working directory to be able to find Case**.txt files
        os.chdir(path)
    def get_data(self,file_name):
        FILE = open(file_name,"r")
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
    def rerun_MST(self, file_log = "MAIN_LOGGING.txt", show_grid=False,distance_measure = 0,recompile=True,corners = 0):
        DPM.PRINT_DARP = True
        DPM.PRINT_TREE = True
        DPM.PATH_COLOR = 'k'
        DPM.TARGET_FINDING = self.TARGET_FINDING
        
        if(corners==0):
            DPM.PRINT_PATH = False
        elif(corners==1):
            DPM.PRINT_PATH = True
            DPM.PRINT_HALF_SHIFTS = False
            DPM.PRINT_DYNAMIC_CONSTRAINTS = False
        elif(corners==2):
            DPM.PRINT_PATH = True
            DPM.PRINT_HALF_SHIFTS = True
            DPM.PRINT_DYNAMIC_CONSTRAINTS = False
        elif(corners==3):
            DPM.PRINT_PATH = True
            DPM.PRINT_HALF_SHIFTS = True
            DPM.PRINT_DYNAMIC_CONSTRAINTS = True

        DPM.algorithm_start(recompile=recompile)
        
        RA = DPM.Run_Algorithm(self.Grid, self.rip, self.dcells, self.Imp, show_grid, dist_meas=distance_measure,log_active=True,log_filename=file_log,target_active=False)
        RA.set_continuous(self.rip_sml,self.rip_cont,self.tp_cont)
        RA.rerun_MST_only(self.A,self.Ilabel)
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

if __name__ == "__main__":
    show_grid = True

    TCC = target_case_checker_MST()
    TCC.get_data("TARGET_CASES/Case01.txt")
    TCC.rerun_MST(show_grid=show_grid,distance_measure=2,recompile=True,corners=1)
    if (show_grid == True):
        plt.show()
