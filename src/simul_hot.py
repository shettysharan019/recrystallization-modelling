# this is your simulation_hot.py file:
import numpy as np
import main_new as main # corrected import
import matplotlib.pyplot as plt
import random
import tkinter as tk
from tkinter import Canvas
import pandas as pd
from main_new import global_s as s # corrected import and name
import math
from tkinter import *
from PIL import Image, ImageTk, EpsImagePlugin
path = "C:/Users/shett/Downloads/Final Year Porject (FYP)/src/optimised_files/output/crystallography_analysis_20250301_174237.csv"

df = pd.read_csv(path)

df = df.to_numpy()

color =["red","blue","cyan","yellow","purple","pink","orange","green","brown","grey","black"]

number_of_grains = 5

M_m = 10

for i in range(0,len(df[:,0])-1): # corrected index range
    if (df[i+1,0]) - (df[i,0]) != 0:
        stepsize_x = df[i+1,0] - df[i,0]
        break

for i in range(0,len(df[:,1])-1): # corrected index range
    if (df[i+1,1]) - (df[i,1]) != 0:
        stepsize_y = df[i+1,1] - df[i,1]
        break
print(stepsize_x,stepsize_y)

df[:,0 ] = (df[:, 0] / (stepsize_x)).astype(int)
df[:,1] = (df[:, 1] / (stepsize_y)).astype(int)

r,c = int(np.max(df[:,0])),int(np.max(df[:,1]))  ##r = x , c= y

EA = np.zeros((r+1,c+1,4))
s = np.zeros((r+1,c+1,3))

for i in df:
    s[int(i[0])][int(i[1])][0] = i[2]  ## theta (average misorientation)
    s[int(i[0])][int(i[1])][1] = i[3]  ## KAM
    s[int(i[0])][int(i[1])][2] = i[5]  ## SE


lattice_angle = np.zeros((r+1,c+1,1))
for i in df:
    lattice_angle[int(i[0])][int(i[1])][0] = i[2]

lattice_status = np.zeros((r+1,c+1),object) ## Stores the color for display makes matrix with 0 but can now store any object

class grain:
    def __init__(self,name,eulerangles,color = "red"):
        self.eulerangles = eulerangles
        self.GB = []
        self.newgrainspx = []
        self.name = name
        self.color = color
    def isGB(self,x,y):
        #if fetchEA(x,y) == self.eulerangles:
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    if i == 0 and j == 0: # avoid checking the pixel itself
                        continue
                    if fetchEA(x+i,y+j) != fetchEA(x,y):
                        return True
                    else: pass
            return False

        #else: pass

    def updateGB(self):
        new_gb = []
        for i in self.GB:
            if grain.isGB(self,i[0],i[1]):
                new_gb.append(i) # keep it in GB
            #else: #remove from GB is not GB anymore
            #    self.GB.remove(i) # avoid removing while iterating, create new list and assign at the end.

        for i in self.newgrainspx:
            new_gb.append(i) # add new grain pixels into GB list

        self.GB = new_gb
        self.newgrainspx = [] # clear new grain pixels after updating GB list.


def fetchEA(x,y):
    a = x%(r+1)
    b = y%(c+1)
    return [EA[a,b,0],EA[a,b,1],EA[a,b,2]]

def mobility(misorientation):
    B = 5
    K = 5
    M_m = 10
    return M_m *(1-(math.exp(-1*B*((misorientation/main.theta_m)**K))))


def del_E(EA_M,EA_1,coords_px):  ## EA_M is the euler angles of the grain, EA_1 the eulerangles of the pixel to be changed
    SE_i = 0
    SE_f = 0
    for x in [-1,0,1]:
        for y in [-1,0,1]:
            if x == 0 and y == 0: # avoid checking the pixel itself
                continue
            if (coords_px[0] + x < 0 or coords_px[0] + x > r) or (coords_px[1] + y < 0 or coords_px[1] + y > c):
                continue # skip out of bound indices
            SE_i = SE_i + main.stored_energy(main.theta(np.matmul(main.g(EA_1[0],EA_1[1],EA_1[2]),np.linalg.inv(main.g(fetchEA(coords_px[0]+x,coords_px[1]+y)[0],fetchEA(coords_px[0]+x,coords_px[1]+y)[1],fetchEA(coords_px[0]+x,coords_px[1]+y)[2]))))) ##Might not work
    for x in [-1,0,1]:
        for y in [-1,0,1]:
            if x == 0 and y == 0: # avoid checking the pixel itself
                continue
            if (coords_px[0] + x < 0 or coords_px[0] + x > r) or (coords_px[1] + y < 0 or coords_px[1] + y > c):
                continue # skip out of bound indices
            SE_f = SE_f + main.stored_energy(main.theta(np.matmul(main.g(EA_M[0],EA_M[1],EA_M[2]),np.linalg.inv(main.g(fetchEA(coords_px[0]+x,coords_px[1]+y)[0],fetchEA(coords_px[0]+x,coords_px[1]+y)[1],fetchEA(coords_px[0]+x,coords_px[1]+y)[2]))))) ##Might not work

    return SE_f - SE_i


def probability(del_E, misorientation):
    if del_E <= 0 :
        return 0.8 # increased probablity value
    else:
        return 0.8 * (np.exp(-1*del_E)) # increased probablity value and added del_E factor

def state_change(current_grain,coords_px): # renamed grain to current_grain to avoid confusion
    pixel_state_initial = fetchEA(coords_px[0],coords_px[1])
    prob = probability(del_E(current_grain.eulerangles, pixel_state_initial,coords_px),np.degrees(main.theta(np.matmul(main.g(current_grain.eulerangles[0],current_grain.eulerangles[1],current_grain.eulerangles[2]),np.linalg.inv(main.g(pixel_state_initial[0],pixel_state_initial[1],pixel_state_initial[2]))))))
    #print(x)
    if random.uniform(0, 1) <= prob:
        EA[coords_px[0]%(r+1),coords_px[1]%(c+1),0] = current_grain.eulerangles[0] # Mod to wrap around
        EA[coords_px[0]%(r+1),coords_px[1]%(c+1),1] = current_grain.eulerangles[1]
        EA[coords_px[0]%(r+1),coords_px[1]%(c+1),2] = current_grain.eulerangles[2]
        current_grain.newgrainspx.append([coords_px[0]%(r+1),coords_px[1]%(c+1)])

        lattice_status[coords_px[0]%(r+1),coords_px[1]%(c+1)] = current_grain.color  ## Mod to wrap around
    else: pass


def print_euler_angles():
    with open(f"sim_output_n={number_of_grains}.txt", "w") as f:
        f.write("phi1,phi,phi2,X,Y,IQ\n")

    for x in range(0, r+1):
        for y in range(0, c+1):
            with open(f"sim_output_n={number_of_grains}.txt", "a") as f:
                f.write("%s,%s,%s,%s,%s,%s\n"%(EA[x, y, 0], EA[x, y, 1], EA[x, y, 2],x*stepsize_x, y*stepsize_y,60))


def generate_random_color():
    # Generate random RGB values
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)

    # Convert to hexadecimal and format as a color string
    color_string = "#{:02X}{:02X}{:02X}".format(red, green, blue)

    return color_string

def save_canvas_image(filename="sim_output.png"): # added filename argument and default filename
    # Create a PostScript file from the canvas
    Canvas.postscript(file="output_image.eps", colormode='color')

    # Use Pillow (PIL) to convert the PostScript file to an image (e.g., PNG)
    EpsImagePlugin.gs_windows_binary = r'C:/Program Files (x86)/gs/gs10.02.1/bin/gswin32c.exe'  # Set the Ghostscript executable path
    img = Image.open("output_image.eps")
    img.save(filename, format="png") # use provided filename
    img.close()
    return filename # return saved filename for gui display


def update_display():
    Canvas.delete("all")
    for i in range(r+1): # corrected range to include r and c
        for j in range(c+1): # corrected range to include r and c
            if lattice_status[i, j] == 0:
                color = 'white'
            else:
                color= lattice_status[i,j]
            Canvas.create_rectangle(i * pixel_size, j * pixel_size, (i+1) * pixel_size, (j+1) * pixel_size, fill=color, outline="") # removed outline for better visual
    Canvas.update()


grains = []


def monte_carlo_step(n=5):
    m=0

    while m < n:
        for i in grains:
            #print(i.name) # removed print statement
            for j in i.GB:
                #print(j)  # removed print statement
                for x in [-1,0,1]:
                    #print(x) # removed print statement
                    for y in [-1,0,1]:
                        #print(y) # removed print statement
                        if x == 0 and y == 0: # avoid checking the GB pixel itself
                            continue
                        if 0 <= (j[0]+x) < (r+1) and 0 <= (j[1]+y) < (c+1): # check boundary conditions
                            if lattice_status[(j[0]+x)%(r+1),(j[1]+y)%(c+1)] == 0:
                                if i.eulerangles != fetchEA(j[0]+x,j[1]+y):
                                    #print(fetchEA(j[0]+x,j[1]+y)) # removed print statement
                                    state_change(i,[j[0]+x,j[1]+y])
                                    #print(fetchEA(j[0]+x,j[1]+y)) # removed print statement

                                    #print(i.name) # removed print statement
                                    #print(i.GB) # removed print statement
                                    #print("GB updated") # removed print statement
        for i in grains: # update GB for all grains after each monte carlo step
            i.updateGB()
        m +=1
    update_display()

def run_all_steps(num_steps=100): # new function to run multiple steps at once
    for _ in range(num_steps):
        monte_carlo_step(n=number_of_grains) # run monte carlo for number of grains times in each step

def init_grain_structure(): # function to initialize grain structure at the begining and when reset is needed
    global grains, lattice_status, EA # ensure global variables are used
    grains = []
    lattice_status = np.zeros((r+1,c+1),object) # reset lattice status
    EA = np.zeros((r+1,c+1,3)) # reset EA also if needed, if you want to restart from scratch. remove if you want to continue from the existing EA
    for i in range(1, number_of_grains + 1):
        nuclii_x = np.random.randint(0,r) # corrected range to be within 0 to r (exclusive of r+1)
        nuclii_y = np.random.randint(0,c) # corrected range to be within 0 to c (exclusive of c+1)
        obj_name = f"grain {i}"
        new_object = grain(obj_name,fetchEA(nuclii_x,nuclii_y),color= generate_random_color())
        new_object.GB.append([nuclii_x,nuclii_y])
        grains.append(new_object)
        lattice_status[nuclii_x,nuclii_y] = new_object.color
    update_display() # update display after initialization

init_grain_structure() # initialize grain structure at start

root = tk.Tk()
root.title("Random nucleation")

pixel_size = 5 # reduced pixel size for faster display in loop and better view.
canvas = Canvas(root, width=(r+1)*pixel_size, height=(c+1)*pixel_size, bg='white') # set white background
Canvas.pack()

step_button = tk.Button(root, text="Monte Carlo Step", command=monte_carlo_step)
all_steps_button = tk.Button(root, text="Run All Steps", command=run_all_steps) # new button
print_button = tk.Button(root, text= "Print Euler angles", command = print_euler_angles)
save_button = tk.Button(root, text= "Save image", command = save_canvas_image)
reset_button = tk.Button(root, text= "Reset", command = init_grain_structure) # Reset button to re-initialize grain structure

step_button.pack(side=tk.LEFT, padx=5)
all_steps_button.pack(side=tk.LEFT, padx=5) # pack new button
print_button.pack(side=tk.LEFT, padx=5)
save_button.pack(side=tk.LEFT, padx=5)
reset_button.pack(side=tk.LEFT, padx=5) # pack reset button


update_display()

root.mainloop()