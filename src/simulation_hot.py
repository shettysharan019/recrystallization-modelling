import numpy as np
import pandas as pd
import random
import math
from PIL import Image, ImageTk, EpsImagePlugin
from main import GrainAnalyzer, SimulationParameters
import os

class HotSimulation:
    def __init__(self, dataframe: pd.DataFrame, s_array: np.ndarray, canvas, params: SimulationParameters, num_grains: int = 100, pixel_size: int = 10, mobility_m: float = 10.0):
        self.df = dataframe.to_numpy()
        self.s_array = s_array
        self.canvas = canvas
        self.params = params
        self.num_grains = num_grains
        self.pixel_size = pixel_size
        self.mobility_m = mobility_m
        self.r, self.c = s_array.shape[:2]
        self.EA = np.zeros((self.r, self.c, 3))
        self.lattice_status = np.zeros((self.r, self.c), object)
        for x in range(self.r):
            for y in range(self.c):
                self.EA[x, y] = s_array[x, y, :3]
        self.grains = []
        self.grain_analyzer = GrainAnalyzer(self.params)

    def fetchEA(self, x, y):
        a = x % self.r
        b = y % self.c
        return self.EA[a, b]

    def mobility(self, misorientation):
        B = 5
        K = 5
        theta_m_rad = np.radians(self.params.theta_M)
        return self.mobility_m * (1 - (math.exp(-1 * B * ((misorientation / theta_m_rad) ** K))))

    def del_E(self, EA_M, EA_1, coords_px):
        SE_i = 0
        SE_f = 0
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if x == 0 and y == 0:
                    continue
                nx, ny = coords_px[0] + x, coords_px[1] + y
                EA_neighbor = self.fetchEA(nx, ny)
                misorientation_initial = self.grain_analyzer.compute_misorientation(
                    np.matmul(self.grain_analyzer.compute_orientation_matrix(*EA_1),
                              np.linalg.inv(self.grain_analyzer.compute_orientation_matrix(*EA_neighbor))))
                SE_i += self.grain_analyzer.stored_energy(misorientation_initial)
                misorientation_final = self.grain_analyzer.compute_misorientation(
                    np.matmul(self.grain_analyzer.compute_orientation_matrix(*EA_M),
                              np.linalg.inv(self.grain_analyzer.compute_orientation_matrix(*EA_neighbor))))
                SE_f += self.grain_analyzer.stored_energy(misorientation_final)
        return SE_f - SE_i

    def probability(self, del_E, misorientation):
        theta_m_rad = np.radians(self.params.theta_M)
        stored_energy_val = self.grain_analyzer.stored_energy(misorientation)
        if del_E <= 0:
            return (self.mobility(misorientation) * stored_energy_val * 2) / (self.mobility_m * 10.0)
        else:
            return (self.mobility(misorientation) * stored_energy_val * 2) * (np.exp(-1 * del_E)) / (self.mobility_m * 10.0)

    def state_change(self, current_grain, coords_px):
        x_coord = coords_px[0]
        y_coord = coords_px[1]
        pixel_state_initial = self.fetchEA(x_coord, y_coord)
        misorientation_val = self.grain_analyzer.compute_misorientation(
            np.matmul(self.grain_analyzer.compute_orientation_matrix(*current_grain['euler_angles']),
                      np.linalg.inv(self.grain_analyzer.compute_orientation_matrix(*pixel_state_initial))))
        prob = self.probability(self.del_E(current_grain['euler_angles'], pixel_state_initial, coords_px), misorientation_val)
        if random.uniform(0, 1) <= prob:
            x, y = coords_px[0] % self.r, coords_px[1] % self.c
            self.EA[x, y] = current_grain['euler_angles']
            current_grain['new_grainspx'].append([x, y])
            self.lattice_status[x, y] = current_grain['color']

    def updateGB(self, grain):
        new_gb = []
        for coord in grain['GB']:  # coord is a list [x, y]
            # Pass individual x and y coordinates
            if self.isGB(coord[0], coord[1], grain['euler_angles']):
                new_gb.append(coord)
        for coord in grain['new_grainspx']:
            new_gb.append(coord)
        grain['GB'] = new_gb
        grain['new_grainspx'] = []

    def isGB(self, x, y, euler_angles):
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                nx, ny = x + i, y + j
                if not np.array_equal(self.fetchEA(nx, ny), euler_angles):
                    return True
        return False

    def print_euler_angles(self):
        try:
            with open(f"sim_output_n={self.num_grains}_hot.txt", "w") as f:
                f.write("phi1,phi,phi2,X,Y,IQ\n")
                for x in range(self.r):
                    for y in range(self.c):
                        f.write(f"{self.EA[x, y, 0]},{self.EA[x, y, 1]},{self.EA[x, y, 2]},{x * 1},{y * 1},60\n")
            print("Euler angles printed to sim_output_n_hot file.")
        except Exception as e:
            print(f"Error printing Euler angles: {e}")

    def generate_random_color(self):
        return f'#{random.randint(0, 0xFFFFFF):06x}'

    def save_canvas_image(self, filename="sim_output_hot.png"):
        try:
            self.canvas.postscript(file="output_image.eps", colormode='color')
            EpsImagePlugin.gs_windows_binary = r'"C:/Program Files/gs/gs10.04.0/bin/gswin64.exe"'  # Adjust path if needed
            img = Image.open("output_image.eps")
            img.save(filename, format="png")
            img.close()
            os.remove("output_image.eps")
            print(f"Monte Carlo Image saved to {filename}")
        except FileNotFoundError:
            print("Ghostscript not found. Please install it and ensure the path in code is correct.")
        except Exception as e:
            print(f"Error saving Monte Carlo image: {str(e)}")

    def update_display(self):
        self.canvas.delete("all")
        for i in range(self.r):
            for j in range(self.c):
                color = self.lattice_status[i, j] if self.lattice_status[i, j] != 0 else 'white'
                self.canvas.create_rectangle(i * self.pixel_size, j * self.pixel_size, (i + 1) * self.pixel_size,
                                             (j + 1) * self.pixel_size, fill=color, outline="")
        self.canvas.update()

    def initialize_grains(self):
        self.grains = []
        self.lattice_status = np.zeros((self.r, self.c), object)
        nucleation_sites = []
        for i in range(self.num_grains):
            while True:
                nuclii_x = random.randint(0, self.r - 1)
                nuclii_y = random.randint(0, self.c - 1)
                if self.lattice_status[nuclii_x, nuclii_y] == 0:
                    nucleation_sites.append((nuclii_x, nuclii_y))
                    break
        for i, (nx, ny) in enumerate(nucleation_sites):
            initial_ea = self.fetchEA(nx, ny).copy()
            color = self.generate_random_color()
            grain = {'name': f"grain {i+1}", 'euler_angles': initial_ea, 'GB': [[nx, ny]], 'new_grainspx': [], 'color': color}
            self.grains.append(grain)
            self.lattice_status[nx, ny] = color
        self.update_display()

    def monte_carlo_step(self, num_steps=30):
        m = 0
        while m < num_steps:
            for i in list(self.grains):
                for j in list(i['GB']):
                    for x in [-1, 0, 1]:
                        for y in [-1, 0, 1]:
                            if x == 0 and y == 0:
                                continue
                            coords_to_check = [(j[0] + x) % self.r, (j[1] + y) % self.c]
                            if self.lattice_status[coords_to_check[0], coords_to_check[1]] == 0:
                                self.state_change(i, coords_to_check)
                self.updateGB(i)
            m += 1
            self.update_display()

    def run_all_steps(self, num_steps=200):
        for _ in range(num_steps):
            self.monte_carlo_step(num_steps=len(self.grains))

if __name__ == '__main__':
    # Example usage (for testing)
    dummy_data = {
        'X': np.arange(10),
        'Y': np.arange(10),
        'Phi1': np.random.rand(100),
        'Phi': np.random.rand(100),
        'Phi2': np.random.rand(100),
        'SE': np.random.rand(100)
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_s_array = np.random.rand(10, 10, 3)
    import tkinter as tk
    root = tk.Tk()
    canvas = tk.Canvas(root, width=500, height=500, bg='white')
    canvas.pack()
    params = SimulationParameters()
    hot_sim = HotSimulation(dummy_df, dummy_s_array, canvas, params)
    hot_sim.initialize_grains()
    root.mainloop()