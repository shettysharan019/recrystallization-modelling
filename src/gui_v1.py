import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from PIL import Image, ImageTk, EpsImagePlugin  # Added EpsImagePlugin for saving canvas
from typing import Optional, Literal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from main import run_main_analysis, SimulationParameters, GrainAnalyzer  # Corrected import to include GrainAnalyzer and SimulationParameters
import threading
from energy_distribution import create_energy_distribution_plot
from datetime import datetime
import random
import math

# Global sigma_m - Define here if needed for Monte Carlo, or better to pass as parameter if necessary
sigma_m = 10.0


class MaterialAnalysisGUI:
    output_files_main_analysis = None  # Class variable to store main analysis output file paths

    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Material Analysis Tool")
        self.root.geometry("1400x900")  # Increased width to accommodate Monte Carlo section

        # Initialize variables
        self.input_file: Optional[str] = None
        self.output_file: Optional[str] = None
        self.current_plots = {}
        self.processing = False  # Flag to track processing state

        # Monte Carlo Simulation Variables (initialized to None initially)
        self.mc_df = None
        self.mc_stepsize_x = None
        self.mc_stepsize_y = None
        self.mc_r = None
        self.mc_c = None
        self.mc_init_energy_array = None
        self.mc_sigma_energy = None
        self.mc_mean_energy = None
        self.mc_max_energy = None
        self.mc_nucleation_site = []
        self.mc_EA = None  # Energy array for MC sim
        self.mc_s = None  # s array from main analysis, used for initialization
        self.mc_lattice_angle = None
        self.mc_lattice_status = None
        self.mc_grains = []
        self.mc_pixel_size = 10  # Define pixel size here
        self.mc_canvas = None  # Canvas for Monte Carlo simulation
        self.mc_number_of_grains = 100  # Default number of grains for MC sim
        self.mc_M_m = 10  # Default mobility parameter

        self.setup_gui()

    def setup_gui(self):
        """Initialize the main GUI layout"""
        # Create main container with three columns
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Configure grid layout for main container (3 columns now)
        self.main_container.grid_columnconfigure(0, weight=1)  # Setup Section
        self.main_container.grid_columnconfigure(1, weight=3)  # Visualization Section (reduced weight)
        self.main_container.grid_columnconfigure(2, weight=2)  # Analysis Section (increased weight)
        self.main_container.grid_rowconfigure(0, weight=1)

        # Create the three main sections
        self.setup_section = self.create_setup_section()
        self.visualization_section = self.create_visualization_section()
        self.analysis_section = self.create_analysis_section()

    def create_setup_section(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self.main_container)  # Fallback if base class method is not available
        frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Title
        ctk.CTkLabel(frame, text="Setup & Configuration", font=("Arial", 16, "bold")).pack(pady=10)

        # Theta_m input
        theta_frame = ctk.CTkFrame(frame)
        theta_frame.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(theta_frame, text="Theta_m value:").pack(side=tk.LEFT, padx=5)
        self.theta_m_var = tk.StringVar(value="15")
        ctk.CTkEntry(theta_frame, textvariable=self.theta_m_var, width=80).pack(side=tk.LEFT, padx=5)

        # Basic Parameters Section
        basic_frame = ctk.CTkFrame(frame)
        basic_frame.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(basic_frame, text="Basic Parameters", font=("Arial", 12, "bold")).pack(pady=5)

        # Tolerance angle
        tol_frame = ctk.CTkFrame(basic_frame)
        tol_frame.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(tol_frame, text="Tolerance Angle:").pack(side=tk.LEFT, padx=5)
        self.tolerance_var = tk.StringVar(value="5")
        ctk.CTkEntry(tol_frame, textvariable=self.tolerance_var, width=80).pack(side=tk.LEFT, padx=5)

        # Grain boundary energy
        gb_frame = ctk.CTkFrame(basic_frame)
        gb_frame.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(gb_frame, text="Grain Boundary Energy:").pack(side=tk.LEFT, padx=5)
        self.gb_energy_var = tk.StringVar(value="1")
        ctk.CTkEntry(gb_frame, textvariable=self.gb_energy_var, width=80).pack(side=tk.LEFT, padx=5)

        # Advanced Parameters Section
        adv_frame = ctk.CTkFrame(frame)
        adv_frame.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(adv_frame, text="Advanced Parameters", font=("Arial", 12, "bold")).pack(pady=5)

        # Temperature
        temp_frame = ctk.CTkFrame(adv_frame)
        temp_frame.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(temp_frame, text="Temperature (K):").pack(side=tk.LEFT, padx=5)
        self.temperature_var = tk.StringVar(value="300")
        ctk.CTkEntry(temp_frame, textvariable=self.temperature_var, width=80).pack(side=tk.LEFT, padx=5)

        # Iteration steps
        iter_frame = ctk.CTkFrame(adv_frame)
        iter_frame.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(iter_frame, text="Iteration Steps:").pack(side=tk.LEFT, padx=5)
        self.iterations_var = tk.StringVar(value="1000")
        ctk.CTkEntry(iter_frame, textvariable=self.iterations_var, width=80).pack(side=tk.LEFT, padx=5)

        # Visualization Parameters Section
        vis_frame = ctk.CTkFrame(frame)
        vis_frame.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(vis_frame, text="Visualization Parameters", font=("Arial", 12, "bold")).pack(pady=5)

        # Color palette
        palette_frame = ctk.CTkFrame(vis_frame)
        palette_frame.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(palette_frame, text="Color Palette:").pack(side=tk.LEFT, padx=5)
        self.palette_var = tk.StringVar(value="plasma")
        palette_options = ["plasma", "inferno", "magma", "viridis"]
        palette_menu = ctk.CTkOptionMenu(
            palette_frame,
            variable=self.palette_var,
            values=palette_options
        )
        palette_menu.pack(side=tk.LEFT, padx=5)

        # Contour levels
        contour_frame = ctk.CTkFrame(vis_frame)
        contour_frame.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(contour_frame, text="Contour Levels:").pack(side=tk.LEFT, padx=5)
        self.contour_var = tk.StringVar(value="10")
        ctk.CTkEntry(contour_frame, textvariable=self.contour_var, width=80).pack(side=tk.LEFT, padx=5)

        # Input file selection
        ctk.CTkButton(
            frame,
            text="Select Input File",
            command=self.select_input_file
        ).pack(pady=5, padx=10, fill=tk.X)

        self.input_label = ctk.CTkLabel(frame, text="No file selected", wraplength=250)
        self.input_label.pack(pady=5)

        # Process buttons Frame
        self.process_frame = ctk.CTkFrame(frame)
        self.process_frame.pack(pady=5, padx=10, fill=tk.X)

        # Loading indicator (initially hidden)
        self.loading_label = ctk.CTkLabel(
            self.process_frame,
            text="Processing...",
            text_color="gray"
        )

        # Process buttons
        self.hot_button = ctk.CTkButton(
            self.process_frame,
            text="Process Hot-Rolled",
            command=lambda: self.start_processing("hot"),
            fg_color="#DB804E"
        )
        self.hot_button.pack(pady=2, fill=tk.X)

        self.cold_button = ctk.CTkButton(
            self.process_frame,
            text="Process Cold-Rolled",
            command=lambda: self.start_processing("cold"),
            fg_color="#5E8CAD"
        )
        self.cold_button.pack(pady=2, fill=tk.X)

        return frame

    def create_visualization_section(self) -> ctk.CTkFrame:
        """Create the visualization section with plot tabs and Monte Carlo Canvas"""
        frame = ctk.CTkFrame(self.main_container)
        frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        # Title
        ctk.CTkLabel(frame, text="Visualization", font=("Arial", 16, "bold")).pack(pady=10)

        # Tab view for plots
        self.tab_view = ctk.CTkTabview(frame)
        self.tab_view.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add tabs (including Monte Carlo Simulation tab)
        self.kam_tab = self.tab_view.add("KAM Plot")
        self.energy_tab = self.tab_view.add("Energy Plot")
        self.misorientation_tab = self.tab_view.add("Misorientation")
        self.energy_distribution_tab = self.tab_view.add("Energy Distribution")
        self.iq_tab = self.tab_view.add("IQ Plot")  # Tab for IQ Plot
        self.monte_carlo_tab = self.tab_view.add("Monte Carlo Simulation")  # Tab for Monte Carlo

        # Initialize plot containers
        self.plot_containers = {
            "KAM Plot": self.kam_tab,
            "Energy Plot": self.energy_tab,
            "Misorientation": self.misorientation_tab,
            "Energy Distribution": self.energy_distribution_tab,
            "IQ Plot": self.iq_tab,
            "Monte Carlo Simulation": self.monte_carlo_tab  # Container for Monte Carlo plot
        }

        # Monte Carlo Canvas and Buttons in Monte Carlo Tab
        self.setup_monte_carlo_tab()

        return frame

    def setup_monte_carlo_tab(self):
        """Set up canvas and buttons in the Monte Carlo Simulation Tab"""
        mc_frame = self.monte_carlo_tab  # Use the Monte Carlo Tab as the frame

        # Canvas for Monte Carlo Display
        self.mc_canvas = ctk.CTkCanvas(mc_frame, width=400, height=300, bg='white')  # Adjust size as needed
        self.mc_canvas.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Buttons Frame for Monte Carlo Controls
        mc_button_frame = ctk.CTkFrame(mc_frame)
        mc_button_frame.pack(pady=5, padx=10, fill=tk.X)

        # Monte Carlo Step Button
        step_button = ctk.CTkButton(mc_button_frame, text="Monte Carlo Step", command=self.monte_carlo_step)
        step_button.pack(side=tk.LEFT, padx=5)

        # Run All Steps Button
        run_all_button = ctk.CTkButton(mc_button_frame, text="Run All Steps", command=self.run_all_monte_carlo_steps)  # New button
        run_all_button.pack(side=tk.LEFT, padx=5)

        # Print Euler Angles Button
        print_button = ctk.CTkButton(mc_button_frame, text="Print Euler Angles", command=self.print_euler_angles)
        print_button.pack(side=tk.LEFT, padx=5)

        # Save Image Button
        save_button = ctk.CTkButton(mc_button_frame, text="Save Image", command=self.save_canvas_image)
        save_button.pack(side=tk.LEFT, padx=5)

    def create_analysis_section(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self.main_container)  # Fallback if base class method is not available
        frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        # Title
        ctk.CTkLabel(frame, text="Analysis & Export", font=("Arial", 16, "bold")).pack(pady=10)

        # Statistics display
        self.stats_frame = ctk.CTkFrame(frame)
        self.stats_frame.pack(fill=tk.X, padx=10, pady=5)

        # Create labels for statistics
        self.create_stat_label("Average Grain Size:", "N/A")
        self.create_stat_label("Grain Count:", "N/A")
        self.create_stat_label("Avg. Misorientation:", "N/A")
        self.create_stat_label("Recryst. Correction:", "N/A")
        self.create_stat_label("Avg. Energy Density:", "N/A")

        # Export settings
        export_settings = ctk.CTkFrame(frame)
        export_settings.pack(fill=tk.X, padx=10, pady=10)

        ctk.CTkLabel(export_settings, text="Export Settings").pack()

        # Resolution settings
        res_frame = ctk.CTkFrame(export_settings)
        res_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(res_frame, text="DPI:").pack(side=tk.LEFT, padx=5)
        self.dpi_var = tk.StringVar(value="300")
        ctk.CTkEntry(res_frame, textvariable=self.dpi_var, width=80).pack(side=tk.LEFT, padx=5)

        # Format selection
        self.format_var = tk.StringVar(value="JPG")
        ctk.CTkRadioButton(export_settings, text="JPG", variable=self.format_var, value="JPG").pack()
        ctk.CTkRadioButton(export_settings, text="TIFF", variable=self.format_var, value="TIFF").pack()

        # Export button
        ctk.CTkButton(
            frame,
            text="Export All",
            command=self.export_results
        ).pack(pady=10, padx=10, fill=tk.X)

        return frame

    def create_stat_label(self, text: str, value: str):
        frame = ctk.CTkFrame(self.stats_frame)  # Fallback
        frame.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(frame, text=text).pack(side=tk.LEFT, padx=5)
        ctk.CTkLabel(frame, text=value).pack(side=tk.RIGHT, padx=5)
        return frame

    def select_input_file(self):
        filename = filedialog.askopenfilename(  # Fallback
            title="Select Input File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.input_file = filename
            self.input_label.configure(text=os.path.basename(filename))
        return filename

    def update_plots(self, plot_data: dict, plot_title: str):
        try:
            if plot_title not in self.plot_containers:
                raise ValueError(f"Invalid plot type: {plot_title}")

            container_tab = self.plot_containers[plot_title]

            # Clear existing plot if it exists
            if plot_title in self.current_plots:
                self.current_plots[plot_title].get_tk_widget().destroy()

            # Load the image using PIL
            image_path = plot_data['data']
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Plot image not found: {image_path}")

            # Create figure and axis
            fig, ax = plt.subplots(figsize=(8, 6))

            # Read and display the image
            img = plt.imread(image_path)
            ax.imshow(img)
            ax.axis('off')  # Hide axes

            # Set title based on plot type - removed redundant title setting, using tab name as title
            ax.set_title(plot_title)

            # Create canvas and display plot
            canvas = FigureCanvasTkAgg(fig, master=container_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Store canvas reference
            self.current_plots[plot_title] = canvas

        except FileNotFoundError as e:
            self.show_error(f"Plot file not found: {str(e)}")
        except Exception as e:
            self.show_error(f"Error updating {plot_title}: {str(e)}")

    def finish_processing(self):
        self.processing = False
        self.hot_button.configure(state="normal")
        self.cold_button.configure(state="normal")
        self.loading_label.pack_forget()
        self.show_info("Processing completed successfully!")

    def export_results(self):
        export_dir = filedialog.askdirectory(title="Select Export Directory")  # Fallback
        if not export_dir:
            return

        try:
            dpi = int(self.dpi_var.get())
            format_ext = self.format_var.get().lower()

            # Export plots
            for name, canvas in self.current_plots.items():
                fig = canvas.figure
                fig.savefig(
                    os.path.join(export_dir, f"{name}.{format_ext}"),
                    dpi=dpi,
                    format=format_ext
                )

            # Export statistics and other data
            # Add actual export code here

            self.show_info("Export completed successfully!")
        except Exception as e:
            self.show_error(f"Export failed: {str(e)}")
            
    def fetchEA(self, x: int, y: int) -> list:
        """Fetch Euler Angles from the mc_EA array."""
        x = x % (self.mc_r + 1)
        y = y % (self.mc_c + 1)
        return [
            self.mc_EA[x, y, 0],  # phi1
            self.mc_EA[x, y, 1],  # phi
            self.mc_EA[x, y, 2]   # phi2
        ]

    def show_error(self, message: str):
        messagebox.showerror("Error", message)  # Fallback

    def show_info(self, message: str):
        messagebox.showinfo("Information", message)  # Fallback

    ####################################################################################################################
    ###################################### Monte Carlo Simulation Functions ##########################################
    ####################################################################################################################

    class Grain:  # Grain Class - moved from simulation_hot.py and adjusted for class context
        def __init__(self, name, eulerangles, color="red"):
            self.eulerangles = eulerangles
            self.GB = []
            self.newgrainspx = []
            self.name = name
            self.color = color

        def isGB(self, gui_instance):  # Pass gui_instance to access GUI level variables
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if gui_instance.fetchEA(self.coords_px[0] + i, self.coords_px[1] + j) != gui_instance.fetchEA(self.coords_px[0], self.coords_px[1]):
                        return True
            return False

        def updateGB(self, gui_instance):  # Pass gui_instance
            new_gb = []
            for i in self.GB:
                # Temporarily set coords_px for isGB to work correctly within Grain Class
                self.coords_px = i
                if self.isGB(gui_instance):
                    new_gb.append(i)
            self.GB = new_gb  # Update GB list by filtering out non-GB pixels

            for i in self.newgrainspx:  # Add newly grown pixels to GB
                self.GB.append(i)
            self.newgrainspx = []  # Reset new grain pixels after updating GB

        def mobility(self, misorientation, gui_instance):
            B = 5
            K = 5
            M_m = gui_instance.mc_M_m  # Use class level M_m
            theta_m_main = float(gui_instance.theta_m_var.get())  # Get theta_m from GUI input
            return M_m * (1 - (math.exp(-1 * B * ((misorientation / np.radians(theta_m_main)) ** K))))

        def del_E(self, EA_M, EA_1, coords_px, grain_analyzer, gui_instance):  # Pass GrainAnalyzer and gui_instance
            SE_i = 0
            SE_f = 0
            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    if (coords_px[0] + x >= gui_instance.mc_r) or (coords_px[1] + y >= gui_instance.mc_c):
                        pass  # Handle boundaries if needed, currently pass
                    else:
                        EA_neighbor = gui_instance.fetchEA(coords_px[0] + x, coords_px[1] + y)
                        del_g_initial = np.matmul(GrainAnalyzer.compute_orientation_matrix(EA_1[0], EA_1[1], EA_1[2]),
                                                    np.linalg.inv(GrainAnalyzer.compute_orientation_matrix(EA_neighbor[0], EA_neighbor[1], EA_neighbor[2])))
                        misorientation_initial = GrainAnalyzer.compute_misorientation(del_g_initial)
                        SE_i = SE_i + grain_analyzer.stored_energy(misorientation_initial)

            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    if (coords_px[0] + x >= gui_instance.mc_r) or (coords_px[1] + y >= gui_instance.mc_c):
                        pass  # Handle boundaries if needed, currently pass
                    else:
                        EA_neighbor = gui_instance.fetchEA(coords_px[0] + x, coords_px[1] + y)
                        del_g_final = np.matmul(GrainAnalyzer.compute_orientation_matrix(EA_M[0], EA_M[1], EA_M[2]),
                                                  np.linalg.inv(GrainAnalyzer.compute_orientation_matrix(EA_neighbor[0], EA_neighbor[1], EA_neighbor[2])))
                        misorientation_final = GrainAnalyzer.compute_misorientation(del_g_final)
                        SE_f = SE_f + grain_analyzer.stored_energy(misorientation_final)

            return SE_f - SE_i

        def probability(self, del_E, misorientation, gui_instance):
            if del_E <= 0:
                theta_m_main = float(gui_instance.theta_m_var.get())  # Get theta_m from GUI input
                return (self.mobility(misorientation, gui_instance) * GrainAnalyzer(SimulationParameters(theta_M=theta_m_main)).stored_energy(misorientation) * 2) / (gui_instance.mc_M_m * sigma_m)
            else:
                theta_m_main = float(gui_instance.theta_m_var.get())  # Get theta_m from GUI input
                return (self.mobility(misorientation, gui_instance) * GrainAnalyzer(SimulationParameters(theta_M=theta_m_main)).stored_energy(misorientation) * 2) * (np.exp(-1 * del_E)) / (gui_instance.mc_M_m * sigma_m)

        def state_change(self, coords_px, gui_instance, grain_analyzer):
            EA_at_coords = gui_instance.fetchEA(coords_px[0], coords_px[1])
            pixel_state_initial = gui_instance.fetchEA(coords_px[0], coords_px[1])
            misorientation_val = np.degrees(GrainAnalyzer.compute_misorientation(
                np.matmul(
                    GrainAnalyzer.compute_orientation_matrix(self.eulerangles[0], self.eulerangles[1], self.eulerangles[2]),
                    np.linalg.inv(GrainAnalyzer.compute_orientation_matrix(pixel_state_initial[0], pixel_state_initial[1], pixel_state_initial[2]))
                )
            ))
            prob = self.probability(self.del_E(self.eulerangles, EA_at_coords, coords_px, grain_analyzer, gui_instance), np.radians(misorientation_val), gui_instance)

            if random.uniform(0, 1) <= prob:
                gui_instance.mc_EA[coords_px[0] % (gui_instance.mc_r + 1), coords_px[1] % (gui_instance.mc_c + 1), 0] = self.eulerangles[0]
                gui_instance.mc_EA[coords_px[0] % (gui_instance.mc_r + 1), coords_px[1] % (gui_instance.mc_c + 1), 1] = self.eulerangles[1]
                gui_instance.mc_EA[coords_px[0] % (gui_instance.mc_r + 1), coords_px[1] % (gui_instance.mc_c + 1), 2] = self.eulerangles[2]
                self.newgrainspx.append([coords_px[0] % (gui_instance.mc_r + 1), coords_px[1] % (gui_instance.mc_c + 1)])
                gui_instance.mc_lattice_status[coords_px[0] % (gui_instance.mc_r + 1), coords_px[1] % (gui_instance.mc_c + 1)] = self.color
            else:
                pass

    def print_euler_angles(self):
        try:
            with open(f"sim_output_n={self.mc_number_of_grains}.txt", "w") as f:
                f.write("phi1,phi,phi2,X,Y,IQ\n")

            for x in range(0, self.mc_r + 1):
                for y in range(0, self.mc_c + 1):
                    with open(f"sim_output_n={self.mc_number_of_grains}.txt", "a") as f:
                        f.write("%s,%s,%s,%s,%s,%s\n" % (self.mc_EA[x, y, 0], self.mc_EA[x, y, 1], self.mc_EA[x, y, 2], x * self.mc_stepsize_x, y * self.mc_stepsize_y, 60))
            self.show_info("Euler angles printed to sim_output_n file.")
        except Exception as e:
            self.show_error(f"Error printing Euler angles: {e}")

    def generate_random_color(self):
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        color_string = "#{:02X}{:02X}{:02X}".format(red, green, blue)
        return color_string

    def save_canvas_image(self):
        try:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                        filetypes=[("PNG file", "*.png"), ("All files", "*.*")])
            if file_path:
                EpsImagePlugin.gs_windows_binary = r'C:\Program Files (x86)\gs\gs10.02.1\bin\gswin32c.exe'  # Set Ghostscript path - IMPORTANT: Adjust this path if needed
                self.mc_canvas.postscript(file="output_image.eps", colormode='color')
                img = Image.open("output_image.eps")
                img.save(file_path, format="png")
                img.close()
                os.remove("output_image.eps")  # Clean up temp eps file
                self.show_info(f"Monte Carlo Image saved to {file_path}")
        except FileNotFoundError:  # Ghostscript not found or path issue
            self.show_error("Ghostscript not found. Please install it and ensure the path in code is correct.")
        except Exception as e:  # Other potential saving errors
            self.show_error(f"Error saving Monte Carlo image: {str(e)}")

    def update_display(self):
        try:
            self.mc_canvas.delete("all")
            for i in range(self.mc_r):
                for j in range(self.mc_c):
                    color = self.mc_lattice_status[i, j] if self.mc_lattice_status[i, j] != 0 else 'white'  # Default to white if no color assigned
                    self.mc_canvas.create_rectangle(i * self.mc_pixel_size, j * self.mc_pixel_size, (i + 1) * self.mc_pixel_size, (j + 1) * self.mc_pixel_size, fill=color, outline="")  # removed outline for cleaner look
            self.mc_canvas.update()
        except Exception as e:
            self.show_error(f"Error updating Monte Carlo display: {str(e)}")

    def monte_carlo_step(self, num_steps=30):
        if not self.mc_grains or not self.mc_nucleation_site:
            self.show_error("Monte Carlo simulation not initialized. Please process input file first.")
            return

        try:
            grain_analyzer = GrainAnalyzer(SimulationParameters())  # Instance of GrainAnalyzer needed for energy calculations

            m = 0
            while m < num_steps:
                for i in self.mc_grains:
                    for j in list(i.GB):  # Iterate over a copy to avoid modification during iteration
                        for x in [-1, 0, 1]:
                            for y in [-1, 0, 1]:
                                coords_to_check = [(j[0] + x) % (self.mc_r + 1), (j[1] + y) % (self.mc_c + 1)]
                                if self.mc_lattice_status[coords_to_check[0], coords_to_check[1]] == 0:  # Check for empty lattice site
                                    EA_at_coords = self.fetchEA(coords_to_check[0], coords_to_check[1])
                                    if i.eulerangles.tolist() != EA_at_coords:  # Check if pixel is not already part of this grain (comparing as lists for numpy arrays)
                                        i.coords_px = coords_to_check # Set coords_px for the Grain object temporarily for energy and mobility calculations
                                        i.state_change(coords_to_check, self, grain_analyzer) # Pass gui_instance and grain_analyzer
                    i.updateGB(self)  # Pass self (GUI instance) to updateGB
                m += 1
            self.update_display()
        except Exception as e:
            self.show_error(f"Error in Monte Carlo Step: {str(e)}")

    def run_all_monte_carlo_steps(self, num_steps=100):
        if not self.mc_grains or not self.mc_nucleation_site:
            self.show_error("Monte Carlo simulation not initialized. Please process input file first.")
            return

        try:
            # Run all steps at once and update display once
            self.monte_carlo_step(num_steps=num_steps)
            self.show_info(f"Completed {num_steps} Monte Carlo steps.")
        except Exception as e:
            self.show_error(f"Error in Run All Monte Carlo Steps: {e}")

    def initialize_monte_carlo_simulation(self):
        """Initializes Monte Carlo simulation variables and grains after main analysis."""
        if self.output_files_main_analysis and 'csv' in self.output_files_main_analysis and 's_array' in self.output_files_main_analysis:
            try:
                # Load data from main analysis outputs
                df_path = self.output_files_main_analysis['csv']
                s_array_path = self.output_files_main_analysis['s_array']

                self.mc_df = pd.read_csv(df_path)
                self.mc_df = self.mc_df.to_numpy()
                self.mc_s = np.load(s_array_path)  # Load s_array from saved .npy file

                # Determine stepsizes, dimensions, energy statistics, nucleation sites as in simulation_hot.py using mc_df
                for i in range(len(self.mc_df[:, 0]) - 1):  # Corrected loop range
                    if abs(self.mc_df[i + 1, 0] - self.mc_df[i, 0]) > 1e-6:
                        self.mc_stepsize_x = abs(self.mc_df[i + 1, 0] - self.mc_df[i, 0])
                        break

                for i in range(len(self.mc_df[:, 1]) - 1):  # Corrected loop range
                    if abs(self.mc_df[i + 1, 1] - self.mc_df[i, 1]) > 1e-6:
                        self.mc_stepsize_y = abs(self.mc_df[i + 1, 1] - self.mc_df[i, 1])
                        break

                self.mc_df[:, 0] = (self.mc_df[:, 0] / (self.mc_stepsize_x)).astype(int)
                self.mc_df[:, 1] = (self.mc_df[:, 1] / (self.mc_stepsize_y)).astype(int)

                self.mc_r, self.mc_c = int(np.max(self.mc_df[:, 0])) + 1, int(np.max(self.mc_df[:, 1])) + 1

                self.mc_init_energy_array = self.mc_df[:, 5]

                self.mc_sigma_energy = np.std(self.mc_init_energy_array)
                self.mc_mean_energy = np.mean(self.mc_init_energy_array)
                self.mc_max_energy = np.max(self.mc_init_energy_array)

                self.mc_nucleation_site = []
                for i in self.mc_df:
                    if i[5] >= (self.mc_mean_energy + 2 * self.mc_sigma_energy):
                        self.mc_nucleation_site.append([i[0], i[1]])

                self.mc_EA = np.zeros((self.mc_r + 1, self.mc_c + 1, 3))
                self.mc_lattice_angle = np.zeros((self.mc_r + 1, self.mc_c + 1, 1))
                self.mc_lattice_status = np.zeros((self.mc_r + 1, self.mc_c + 1), object)  # Stores colors

                # Initialize EA from s_array (using s array from main analysis)
                for x in range(self.mc_r + 1):
                    for y in range(self.mc_c + 1):
                        self.mc_EA[x, y, 0] = self.mc_s[x, y, 0]  # phi1
                        self.mc_EA[x, y, 1] = self.mc_s[x, y, 1]  # phi
                        self.mc_EA[x, y, 2] = self.mc_s[x, y, 2]  # phi2

                self.mc_grains = []  # Initialize grains list
                for i in range(1, self.mc_number_of_grains + 1):
                    a = np.random.randint(0, len(self.mc_nucleation_site))
                    nuclii_x = int(self.mc_nucleation_site[a][0])
                    nuclii_y = int(self.mc_nucleation_site[a][1])
                    obj_name = f"grain {i}"
                    initial_ea = self.fetchEA(nuclii_x, nuclii_y)
                    new_grain_obj = self.Grain(obj_name, np.array(initial_ea), color=self.generate_random_color())
                    new_grain_obj.GB.append([nuclii_x, nuclii_y])
                    self.mc_grains.append(new_grain_obj)
                    self.mc_lattice_status[nuclii_x, nuclii_y] = new_grain_obj.color

                self.update_display()  # Initial display of grain structure

                self.show_info("Monte Carlo simulation initialized.")

            except FileNotFoundError:
                self.show_error("Output files from main analysis not found.")
            except Exception as e:
                self.show_error(f"Error initializing Monte Carlo simulation: {str(e)}")
        else:
            self.show_error("Main analysis output files not found. Run 'Process Hot-Rolled' or 'Process Cold-Rolled' first.")

    def start_processing(self, process_type):
        """Starts the data processing in a separate thread."""
        if self.processing:
            self.show_info("Processing already in progress. Please wait.")
            return

        if not self.input_file:
            self.show_error("Please select an input file first.")
            return

        try:
            # Get parameters from GUI inputs
            params = SimulationParameters(
                theta_M=float(self.theta_m_var.get()),
                tolerance_angle=float(self.tolerance_var.get()),
                grain_boundary_energy=float(self.gb_energy_var.get()),
                temperature=float(self.temperature_var.get()),
                iteration_steps=int(self.iterations_var.get()),
                color_palette=self.palette_var.get(),
                contour_levels=int(self.contour_var.get())
            )

            self.processing = True
            self.hot_button.configure(state="disabled")
            self.cold_button.configure(state="disabled")
            self.loading_label.pack(pady=5)
            self.root.update_idletasks()  # Force GUI update to show loading label

            # Start processing thread
            thread = threading.Thread(
                target=self.process_data_thread,
                args=(self.input_file, process_type, params)  # Pass parameters and process_type
            )
            thread.start()

        except ValueError as ve:
            self.finish_processing()  # Ensure buttons are re-enabled even if error occurs
            self.show_error(f"Invalid input value: {str(ve)}")
        except Exception as e:
            self.finish_processing()  # Ensure buttons are re-enabled even if error occurs
            self.show_error(f"Error starting processing: {str(e)}")

    def process_data_thread(self, input_file, process_type, params):
        """Process data in a thread to prevent GUI blocking."""
        try:
            # Run main analysis and get output file paths
            output_files = run_main_analysis(input_file, process_type, params)  # Pass params and process_type

            # Generate plots and energy distribution plot
            plot_data_mapping = [
                ('Energy Plot', 'stored_energy'),
                ('Misorientation', 'average_misorientation'),
                ('KAM Plot', 'kam'),
                ('IQ Plot', 'iq')
            ]

            plot_output_data = {}
            for plot_title, output_key in plot_data_mapping:
                if output_key in output_files:
                    plot_output_data[plot_title] = {'data': output_files[output_key]}
            
            energy_dist_plot_path = create_energy_distribution_plot(output_files['csv'], GrainAnalyzer(params).output_dir)  # Pass output_dir
            plot_output_data['Energy Distribution'] = {'data': energy_dist_plot_path}  # Add energy distribution plot path

            # Update plots in GUI
            self.root.after(0, self.update_gui_plots, plot_output_data)

            # Initialize Monte Carlo Simulation after main analysis
            self.output_files_main_analysis = output_files  # Store output files for MC initialization
            self.root.after(0, self.initialize_monte_carlo_simulation)  # Initialize MC after plots are updated

            # Finish processing and update GUI components
            self.root.after(0, self.finish_processing)

        except RuntimeError as re:
            self.root.after(0, self.finish_processing)  # Ensure processing flag is reset even on error
            self.root.after(0, self.show_error, str(re))  # Show error in GUI thread

    def update_gui_plots(self, plot_data):
        """Update plots in the GUI based on processed data."""
        for plot_title, data in plot_data.items():
            self.update_plots(data, plot_title)  # Call plot update for each plot type

    def run_process_thread(self, process_type: Literal["hot", "cold"]):
        output_files = {}
        try:
            # Get parameters from GUI
            params = SimulationParameters(
                theta_M=float(self.theta_m_var.get()),
                tolerance_angle=float(self.tolerance_var.get()),
                grain_boundary_energy=float(self.gb_energy_var.get()),
                temperature=float(self.temperature_var.get()),
                iteration_steps=int(self.iterations_var.get()),
                color_palette=self.palette_var.get(),
                contour_levels=int(self.contour_var.get())
            )

            # Run main analysis first
            main_output = run_main_analysis(self.input_file, process_type, params) # Pass process_type and params
            output_files.update(main_output)

            # Run energy distribution analysis
            energy_distribution_plot_path = create_energy_distribution_plot(
                output_files['csv'],
                GrainAnalyzer(params).output_dir # Use GrainAnalyzer with params to get output_dir
            )
            output_files['energy_distribution'] = energy_distribution_plot_path

            plot_data_mapping_main = {
                "KAM Plot": {'data': output_files['kam']},
                "Energy Plot": {'data': output_files['stored_energy']},
                "Misorientation": {'data': output_files['average_misorientation']},
                "Energy Distribution": {'data': output_files['energy_distribution']},
                "IQ Plot": {'data': output_files['iq']}  # Use 'iq' key here which is lowercase from main.py output_files
            }

            for plot_title, data in plot_data_mapping_main.items():
                self.root.after(0, self.update_plots, data, plot_title)

            if process_type == "hot":
                # No direct Monte Carlo plot path returned anymore. Simulation is run and displayed on canvas.
                pass  # Monte Carlo simulation is initialized and run within GUI now

            elif process_type == "cold":
                # Placeholder for cold rolled simulation
                monte_carlo_plot_path = "path/to/cold_rolled_simulation_plot.png"  # Replace with actual cold rolled sim function
                self.root.after(0, self.update_plots, {'data': monte_carlo_plot_path}, "Monte Carlo Simulation")
            else:
                raise ValueError(f"Unknown process type: {process_type}")

            # Store output file paths from main analysis so Monte Carlo can access them later
            self.output_files_main_analysis = output_files  # Store output file paths

            # Initialize Monte Carlo Simulation after main analysis is complete and files are saved
            self.root.after(0, self.initialize_monte_carlo_simulation)  # Initialize MC in main thread

        except FileNotFoundError as fnf_error:
            self.root.after(0, self.show_error, f"File not found: {str(fnf_error)}")
        except ValueError as ve:
            self.root.after(0, self.show_error, f"Invalid parameter: {str(ve)}")
        except pd.errors.EmptyDataError as empty_data_error:
            self.root.after(0, self.show_error, f"No data in input file: {str(empty_data_error)}")
        except Exception as e:
            self.root.after(0, self.show_error, f"Processing failed: {str(e)}")
        finally:
            self.root.after(0, self.finish_processing)

    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = MaterialAnalysisGUI()
    app.run()