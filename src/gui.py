import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from PIL import Image, ImageTk, EpsImagePlugin
from typing import Optional, Literal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from main import run_main_analysis, SimulationParameters, GrainAnalyzer
import threading
from energy_distribution import create_energy_distribution_plot
from datetime import datetime
import random
import math

# Import simulation logics
import simulation_hot
import simulation_cold

class MaterialAnalysisGUI:
    output_files_main_analysis = None
    mc_hot_simulation = None
    mc_cold_simulation = None

    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Material Analysis Tool")
        self.root.geometry("1400x900")
        self.input_file: Optional[str] = None
        self.output_file: Optional[str] = None
        self.current_plots = {}
        self.processing = False
        self.mc_canvas = None
        self.setup_gui()

    def setup_gui(self):
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.main_container.grid_columnconfigure(0, weight=1, minsize=300)  # Setup section
        self.main_container.grid_columnconfigure(1, weight=3, minsize=800)  # Visualization
        self.main_container.grid_columnconfigure(2, weight=1, minsize=300)  # Analysis
        self.main_container.grid_rowconfigure(0, weight=1)
        self.setup_section = self.create_setup_section()
        self.visualization_section = self.create_visualization_section()
        self.analysis_section = self.create_analysis_section()

    def create_setup_section(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self.main_container)
        frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        ctk.CTkLabel(frame, text="Setup & Configuration", font=("Arial", 16, "bold")).pack(pady=10)
        theta_frame = ctk.CTkFrame(frame)
        theta_frame.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(theta_frame, text="Theta_m value:").pack(side=tk.LEFT, padx=5)
        self.theta_m_var = tk.StringVar(value="15")
        ctk.CTkEntry(theta_frame, textvariable=self.theta_m_var, width=80).pack(side=tk.LEFT, padx=5)
        basic_frame = ctk.CTkFrame(frame)
        basic_frame.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(basic_frame, text="Basic Parameters", font=("Arial", 12, "bold")).pack(pady=5)
        tol_frame = ctk.CTkFrame(basic_frame)
        tol_frame.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(tol_frame, text="Tolerance Angle (KAM):").pack(side=tk.LEFT, padx=5)
        self.tolerance_var = tk.StringVar(value="5")
        ctk.CTkEntry(tol_frame, textvariable=self.tolerance_var, width=80).pack(side=tk.LEFT, padx=5)
        gb_frame = ctk.CTkFrame(basic_frame)
        gb_frame.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(gb_frame, text="Grain Boundary Energy:").pack(side=tk.LEFT, padx=5)
        self.gb_energy_var = tk.StringVar(value="1")
        ctk.CTkEntry(gb_frame, textvariable=self.gb_energy_var, width=80).pack(side=tk.LEFT, padx=5)
        adv_frame = ctk.CTkFrame(frame)
        adv_frame.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(adv_frame, text="Advanced Parameters", font=("Arial", 12, "bold")).pack(pady=5)
        grains_frame = ctk.CTkFrame(adv_frame)
        grains_frame.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(grains_frame, text="Number of Grains:").pack(side=tk.LEFT, padx=5)
        self.num_grains_var = tk.StringVar(value="100")
        ctk.CTkEntry(grains_frame, textvariable=self.num_grains_var, width=80).pack(side=tk.LEFT, padx=5)
        temp_frame = ctk.CTkFrame(adv_frame)
        temp_frame.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(temp_frame, text="Temperature (K):").pack(side=tk.LEFT, padx=5)
        self.temperature_var = tk.StringVar(value="300")
        ctk.CTkEntry(temp_frame, textvariable=self.temperature_var, width=80).pack(side=tk.LEFT, padx=5)
        
        iter_frame = ctk.CTkFrame(adv_frame)
        iter_frame.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(iter_frame, text="Iteration Steps:").pack(side=tk.LEFT, padx=5)
        self.iterations_var = tk.StringVar(value="1000")
        ctk.CTkEntry(iter_frame, textvariable=self.iterations_var, width=80).pack(side=tk.LEFT, padx=5)
        vis_frame = ctk.CTkFrame(frame)
        vis_frame.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(vis_frame, text="Visualization Parameters", font=("Arial", 12, "bold")).pack(pady=5)
        palette_frame = ctk.CTkFrame(vis_frame)
        palette_frame.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(palette_frame, text="Color Palette:").pack(side=tk.LEFT, padx=5)
        self.palette_var = tk.StringVar(value="plasma")
        palette_options = ["plasma", "inferno", "magma", "viridis"]
        palette_menu = ctk.CTkOptionMenu(palette_frame, variable=self.palette_var, values=palette_options)
        palette_menu.pack(side=tk.LEFT, padx=5)
        contour_frame = ctk.CTkFrame(vis_frame)
        contour_frame.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(contour_frame, text="Contour Levels:").pack(side=tk.LEFT, padx=5)
        self.contour_var = tk.StringVar(value="10")
        ctk.CTkEntry(contour_frame, textvariable=self.contour_var, width=80).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(frame, text="Select Input File", command=self.select_input_file).pack(pady=5, padx=10, fill=tk.X)
        self.input_label = ctk.CTkLabel(frame, text="No file selected", wraplength=250)
        self.input_label.pack(pady=5)
        self.process_frame = ctk.CTkFrame(frame)
        self.process_frame.pack(pady=5, padx=10, fill=tk.X)
        self.loading_label = ctk.CTkLabel(self.process_frame, text="Processing...", text_color="gray")
        self.hot_button = ctk.CTkButton(self.process_frame, text="Process Hot-Rolled", command=lambda: self.start_processing("hot"), fg_color="#DB804E")
        self.hot_button.pack(pady=2, fill=tk.X)
        self.cold_button = ctk.CTkButton(self.process_frame, text="Process Cold-Rolled", command=lambda: self.start_processing("cold"), fg_color="#5E8CAD")
        self.cold_button.pack(pady=2, fill=tk.X)
        return frame

    def create_visualization_section(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self.main_container)
        frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        ctk.CTkLabel(frame, text="Visualization", font=("Arial", 16, "bold")).pack(pady=10)
        self.tab_view = ctk.CTkTabview(frame)
        self.tab_view.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.kam_tab = self.tab_view.add("KAM Plot")
        self.energy_tab = self.tab_view.add("Energy Plot")
        self.misorientation_tab = self.tab_view.add("Misorientation")
        self.energy_distribution_tab = self.tab_view.add("Energy Distribution")
        self.iq_tab = self.tab_view.add("IQ Plot")
        self.monte_carlo_tab = self.tab_view.add("Monte Carlo Simulation")
        self.plot_containers = {
            "KAM Plot": self.kam_tab,
            "Energy Plot": self.energy_tab,
            "Misorientation": self.misorientation_tab,
            "Energy Distribution": self.energy_distribution_tab,
            "IQ Plot": self.iq_tab,
            "Monte Carlo Simulation": self.monte_carlo_tab
        }
        self.setup_monte_carlo_tab()
        return frame

    def setup_monte_carlo_tab(self):
        mc_frame = self.monte_carlo_tab
        self.mc_canvas = ctk.CTkCanvas(mc_frame, width=400, height=300, bg='white')
        self.mc_canvas.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        mc_button_frame = ctk.CTkFrame(mc_frame)
        mc_button_frame.pack(pady=5, padx=10, fill=tk.X)
        
        # Store buttons as instance variables
        self.step_button = ctk.CTkButton(mc_button_frame, text="Monte Carlo Step", command=self.monte_carlo_step)
        self.step_button.pack(side=tk.LEFT, padx=5)
        self.run_all_button = ctk.CTkButton(mc_button_frame, text="Run All Steps", command=self.run_all_monte_carlo_steps)
        self.run_all_button.pack(side=tk.LEFT, padx=5)
        self.print_button = ctk.CTkButton(mc_button_frame, text="Print Euler Angles", command=self.print_euler_angles)
        self.print_button.pack(side=tk.LEFT, padx=5)
        self.save_button = ctk.CTkButton(mc_button_frame, text="Save Image", command=self.save_canvas_image)
        self.save_button.pack(side=tk.LEFT, padx=5)

    def create_analysis_section(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self.main_container)
        frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        ctk.CTkLabel(frame, text="Analysis & Export", font=("Arial", 16, "bold")).pack(pady=10)
        self.stats_frame = ctk.CTkFrame(frame)
        self.stats_frame.pack(fill=tk.X, padx=10, pady=5)
        self.create_stat_label("Average Grain Size:", "N/A")
        self.create_stat_label("Grain Count:", "N/A")
        self.create_stat_label("Avg. Misorientation:", "N/A")
        self.create_stat_label("Recryst. Correction:", "N/A")
        self.create_stat_label("Avg. Energy Density:", "N/A")
        export_settings = ctk.CTkFrame(frame)
        export_settings.pack(fill=tk.X, padx=10, pady=10)
        ctk.CTkLabel(export_settings, text="Export Settings").pack()
        res_frame = ctk.CTkFrame(export_settings)
        res_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(res_frame, text="DPI:").pack(side=tk.LEFT, padx=5)
        self.dpi_var = tk.StringVar(value="300")
        ctk.CTkEntry(res_frame, textvariable=self.dpi_var, width=80).pack(side=tk.LEFT, padx=5)
        self.format_var = tk.StringVar(value="JPG")
        ctk.CTkRadioButton(export_settings, text="JPG", variable=self.format_var, value="JPG").pack()
        ctk.CTkRadioButton(export_settings, text="TIFF", variable=self.format_var, value="TIFF").pack()
        ctk.CTkButton(frame, text="Export All", command=self.export_results).pack(pady=10, padx=10, fill=tk.X)
        return frame

    def create_stat_label(self, text: str, value: str):
        frame = ctk.CTkFrame(self.stats_frame)
        frame.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(frame, text=text).pack(side=tk.LEFT, padx=5)
        ctk.CTkLabel(frame, text=value).pack(side=tk.RIGHT, padx=5)
        return frame

    def select_input_file(self):
        filename = filedialog.askopenfilename(title="Select Input File", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if filename:
            self.input_file = filename
            self.input_label.configure(text=os.path.basename(filename))
        return filename

    def update_plots(self, plot_data: dict, plot_title: str):
        try:
            if plot_title not in self.plot_containers:
                raise ValueError(f"Invalid plot type: {plot_title}")
            container_tab = self.plot_containers[plot_title]
            if plot_title in self.current_plots:
                self.current_plots[plot_title].get_tk_widget().destroy()
            image_path = plot_data['data']
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Plot image not found: {image_path}")
            fig, ax = plt.subplots(figsize=(8, 6))
            img = plt.imread(image_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(plot_title)
            canvas = FigureCanvasTkAgg(fig, master=container_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
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
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        if not export_dir:
            return
        try:
            dpi = int(self.dpi_var.get())
            format_ext = self.format_var.get().lower()
            for name, canvas in self.current_plots.items():
                fig = canvas.figure
                fig.savefig(os.path.join(export_dir, f"{name}.{format_ext}"), dpi=dpi, format=format_ext)
            self.show_info("Export completed successfully!")
        except Exception as e:
            self.show_error(f"Export failed: {str(e)}")

    def show_error(self, message: str):
        messagebox.showerror("Error", message)

    def show_info(self, message: str):
        messagebox.showinfo("Information", message)

    def start_processing(self, process_type):
        if self.processing:
            self.show_info("Processing already in progress. Please wait.")
            return
        if not self.input_file:
            self.show_error("Please select an input file first.")
            return
        try:
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
            self.root.update_idletasks()
            thread = threading.Thread(target=self.process_data_thread, args=(self.input_file, process_type, params))
            thread.start()
        except ValueError as ve:
            self.finish_processing()
            self.show_error(f"Invalid input value: {str(ve)}")
        except Exception as e:
            self.finish_processing()
            self.show_error(f"Error starting processing: {str(e)}")

    def process_data_thread(self, input_file, process_type, params):
        try:
            output_files = run_main_analysis(input_file, process_type, params)
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
            energy_dist_plot_path = create_energy_distribution_plot(output_files['csv'], GrainAnalyzer(params).output_dir)
            plot_output_data['Energy Distribution'] = {'data': energy_dist_plot_path}
            self.root.after(0, self.update_gui_plots, plot_output_data)
            self.output_files_main_analysis = output_files
            if process_type == "hot":
                self.root.after(0, self.initialize_hot_simulation)
            elif process_type == "cold":
                self.root.after(0, self.initialize_cold_simulation)
            self.root.after(0, self.finish_processing)
        except RuntimeError as re:
            self.root.after(0, self.finish_processing)
            self.root.after(0, self.show_error, str(re))

    def update_gui_plots(self, plot_data):
        for plot_title, data in plot_data.items():
            self.update_plots(data, plot_title)

    def initialize_hot_simulation(self):
        if self.output_files_main_analysis and 'csv' in self.output_files_main_analysis and 's_array' in self.output_files_main_analysis:
            try:
                num_grains = int(self.num_grains_var.get())
                df_path = self.output_files_main_analysis['csv']
                s_array_path = self.output_files_main_analysis['s_array']
                df = pd.read_csv(df_path)
                s_array = np.load(s_array_path)
                params = SimulationParameters(theta_M=float(self.theta_m_var.get()))
                self.mc_hot_simulation = simulation_hot.HotSimulation(df, s_array, self.mc_canvas, params, num_grains=num_grains)
                self.mc_hot_simulation.initialize_grains()
            except FileNotFoundError:
                self.show_error("Output files from main analysis not found for hot simulation.")
            except Exception as e:
                self.show_error(f"Error initializing hot simulation: {str(e)}")
        else:
            self.show_error("Main analysis output files not found. Run 'Process Hot-Rolled' first.")

    def initialize_cold_simulation(self):
        if self.output_files_main_analysis and 'csv' in self.output_files_main_analysis and 's_array' in self.output_files_main_analysis:
            try:
                df_path = self.output_files_main_analysis['csv']
                s_array_path = self.output_files_main_analysis['s_array']
                df = pd.read_csv(df_path)
                s_array = np.load(s_array_path)
                params = SimulationParameters(theta_M=float(self.theta_m_var.get()))
                self.mc_cold_simulation = simulation_cold.ColdSimulation(df, s_array, self.mc_canvas, params)
                self.mc_cold_simulation.initialize_grains()
            except FileNotFoundError:
                self.show_error("Output files from main analysis not found for cold simulation.")
            except Exception as e:
                self.show_error(f"Error initializing cold simulation: {str(e)}")
        else:
            self.show_error("Main analysis output files not found. Run 'Process Cold-Rolled' first.")

    def set_mc_buttons_state(self, enabled: bool):
        """Enable or disable Monte Carlo buttons"""
        state = "normal" if enabled else "disabled"
        self.step_button.configure(state=state)
        self.run_all_button.configure(state=state)
        self.print_button.configure(state=state)
        self.save_button.configure(state=state)

    def monte_carlo_step(self):
        if self.tab_view.get() == "Monte Carlo Simulation":
            self.set_mc_buttons_state(False)
            try:
                if self.mc_hot_simulation:
                    self.mc_hot_simulation.monte_carlo_step()
                elif self.mc_cold_simulation:
                    self.mc_cold_simulation.monte_carlo_step()
            finally:
                self.set_mc_buttons_state(True)
                self.calculate_and_display_grain_size()

    def run_all_monte_carlo_steps(self):
        if self.tab_view.get() == "Monte Carlo Simulation":
            self.set_mc_buttons_state(False)
            try:
                if self.mc_hot_simulation:
                    self.mc_hot_simulation.run_all_steps()
                elif self.mc_cold_simulation:
                    self.mc_cold_simulation.run_all_steps()
            finally:
                self.set_mc_buttons_state(True)
                self.calculate_and_display_grain_size()

    def calculate_and_display_grain_size(self):
        """Calculate grain size and update the GUI"""
        try:
            if self.mc_hot_simulation:
                simulation = self.mc_hot_simulation
            elif self.mc_cold_simulation:
                simulation = self.mc_cold_simulation
            else:
                return
                
            # Get the current EA data
            ea_data = simulation.EA
            r, c = ea_data.shape[:2]
            
            # Calculate orientation matrices
            G = np.zeros((r, c, 2, 3, 3))
            for x in range(r):
                for y in range(c):
                    phi1, phi, phi2 = ea_data[x, y]
                    G[x, y, 0] = GrainAnalyzer.compute_orientation_matrix(phi1, phi, phi2)
                    G[x, y, 1] = np.linalg.inv(G[x, y, 0])
            
            # Calculate grain size
            from grain_size import calculate_grain_stats
            grain_stats = calculate_grain_stats(
                r, c, 
                1, 1,  # Assuming step size of 1 for the simulation
                G, 
                GrainAnalyzer.compute_misorientation
            )
            
            # Update the GUI with grain size stats
            self.update_stat_labels({
                "Average Grain Size:": f"{grain_stats['avg_x']:.2f} µm (X), {grain_stats['avg_y']:.2f} µm (Y)",
                "Grain Count:": str(len(simulation.grains))
            })
            
        except Exception as e:
            print(f"Error calculating grain size: {e}")
            
    def update_stat_labels(self, updates: dict):
        """Update specific statistic labels in the GUI"""
        for widget in self.stats_frame.winfo_children():
            if isinstance(widget, ctk.CTkFrame):
                label = widget.winfo_children()[0]
                if label.cget("text") in updates:
                    value_label = widget.winfo_children()[1]
                    value_label.configure(text=updates[label.cget("text")])

    def print_euler_angles(self):
        if self.mc_hot_simulation:
            self.mc_hot_simulation.print_euler_angles()
        elif self.mc_cold_simulation:
            self.mc_cold_simulation.print_euler_angles()

    def save_canvas_image(self):
        if self.mc_hot_simulation:
            self.mc_hot_simulation.save_canvas_image()
        elif self.mc_cold_simulation:
            self.mc_cold_simulation.save_canvas_image()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = MaterialAnalysisGUI()
    app.run()