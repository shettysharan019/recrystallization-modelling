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
import csv

# Import simulation logics
import simulation_hot

class MaterialAnalysisGUI:
    output_files_main_analysis = None
    mc_hot_simulation = None

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
        
        # Phase transition animation state
        self.phase_transition_animation_active = False
        self.phase_animation_frame = 0
        self.phase_animation_colors = ["#FF5500", "#FFB380", "#FFCCAA", "#FFE6D5", "#FFFFFF", "#FFE6D5", "#FFCCAA", "#FFB380", "#FF5500"]

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
        ctk.CTkLabel(gb_frame, text="Grain Boundary Mobility:").pack(side=tk.LEFT, padx=5)
        self.gb_energy_var = tk.StringVar(value="1")
        ctk.CTkEntry(gb_frame, textvariable=self.gb_energy_var, width=80).pack(side=tk.LEFT, padx=5)
        adv_frame = ctk.CTkFrame(frame)
        adv_frame.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(adv_frame, text="Advanced Parameters", font=("Arial", 12, "bold")).pack(pady=5)
        grains_frame = ctk.CTkFrame(adv_frame)
        grains_frame.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(grains_frame, text="Number of Initiation Sites:").pack(side=tk.LEFT, padx=5)
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
        
        # Add grain growth parameters section - Commented out
        '''
        growth_frame = ctk.CTkFrame(frame)
        growth_frame.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(growth_frame, text="Grain Growth Parameters", font=("Arial", 12, "bold")).pack(pady=5)
        
        # Activation energy
        activation_frame = ctk.CTkFrame(growth_frame)
        activation_frame.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(activation_frame, text="Activation Energy (eV):").pack(side=tk.LEFT, padx=5)
        self.activation_energy_var = tk.StringVar(value="0.5")
        ctk.CTkEntry(activation_frame, textvariable=self.activation_energy_var, width=80).pack(side=tk.LEFT, padx=5)
        
        # Growth factor
        growth_factor_frame = ctk.CTkFrame(growth_frame)
        growth_factor_frame.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(growth_factor_frame, text="Growth Mobility Factor:").pack(side=tk.LEFT, padx=5)
        self.growth_mobility_var = tk.StringVar(value="1.5")
        ctk.CTkEntry(growth_factor_frame, textvariable=self.growth_mobility_var, width=80).pack(side=tk.LEFT, padx=5)
        
        # CSL Boundaries checkbox with info button
        csl_frame = ctk.CTkFrame(growth_frame)
        csl_frame.pack(fill=tk.X, pady=2)
        self.csl_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(csl_frame, text="Use CSL Boundaries", variable=self.csl_var).pack(side=tk.LEFT, padx=5)
        csl_info_button = ctk.CTkButton(csl_frame, text="?", width=20, height=20, 
                                        command=lambda: self.show_info("CSL Boundaries are special grain boundaries with coincidence site lattice structure. They have lower energy and mobility, affecting grain growth rate. Examples include twin boundaries and other special orientation relationships between grains."))
        csl_info_button.pack(side=tk.LEFT, padx=2)
        
        # Triple Junction checkbox with info button
        tj_frame = ctk.CTkFrame(growth_frame)
        tj_frame.pack(fill=tk.X, pady=2)
        self.tj_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(tj_frame, text="Detect Triple Junctions", variable=self.tj_var).pack(side=tk.LEFT, padx=5)
        tj_info_button = ctk.CTkButton(tj_frame, text="?", width=20, height=20, 
                                      command=lambda: self.show_info("Triple junctions are points where three grain boundaries meet. They play an important role in grain growth kinetics by creating drag effects that slow boundary migration. Enabling this option allows more realistic simulation of grain boundary dynamics."))
        tj_info_button.pack(side=tk.LEFT, padx=2)
        '''
        
        # Since grain growth code is commented out, we'll set default values for the variables
        self.activation_energy_var = tk.StringVar(value="0.5")
        self.growth_mobility_var = tk.StringVar(value="1.5")
        self.csl_var = tk.BooleanVar(value=True)
        self.tj_var = tk.BooleanVar(value=True)
        
        # Simulation phase control - force recrystallization only
        phase_frame = ctk.CTkFrame(frame)
        phase_frame.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(phase_frame, text="Simulation Phase:").pack(side=tk.LEFT, padx=5)
        self.sim_phase_var = tk.StringVar(value="recrystallization")
        phase_label = ctk.CTkLabel(phase_frame, text="Recrystallization only", text_color="#4E84DB")
        phase_label.pack(side=tk.LEFT, padx=5)
        
        vis_frame = ctk.CTkFrame(frame)
        vis_frame.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(vis_frame, text="Visualization Parameters", font=("Arial", 12, "bold")).pack(pady=5)
        bins_frame = ctk.CTkFrame(vis_frame)
        bins_frame.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(bins_frame, text="Energy Dist. Bins:").pack(side=tk.LEFT, padx=5)
        self.bins_var = tk.StringVar(value="30")
        ctk.CTkEntry(bins_frame, textvariable=self.bins_var, width=80).pack(side=tk.LEFT, padx=5)
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
        
        # Canvas for simulation display
        self.mc_canvas = ctk.CTkCanvas(mc_frame, width=400, height=300, bg='white')
        self.mc_canvas.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Add tooltip label for simulation messages
        self.simulation_tooltip = ctk.CTkLabel(
            mc_frame, 
            text="",
            font=("Arial", 10, "italic"),
            text_color="#666666",
            height=20
        )
        self.simulation_tooltip.pack(fill=tk.X, padx=10)
        
        # Add control parameters frame
        mc_control_frame = ctk.CTkFrame(mc_frame)
        mc_control_frame.pack(pady=5, padx=10, fill=tk.X)
        
        # Capture Interval
        ctk.CTkLabel(mc_control_frame, text="Capture Interval:").pack(side=tk.LEFT, padx=5)
        self.capture_interval_var = tk.StringVar(value="10")
        ctk.CTkEntry(mc_control_frame, textvariable=self.capture_interval_var, width=80).pack(side=tk.LEFT, padx=5)
        
        # JPEG Quality
        ctk.CTkLabel(mc_control_frame, text="JPEG Quality:").pack(side=tk.LEFT, padx=5)
        self.quality_var = tk.StringVar(value="95")
        ctk.CTkEntry(mc_control_frame, textvariable=self.quality_var, width=60).pack(side=tk.LEFT, padx=5)
        
        # Save Euler Angles checkbox
        self.save_ea_with_img_var = tk.BooleanVar(value=False)
        save_ea_checkbox = ctk.CTkCheckBox(
            mc_control_frame, 
            text="Save EA with Images", 
            variable=self.save_ea_with_img_var
        )
        save_ea_checkbox.pack(side=tk.LEFT, padx=10)
        
        # Buttons frame
        mc_button_frame = ctk.CTkFrame(mc_frame)
        mc_button_frame.pack(pady=5, padx=10, fill=tk.X)
        
        # Core simulation buttons - larger size
        self.step_button = ctk.CTkButton(
            mc_button_frame, 
            text="Step", 
            command=self.monte_carlo_step, 
            width=100,
            height=30
        )
        self.step_button.pack(side=tk.LEFT, padx=5)
        
        self.run_all_button = ctk.CTkButton(
            mc_button_frame, 
            text="Run All", 
            command=self.run_all_monte_carlo_steps, 
            width=100,
            height=30
        )
        self.run_all_button.pack(side=tk.LEFT, padx=5)
        
        self.print_button = ctk.CTkButton(
            mc_button_frame, 
            text="Save EA", 
            command=self.print_euler_angles, 
            width=100,
            height=30
        )
        self.print_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ctk.CTkButton(
            mc_button_frame, 
            text="Save Img", 
            command=self.save_canvas_image, 
            width=100,
            height=30
        )
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # Add a small separator
        separator = ctk.CTkFrame(mc_button_frame, width=2, height=20)
        separator.pack(side=tk.LEFT, padx=8)
        
        # Phase-specific buttons
        self.rx_step_button = ctk.CTkButton(
            mc_button_frame, 
            text="Recrystallization Only", 
            command=self.run_recrystallization_only,
            fg_color="#4E84DB",
            width=160,
            height=30
        )
        self.rx_step_button.pack(side=tk.LEFT, padx=5)

    def create_analysis_section(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self.main_container)
        frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        ctk.CTkLabel(frame, text="Analysis & Export", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Create a section for simulation status
        self.phase_info_frame = ctk.CTkFrame(frame)
        self.phase_info_frame.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(self.phase_info_frame, text="Simulation Status", font=("Arial", 12, "bold")).pack(pady=5)
        
        # Phase info section - simplified with only essential info
        self.create_stat_label(self.phase_info_frame, "Current Phase:", "Not started")
        self.create_stat_label(self.phase_info_frame, "MCS Count:", "0")
        self.create_stat_label(self.phase_info_frame, "Average Grain Size:", "N/A")
        
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
        
        # Add button to export simulation data
        self.export_sim_button = ctk.CTkButton(
            frame, 
            text="Export Simulation Data", 
            command=self.export_simulation_data,
            fg_color="#5081B8"
        )
        self.export_sim_button.pack(pady=5, padx=10, fill=tk.X)
        
        return frame

    def create_stat_label(self, parent_frame, text: str, value: str):
        frame = ctk.CTkFrame(parent_frame)
        frame.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(frame, text=text).pack(side=tk.LEFT, padx=5)
        label = ctk.CTkLabel(frame, text=value)
        label.pack(side=tk.RIGHT, padx=5)
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

    def export_simulation_data(self):
        """Export comprehensive simulation data to CSV and images"""
        if not self.mc_hot_simulation or not hasattr(self.mc_hot_simulation, 'grain_stats_history'):
            self.show_info("No simulation data available to export.")
            return
            
        # Ask user for export directory
        export_dir = filedialog.askdirectory(title="Select Export Directory for Simulation Data")
        if not export_dir:
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create subdirectory for this export
            sim_export_dir = os.path.join(export_dir, f"simulation_export_{timestamp}")
            os.makedirs(sim_export_dir, exist_ok=True)
            
            # Export grain statistics to CSV
            stats_path = os.path.join(sim_export_dir, "grain_statistics.csv")
            self.export_grain_stats_to_file(stats_path)
            
            # Export current simulation image
            sim_image_path = os.path.join(sim_export_dir, "simulation_state.jpg")
            self.mc_hot_simulation.save_canvas_image(sim_image_path)
            
            # Export Euler angles
            ea_path = os.path.join(sim_export_dir, "euler_angles.txt")
            self.mc_hot_simulation.print_euler_angles(ea_path)
            
            # Create a summary report
            with open(os.path.join(sim_export_dir, "simulation_summary.txt"), 'w') as f:
                f.write(f"Simulation Summary - {timestamp}\n")
                f.write("="*50 + "\n\n")
                f.write(f"Current Phase: {self.mc_hot_simulation.simulation_phase}\n")
                f.write(f"Monte Carlo Steps: {self.mc_hot_simulation.current_mcs}\n")
                f.write(f"Number of Grains: {len([g for g in self.mc_hot_simulation.grains if len(g['GB']) > 0 or len(g['interior']) > 0])}\n")
                
                # Grain growth related code is commented out
                '''
                if self.mc_hot_simulation.phase_transition_mcs is not None:
                    f.write(f"Phase Transition at MCS: {self.mc_hot_simulation.phase_transition_mcs}\n")
                    f.write(f"Growth Time (MCS): {self.mc_hot_simulation.current_mcs - self.mc_hot_simulation.phase_transition_mcs}\n")
                '''
                
                # Add parameters used
                f.write("\nSimulation Parameters:\n")
                f.write(f"Theta_M: {self.mc_hot_simulation.params.theta_M}\n")
                f.write(f"Temperature: {self.mc_hot_simulation.params.temperature} K\n")
                
                # Grain growth related code is commented out
                '''
                f.write(f"Activation Energy: {self.mc_hot_simulation.activation_energy} eV\n")
                f.write(f"Growth Mobility Factor: {self.mc_hot_simulation.growth_mobility_factor}\n")
                f.write(f"Triple Junction Detection: {self.mc_hot_simulation.detect_triple_junctions}\n")
                f.write(f"CSL Boundaries: {self.mc_hot_simulation.use_csl_boundaries}\n")
                '''
            
            self.show_info(f"Simulation data exported successfully to {sim_export_dir}")
            
        except Exception as e:
            self.show_error(f"Error exporting simulation data: {str(e)}")

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
            energy_dist_plot_path = create_energy_distribution_plot(
                output_files['csv'], 
                GrainAnalyzer(params).output_dir,
                bins=int(self.bins_var.get())
            )
            plot_output_data['Energy Distribution'] = {'data': energy_dist_plot_path}
            self.root.after(0, self.update_gui_plots, plot_output_data)
            self.output_files_main_analysis = output_files
            self.root.after(0, self.initialize_hot_simulation)
            self.root.after(0, self.finish_processing)
        except ValueError:
            self.root.after(0, self.show_error, "Invalid bin count. Using default (30).")
            energy_dist_plot_path = create_energy_distribution_plot(output_files['csv'], GrainAnalyzer(params).output_dir)
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
                
                # Create parameters with additional properties
                params = SimulationParameters(
                    theta_M=float(self.theta_m_var.get()),
                    temperature=float(self.temperature_var.get())
                )
                
                # Grain growth related code is commented out
                '''
                # Add additional parameters from UI
                params.activation_energy = float(self.activation_energy_var.get())
                params.growth_mobility_factor = float(self.growth_mobility_var.get())
                '''
                
                # Create the simulation
                self.mc_hot_simulation = simulation_hot.HotSimulation(
                    df, s_array, self.mc_canvas, params, 
                    num_grains=num_grains, 
                    mobility_m=float(self.gb_energy_var.get())
                )
                
                # Force recrystallization phase only
                self.mc_hot_simulation.simulation_phase = 'recrystallization'
                
                # Grain growth related code is commented out
                '''
                # Set additional parameters
                self.mc_hot_simulation.activation_energy = float(self.activation_energy_var.get())
                self.mc_hot_simulation.growth_mobility_factor = float(self.growth_mobility_var.get())
                self.mc_hot_simulation.detect_triple_junctions = self.tj_var.get()
                self.mc_hot_simulation.use_csl_boundaries = self.csl_var.get()
                '''
                
                # Initialize the simulation
                self.mc_hot_simulation.initialize_grains()
                
                # Update the phase display
                self.update_phase_display()
                
                # Show simulation tooltip
                self.update_simulation_tooltip("Simulation initialized successfully. Ready to run.")
            except FileNotFoundError:
                self.show_error("Output files from main analysis not found for hot simulation.")
            except Exception as e:
                self.show_error(f"Error initializing hot simulation: {str(e)}")
        else:
            self.show_error("Main analysis output files not found. Run 'Process Hot-Rolled' first.")

    def update_simulation_tooltip(self, message):
        """Update the tooltip message in the Monte Carlo tab"""
        self.simulation_tooltip.configure(text=message)
        self.root.update_idletasks()

    def update_phase_display(self):
        """
        Update display with current simulation status.
        Updates both the status panel and the MCS counter.
        """
        if not self.mc_hot_simulation:
            return
        
        # Get the current phase - always recrystallization since grain growth is disabled
        current_phase = self.mc_hot_simulation.simulation_phase
        
        # Initialize statistics dictionary with current values
        stats = {
            "Current Phase:": current_phase.capitalize(),
            "MCS Count:": str(self.mc_hot_simulation.current_mcs),
            "Average Grain Size:": "N/A"
        }
        
        # Update grain size if available
        if hasattr(self.mc_hot_simulation, 'grain_stats_history') and self.mc_hot_simulation.grain_stats_history:
            latest_stats = self.mc_hot_simulation.grain_stats_history[-1]
            
            if 'grain_size_stats' in latest_stats and 'mean' in latest_stats['grain_size_stats']:
                stats["Average Grain Size:"] = f"{latest_stats['grain_size_stats']['mean']:.2f} px"
        
        # Update all statistics
        self.update_stat_labels_in_frame(self.phase_info_frame, stats)

    def update_stat_labels_in_frame(self, frame, updates: dict):
        """Update all statistic labels in the given frame"""
        for widget in frame.winfo_children():
            if isinstance(widget, ctk.CTkFrame):
                label_widgets = [w for w in widget.winfo_children() if isinstance(w, ctk.CTkLabel)]
                if len(label_widgets) >= 2:
                    label_text = label_widgets[0].cget("text")
                    if label_text in updates:
                        value = updates[label_text]
                        label_widgets[1].configure(text=value)
                        
                        # Add color highlight for phase
                        if label_text == "Current Phase:":
                            if value == "Recrystallization":
                                label_widgets[1].configure(text_color="#4E84DB")  # Blue for recrystallization
                            else:
                                label_widgets[1].configure(text_color="#FF5500")  # Orange for growth

    def set_mc_buttons_state(self, enabled: bool):
        """Enable or disable Monte Carlo buttons"""
        state = "normal" if enabled else "disabled"
        self.step_button.configure(state=state)
        self.run_all_button.configure(state=state)
        self.rx_step_button.configure(state=state)
        self.print_button.configure(state=state)
        self.save_button.configure(state=state)

    def monte_carlo_step(self):
        if not self.mc_hot_simulation:
            self.show_error("Simulation not initialized. Run 'Process Hot-Rolled' first.")
            return
            
        # Allow any tab to be active
        self.set_mc_buttons_state(False)
        try:
            # Perform the Monte Carlo step
            self.mc_hot_simulation.monte_carlo_step()
            
            # Update phase display
            self.update_phase_display()
            
        finally:
            self.set_mc_buttons_state(True)

    def run_all_monte_carlo_steps(self):
        if not self.mc_hot_simulation:
            self.show_error("Simulation not initialized. Run 'Process Hot-Rolled' first.")
            return
            
        # Don't restrict tab switching
        self.set_mc_buttons_state(False)
        try:
            # Default number of steps for "Run All" since Total Steps field was removed
            total_steps = 500
            capture_interval = int(self.capture_interval_var.get())
            
            # Update tooltip to indicate simulation is running
            self.update_simulation_tooltip("Running simulation... please wait.")
            
            # Run the simulation in a separate thread to prevent UI freezing
            thread = threading.Thread(
                target=self.run_simulation_thread,
                args=(total_steps, capture_interval)
            )
            thread.daemon = True  # This ensures the thread will exit when the main program exits
            thread.start()
        except ValueError as ve:
            self.show_error(f"Invalid step values: {str(ve)}")
            self.set_mc_buttons_state(True)
        except Exception as e:
            self.show_error(f"Error running simulation: {str(e)}")
            self.set_mc_buttons_state(True)
    
    def run_simulation_thread(self, total_steps, capture_interval):
        """Run simulation in a separate thread and update UI when complete"""
        try:
            # Set up a counter to update UI periodically
            update_interval = 10  # Update UI every 10 steps
            
            # Track captures for saving Euler angles
            captures_dir = None
            if self.save_ea_with_img_var.get():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                captures_dir = os.path.join("MC_Captures", f"capture_{timestamp}")
                os.makedirs(captures_dir, exist_ok=True)
            
            # Run simulation step by step to allow for UI updates
            for step in range(total_steps):
                # Grain growth code is commented out
                '''
                # Check if recrystallization is complete
                if self.mc_hot_simulation.simulation_phase == 'recrystallization' and self.mc_hot_simulation.is_lattice_filled():
                    self.mc_hot_simulation.simulation_phase = 'growth'
                    self.mc_hot_simulation.phase_transition_mcs = self.mc_hot_simulation.current_mcs
                    self.mc_hot_simulation.transition_to_growth()
                '''
                
                # Always use recrystallization step
                self.mc_hot_simulation.recrystallization_step()
                self.mc_hot_simulation.current_mcs += 1
                
                # Update display and stats periodically to keep UI responsive
                if step % update_interval == 0:
                    self.root.after(0, self.update_display_during_run)
                
                # Capture states at specified intervals if requested
                if step % capture_interval == 0:
                    self.mc_hot_simulation.collect_grain_statistics()
                    
                    # Save image and euler angles at each capture point if enabled
                    if self.save_ea_with_img_var.get() and captures_dir:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                        img_path = os.path.join(captures_dir, f"rx_mcs{self.mc_hot_simulation.current_mcs}_{timestamp}.jpg")
                        self.mc_hot_simulation.save_canvas_image(img_path)
                        
                        ea_path = os.path.join(captures_dir, f"euler_mcs{self.mc_hot_simulation.current_mcs}_{timestamp}.txt")
                        self.mc_hot_simulation.print_euler_angles(ea_path)
            
            # Final update after all steps
            self.root.after(0, self.simulation_complete_callback)
            
        except Exception as e:
            self.root.after(0, self.show_error, f"Error in simulation: {str(e)}")
            self.root.after(0, self.set_mc_buttons_state, True)

    def update_display_during_run(self):
        """Update display during simulation run without blocking the UI"""
        self.mc_hot_simulation.update_display()
        self.update_phase_display()

    def simulation_complete_callback(self):
        """Called when the simulation thread completes"""
        # Update UI after simulation completes
        self.mc_hot_simulation.update_display()
        self.update_phase_display()
        self.set_mc_buttons_state(True)
        self.update_simulation_tooltip("Simulation completed successfully!")
    
    def run_recrystallization_only(self):
        """Run simulation until recrystallization is complete"""
        if not self.mc_hot_simulation:
            self.show_error("Simulation not initialized. Run 'Process Hot-Rolled' first.")
            return
        
        # Force simulation phase to recrystallization
        self.mc_hot_simulation.simulation_phase = 'recrystallization'
        self.update_phase_display()
        
        # Run simulation until the lattice is filled
        self.set_mc_buttons_state(False)
        try:
            # Default number of steps
            max_steps = 500
            
            # Update tooltip
            self.update_simulation_tooltip("Running recrystallization phase... please wait.")
            
            # Run in a separate thread
            thread = threading.Thread(
                target=self.run_rx_only_thread,
                args=(max_steps,)
            )
            thread.daemon = True  # This ensures the thread will exit when the main program exits
            thread.start()
        except ValueError as ve:
            self.show_error(f"Invalid step values: {str(ve)}")
            self.set_mc_buttons_state(True)
        except Exception as e:
            self.show_error(f"Error running simulation: {str(e)}")
            self.set_mc_buttons_state(True)
    
    def run_rx_only_thread(self, max_steps):
        """Run recrystallization phase in a separate thread"""
        try:
            steps_taken = 0
            update_interval = 10  # Update UI every 10 steps
            
            captures_dir = None
            if self.save_ea_with_img_var.get():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                captures_dir = os.path.join("MC_Captures", f"capture_{timestamp}")
                os.makedirs(captures_dir, exist_ok=True)
                
            capture_interval = int(self.capture_interval_var.get())
            
            while (not self.mc_hot_simulation.is_lattice_filled() and 
                  steps_taken < max_steps):
                self.mc_hot_simulation.recrystallization_step()
                steps_taken += 1
                self.mc_hot_simulation.current_mcs += 1
                
                # Update UI periodically to keep responsive
                if steps_taken % update_interval == 0:
                    self.root.after(0, self.update_display_during_run)
                
                # Capture states at specified intervals if requested
                if self.save_ea_with_img_var.get() and captures_dir and steps_taken % capture_interval == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                    img_path = os.path.join(captures_dir, f"rx_mcs{self.mc_hot_simulation.current_mcs}_{timestamp}.jpg")
                    self.mc_hot_simulation.save_canvas_image(img_path)
                    
                    ea_path = os.path.join(captures_dir, f"euler_mcs{self.mc_hot_simulation.current_mcs}_{timestamp}.txt")
                    self.mc_hot_simulation.print_euler_angles(ea_path)
                    
            # Final update
            self.root.after(0, self.mc_hot_simulation.update_display)
            self.root.after(0, self.update_phase_display)
            self.root.after(0, self.simulation_complete_callback)
            
            # Let the user know if we hit max steps
            if steps_taken >= max_steps and not self.mc_hot_simulation.is_lattice_filled():
                self.root.after(0, self.show_info, 
                               f"Reached maximum steps ({max_steps}) before completing recrystallization.")
        except Exception as e:
            self.root.after(0, self.show_error, f"Error in recrystallization: {str(e)}")
            self.root.after(0, self.set_mc_buttons_state, True)

    def export_grain_stats_to_file(self, filename):
        """Export grain statistics to a CSV file"""
        with open(filename, 'w', newline='') as csvfile:
            # Determine all possible fields by inspecting the data
            all_stats = self.mc_hot_simulation.grain_stats_history
            
            # Basic fields that are always present
            fieldnames = ['mcs', 'phase', 'num_grains', 'avg_grain_size']
            
            # Check for grain size stats fields
            if all_stats and 'grain_size_stats' in all_stats[0]:
                for key in all_stats[0]['grain_size_stats'].keys():
                    fieldnames.append(f'size_{key}')
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for stats in all_stats:
                row = {
                    'mcs': stats['mcs'],
                    'phase': stats['phase'],
                    'num_grains': stats['num_grains'],
                    'avg_grain_size': stats['avg_grain_size']
                }
                
                # Add grain size stats
                if 'grain_size_stats' in stats:
                    for key, value in stats['grain_size_stats'].items():
                        row[f'size_{key}'] = value
                
                writer.writerow(row)

    def print_euler_angles(self):
        if not self.mc_hot_simulation:
            self.show_error("Simulation not initialized. Run 'Process Hot-Rolled' first.")
            return
        self.mc_hot_simulation.print_euler_angles()

    def save_canvas_image(self):
        if not self.mc_hot_simulation:
            self.show_error("Simulation not initialized. Run 'Process Hot-Rolled' first.")
            return
            
        # Ask for filename if needed
        filename = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if not filename:
            return
            
        # Save the canvas image
        self.mc_hot_simulation.save_canvas_image(filename)
        
        # Save Euler angles if option is checked
        if self.save_ea_with_img_var.get():
            # Create matching filename for Euler angles
            base_name = os.path.splitext(filename)[0]
            ea_filename = f"{base_name}_euler.txt"
            self.mc_hot_simulation.print_euler_angles(ea_filename)
            self.show_info(f"Image saved to {filename}\nEuler angles saved to {ea_filename}")
        else:
            self.show_info(f"Image saved to {filename}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = MaterialAnalysisGUI()
    app.run()