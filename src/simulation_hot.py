import numpy as np
import pandas as pd
import random
import math
from PIL import Image, ImageTk, EpsImagePlugin
from main import GrainAnalyzer, SimulationParameters
import os
from datetime import datetime
import aspose.imaging as ai

class HotSimulation:
    def __init__(self, dataframe: pd.DataFrame, s_array: np.ndarray, canvas, params: SimulationParameters, num_grains, pixel_size: int = 10, mobility_m: float = 10.0):
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
        # Phase tracking - 'recrystallization' or 'growth'
        self.simulation_phase = 'recrystallization'
        # Boltzmann constant in J/K
        self.boltzmann_constant = 1.380649e-23
        # Keep track of MCS (Monte Carlo Steps)
        self.current_mcs = 0
        # Store grain statistics over time
        self.grain_stats_history = []
        # Grain ID tracking for statistics
        self.grain_id_map = np.zeros((self.r, self.c), dtype=int)
        # Track phase transition point
        self.phase_transition_mcs = None
        # Growth exponent tracking
        self.growth_exponent = None
        # Set activation energy from parameters (use provided or default)
        self.activation_energy = getattr(params, 'activation_energy', 0.5)  # eV
        # Growth mobility factor - controls relative speed of growth phase
        self.growth_mobility_factor = getattr(params, 'growth_mobility_factor', 1.5)
        # Triple junction detection flag
        self.detect_triple_junctions = True
        # CSL boundary special treatment flag
        self.use_csl_boundaries = True
        # Initialize CSL angle list for special boundaries
        self.csl_angles = np.array([8.66, 16.26, 22.62, 28.07, 36.87, 38.94])  # Σ3, Σ5, Σ7, Σ9, Σ11 boundaries

    def fetchEA(self, x, y):
        a = x % self.r
        b = y % self.c
        return self.EA[a, b]

    def mobility(self, misorientation):
        """
        Enhanced mobility calculation with improved temperature dependence and 
        orientation dependence. Includes special treatment for coincidence site lattice (CSL)
        boundaries if enabled.
        """
        # Convert misorientation to degrees for easier comparison
        misorientation_deg = np.degrees(misorientation)
        theta_m_rad = np.radians(self.params.theta_M)
        
        # Arrhenius temperature dependence:
        # M = M0 * exp(-Q/RT)
        # where Q is activation energy, R is gas constant, T is temperature
        # kB is Boltzmann constant, eV to J conversion is 1.602176634e-19
        kB_eV = 8.617333262e-5  # Boltzmann constant in eV/K
        temperature_factor = np.exp(-self.activation_energy / (kB_eV * self.params.temperature))
        
        # Base mobility based on Read-Shockley model
        # Enhanced with smoother function for high-angle boundaries
        if misorientation_deg < self.params.theta_M:
            # Low-angle boundary - Read-Shockley model (modified)
            rel_angle = misorientation_deg / self.params.theta_M
            base_mobility = rel_angle * (1 - np.log(rel_angle + 1e-10))
        else:
            # High-angle boundary - constant mobility
            base_mobility = 1.0
        
        # Check for CSL boundaries if enabled
        csl_factor = 1.0
        if self.use_csl_boundaries:
            # Lower mobility for special boundaries (CSL)
            # Check if misorientation is close to a CSL boundary angle
            angle_diffs = np.abs(self.csl_angles - misorientation_deg)
            if np.min(angle_diffs) < 2.0:  # Within 2 degrees of a CSL boundary
                # Reduce mobility for special boundaries (they move slower)
                csl_factor = 0.4
        
        # Phase-dependent mobility adjustment
        phase_factor = self.growth_mobility_factor if self.simulation_phase == 'growth' else 1.0
        
        # Combine all factors
        final_mobility = self.mobility_m * temperature_factor * base_mobility * csl_factor * phase_factor
        
        # Ensure minimum mobility
        return max(final_mobility, 0.01 * self.mobility_m)

    def del_E(self, EA_M, EA_1, coords_px):
        """
        Calculate energy change for a potential state change.
        Enhanced with orientation-dependent boundary energy.
        """
        SE_i = 0  # Initial energy
        SE_f = 0  # Final energy
        
        # Number of neighbors to consider (8 for standard Moore neighborhood)
        num_neighbors = 0
        local_curvature = 0
        
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if x == 0 and y == 0:
                    continue
                nx, ny = coords_px[0] + x, coords_px[1] + y
                EA_neighbor = self.fetchEA(nx, ny)
                num_neighbors += 1
                
                # Initial energy calculation
                misorientation_initial = self.grain_analyzer.compute_misorientation(
                    np.matmul(self.grain_analyzer.compute_orientation_matrix(*EA_1),
                              np.linalg.inv(self.grain_analyzer.compute_orientation_matrix(*EA_neighbor))))
                initial_energy = self.grain_analyzer.stored_energy(misorientation_initial)
                SE_i += initial_energy
                
                # Final energy calculation
                misorientation_final = self.grain_analyzer.compute_misorientation(
                    np.matmul(self.grain_analyzer.compute_orientation_matrix(*EA_M),
                              np.linalg.inv(self.grain_analyzer.compute_orientation_matrix(*EA_neighbor))))
                final_energy = self.grain_analyzer.stored_energy(misorientation_final)
                SE_f += final_energy
                
                # Count if neighbor has same orientation as the growing grain
                # Used for curvature estimation
                if np.array_equal(EA_neighbor, EA_M):
                    local_curvature -= 1  # Same orientation, boundary curves inward
                else:
                    local_curvature += 1  # Different orientation, boundary curves outward
        
        # Energy change due to crystallographic misorientation
        energy_diff = SE_f - SE_i
        
        # Consider additional curvature effect during growth phase
        if self.simulation_phase == 'growth':
            # Normalize curvature effect based on neighbor count
            normalized_curvature = local_curvature / num_neighbors if num_neighbors > 0 else 0
            
            # Curvature-driven component scales with local curvature
            # Positive curvature makes energy change less favorable
            # Negative curvature makes energy change more favorable
            curvature_component = normalized_curvature * 0.2 * np.mean([SE_i, SE_f])
            
            # Add curvature component to energy difference
            energy_diff += curvature_component
        
        return energy_diff

    def calculate_local_curvature(self, grain, x, y):
        """
        Enhanced calculation of local curvature at a boundary point.
        Uses a larger neighborhood and weighted distances for more accurate curvature estimation.
        """
        # Initialize arrays for same-grain and different-grain neighbors
        same_grain_coords = []
        diff_grain_coords = []
        center = np.array([x, y])
        
        # Expanded neighborhood for better curvature estimation (2 cells radius)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                    
                nx, ny = (x + dx) % self.r, (y + dy) % self.c
                neighbor_coords = np.array([nx, ny])
                weight = 1.0 / (np.linalg.norm(np.array([dx, dy])) + 0.1)  # Weight by inverse distance
                
                # Check if EA matches the grain's EA
                if np.array_equal(self.fetchEA(nx, ny), grain['euler_angles']):
                    same_grain_coords.append((neighbor_coords, weight))
                elif self.lattice_status[nx, ny] != 0:  # Not empty and not same grain
                    diff_grain_coords.append((neighbor_coords, weight))
        
        # Calculate weighted counts
        same_grain_weight = sum(w for _, w in same_grain_coords)
        diff_grain_weight = sum(w for _, w in diff_grain_coords)
        
        # Calculate weighted center positions for same-grain and diff-grain neighbors
        if same_grain_coords:
            same_grain_center = np.average([coord for coord, _ in same_grain_coords], axis=0, 
                                          weights=[w for _, w in same_grain_coords])
        else:
            same_grain_center = center
            
        if diff_grain_coords:
            diff_grain_center = np.average([coord for coord, _ in diff_grain_coords], axis=0,
                                          weights=[w for _, w in diff_grain_coords])
        else:
            diff_grain_center = center
        
        # Calculate the vector from same-grain center to diff-grain center
        # This vector points in the direction of positive curvature
        curvature_vector = diff_grain_center - same_grain_center
        curvature_magnitude = np.linalg.norm(curvature_vector)
        
        # Scale curvature by the relative difference in weighted counts
        if same_grain_weight + diff_grain_weight > 0:
            count_factor = (diff_grain_weight - same_grain_weight) / (same_grain_weight + diff_grain_weight)
        else:
            count_factor = 0
            
        # Combine magnitude and count effects
        curvature = curvature_magnitude * count_factor
        
        # Detect and handle triple junctions if enabled
        if self.detect_triple_junctions and len(diff_grain_coords) >= 5:
            # Count unique different grains in the neighborhood
            neighbor_grains = set()
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = (x + dx) % self.r, (y + dy) % self.c
                    if self.lattice_status[nx, ny] != 0 and self.lattice_status[nx, ny] != grain['color']:
                        neighbor_grains.add(self.lattice_status[nx, ny])
            
            # If we have 3 or more different grains meeting, it's a triple/multiple junction
            if len(neighbor_grains) >= 2:
                # Triple junctions have special dynamics - typically lower mobility
                # Reduce the effective curvature to reflect junction drag
                curvature *= 0.5
                
        return curvature

    def probability(self, del_E, misorientation):
        """
        Calculate transition probability with enhanced Boltzmann statistics.
        Includes temperature, orientation, and curvature effects.
        """
        MIN_PROB = 0.01  # 1% minimum chance
        stored_energy_val = self.grain_analyzer.stored_energy(misorientation)
        
        # Normalize temperature to a reasonable scale
        kT = self.boltzmann_constant * self.params.temperature / 300.0
        
        if self.simulation_phase == 'recrystallization':
            # During recrystallization, focus on stored energy reduction
            # Higher stored energy means faster recrystallization
            if del_E <= 0:
                # Energy-reducing transitions are favorable
                calculated_prob = (self.mobility(misorientation) * stored_energy_val * 2) / (self.mobility_m * 10.0)
            else:
                # Unfavorable transitions still possible via thermal fluctuations
                calculated_prob = (self.mobility(misorientation) * stored_energy_val * 2) * (np.exp(-1 * del_E / kT)) / (self.mobility_m * 10.0)
        else:  # Grain growth phase
            # During grain growth, focus on curvature-driven boundary migration
            # Energy change already includes curvature effects
            mobility_factor = self.mobility(misorientation)
            
            if del_E <= 0:
                # Energy-reducing moves are highly favorable
                # But still scale with mobility to account for boundary character
                calculated_prob = 0.8 * mobility_factor
            else:
                # Proper Boltzmann statistics for unfavorable moves
                # Exponential decay with increasing energy barrier
                calculated_prob = mobility_factor * np.exp(-1 * del_E / kT)
                
        return max(calculated_prob, MIN_PROB)

    def state_change(self, current_grain, coords_px):
        """
        Handle state change for a pixel with appropriate physics.
        Enhanced with orientation dependence and curvature effects.
        """
        x_coord = coords_px[0]
        y_coord = coords_px[1]
        pixel_state_initial = self.fetchEA(x_coord, y_coord)
        
        # Calculate misorientation
        misorientation_val = self.grain_analyzer.compute_misorientation(
            np.matmul(self.grain_analyzer.compute_orientation_matrix(*current_grain['euler_angles']),
                      np.linalg.inv(self.grain_analyzer.compute_orientation_matrix(*pixel_state_initial))))
        
        # Calculate energy change
        energy_change = self.del_E(current_grain['euler_angles'], pixel_state_initial, coords_px)
        
        # During grain growth, also consider local curvature explicitly
        if self.simulation_phase == 'growth' and self.lattice_status[x_coord, y_coord] != 0:
            curvature = self.calculate_local_curvature(current_grain, x_coord, y_coord)
            
            # Apply curvature-based adjustment to energy change
            # Negative curvature (boundary curved away from grain) makes growth more favorable
            energy_change *= (1.0 - (curvature * 0.15))
        
        # Calculate probability of state change
        prob = self.probability(energy_change, misorientation_val)
        
        # Apply Monte Carlo criterion
        if random.uniform(0, 1) <= prob:
            x, y = coords_px[0] % self.r, coords_px[1] % self.c
            self.EA[x, y] = current_grain['euler_angles']
            current_grain['new_grainspx'].append([x, y])
            
            # Update the lattice status and grain ID map
            previous_color = self.lattice_status[x, y]
            self.lattice_status[x, y] = current_grain['color']
            self.grain_id_map[x, y] = current_grain['id']
            
            # If this was a growth event (replacing another grain), update that grain's boundary
            if self.simulation_phase == 'growth' and previous_color != 0:
                for grain in self.grains:
                    if grain['color'] == previous_color:
                        # Remove this pixel from the other grain's interior
                        if [x, y] in grain['interior']:
                            grain['interior'].remove([x, y])
                        # Add to boundary if it wasn't already
                        if [x, y] not in grain['GB']:
                            grain['GB'].append([x, y])
                        break

    def updateGB(self, grain):
        """Update the grain boundary and interior pixels"""
        new_gb = []
        new_interior = []
        
        # Go through current boundary points
        for coord in grain['GB']:  # coord is a list [x, y]
            # Check if still a boundary point
            if self.isGB(coord[0], coord[1], grain['euler_angles']):
                new_gb.append(coord)
            else:
                # If no longer a boundary point, it's now interior
                new_interior.append(coord)
        
        # Add newly acquired boundary points
        for coord in grain['new_grainspx']:
            if self.isGB(coord[0], coord[1], grain['euler_angles']):
                new_gb.append(coord)
            else:
                new_interior.append(coord)
        
        # Update the grain's data
        grain['GB'] = new_gb
        
        # Initialize interior list if it doesn't exist
        if 'interior' not in grain:
            grain['interior'] = []
            
        # Update interior points
        grain['interior'].extend(new_interior)
        grain['new_grainspx'] = []

    def isGB(self, x, y, euler_angles):
        """Determine if a point is on a grain boundary"""
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                nx, ny = (x + i) % self.r, (y + j) % self.c
                if not np.array_equal(self.fetchEA(nx, ny), euler_angles):
                    return True
        return False

    def print_euler_angles(self, filename=None):
        """Save Euler angles with matching timestamp"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            if filename is None:
                filename = f"euler_{timestamp}.txt"
                
            with open(filename, "w") as f:
                f.write("phi1,phi,phi2,X,Y,IQ\n")
                for x in range(self.r):
                    for y in range(self.c):
                        f.write(f"{self.EA[x,y,0]},{self.EA[x,y,1]},{self.EA[x,y,2]},{x},{y},60\n")
        except Exception as e:
            print(f"Error saving Euler angles: {str(e)}")

    def generate_random_color(self):
        return f'#{random.randint(0, 0xFFFFFF):06x}'

    def save_canvas_image(self, filename=None):
        """Save canvas as JPG using Aspose.Imaging"""
        try:
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            
            # Default filename if not provided
            if filename is None:
                filename = f"sim_output_{timestamp}.jpg"
                
            # Create EPS version first
            eps_path = f"temp_{timestamp}.eps"
            self.canvas.postscript(file=eps_path, colormode='color')
            
            # Convert EPS to JPG using Aspose
            with ai.Image.load(eps_path) as image:
                options = ai.imageoptions.JpegOptions()
                options.quality = 95  # Set JPEG quality
                image.save(filename, options)
                
            # Cleanup temporary EPS
            os.remove(eps_path)
            
        except Exception as e:
            print(f"Error saving image: {str(e)}")
        finally:
            if os.path.exists(eps_path):
                os.remove(eps_path)

    def update_display(self):
        """
        Update the visual display of the simulation.
        Enhanced with better visual indicators for phase transitions.
        """
        self.canvas.delete("all")
        
        # Draw the lattice
        for i in range(self.r):
            for j in range(self.c):
                color = self.lattice_status[i, j] if self.lattice_status[i, j] != 0 else 'white'
                self.canvas.create_rectangle(i * self.pixel_size, j * self.pixel_size, 
                                           (i + 1) * self.pixel_size, (j + 1) * self.pixel_size, 
                                           fill=color, outline="")
        
        # Add phase indicator if in growth phase
        if self.simulation_phase == 'growth':
            # Add a growth phase indicator border
            border_width = 4
            self.canvas.create_rectangle(0, 0, self.r * self.pixel_size, self.c * self.pixel_size,
                                       outline="#FF5500", width=border_width)
            
            # Add phase text indicator
            self.canvas.create_text(self.r * self.pixel_size - 60, 20, 
                                   text="Growth Phase", 
                                   fill="#FF5500", 
                                   font=("Arial", 12, "bold"))
            
            # Add MCS counter
            self.canvas.create_text(60, 20, 
                                   text=f"MCS: {self.current_mcs}", 
                                   fill="#000000", 
                                   font=("Arial", 10))
        
        self.canvas.update()

    def initialize_grains(self):
        """Initialize grains for the simulation"""
        self.grains = []
        self.lattice_status = np.zeros((self.r, self.c), object)
        self.grain_id_map = np.zeros((self.r, self.c), dtype=int)
        nucleation_sites = []
        
        # Create random nucleation sites
        for i in range(self.num_grains):
            while True:
                nuclii_x = random.randint(0, self.r - 1)
                nuclii_y = random.randint(0, self.c - 1)
                if self.lattice_status[nuclii_x, nuclii_y] == 0:
                    nucleation_sites.append((nuclii_x, nuclii_y))
                    break
        
        # Initialize grains at nucleation sites
        for i, (nx, ny) in enumerate(nucleation_sites):
            initial_ea = self.fetchEA(nx, ny).copy()
            color = self.generate_random_color()
            grain = {
                'id': i + 1,  # Assign unique ID to each grain
                'name': f"grain {i+1}", 
                'euler_angles': initial_ea, 
                'GB': [[nx, ny]], 
                'interior': [],
                'new_grainspx': [], 
                'color': color
            }
            self.grains.append(grain)
            self.lattice_status[nx, ny] = color
            self.grain_id_map[nx, ny] = i + 1
            
        self.update_display()
        self.simulation_phase = 'recrystallization'
        self.current_mcs = 0
        self.phase_transition_mcs = None
        self.grain_stats_history = []

    def monte_carlo_step(self, num_steps=30):
        """Perform Monte Carlo steps with phase-appropriate logic"""
        m = 0
        while m < num_steps:
            # Check if recrystallization is complete
            if self.simulation_phase == 'recrystallization' and self.is_lattice_filled():
                print(f"Recrystallization complete at MCS {self.current_mcs}. Transitioning to grain growth phase.")
                self.simulation_phase = 'growth'
                # Record the MCS at which phase transition occurred
                self.phase_transition_mcs = self.current_mcs
                self.transition_to_growth()
            
            # Apply appropriate Monte Carlo steps based on current phase
            if self.simulation_phase == 'recrystallization':
                self.recrystallization_step()
            else:
                self.grain_growth_step()
                
            m += 1
            self.current_mcs += 1
            
            # Update display and collect statistics
            self.update_display()
            self.collect_grain_statistics()

    def recrystallization_step(self):
        """Monte Carlo step for recrystallization phase"""
        for grain in list(self.grains):
            for boundary_point in list(grain['GB']):
                for x in [-1, 0, 1]:
                    for y in [-1, 0, 1]:
                        if x == 0 and y == 0:
                            continue
                        coords_to_check = [(boundary_point[0] + x) % self.r, (boundary_point[1] + y) % self.c]
                        # Only try to fill empty spaces during recrystallization
                        if self.lattice_status[coords_to_check[0], coords_to_check[1]] == 0:
                            self.state_change(grain, coords_to_check)
            self.updateGB(grain)

    def grain_growth_step(self):
        """
        Enhanced Monte Carlo step for grain growth phase.
        Implements improved boundary dynamics and triple junction handling.
        """
        # In grain growth, we consider all boundary points regardless of surrounding empty space
        for grain in list(self.grains):
            # Skip extinct grains (those with no boundary points)
            if len(grain['GB']) == 0:
                continue
                
            # Process boundary points
            for boundary_point in list(grain['GB']):
                # Consider all neighbors for potential growth
                for x in [-1, 0, 1]:
                    for y in [-1, 0, 1]:
                        if x == 0 and y == 0:
                            continue
                        coords_to_check = [(boundary_point[0] + x) % self.r, (boundary_point[1] + y) % self.c]
                        
                        # Only try to capture pixels that belong to different grains
                        check_x, check_y = coords_to_check
                        if (self.lattice_status[check_x, check_y] != 0 and 
                            self.lattice_status[check_x, check_y] != grain['color']):
                            self.state_change(grain, coords_to_check)
                            
            # Update boundary and interior points
            self.updateGB(grain)
            
            # Remove extinct grains (those that have been completely consumed)
            if len(grain['GB']) == 0 and len(grain['interior']) == 0:
                self.grains.remove(grain)
        
        # Calculate growth exponent periodically during growth phase
        if self.current_mcs % 50 == 0 and len(self.grain_stats_history) > 5:
            self.calculate_growth_exponent()

    def transition_to_growth(self):
        """
        Enhanced handling of the transition from recrystallization to grain growth.
        Updates grain data and prepares for growth phase analysis.
        """
        # Update all grains to properly mark interior and boundary points
        for grain in self.grains:
            interior_points = []
            boundary_points = []
            
            # Scan the entire lattice to identify all points belonging to this grain
            for x in range(self.r):
                for y in range(self.c):
                    if self.lattice_status[x, y] == grain['color']:
                        if self.isGB(x, y, grain['euler_angles']):
                            boundary_points.append([x, y])
                        else:
                            interior_points.append([x, y])
            
            # Update the grain data
            grain['GB'] = boundary_points
            grain['interior'] = interior_points
            grain['new_grainspx'] = []
        
        # Print transition message to console
        msg = f"Transitioned to growth phase at MCS {self.current_mcs}"
        print("=" * len(msg))
        print(msg)
        print(f"Number of grains at transition: {len(self.grains)}")
        print("=" * len(msg))

    def is_lattice_filled(self):
        """Check if the lattice is completely filled (no white spaces)"""
        return not np.any(self.lattice_status == 0)

    def calculate_growth_exponent(self):
        """
        Calculate the grain growth exponent using data from the growth phase.
        The growth exponent 'n' appears in the relation d^n - d0^n = kt,
        where d is grain size, d0 is initial grain size, k is rate constant, t is time.
        """
        try:
            # Filter for growth phase data only
            growth_data = [stats for stats in self.grain_stats_history if stats['phase'] == 'growth']
            
            if len(growth_data) < 5:
                # Not enough data points yet
                return None
                
            # Extract time (MCS) and grain size data
            times = np.array([stats['mcs'] - self.phase_transition_mcs for stats in growth_data])
            sizes = np.array([stats['grain_size_stats']['mean'] for stats in growth_data])
            
            # Avoid log(0) issues
            times = times + 1  # Shift to avoid t=0
            
            # Apply log transformation for power law analysis
            log_times = np.log(times)
            log_sizes = np.log(sizes)
            
            # Linear regression on log-log data
            if len(log_times) > 1 and len(log_sizes) > 1:
                slope, intercept = np.polyfit(log_times, log_sizes, 1)
                r_squared = np.corrcoef(log_times, log_sizes)[0, 1]**2
                
                # In d^n = d0^n + kt, if we assume d0 is small relative to d during growth,
                # we get d^n ~ kt, or d ~ t^(1/n)
                # In log form: log(d) ~ (1/n)*log(t) + const
                # So growth_exponent = 1/slope
                
                growth_exponent = None
                if slope > 0:
                    growth_exponent = 1/slope
                
                # Store the calculated growth exponent
                self.growth_exponent = {
                    'n': growth_exponent,
                    'r_squared': r_squared,
                    'slope': slope
                }
                
                print(f"Updated growth exponent: n ≈ {growth_exponent:.3f} (R² = {r_squared:.3f})")
                return growth_exponent
            
        except Exception as e:
            print(f"Error calculating growth exponent: {str(e)}")
            return None

    def collect_grain_statistics(self):
        """
        Enhanced collection of statistics about current grain structure.
        Includes additional metrics for grain growth analysis.
        """
        # Count the number of grains
        active_grains = [g for g in self.grains if len(g['GB']) > 0 or len(g['interior']) > 0]
        num_grains = len(active_grains)
        
        # Calculate average grain size (in pixels)
        if num_grains > 0:
            total_pixels = self.r * self.c
            avg_grain_size = total_pixels / num_grains
        else:
            avg_grain_size = 0
            
        # Calculate grain sizes and normalized distribution
        grain_sizes = [len(g['GB']) + len(g['interior']) for g in active_grains]
        
        # Avoid division by zero
        if grain_sizes and sum(grain_sizes) > 0:
            grain_sizes_normalized = [size / sum(grain_sizes) for size in grain_sizes]
        else:
            grain_sizes_normalized = []
        
        # Calculate grain size statistics
        if grain_sizes:
            avg_size = np.mean(grain_sizes)
            median_size = np.median(grain_sizes)
            max_size = max(grain_sizes)
            min_size = min(grain_sizes)
            std_dev = np.std(grain_sizes)
            # Coefficient of variation (measure of size distribution uniformity)
            cv = std_dev / avg_size if avg_size > 0 else 0
        else:
            avg_size = median_size = max_size = min_size = std_dev = cv = 0
        
        # Calculate aspect ratio of grains (approximate)
        aspect_ratios = []
        for grain in active_grains:
            if not grain['GB'] and not grain['interior']:
                continue
                
            # Combine boundary and interior points
            all_points = grain['GB'] + grain['interior']
            if len(all_points) < 3:
                continue
                
            # Convert to numpy array for easier calculation
            points = np.array(all_points)
            
            # Calculate min/max x and y coordinates
            min_x, max_x = np.min(points[:,0]), np.max(points[:,0])
            min_y, max_y = np.min(points[:,1]), np.max(points[:,1])
            
            # Calculate aspect ratio as max dimension / min dimension
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            
            if width > 0 and height > 0:
                aspect = max(width/height, height/width)
                aspect_ratios.append(aspect)
        
        avg_aspect_ratio = np.mean(aspect_ratios) if aspect_ratios else 1.0
        
        # Store the statistics
        stats = {
            'mcs': self.current_mcs,
            'phase': self.simulation_phase,
            'num_grains': num_grains,
            'avg_grain_size': avg_grain_size,
            'grain_size_stats': {
                'mean': avg_size,
                'median': median_size,
                'max': max_size,
                'min': min_size,
                'std_dev': std_dev,
                'cv': cv
            },
            'avg_aspect_ratio': avg_aspect_ratio,
            'grain_sizes': grain_sizes
        }
        
        # Add growth exponent information if available
        if self.growth_exponent is not None:
            stats['growth_exponent'] = self.growth_exponent['n']
            stats['growth_r_squared'] = self.growth_exponent['r_squared']
        
        # Add phase transition MCS if applicable
        if self.phase_transition_mcs is not None:
            stats['phase_transition_mcs'] = self.phase_transition_mcs
        
        self.grain_stats_history.append(stats)
        return stats

    def run_all_steps(self, total_steps=500, capture_interval=10):
        """Run simulation for specified number of steps with automatic phase transition"""
        root_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.capture_dir = None
        capture_active = False
        steps_since_capture = 0
        
        # Ensure initial MCS is 0
        self.current_mcs = 0

        for step in range(1, total_steps + 1):
            # Check for phase transition if in recrystallization
            if self.simulation_phase == 'recrystallization' and self.is_lattice_filled():
                print(f"Recrystallization complete at MCS {self.current_mcs}. Transitioning to grain growth.")
                self.simulation_phase = 'growth'
                self.phase_transition_mcs = self.current_mcs
                self.transition_to_growth()
                
                # Create a new capture directory for grain growth phase if needed
                if self.capture_dir is None:
                    self.capture_dir = os.path.join("MC_Captures", f"capture_{root_timestamp}")
                    os.makedirs(self.capture_dir, exist_ok=True)
                
                # Always capture the transition point
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                self.save_canvas_image(os.path.join(self.capture_dir, f"transition_to_growth_{timestamp}.jpg"))
                self.print_euler_angles(os.path.join(self.capture_dir, f"euler_transition_{timestamp}.txt"))
            
            # Run appropriate Monte Carlo step based on current phase
            if self.simulation_phase == 'recrystallization':
                self.recrystallization_step()
            else:
                self.grain_growth_step()
                
            self.current_mcs += 1
            
            # Setup capture directory if needed
            if self.capture_dir is None and (self.is_lattice_filled() or step == total_steps):
                self.capture_dir = os.path.join("MC_Captures", f"capture_{root_timestamp}")
                os.makedirs(self.capture_dir, exist_ok=True)
                capture_active = True
                steps_since_capture = 0
                
            if capture_active:
                steps_since_capture += 1
                if steps_since_capture % capture_interval == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                    phase_prefix = 'rx' if self.simulation_phase == 'recrystallization' else 'growth'
                    self.save_canvas_image(os.path.join(self.capture_dir, f"{phase_prefix}_mcs{self.current_mcs}_{timestamp}.jpg"))
                    self.print_euler_angles(os.path.join(self.capture_dir, f"euler_mcs{self.current_mcs}_{timestamp}.txt"))
                    
                    # Also save grain statistics
                    stats = self.collect_grain_statistics()
                    stats_path = os.path.join(self.capture_dir, f"grain_stats_mcs{self.current_mcs}.txt")
                    self.save_grain_statistics(stats, stats_path)

            self.update_display()
            
        # Calculate final growth exponent
        self.calculate_growth_exponent()

    def save_grain_statistics(self, stats, filename):
        """Save grain statistics to a text file with enhanced metrics"""
        try:
            with open(filename, 'w') as f:
                f.write(f"MCS: {stats['mcs']}\n")
                f.write(f"Phase: {stats['phase']}\n")
                f.write(f"Number of grains: {stats['num_grains']}\n")
                f.write(f"Average grain size: {stats['avg_grain_size']:.2f} pixels\n")
                f.write(f"Mean grain size: {stats['grain_size_stats']['mean']:.2f} pixels\n")
                f.write(f"Median grain size: {stats['grain_size_stats']['median']:.2f} pixels\n")
                f.write(f"Max grain size: {stats['grain_size_stats']['max']:.2f} pixels\n")
                f.write(f"Min grain size: {stats['grain_size_stats']['min']:.2f} pixels\n")
                f.write(f"Size std deviation: {stats['grain_size_stats']['std_dev']:.2f} pixels\n")
                f.write(f"Size coefficient of variation: {stats['grain_size_stats']['cv']:.4f}\n")
                f.write(f"Average aspect ratio: {stats['avg_aspect_ratio']:.2f}\n")
                
                if 'growth_exponent' in stats:
                    f.write(f"Growth exponent (n): {stats['growth_exponent']:.4f}\n")
                    f.write(f"R-squared: {stats['growth_r_squared']:.4f}\n")
                
                if 'phase_transition_mcs' in stats:
                    f.write(f"Phase transition occurred at: MCS {stats['phase_transition_mcs']}\n")
        except Exception as e:
            print(f"Error saving grain statistics: {str(e)}")

    def generate_grain_growth_plot(self, output_path=None):
        """
        Enhanced plot showing grain growth statistics over time.
        Includes phase transition marker and growth exponent info.
        """
        import matplotlib.pyplot as plt
        
        if not self.grain_stats_history:
            print("No grain statistics available for plotting")
            return None
            
        # Extract data for plotting
        mcs_values = [stats['mcs'] for stats in self.grain_stats_history]
        num_grains = [stats['num_grains'] for stats in self.grain_stats_history]
        avg_size = [stats['grain_size_stats']['mean'] if 'grain_size_stats' in stats else 0 
                    for stats in self.grain_stats_history]
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot number of grains vs MCS
        ax1.plot(mcs_values, num_grains, 'b-', linewidth=2)
        ax1.set_ylabel('Number of Grains')
        ax1.set_title('Grain Evolution during Simulation')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot average grain size vs MCS
        ax2.plot(mcs_values, avg_size, 'r-', linewidth=2)
        ax2.set_ylabel('Average Grain Size (pixels)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add log-log plot of grain size vs MCS for growth phase
        if self.phase_transition_mcs is not None:
            growth_stats = [stats for stats in self.grain_stats_history if stats['phase'] == 'growth']
            
            if growth_stats:
                growth_mcs = [stats['mcs'] - self.phase_transition_mcs + 1 for stats in growth_stats]  # +1 to avoid log(0)
                growth_sizes = [stats['grain_size_stats']['mean'] for stats in growth_stats]
                
                # Log-log plot
                ax3.loglog(growth_mcs, growth_sizes, 'mo-', linewidth=2)
                ax3.set_xlabel('Monte Carlo Steps since Growth Phase (log scale)')
                ax3.set_ylabel('Average Grain Size (log scale)')
                ax3.set_title('Growth Kinetics (Log-Log Plot)')
                ax3.grid(True, linestyle='--', alpha=0.7)
                
                # Add fit line if we have calculated the growth exponent
                if self.growth_exponent is not None and self.growth_exponent['n'] is not None:
                    # For plotting the fit line
                    x_fit = np.linspace(min(growth_mcs), max(growth_mcs), 100)
                    slope = self.growth_exponent['slope']
                    intercept = np.log(growth_sizes[0]) - slope * np.log(growth_mcs[0])
                    y_fit = np.exp(slope * np.log(x_fit) + intercept)
                    
                    ax3.loglog(x_fit, y_fit, 'k--', linewidth=1.5, 
                            label=f"n ≈ {self.growth_exponent['n']:.2f}, R² = {self.growth_exponent['r_squared']:.3f}")
                    ax3.legend(loc='best')
            else:
                # No growth phase data yet
                ax3.set_xlabel('Monte Carlo Steps (MCS)')
                ax3.text(0.5, 0.5, 'Awaiting grain growth phase data', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax3.transAxes)
        else:
            # No phase transition yet
            ax3.set_xlabel('Monte Carlo Steps (MCS)')
            ax3.text(0.5, 0.5, 'Awaiting phase transition to growth', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax3.transAxes)
        
        # Add vertical line at phase transition if it occurred
        if self.phase_transition_mcs is not None:
            ax1.axvline(x=self.phase_transition_mcs, color='g', linestyle='--', 
                      label='Rx → Growth')
            ax2.axvline(x=self.phase_transition_mcs, color='g', linestyle='--')
            ax1.legend(loc='best')
            
            # Add annotations
            ax1.annotate('Recrystallization', 
                       xy=(self.phase_transition_mcs/2, ax1.get_ylim()[1]*0.9),
                       ha='center', va='center',
                       bbox=dict(boxstyle='round', fc='lightblue', alpha=0.7))
                       
            if max(mcs_values) > self.phase_transition_mcs:
                ax1.annotate('Grain Growth', 
                           xy=((max(mcs_values) + self.phase_transition_mcs)/2, ax1.get_ylim()[1]*0.9),
                           ha='center', va='center',
                           bbox=dict(boxstyle='round', fc='lightcoral', alpha=0.7))
        
        plt.tight_layout()
        
        # Save or show the plot
        if output_path:
            plt.savefig(output_path, dpi=300)
            plt.close()
            return output_path
        else:
            plt.show()
            return None

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