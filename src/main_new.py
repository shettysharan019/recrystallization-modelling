import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import csv
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

global_s = None

@dataclass
class SimulationParameters:
    """Data class to hold simulation parameters"""
    theta_M: float = 15.0
    tolerance_angle: float = 5.0
    grain_boundary_energy: float = 1.0
    temperature: float = 300.0
    iteration_steps: int = 1000
    color_palette: str = 'plasma'
    contour_levels: int = 10
    
class GrainAnalyzer:
    def __init__(self, params: SimulationParameters):
        self.params = params
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    @staticmethod
    def compute_orientation_matrix(phi1: float, phi: float, phi2: float) -> np.ndarray:
        """Compute orientation matrix for given Euler angles using vectorized operations."""
        cos_phi1, sin_phi1 = np.cos(phi1), np.sin(phi1)
        cos_phi2, sin_phi2 = np.cos(phi2), np.sin(phi2)
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        
        g1 = np.array([
            [cos_phi1, sin_phi1, 0],
            [-sin_phi1, cos_phi1, 0],
            [0, 0, 1]
        ])
        
        g2 = np.array([
            [cos_phi2, sin_phi2, 0],
            [-sin_phi2, cos_phi2, 0],
            [0, 0, 1]
        ])
        
        g_mid = np.array([
            [1, 0, 0],
            [0, cos_phi, sin_phi],
            [0, -sin_phi, cos_phi]
        ])
        
        return np.matmul(g2, np.matmul(g_mid, g1))

    @staticmethod
    def compute_misorientation(del_g: np.ndarray, symmetry_ops: Optional[List[np.ndarray]] = None) -> float:
        """Compute minimum misorientation angle considering symmetry operations."""
        if symmetry_ops is None:
            symmetry_ops = [
                np.eye(3),
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
                np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
                np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),
                np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
                np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]),
                np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]),
                np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
                np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
                np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]),
                np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
                np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
                np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
                np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]),
                np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
                np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
                np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
                np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
                np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
                np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
                np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            ]
        
        min_val = float('inf')
        for op in symmetry_ops:
            try:
                trace_val = np.trace(np.matmul(op, del_g))
                n_val = np.arccos(np.clip((trace_val - 1) / 2, -1, 1))
                min_val = min(min_val, n_val)
            except Exception:
                continue
        
        return min_val if min_val != float('inf') else 0

    def stored_energy(self, theta_val: float) -> float:
        """Calculate stored energy based on misorientation angle."""
        deg_theta = np.degrees(theta_val)
        if deg_theta < self.params.tolerance_angle:
            return 0
        elif deg_theta > self.params.theta_M:
            return self.params.grain_boundary_energy / 2
        return (self.params.grain_boundary_energy * 
                (deg_theta / self.params.theta_M) * 
                (1 - np.log(deg_theta / self.params.theta_M))) / 2

    def process_grid(self, s: np.ndarray, G: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process crystallographic data using vectorized operations where possible."""
        r, c = s.shape[:2]
        stored_energy_values = np.zeros((r, c))
        average_misorientation = np.zeros((r, c))
        kam = np.zeros((r, c))
        
        neighbor_offsets = np.array([(-1,-1), (-1,0), (-1,1), (0,-1), 
                                   (0,1), (1,-1), (1,0), (1,1)])
        
        for x in range(1, r-1):
            for y in range(1, c-1):
                if s[x, y, 3] < 0.1:
                    continue
                    
                misorientation_values = []
                valid_neighbors = 0
                
                for dx, dy in neighbor_offsets:
                    nx, ny = x + dx, y + dy
                    if s[nx, ny, 3] < 0.1:
                        continue
                        
                    misorientation = self.compute_misorientation(
                        np.matmul(G[x, y, 0], G[nx, ny, 1])
                    )
                    misorientation_values.append(misorientation)
                    
                    if np.degrees(misorientation) < self.params.theta_M:
                        kam[x, y] += misorientation
                        valid_neighbors += 1
                
                if misorientation_values:
                    misorientation_array = np.array(misorientation_values)
                    average_misorientation[x, y] = np.mean(misorientation_array)
                    stored_energy_values[x, y] = np.mean([self.stored_energy(m) for m in misorientation_array])
                    
                    if valid_neighbors > 0:
                        kam[x, y] /= valid_neighbors
        
        return stored_energy_values, average_misorientation, kam

    def analyze_data(self, input_filepath: str) -> Dict[str, str]:
        """Main processing pipeline with improved error handling and memory management."""
        global global_s
        try:
            # Load and preprocess data
            df = pd.read_csv(input_filepath)
            df.columns = df.columns.str.strip().str.lower()
            
            # Validate columns
            expected_columns = ['phi1', 'phi', 'phi2', 'x', 'y', 'iq']
            if not all(col in df.columns for col in expected_columns):
                raise ValueError("Input file missing required columns")
            
            # Convert to numpy array and process coordinates
            data = df[expected_columns].to_numpy()
            
            stepsize_x = np.min(np.diff(np.unique(data[:, 3])))
            stepsize_y = np.min(np.diff(np.unique(data[:, 4])))
            
            data[:, 3] = (data[:, 3] / stepsize_x).astype(int)
            data[:, 4] = (data[:, 4] / stepsize_y).astype(int)
            
            r = int(np.max(data[:, 3])) + 1
            c = int(np.max(data[:, 4])) + 1
            
            # Initialize arrays
            s = np.zeros((r, c, 4))
            global_s = s
            IQ = np.zeros((r, c))
            G = np.zeros((r, c, 2, 3, 3))
            
            # Populate arrays
            for phi1, phi, phi2, x, y, iq in data:
                x, y = int(x), int(y)
                s[x, y, :3] = [phi1, phi, phi2]
                s[x, y, 3] = 1
                IQ[x, y] = iq
                
                G[x, y, 0] = self.compute_orientation_matrix(phi1, phi, phi2)
                G[x, y, 1] = np.linalg.inv(G[x, y, 0])
            
            # Normalize IQ values
            IQ = 100 * (IQ - np.min(IQ)) / (np.ptp(IQ) + 1e-10)
            
            # Process data
            stored_energy_values, average_misorientation, kam = self.process_grid(s, G)
            
            # Generate timestamp for output files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save results to CSV
            output_file = os.path.join(self.output_dir, f"crystallography_analysis_{timestamp}.csv")
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["X", "Y", "Theta", "KAM", "IQ", "Stored Energy"])
                
                for x in range(1, r-1):
                    for y in range(1, c-1):
                        if s[x, y, 3] >= 0.1:
                            writer.writerow([
                                x, y,
                                average_misorientation[x, y],
                                kam[x, y],
                                IQ[x, y],
                                stored_energy_values[x, y]
                            ])
            
            # Create and save visualizations
            plot_data = [
                (stored_energy_values, self.params.color_palette, 'Stored Energy'),
                (average_misorientation, self.params.color_palette, 'Average Misorientation'),
                (kam, 'seismic', 'KAM'),
                (IQ, 'seismic', 'Image Quality')
            ]
            
            output_files = {'csv': output_file}
            
            for data_arr, cmap, title in plot_data:
                plt.figure(figsize=(10, 8))
                plt.imshow(data_arr, cmap=cmap, aspect='auto')
                plt.colorbar(label=title)
                plt.title(title)
                plt.tight_layout()
                
                image_file = os.path.join(self.output_dir, f"{title.replace(' ', '_')}_{timestamp}.png")
                plt.savefig(image_file, dpi=300)
                plt.close()
                
                output_files[title.lower().replace(' ', '_')] = image_file
            
            # Save the processed s array for simulation_hot.py
            s_output_file = os.path.join(self.output_dir, f"processed_s_array_{timestamp}.npy")
            np.save(s_output_file, s)
            output_files['s_array'] = s_output_file
            
            return output_files
            
        except Exception as e:
            raise RuntimeError(f"Error during processing: {str(e)}")

# Function to be called from GUI
def run_analysis(input_filepath: str, params: Optional[SimulationParameters] = None) -> Dict[str, str]:
    """
    Run the grain analysis with the given parameters.
    Returns a dictionary of output file paths.
    """
    if params is None:
        params = SimulationParameters()
    
    analyzer = GrainAnalyzer(params)
    return analyzer.analyze_data(input_filepath)