########################### IMPORTS ######################################
import pandas as pd
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

######################## READING THE DATAFILE ###########################
# The first column of the data file will be read as Phi_1, 2nd - Phi, 3rd - Phi_2, 4th - x, 5th - y, 6th - IQ

######################## GLOBAL VARIABLES ###############################
global_theta_m = 15.0  # 15 degree is the critical value for misorientation
global_sigma_m = 10.0  # Global sigma_m
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
        """Initialize GrainAnalyzer with simulation parameters"""
        self.params = params
        # Use __file__ for path resolution
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
        theta_m_val = self.params.theta_M
        
        if deg_theta < self.params.tolerance_angle:
            return 0
        elif deg_theta > theta_m_val:
            return self.params.grain_boundary_energy / 2
            
        return (self.params.grain_boundary_energy *
                (deg_theta / theta_m_val) *
                (1 - np.log(deg_theta / theta_m_val))) / 2

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
                    if nx < 0 or nx >= r or ny < 0 or ny >= c or s[nx, ny, 3] < 0.1:
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

    def analyze_data(self, input_file_path: str) -> Dict[str, str]:
        """Main processing pipeline with improved error handling and memory management."""
        global global_s

        try:
            # Load and preprocess data
            df = pd.read_csv(input_file_path)
            df.columns = [col.strip().lower() for col in df.columns]

            expected_columns = ['phi1', 'phi', 'phi2', 'x', 'y', 'iq']
            found_columns = [col for col in expected_columns if col in df.columns]

            if len(found_columns) < len(expected_columns):
                missing = set(expected_columns) - set(found_columns)
                if len(df.columns) >= len(expected_columns):
                    column_mapping = {df.columns[i]: expected_columns[i] for i in range(min(len(df.columns), len(expected_columns)))}
                    df = df.rename(columns=column_mapping)
                else:
                    raise ValueError(f"Input file missing required columns: {missing}")

            data = df[expected_columns].to_numpy()

            x_coords = np.unique(data[:, 3])
            y_coords = np.unique(data[:, 4])
            
            stepsize_x = np.min(np.diff(x_coords)) if len(x_coords) > 1 else 1.0
            stepsize_y = np.min(np.diff(y_coords)) if len(y_coords) > 1 else 1.0

            data[:, 3] = np.round(data[:, 3] / stepsize_x).astype(int)
            data[:, 4] = np.round(data[:, 4] / stepsize_y).astype(int)

            r = int(np.max(data[:, 3])) + 1
            c = int(np.max(data[:, 4])) + 1

            s_matrix = np.zeros((r, c, 4))
            global_s = s_matrix  # Store processed s matrix globally
            IQ_matrix = np.zeros((r, c))
            G_matrices = np.zeros((r, c, 2, 3, 3))

            for phi1, phi, phi2, x, y, iq in data:
                x, y = int(x), int(y)
                if x < 0 or x >= r or y < 0 or y >= c:
                    continue
                    
                s_matrix[x, y, :3] = [phi1, phi, phi2]
                s_matrix[x, y, 3] = 1  # Valid data flag
                IQ_matrix[x, y] = iq

                G_matrices[x, y, 0] = self.compute_orientation_matrix(phi1, phi, phi2)
                G_matrices[x, y, 1] = np.linalg.inv(G_matrices[x, y, 0])

            if np.max(IQ_matrix) > np.min(IQ_matrix):
                IQ_matrix = 100 * (IQ_matrix - np.min(IQ_matrix)) / (np.max(IQ_matrix) - np.min(IQ_matrix))

            stored_energy_values, average_misorientation, kam = self.process_grid(s_matrix, G_matrices)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.output_dir, f"analysis_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)

            output_csv_path = os.path.join(output_dir, f"crystallography_analysis_{timestamp}.csv")
            with open(output_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["X", "Y", "Theta", "KAM", "IQ", "Stored Energy"])
                for x in range(1, r-1):
                    for y in range(1, c-1):
                        if s_matrix[x, y, 3] >= 0.1:
                            writer.writerow([
                                x * stepsize_x, y * stepsize_y,
                                average_misorientation[x, y],
                                kam[x, y],
                                IQ_matrix[x, y],
                                stored_energy_values[x, y]
                            ])

            plot_data = [
                (stored_energy_values, self.params.color_palette, 'Stored Energy'),
                (average_misorientation, self.params.color_palette, 'Average Misorientation'),
                (kam, 'seismic', 'KAM'),
                (IQ_matrix, 'gray', 'IQ')
            ]

            output_files = {'csv': output_csv_path}
            for data_arr, cmap, title in plot_data:
                plt.figure(figsize=(10, 8))
                plt.imshow(data_arr.T, cmap=cmap, aspect='equal', origin='lower', extent=[0, r*stepsize_x, 0, c*stepsize_y])
                plt.colorbar(label=title)
                plt.title(title)
                plt.xlabel("X-position (µm)")
                plt.ylabel("Y-position (µm)")
                plt.tight_layout()
                image_file = os.path.join(output_dir, f"{title.replace(' ', '')}{timestamp}.png")
                plt.savefig(image_file, dpi=300)
                plt.close()
                key = title.lower().replace(' ', '_')
                output_files[key] = image_file

            s_output_file = os.path.join(output_dir, f"processed_s_array_{timestamp}.npy")
            np.save(s_output_file, s_matrix)
            output_files['s_array'] = s_output_file

            return output_files

        except Exception as e:
            raise RuntimeError(f"Error during processing: {str(e)}")

def run_main_analysis(input_file_path: str, process_type: str, params: SimulationParameters, output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Main function to run the crystallographic analysis.
    """
    try:
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", f"{process_type}_analysis_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)

        analyzer = GrainAnalyzer(params)
        output_files = analyzer.analyze_data(input_file_path)
        return output_files

    except Exception as e:
        raise RuntimeError(f"Error in main analysis: {str(e)}")

if __name__ == "__main__":
    # When running directly, use default parameters and interactive input if desired.
    try:
        default_input_path = "C:/Users/shett/Downloads/Final Year Porject (FYP)/data/input/HR_complete.csv"
        input_csv_path = input(f"Enter path to input CSV file (or press Enter for default '{default_input_path}'): ").strip()
        if not input_csv_path:
            input_csv_path = default_input_path

        if not os.path.exists(input_csv_path):
            print(f"Error: Input file not found at {input_csv_path}. Please update the path.")
        else:
            process_type = input("Enter process type (hot/cold): ").strip().lower()
            if process_type not in ["hot", "cold"]:
                process_type = "hot"
                print(f"Using default process type: {process_type}")

            params = SimulationParameters(theta_M=15.0)
            output_files = run_main_analysis(input_csv_path, process_type, params)
            print("Analysis completed successfully.")
            print(f"Output files directory: {os.path.dirname(output_files['csv'])}")
            print("Output files:", output_files)

            from energy_distribution import create_energy_distribution_plot
            energy_plot = create_energy_distribution_plot(output_files['csv'], os.path.dirname(output_files['csv']))
            print(f"Energy distribution plot created: {energy_plot}")

    except Exception as e:
        print(f"Analysis failed: {e}")