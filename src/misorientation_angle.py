from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from numba import njit

class CrystalMisorientation:
    """
    A class to compute misorientation angles in crystallographic structures
    
    Attributes:
        theta_m (float): Critical misorientation angle (default: 15 degrees)
        symmetry_matrices (np.ndarray): Predefined symmetry rotation matrices
    """
    
    SYMMETRY_MATRICES = np.array([
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
        [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
        [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
        [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
        [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
        [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        [[0, 0, -1], [0, -1, 0], [-1, 0, 0]],
        [[0, 0, 1], [0, -1, 0], [1, 0, 0]],
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        [[-1, 0, 0], [0, 0, -1], [0, -1, 0]],
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
        [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    ], dtype=np.float64)
    
    def __init__(self, theta_m: float = 15):
        """
        Initialize the misorientation analysis
        
        Args:
            theta_m (float): Critical misorientation angle in degrees
        """
        self.theta_m = theta_m
    
    @staticmethod
    @njit
    def calculate_rotation_matrix(phi1: float, phi: float, phi2: float) -> np.ndarray:
        """
        Calculate 3D rotation matrix using Euler angles
        
        Args:
            phi1 (float): First Euler angle
            phi (float): Second Euler angle
            phi2 (float): Third Euler angle
        
        Returns:
            np.ndarray: 3x3 rotation matrix
        """
        # Optimized rotation matrix computation
        cos_phi1, sin_phi1 = np.cos(phi1), np.sin(phi1)
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        cos_phi2, sin_phi2 = np.cos(phi2), np.sin(phi2)
        
        g_one = np.array([
            [cos_phi1, sin_phi1, 0],
            [-sin_phi1, cos_phi1, 0],
            [0, 0, 1]
        ])
        
        g_two = np.array([
            [cos_phi2, sin_phi2, 0],
            [-sin_phi2, cos_phi2, 0],
            [0, 0, 1]
        ])
        
        g = np.array([
            [1, 0, 0],
            [0, cos_phi, sin_phi],
            [0, -sin_phi, cos_phi]
        ])
        
        return np.matmul(g_two, np.matmul(g, g_one))
    
    @staticmethod
    @njit
    def calculate_misorientation_angle(del_g: np.ndarray) -> float:
        """
        Calculate minimum misorientation angle considering symmetry
        
        Args:
            del_g (np.ndarray): Rotation matrix difference
        
        Returns:
            float: Minimum misorientation angle
        """
        min_angle = np.inf
        
        for sym_matrix in CrystalMisorientation.SYMMETRY_MATRICES:
            # Compute trace-based angle calculation
            rotation_matrix = np.matmul(sym_matrix, del_g)
            trace = np.trace(rotation_matrix)
            angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            
            min_angle = min(min_angle, angle)
        
        return min_angle
    
    def analyze_misorientation(self, data_path: str) -> np.ndarray:
        """
        Analyze misorientation for entire crystal structure
        
        Args:
            data_path (str): Path to Excel file
        
        Returns:
            np.ndarray: Misorientation angles for each point
        """
        # Read and preprocess data
        df = pd.read_excel(data_path)
        
        # Verify column names
        expected_columns = ['Phi1', 'Phi', 'Phi2', 'X', 'Y']
        if not all(col in df.columns for col in expected_columns):
            raise ValueError(f"Expected columns {expected_columns}, but got {df.columns}")
        
        # Compute step size
        df = df.sort_values(['X', 'Y'])
        step_size = df['X'].diff().dropna().min()
        
        # Convert coordinates to integer grid
        df['X_grid'] = ((df['X'] - df['X'].min()) / step_size).astype(int)
        df['Y_grid'] = ((df['Y'] - df['Y'].min()) / step_size).astype(int)
        
        # Get grid dimensions
        max_x = df['X_grid'].max()
        max_y = df['Y_grid'].max()
        
        # Initialize data structures
        grid_data = np.full((max_x + 1, max_y + 1, 4), np.nan, dtype=np.float64)
        rotation_tensors = np.zeros((max_x + 1, max_y + 1, 2, 3, 3), dtype=np.float64)
        
        # Populate grid data
        for _, row in df.iterrows():
            x, y = int(row['X_grid']), int(row['Y_grid'])
            grid_data[x, y, 0] = row['Phi1']
            grid_data[x, y, 1] = row['Phi']
            grid_data[x, y, 2] = row['Phi2']
            grid_data[x, y, 3] = x
        
        # Compute rotation matrices
        for x in range(max_x + 1):
            for y in range(max_y + 1):
                if not np.isnan(grid_data[x, y, 0]):
                    rot_matrix = self.calculate_rotation_matrix(
                        grid_data[x, y, 0], 
                        grid_data[x, y, 1], 
                        grid_data[x, y, 2]
                    )
                    rotation_tensors[x, y, 0] = rot_matrix
                    rotation_tensors[x, y, 1] = np.linalg.inv(rot_matrix)
        
        # Compute misorientation angles
        misorientation_angles = np.zeros((max_x + 1, max_y + 1), dtype=np.float64)
        
        for x in range(1, max_x):
            for y in range(1, max_y):
                # Skip if current point is not valid
                if np.isnan(grid_data[x, y, 0]):
                    continue
                
                local_angles = []
                
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        # Skip invalid neighboring points
                        if (np.isnan(grid_data[x+i, y+j, 0])):
                            continue
                        
                        # Compute misorientation between neighboring points
                        del_g = np.matmul(
                            rotation_tensors[x, y, 0], 
                            rotation_tensors[x+i, y+j, 1]
                        )
                        local_angles.append(self.calculate_misorientation_angle(del_g))
                
                # Compute average, excluding self-comparison
                if local_angles:
                    self_rotation = np.matmul(
                        rotation_tensors[x, y, 0], 
                        rotation_tensors[x, y, 1]
                    )
                    misorientation_angles[x, y] = (
                        np.mean(local_angles) - 
                        self.calculate_misorientation_angle(self_rotation)
                    )
        
        return np.degrees(misorientation_angles)
    
    def save_results(self, misorientation_angles: np.ndarray, 
                     step_size: float, output_path: str = 'misorientation_results.txt'):
        """
        Save misorientation results to a text file
        
        Args:
            misorientation_angles (np.ndarray): Computed misorientation angles
            step_size (float): Spatial resolution of the grid
            output_path (str): Path to save results
        """
        with open(output_path, 'w') as f:
            f.write("X\tY\tMisorientation Angle (degrees)\n")
            for x in range(misorientation_angles.shape[0]):
                for y in range(misorientation_angles.shape[1]):
                    f.write(f"{x*step_size}\t{y*step_size}\t{misorientation_angles[x,y]}\n")

def main():
    # Example usage
    data_path = r"C:/Users/shett/Downloads/Final Year Porject (FYP)/data/input/HR.xlsx"
    analyzer = CrystalMisorientation()
    
    try:
        # Read the Excel file to get the step size
        df = pd.read_excel(data_path)
        df = df.sort_values(['X', 'Y'])
        step_size = df['X'].diff().dropna().min()
        
        # Compute misorientation angles
        misorientation_angles = analyzer.analyze_misorientation(data_path)
        
        # Save results
        analyzer.save_results(misorientation_angles, step_size)
        print("Misorientation analysis completed successfully.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()