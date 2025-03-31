import pandas as pd
import numpy as np
from pathlib import Path

def load_and_process_data(file_path: str) -> tuple[np.ndarray, float]:
    """
    Load and preprocess the HR data file.
    Returns processed data array and step size.
    """
    # Use pandas efficient reading with specific columns
    df = pd.read_excel(
        file_path,
        usecols=['Phi1', 'Phi', 'Phi2', 'X', 'Y']
    )
    
    # Calculate step size from first two X values
    step_size = df['X'].iloc[1] - df['X'].iloc[0]
    
    # Normalize coordinates efficiently
    df[['X', 'Y']] = (df[['X', 'Y']] / step_size).astype(np.int32)
    
    return df.to_numpy(), step_size

def calculate_rotation_matrix(phi1: float, phi: float, phi2: float) -> np.ndarray:
    """
    Calculate rotation matrix using vectorized operations.
    Uses pre-calculated degree conversions for efficiency.
    """
    # Convert to radians once
    phi1_rad = np.radians(phi1)
    phi_rad = np.radians(phi)
    phi2_rad = np.radians(phi2)
    
    # Calculate sine and cosine values once
    cos_phi1, sin_phi1 = np.cos(phi1_rad), np.sin(phi1_rad)
    cos_phi, sin_phi = np.cos(phi_rad), np.sin(phi_rad)
    cos_phi2, sin_phi2 = np.cos(phi2_rad), np.sin(phi2_rad)
    
    # Construct matrices efficiently
    g1 = np.array([
        [cos_phi1, sin_phi1, 0],
        [-sin_phi1, cos_phi1, 0],
        [0, 0, 1]
    ])
    
    g_mid = np.array([
        [1, 0, 0],
        [0, cos_phi, sin_phi],
        [0, -sin_phi, cos_phi]
    ])
    
    g2 = np.array([
        [cos_phi2, sin_phi2, 0],
        [-sin_phi2, cos_phi2, 0],
        [0, 0, 1]
    ])
    
    # Perform matrix multiplication
    return g2 @ g_mid @ g1

def process_euler_angles(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Process Euler angles and create efficient sparse representation.
    Returns coordinates and corresponding rotation matrices.
    """
    coords = data[:, 3:5].astype(np.int32)  # X,Y coordinates
    angles = data[:, 0:3]  # Euler angles
    
    # Calculate rotation matrices for all points at once
    rotation_matrices = np.array([
        calculate_rotation_matrix(phi1, phi, phi2)
        for phi1, phi, phi2 in angles
    ])
    
    return coords, rotation_matrices

def main(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Main function to process HR data and calculate rotation matrices.
    """
    # Load and process data
    data, step_size = load_and_process_data(file_path)
    
    # Process Euler angles
    coords, rotation_matrices = process_euler_angles(data)
    
    return coords, rotation_matrices

if __name__ == "__main__":
    file_path = Path("C:/Users/shett/Downloads/Final Year Porject (FYP)/data/input/HR.xlsx")
    coords, rotation_matrices = main(file_path)
    print(f"Shape of coordinates: {coords.shape}")
    print(f"Shape of rotation matrices: {rotation_matrices.shape}")
    
    # Example: Print first rotation matrix
    if rotation_matrices.size > 0:
        print("\nFirst rotation matrix:")
        print(rotation_matrices[0])