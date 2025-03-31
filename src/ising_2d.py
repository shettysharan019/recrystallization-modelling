import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class IsingModel:
    """
    Efficient 2D Ising Model Simulation with Metropolis Algorithm
    
    Represents magnetic spin interactions in a 2D lattice
    using Monte Carlo simulation techniques.
    """
    
    def __init__(self, size: int = 100, temperature: float = 2.0, seed: int = 42):
        """
        Initialize Ising Model simulation
        
        Args:
            size (int): Lattice grid size
            temperature (float): System temperature
            seed (int): Random seed for reproducibility
        """
        np.random.seed(seed)
        
        # Use NumPy for efficient lattice initialization
        self.size = size
        self.temperature = temperature
        self.lattice = np.random.choice([-1, 1], size=(size, size))
    
    def calculate_energy_change(self, x: int, y: int) -> float:
        """
        Calculate local energy change efficiently using periodic boundary conditions
        
        Args:
            x (int): x-coordinate of spin
            y (int): y-coordinate of spin
        
        Returns:
            float: Energy change for potential spin flip
        """
        # Use NumPy's advanced indexing for neighbor computation
        neighbors = np.array([
            self.lattice[(x+1) % self.size, y],
            self.lattice[(x-1) % self.size, y],
            self.lattice[x, (y+1) % self.size],
            self.lattice[x, (y-1) % self.size]
        ])
        
        # Compute energy change
        return 2 * self.lattice[x, y] * np.sum(neighbors)
    
    def monte_carlo_step(self) -> None:
        """
        Perform a single Monte Carlo simulation step using Metropolis algorithm
        """
        for _ in range(self.size * self.size):
            # Randomly select a spin
            x, y = np.random.randint(0, self.size, 2)
            
            # Calculate energy change
            delta_energy = self.calculate_energy_change(x, y)
            
            # Decision to flip spin based on Metropolis criterion
            if delta_energy <= 0 or np.random.random() < np.exp(-delta_energy / self.temperature):
                self.lattice[x, y] *= -1
    
    def run_simulation(self, steps: int = 1000) -> np.ndarray:
        """
        Run Ising Model simulation for specified steps
        
        Args:
            steps (int): Number of Monte Carlo steps
        
        Returns:
            np.ndarray: Final lattice configuration
        """
        for _ in range(steps):
            self.monte_carlo_step()
        
        return self.lattice
    
    def visualize(self, save_path: str = None) -> None:
        """
        Visualize lattice configuration
        
        Args:
            save_path (str, optional): Path to save visualization
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(self.lattice, cmap='binary', interpolation='nearest')
        plt.colorbar(label='Spin State')
        plt.title(f'Ising Model Lattice (T = {self.temperature})')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

def main():
    # Example usage
    ising = IsingModel(size=100, temperature=2.0)
    final_lattice = ising.run_simulation(steps=100)
    ising.visualize(save_path='ising_simulation.png')

if __name__ == '__main__':
    main()