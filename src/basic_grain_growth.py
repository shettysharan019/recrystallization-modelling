import numpy as np
import tkinter as tk
from tkinter import Canvas
import random
from typing import List, Tuple

class LatticeSimulation:
    """
    Optimized 2D Lattice Simulation with improved performance and memory efficiency
    """
    def __init__(
        self, 
        lattice_size: int = 51, 
        temperature: float = 2.0, 
        num_nuclei: int = 10
    ):
        """
        Initialize the lattice simulation with configurable parameters
        
        Args:
            lattice_size (int): Size of the square lattice
            temperature (float): Temperature parameter for simulation
            num_nuclei (int): Number of initial nuclei to seed
        """
        self.L = lattice_size
        self.T = temperature
        self.states = [0, 1, 2]
        
        # Efficient lattice initialization
        self.lattice = np.zeros((self.L, self.L), dtype=np.uint8)
        
        # Seed initial nuclei with optimized approach
        self._seed_nuclei(num_nuclei)
    
    def _seed_nuclei(self, num_nuclei: int) -> None:
        """
        Seed initial nuclei at random locations
        
        Args:
            num_nuclei (int): Number of nuclei to create
        """
        nuclei_coords = np.random.randint(0, self.L, size=(num_nuclei, 2))
        for x, y in nuclei_coords:
            self.lattice[x, y] = random.choice([1, 2])
    
    def calculate_neighborhood_state(self, x: int, y: int) -> int:
        """
        Calculate the dominant state in the neighborhood
        
        Args:
            x (int): x-coordinate
            y (int): y-coordinate
        
        Returns:
            int: Dominant state (1, 2, or 0 for neutral)
        """
        # Optimize neighborhood checking with NumPy slicing
        neighborhood = self.lattice[
            max(0, x-1):min(self.L, x+2), 
            max(0, y-1):min(self.L, y+2)
        ]
        
        unique, counts = np.unique(neighborhood, return_counts=True)
        
        # Remove zero state from consideration if present
        non_zero_mask = unique != 0
        unique = unique[non_zero_mask]
        counts = counts[non_zero_mask]
        
        if len(unique) == 0:
            return 0
        
        if len(unique) == 1:
            return unique[0]
        
        # Return most frequent state, with random choice as tiebreaker
        max_count_states = unique[counts == counts.max()]
        return random.choice(max_count_states)
    
    def monte_carlo_step(self) -> None:
        """
        Perform a single Monte Carlo step with optimized state transition
        """
        # Create a copy to avoid in-place modifications during iteration
        next_lattice = self.lattice.copy()
        
        for x in range(self.L):
            for y in range(self.L):
                # Skip already occupied sites
                if self.lattice[x, y] != 0:
                    continue
                
                # Determine next state based on neighborhood
                next_state = self.calculate_neighborhood_state(x, y)
                next_lattice[x, y] = next_state
        
        self.lattice = next_lattice
    
    def run_simulation(self, steps: int = 100) -> None:
        """
        Run the simulation for a specified number of steps
        
        Args:
            steps (int): Number of Monte Carlo steps to perform
        """
        for _ in range(steps):
            self.monte_carlo_step()

class LatticeVisualization:
    """
    Visualization class for the Lattice Simulation
    """
    def __init__(
        self, 
        simulation: LatticeSimulation, 
        pixel_size: int = 10
    ):
        """
        Initialize the visualization
        
        Args:
            simulation (LatticeSimulation): Simulation instance to visualize
            pixel_size (int): Size of each lattice pixel
        """
        self.simulation = simulation
        self.pixel_size = pixel_size
        
        self.root = tk.Tk()
        self.root.title("Lattice Simulation")
        
        self.canvas = Canvas(
            self.root, 
            width=simulation.L * pixel_size, 
            height=simulation.L * pixel_size
        )
        self.canvas.pack()
        
        self.step_button = tk.Button(
            self.root, 
            text="Monte Carlo Step", 
            command=self._update_display
        )
        self.step_button.pack()
        
        self._update_display()
    
    def _update_display(self) -> None:
        """
        Update the canvas with the current lattice state
        """
        self.simulation.monte_carlo_step()
        self.canvas.delete("all")
        
        color_map = {0: 'white', 1: 'red', 2: 'green'}
        
        for x in range(self.simulation.L):
            for y in range(self.simulation.L):
                color = color_map[self.simulation.lattice[x, y]]
                self.canvas.create_rectangle(
                    x * self.pixel_size, 
                    y * self.pixel_size, 
                    (x+1) * self.pixel_size, 
                    (y+1) * self.pixel_size, 
                    fill=color
                )
    
    def run(self) -> None:
        """
        Start the visualization
        """
        self.root.mainloop()

def main():
    # Create simulation
    simulation = LatticeSimulation(
        lattice_size=51, 
        temperature=2.0, 
        num_nuclei=10
    )
    
    # Create visualization
    visualization = LatticeVisualization(simulation)
    visualization.run()

if __name__ == "__main__":
    main()