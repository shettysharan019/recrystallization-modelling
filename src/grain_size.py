import numpy as np
import random
from typing import Callable

def calculate_grain_stats(r: int, c: int, stepsize_x: float, stepsize_y: float, 
                         G: np.ndarray, theta_func: Callable, n1: int = 20, n2: int = 20) -> dict:
    """Calculate and return both X and Y grain sizes in a single call"""
    return {
        'avg_x': average_grain_x(n1, r, c, stepsize_x, G, theta_func),
        'avg_y': average_grain_y(n2, r, c, stepsize_y, G, theta_func)
    }
    
def average_grain_x(n1: int, r: int, c: int, stepsize_x: float, 
                   G: np.ndarray, theta_func: Callable) -> float:
    if r < 2 or c < 2:
        return 0.0
    
    try:
        random_y = random.sample(range(0, c), min(n1, c))
        subgrains = []
        for y in random_y:
            i = 0
            for x in range(r-1):  # Prevent index overflow
                if np.degrees(theta_func(np.matmul(G[x,y,0], G[x+1,y,1]))) > 15:
                    i += 1
            subgrains.append(r*stepsize_x/i if i > 0 else r*stepsize_x)
        return sum(subgrains)/len(subgrains)
    except Exception as e:
        print(f"Error calculating X grain size: {e}")
        return 0.0

def average_grain_y(n2: int, r: int, c: int, stepsize_y: float, 
                   G: np.ndarray, theta_func: Callable) -> float:
    if r < 2 or c < 2:
        return 0.0
    
    try:
        random_x = random.sample(range(0, r), min(n2, r))
        subgrains = []
        for x in random_x:
            i = 0
            for y in range(c-1):  # Prevent index overflow
                if np.degrees(theta_func(np.matmul(G[x,y,0], G[x,y+1,1]))) > 15:
                    i += 1
            subgrains.append(c*stepsize_y/i if i > 0 else c*stepsize_y)
        return sum(subgrains)/len(subgrains)
    except Exception as e:
        print(f"Error calculating Y grain size: {e}")
        return 0.0