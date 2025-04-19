import numpy as np
import random
from typing import Callable, Dict, List, Tuple, Union, Optional
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

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

def analyze_grain_growth(grain_id_map: np.ndarray, history: List[Dict]) -> Dict:
    """
    Analyze grain growth patterns from simulation history.
    
    Args:
        grain_id_map: Current grain ID map from simulation
        history: List of grain statistics history
        
    Returns:
        Dictionary containing grain growth analysis results
    """
    try:
        # Skip if no history available
        if not history:
            return {"status": "No history available"}
        
        # Extract data from history
        mcs_values = [stats['mcs'] for stats in history]
        num_grains = [stats['num_grains'] for stats in history]
        
        # Calculate grain growth exponent using log-log linear regression
        # Only use growth phase data if available
        growth_indices = [i for i, stats in enumerate(history) if stats['phase'] == 'growth']
        
        if len(growth_indices) > 5:  # Need enough data points for meaningful regression
            growth_mcs = [mcs_values[i] for i in growth_indices]
            growth_sizes = [history[i]['grain_size_stats']['mean'] for i in growth_indices]
            
            # Apply log transformation for power law analysis
            log_mcs = np.log(np.array(growth_mcs) + 1)  # +1 to avoid log(0)
            log_sizes = np.log(np.array(growth_sizes))
            
            # Linear regression on log-log data
            if len(log_mcs) > 1 and len(log_sizes) > 1:
                slope, intercept = np.polyfit(log_mcs, log_sizes, 1)
                r_squared = np.corrcoef(log_mcs, log_sizes)[0, 1]**2
                
                # The exponent n in the grain growth equation d^n - d0^n = kt
                # is related to the slope of log(d) vs log(t)
                growth_exponent = 1/slope if slope != 0 else float('inf')
            else:
                growth_exponent = None
                r_squared = None
        else:
            growth_exponent = None
            r_squared = None
        
        # Calculate grain size distribution characteristics
        if len(history) > 0:
            latest_stats = history[-1]
            grain_sizes = latest_stats.get('grain_sizes', [])
            
            if grain_sizes:
                normalized_sizes = np.array(grain_sizes) / np.mean(grain_sizes) if np.mean(grain_sizes) > 0 else []
                
                size_distribution = {
                    'mean': np.mean(grain_sizes),
                    'median': np.median(grain_sizes),
                    'std_dev': np.std(grain_sizes),
                    'cv': np.std(grain_sizes) / np.mean(grain_sizes) if np.mean(grain_sizes) > 0 else 0,
                    'max': np.max(grain_sizes),
                    'min': np.min(grain_sizes),
                    'normalized_sizes': normalized_sizes.tolist() if len(normalized_sizes) > 0 else []
                }
            else:
                size_distribution = {'status': 'No grain size data available'}
        else:
            size_distribution = {'status': 'No history data available'}
        
        # Return analysis results
        return {
            'growth_exponent': growth_exponent,
            'growth_r_squared': r_squared,
            'size_distribution': size_distribution,
            'total_mcs': mcs_values[-1] if mcs_values else 0,
            'final_grain_count': num_grains[-1] if num_grains else 0
        }
        
    except Exception as e:
        return {'error': str(e)}

def generate_grain_size_distribution_plot(
    grain_sizes: List[float], 
    output_dir: str, 
    normalized: bool = True,
    bins: int = 20
) -> str:
    """
    Generate a histogram of grain size distribution
    
    Args:
        grain_sizes: List of grain sizes
        output_dir: Directory to save the plot
        normalized: Whether to normalize grain sizes
        bins: Number of histogram bins
        
    Returns:
        Path to the saved plot
    """
    try:
        if not grain_sizes:
            return ""
            
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Normalize grain sizes if requested
        if normalized and np.mean(grain_sizes) > 0:
            data = np.array(grain_sizes) / np.mean(grain_sizes)
            x_label = "Normalized Grain Size (d/<d>)"
            title = "Normalized Grain Size Distribution"
        else:
            data = np.array(grain_sizes)
            x_label = "Grain Size (pixels)"
            title = "Grain Size Distribution"
            
        # Create histogram
        plt.hist(data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add lognormal fit line if enough data points
        if len(data) > 5:
            from scipy import stats
            import numpy as np
            
            # Fit lognormal distribution
            shape, loc, scale = stats.lognorm.fit(data)
            x = np.linspace(min(data), max(data), 100)
            pdf = stats.lognorm.pdf(x, shape, loc, scale)
            
            # Scale PDF to match histogram height
            hist, bin_edges = np.histogram(data, bins=bins)
            max_hist_height = max(hist)
            max_pdf_height = max(pdf)
            scaling_factor = max_hist_height / max_pdf_height if max_pdf_height > 0 else 1
            
            plt.plot(x, pdf * scaling_factor, 'r-', linewidth=2, 
                     label=f'Lognormal Fit (σ={shape:.2f})')
            plt.legend()
        
        # Add labels and title
        plt.xlabel(x_label)
        plt.ylabel("Frequency")
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save and return plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"grain_size_dist_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        return plot_path
        
    except Exception as e:
        print(f"Error generating grain size distribution plot: {e}")
        return ""

def generate_growth_kinetics_plot(
    history: List[Dict], 
    output_dir: str
) -> str:
    """
    Generate plot showing grain growth kinetics
    
    Args:
        history: List of grain statistics history
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot
    """
    try:
        if not history:
            return ""
            
        # Extract data
        growth_indices = [i for i, stats in enumerate(history) if stats['phase'] == 'growth']
        
        if len(growth_indices) < 5:
            return ""  # Not enough growth phase data
            
        growth_mcs = [history[i]['mcs'] for i in growth_indices]
        growth_sizes = [history[i]['grain_size_stats']['mean'] for i in growth_indices]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot grain size vs MCS
        ax1.plot(growth_mcs, growth_sizes, 'bo-')
        ax1.set_xlabel('Monte Carlo Steps')
        ax1.set_ylabel('Average Grain Size (pixels)')
        ax1.set_title('Grain Growth Kinetics')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot log-log for power law relationship
        log_mcs = np.log(np.array(growth_mcs) + 1)  # +1 to avoid log(0)
        log_sizes = np.log(np.array(growth_sizes))
        
        ax2.plot(log_mcs, log_sizes, 'ro-')
        
        # Add linear fit
        if len(log_mcs) > 1:
            slope, intercept = np.polyfit(log_mcs, log_sizes, 1)
            fit_line = slope * log_mcs + intercept
            ax2.plot(log_mcs, fit_line, 'k--', 
                     label=f'Slope: {slope:.3f}\nExponent: {1/slope:.3f}')
            ax2.legend()
        
        ax2.set_xlabel('log(MCS)')
        ax2.set_ylabel('log(Grain Size)')
        ax2.set_title('Power Law Relationship')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"growth_kinetics_{timestamp}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        return plot_path
        
    except Exception as e:
        print(f"Error generating growth kinetics plot: {e}")
        return ""

def calculate_avrami_parameters(
    history: List[Dict]
) -> Dict:
    """
    Calculate Avrami parameters for the recrystallization phase
    
    Args:
        history: List of grain statistics history
        
    Returns:
        Dictionary containing Avrami analysis results
    """
    try:
        # Filter for recrystallization phase data
        rx_indices = [i for i, stats in enumerate(history) if stats['phase'] == 'recrystallization']
        
        if len(rx_indices) < 5:
            return {'status': 'Not enough recrystallization data'}
            
        # Extract MCS values
        mcs_values = [history[i]['mcs'] for i in rx_indices]
        
        # Calculate recrystallized fraction at each step
        # Assuming the lattice is completely filled at the end of recrystallization
        if 'lattice_filled_fraction' in history[rx_indices[0]]:
            rx_fractions = [history[i]['lattice_filled_fraction'] for i in rx_indices]
        else:
            # Estimate based on last rx index being 100% filled
            rx_fractions = [i / rx_indices[-1] for i in range(len(rx_indices))]
        
        # Calculate Avrami parameters
        valid_indices = [i for i, f in enumerate(rx_fractions) if 0 < f < 1]
        
        if len(valid_indices) < 3:
            return {'status': 'Not enough valid data points for Avrami analysis'}
            
        valid_mcs = [mcs_values[i] for i in valid_indices]
        valid_fractions = [rx_fractions[i] for i in valid_indices]
        
        # Calculate ln(-ln(1-X)) vs ln(t) for Avrami plot
        avrami_y = [np.log(-np.log(1 - f)) if 0 < f < 1 else np.nan for f in valid_fractions]
        avrami_x = [np.log(t) if t > 0 else np.nan for t in valid_mcs]
        
        # Filter out nan values
        valid_points = [(x, y) for x, y in zip(avrami_x, avrami_y) if not np.isnan(x) and not np.isnan(y)]
        
        if len(valid_points) < 3:
            return {'status': 'Not enough valid data points after filtering'}
            
        valid_x = [p[0] for p in valid_points]
        valid_y = [p[1] for p in valid_points]
        
        # Linear regression for Avrami parameters
        slope, intercept = np.polyfit(valid_x, valid_y, 1)
        r_squared = np.corrcoef(valid_x, valid_y)[0, 1]**2
        
        # n = slope, k = exp(intercept/n)
        avrami_n = slope
        avrami_k = np.exp(intercept / avrami_n) if avrami_n != 0 else 0
        
        return {
            'avrami_n': avrami_n,
            'avrami_k': avrami_k,
            'r_squared': r_squared,
            'valid_points': len(valid_points),
            'status': 'success'
        }
        
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

if __name__ == "__main__":
    # Example usage for testing
    # Create dummy data
    dummy_sizes = np.random.lognormal(mean=3, sigma=0.5, size=200)
    dummy_history = [
        {'mcs': i*10, 'phase': 'growth', 'num_grains': 100-i, 
         'grain_size_stats': {'mean': 10+i*2, 'median': 9+i*2}} 
        for i in range(20)
    ]
    
    # Test grain size distribution plot
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = generate_grain_size_distribution_plot(dummy_sizes, output_dir)
    print(f"Grain size distribution plot saved at: {plot_path}")
    
    # Test growth kinetics plot
    kinetics_path = generate_growth_kinetics_plot(dummy_history, output_dir)
    print(f"Growth kinetics plot saved at: {kinetics_path}")
    
    # Test grain growth analysis
    dummy_map = np.random.randint(1, 10, size=(50, 50))
    analysis = analyze_grain_growth(dummy_map, dummy_history)
    print("Grain growth analysis:", analysis)