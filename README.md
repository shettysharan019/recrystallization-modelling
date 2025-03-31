# Material Analysis Tool

## Overview
A sophisticated GUI application for materials scientists and engineers to analyze crystallographic properties, grain structures, and simulate microstructural evolution through hot and cold rolling processes. This tool provides comprehensive visualization and analysis capabilities for understanding material behavior at the microstructural level.

## Key Features

### Analysis Capabilities
- **Kernel Average Misorientation (KAM)**: Visualize local misorientation to identify strain concentrations
- **Energy Distribution**: Analyze stored energy patterns across material samples
- **Misorientation Analysis**: Quantify grain boundary characteristics and distributions
- **Image Quality Mapping**: Assess diffraction pattern quality for microstructural features
- **Grain Size Calculation**: Automatic measurement of grain dimensions and distribution

### Simulation Tools
- **Monte Carlo Simulation**: Model grain evolution and recrystallization processes
- **Hot-Rolled Material Simulation**: Specialized algorithms for high-temperature deformation
- **Cold-Rolled Material Simulation**: Simulate strain hardening and recovery mechanisms

### Visualization
- **Interactive Plots**: Multiple visualization options with customizable parameters
- **Adjustable Color Palettes**: Choose from various scientific colormaps (plasma, inferno, magma, viridis)
- **Contour Mapping**: Visualize gradients across material properties
- **Export Capabilities**: Save high-resolution images in multiple formats (JPG, TIFF)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/material-analysis-tool.git
   cd material-analysis-tool
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Starting the Application
```
python src/gui.py
```

### Workflow
1. **Configure Parameters**:
   - Set basic parameters (tolerance angle, grain boundary energy)
   - Adjust advanced settings (temperature, iteration steps)
   - Customize visualization options

2. **Import Data**:
   - Select input CSV files containing orientation data
   - Supported format: [x, y, phi1, phi, phi2, IQ] coordinates

3. **Process Data**:
   - Choose between hot-rolled or cold-rolled processing algorithms
   - Monitor progress through the GUI interface

4. **Analyze Results**:
   - Navigate between visualization tabs to examine different material properties
   - Review statistical data (grain size, count, misorientation)
   - Run Monte Carlo simulations to predict microstructural evolution

5. **Export Results**:
   - Save visualizations in preferred format and resolution
   - Export processed data for further analysis

## Dependencies
- customtkinter
- tkinter
- PIL (Pillow)
- matplotlib
- numpy
- pandas

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements
This tool was developed as part of a Final Year Project in Materials Science and Engineering.

---

*For questions and support, please open an issue on the GitHub repository.*
