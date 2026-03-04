"""
Example demonstrating how to save and load a Grid.

This script performs the following steps:
1. Creates a RegularRectGrid with specified dimensions and resolution.
2. Saves the grid configuration to a YAML file.
3. Loads the grid configuration from the saved file.
4. Compares the edges of the original and loaded grids to ensure they are identical.
"""

import grids
from pathlib import Path

grid = grids.RegularRectGrid(grids.RegularRectGridCfg(
    length=1, height=1, num_cols=10, num_rows=10,
))

path = Path(__file__).parent / "tmp/regular_rect_configs.yaml"
grid.save_configs(path)

loaded_grid: grids.RegularRectGrid = grids.load_grid(path)

# Check that the edges are the same
all_same = True
for edges_i, edges_i_loaded in zip(grid.edges, loaded_grid.edges): 
    all_same = all_same and (edges_i == edges_i_loaded).all()

assert all_same == True