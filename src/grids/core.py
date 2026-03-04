import numpy as np
from abc import ABC, abstractmethod
import yaml

from .utils import *

class GridCfg:
    def get_grid(self):
        return cfg_to_class[type(self)](self)

class Grid(ABC):
    def __init__(self, configs: GridCfg):
        self.configs = configs
        self.compute_edges()
        
        self._shape = tuple(e.size - 1 for e in self.edges) 
        self._shape_t = tuple(s + 2 for s in self.shape)
        
        self._shape_mpl = tuple(reversed(self.shape[:2])) + self.shape[2:]
        self._shape_mpl_t = tuple(s + 2 for s in self._shape_mpl)

        self.num_dims = len(self.edges)

    @abstractmethod
    def compute_edges(self):
        "Creates the member `self._edges`."
        pass
    
    @property
    def shape(self):
        "Shape in each grid dimension, i.e, number of cells in each dimension."
        return self._shape
    
    @property
    def shape_t(self):
        "Shape considering the cells outside the grid."
        return self._shape_t
    
    @property
    def shape_mpl(self):
        '''
        Shape in each grid dimension ordered in the way usually used by matplotlib,
        that is, index 0 is swapped with index 1.
        '''
        return self._shape_mpl
  
    @property
    def shape_mpl_t(self):
        return self._shape_mpl_t

    @property
    def edges(self):
        "Returns the area of each cell in the grid."
        return self._edges

    @abstractmethod
    def coords(self, points: np.ndarray, check_out_of_bounds=True, simplify_shape=False):
        "Calculates the coordinates of the points in `points` on the grid."
        pass

    def count(self, coords: np.ndarray, end_id: np.ndarray=None, remove_out_of_bounds=False, simplify_shape=False):
        '''
        Counts the number of points in each cell of the grid, given the coordinates of the points
        in the grid (`coords`).

        Parameters
        ----------
        end_id:
            1-D array with the number of elements to be considered in `coords`. See the documentation for `self.sum_by_cell`.

        remove_out_of_bounds: 
            If True, removes cells that are out of the grid bounds from the result.
        
        simplify_shape:
            If True, simplifies the shape of the returned array to remove unnecessary dimensions.

        Returns
        -------
        count_grid: ndarray
            Array with the count of points in each cell. The element at index (i, j)
            is the count for the cell located at the i-th row and j-th column of the grid.
            The index (0, 0) is the cell at the lower left corner.
        '''
        coords = adjust_shape(coords, arr_name="coords")

        count_grid_shape = [coords.shape[0], *[i+2 for i in self.shape]]
        count_grid = np.zeros(count_grid_shape, dtype=int)
        for idx, coords_i in enumerate(coords):
            if end_id is not None:
                coords_i = coords_i[:end_id[idx]]

            unique_coords, count = np.unique(coords_i, axis=0, return_counts=True)
            count_grid[idx, unique_coords[:, 0], unique_coords[:, 1]] = count 

        count_grid = np.transpose(count_grid, axes=(0, 2, 1))

        if remove_out_of_bounds:
            count_grid = remove_cells_out_of_bounds(count_grid, many_layers=True)

        if simplify_shape:
            count_grid = simplify_arr_shape(count_grid)
        
        return count_grid

    def sum_by_cell(self, values: np.array, coords: np.array, end_id: np.ndarray=None, zero_value=0, 
        remove_out_of_bounds=False, simplify_shape=False):
        '''
        Sum of values that are in the same cell (have the same coordinate) of the grid.
        Each element in `values` has an associated coordinate in the grid given by `coords`.

        Parameters
        -----------
        values:
            Array with N (number of points) elements, which are the values associated with each coordinate.
            The type of the elements can be another array (in this case, `values` would be a multidimensional array),
            or a user-defined type (in this case, the type must implement `__add__()` and you must provide the zero element in `zero_value`).

        coords:
            Coordinates associated to each element in `values`.

        end_id:
            1-D array with the number of elements to be considered in `coords`. Only the following elements are considered:
            
            >>> coords[layer_id, :end_id[layer_id]]

        remove_out_of_bounds: 
            If True, removes cells that are out of the grid bounds from the result.
        
        simplify_shape:
            If True, simplifies the shape of the returned array to remove unnecessary dimensions.

        Returns
        --------
        values_sum: ndarray
            Array with the sum of values that are in the same cell.
            If `remove_out_of_bounds=True`, its shape is:
            
            (N_l, N_c, [shape of elements in `values`])
            
            where N_l is the number of rows and N_c is the number of columns of the grid, such that
            `values_sum[0, 0, ...]` is the value in the cell at the lower left corner.
            
            If `remove_out_of_bounds=False`, its shape is:
            
            (N_l + 2, N_c + 2, [shape of elements in `values`])

            In this case, the elements `values_sum[-1, col_id, :]`/`values_sum[row_id, -1, ...]` refer to the row/column before the first row/column,
            and the element `values_sum[N_l, col_id, :]`/`values_sum[row_id, N_c, ...]` refers to the row/column just after the last row/column.
        '''
        if len(coords.shape) == 2:
            coords = adjust_shape(coords, arr_name="coords")
            order = len(values.shape)
            values = adjust_shape(values, expected_order=order)

        v_shape = [values.shape[0], *reversed(self.shape[:2]), *self.shape[2:], *values.shape[2:]]
        for idx in range(1, 1+len(self.shape)):
            v_shape[idx] += 2

        values_sum = np.full(v_shape, fill_value=zero_value, dtype=values.dtype)

        layer_ids = list(range(coords.shape[0]))

        if end_id is None:
            for idx in range(coords.shape[1]):
                values_sum[layer_ids, coords[:, idx, 1], coords[:, idx, 0]] += values[:, idx]
        else:
            for layer_id in layer_ids:
                for idx, c in enumerate(coords[layer_id,:end_id[layer_id]]):
                    values_sum[layer_id, c[1], c[0]] += values[layer_id, idx]

        if remove_out_of_bounds:
            values_sum = remove_cells_out_of_bounds(values_sum, many_layers=True)

        if simplify_shape:
            values_sum = simplify_arr_shape(values_sum)
        
        return values_sum

    def mean_by_cell(self, values: np.array, coords: np.array, end_id=None, count: np.array=None,
        simplify_shape=False, remove_out_of_bound=False):
        '''
        Same as `self.sum_by_cell`, but divides the result by
        the count of points in each cell, thus computing the mean per cell.
        '''
        if len(coords.shape) == 2:
            coords = adjust_shape(coords, arr_name="coords")
            order = len(values.shape)
            values = adjust_shape(values, expected_order=order)

        values_mean = self.sum_by_cell(
            values, coords, 
            end_id=end_id,
        )
        
        if count is None:
            count = self.count(coords, end_id=end_id)

        non_zero_mask = count > 0

        # if len(coords.shape) == 3:
        #     num_new_axis = len(values.shape) - 2
        # else:
        #     num_new_axis = len(values.shape) - 1
        num_new_axis = len(values.shape) - 2

        values_mean[non_zero_mask] /= count[non_zero_mask].reshape(-1, *[1 for _ in range(num_new_axis)])
        
        if remove_out_of_bound:
            values_mean = remove_cells_out_of_bounds(values_mean, many_layers=True)
        
        if simplify_shape:
            values_mean = simplify_arr_shape(values_mean)

        return values_mean

    def debug_points(self, ax, points: np.ndarray, check_out_of_bounds=True):
        """
        Visualizes and annotates a set of points on a matplotlib axis with their corresponding grid coordinates.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The matplotlib axis on which to plot the points and annotations.
        
        points : np.ndarray
            An array of point coordinates to be visualized and annotated. Should be of shape (N, 2) for N points.
        
        check_out_of_bounds : bool, optional
            If True, checks whether the points are out of the grid bounds before annotation. Default is True.
        
        Notes
        -----
        - Points are plotted as black dots.
        - Each point is annotated with its grid coordinate (i, j) in red text.
        - Uses `self.coords` to obtain grid coordinates for each point.
        """
        coords = self.coords(points, simplify_shape=True, check_out_of_bounds=check_out_of_bounds)

        ax.scatter(*points.T, c="black")

        # Annotate each point with its grid coordinate
        for p, (i, j) in zip(points, coords):
            ax.text(*p, f"({i},{j})", fontsize=9, color="red", ha="left", va="bottom")

    def get_save_configs(self):
        return self.configs

    def save_configs(self, path):
        with open(path, "w") as f:
            yaml.dump(self.get_save_configs(), f)


class RectangularGridCfg(GridCfg):
    def __init__(self, edges: tuple[np.ndarray]):
        '''
        Rectangular grid given the positions of the cell edges.

        Parameters
        -----------
        edges:
            Tuple with two arrays:

            edges[0]: Positions of the column edges, including the extremes.
            edges[1]: Positions of the row edges, including the extremes.
        '''
        self.edges = edges

    def get_retangular_cfg(self):
        return self

class RectangularGrid(Grid):
    configs: RectangularGridCfg

    def __init__(self, configs: RectangularGridCfg) -> None:
        super().__init__(configs)

        # Length of the grid in each dimension
        self.size = []
        
        # Center of the cells in each dimension
        self.dim_cell_center = []
        
        # Length of the cells in each dimension
        self.dim_cell_size = []
       
        for dim in range(self.num_dims):
            self.size.append(self.edges[dim][-1] - self.edges[dim][0])
            self.dim_cell_center.append((self.edges[dim][1:] + self.edges[dim][:-1])/2)
            self.dim_cell_size.append(self.edges[dim][1:] - self.edges[dim][:-1])

        self.dim_extremes = []
        for e in self.edges:
            self.dim_extremes.append([e[0], e[-1]])

        # Meshgrid of cell centers
        self.meshgrid = np.meshgrid(*self.dim_cell_center)

        # Area of the cells.
        # self.cell_area[i, j] = Area of the cell at row i and column j
        w, h = np.meshgrid(*self.dim_cell_size)
        self.cell_area = w * h

    def compute_edges(self):
        self._edges = self.configs.edges
    
    def get_out_mask(self, coords: np.array):
        "Returns a mask for the points in `coords` that are outside the grid."
        coords = adjust_shape(coords, arr_name="coords")
        x = coords[:, :, 0]
        y = coords[:, :, 1]
        out_x = (x < 0) | (x >= self.shape[0])
        out_y = (y < 0) | (y >= self.shape[1])
        out = out_x | out_y
        return simplify_arr_shape(out)

    def remove_out_of_bounds(self, coords: np.array):
        '''
        Returns an array containing only the coordinates in `coords`
        that are inside the grid.
        '''
        return coords[np.logical_not(self.get_out_mask(coords))]

    def coords(self, points, check_out_of_bounds=True, simplify_shape=False):
        '''
        Calculates the coordinates of the points in `points` on the grid.

        Parameters
        ----------
        points:
            Array of points, which can have the following shapes:
            * (N, 2): 
                N is the number of points and 2 corresponds to the two grid dimensions 
                (0 for the x-axis and 1 for the y-axis).

            * (M, N, 2):
                Essentially a list of M arrays, each with N points.
        
        check_out_of_bounds:
            If True, points that fall outside the grid boundaries will be assigned special values:
            - Points with coordinates less than 0 will be set to -1.
            - Points with coordinates greater than or equal to the grid shape will be set to the grid shape value.
            If False, out-of-bounds points are not checked or modified.

        Returns
        -------
        coords: ndarray
            Array with the same shape as `points`, where the i-th element is the coordinate of the 
            i-th point on the grid. The second index of the shape indicates the coordinate dimension:
            
            coords[i, 0] -> Coordinate on the x-axis (Column) \n
            coords[i, 1] -> Coordinate on the y-axis (Row)
            
            The grid cell at the lower left corner has coordinate (0, 0).
            If `points` has shape (M, N, 2), then the above example becomes `coords[m, i, 0]`.
        '''
        num_indices = len(points.shape)
        if num_indices == 2:
            points = points.reshape(1, *points.shape)
        elif num_indices != 3:
            raise Exception(f"`len(points.shape)` é {len(points.shape)}, mas deveria ser 2 ou 3.")

        coords = np.empty(points.shape, dtype=int)
        for dim in range(self.num_dims):
            # np.searchsorted returns the index where each value would be inserted to maintain order
            # Subtract 1 to get the cell index (since edges are cell boundaries)
            idx = np.searchsorted(self.edges[dim], points[:, :, dim], side='right') - 1
            coords[:, :, dim] = idx

        if check_out_of_bounds:
            for dim in range(self.num_dims):
                coords[:, :, dim][coords[:, :, dim] < 0] = -1
                coords[:, :, dim][coords[:, :, dim] >= self.shape[dim]] = self.shape[dim]

        if simplify_shape and coords.shape[0] == 1:
            coords = coords.reshape(*coords.shape[1:])

        return coords

    def circle_mask(self, radius, center=(0, 0), mode="outside"):
        '''
        Mask of the cells for the circle with center `center` and radius `radius`.
        The mask depends on `mode`, which can take the following values:

        outside: Cells outside the circle.
        inside: Cells inside the circle.
        intersect: Cells intersecting the perimeter of the circle.
        '''
        valid_modes = ["outside", "inside", "intersect"]
        if mode not in valid_modes:
            raise ValueError(f"Valor inválido de `mode`: {mode}. Os valores válidos são: {valid_modes}.")

        x_max = self.meshgrid[0] + self.dim_cell_size[0]/2
        x_min = self.meshgrid[0] - self.dim_cell_size[0]/2
        
        y_max = (self.meshgrid[1].T + self.dim_cell_size[1]/2).T
        y_min = (self.meshgrid[1].T - self.dim_cell_size[1]/2).T
        
        x_max_sqr = np.square(x_max - center[0]) 
        x_min_sqr = np.square(x_min - center[0]) 
        
        y_max_sqr = np.square(y_max - center[1]) 
        y_min_sqr = np.square(y_min - center[1]) 

        d1 = np.sqrt(x_max_sqr + y_max_sqr)
        d2 = np.sqrt(x_max_sqr + y_min_sqr)
        d3 = np.sqrt(x_min_sqr + y_max_sqr)
        d4 = np.sqrt(x_min_sqr + y_min_sqr)

        if mode == "outside":
            return np.logical_not((d1 < radius) | (d2 < radius) | (d3 < radius) | (d4 < radius))
        if mode == "inside":
            return (d1 < radius) & (d2 < radius) & (d3 < radius) & (d4 < radius)
        if mode == "intersect":
            inside = (d1 < radius) & (d2 < radius) & (d3 < radius) & (d4 < radius)
            outside = np.logical_not((d1 < radius) | (d2 < radius) | (d3 < radius) | (d4 < radius))
            return (~outside) & (~inside)

    def plot_grid(self, ax, adjust_lims=True, color="black"):
        from matplotlib.axes import Axes
        from matplotlib.collections import LineCollection
        ax: Axes = ax

        x1 = self.dim_cell_center[0] - self.dim_cell_size[0]/2
        y1 = self.dim_cell_center[1] - self.dim_cell_size[1]/2

        max_x = x1[-1] + self.dim_cell_size[0][-1]
        max_y = y1[-1] + self.dim_cell_size[1][-1]

        lines = []
        for x in x1:
            lines.append([(x, y1[0]), (x, max_y)])
        lines.append([(max_x, y1[0]), (max_x, max_y)])
        for y in y1:
            lines.append([(x1[0], y), (max_x, y)])
        lines.append([(x1[0], max_y), (max_x, max_y)])
        
        if adjust_lims:
            offset = 0.3
            ax.set_xlim(
                self.dim_cell_center[0][0] - self.dim_cell_size[0][0]/2 * (1 + offset),
                self.dim_cell_center[0][-1] + self.dim_cell_size[0][-1]/2 * (1 + offset),
            )
            ax.set_ylim(
                self.dim_cell_center[1][0] - self.dim_cell_size[1][0]/2 * (1 + offset),
                self.dim_cell_center[1][-1] + self.dim_cell_size[1][-1]/2 * (1 + offset),
            )

        ax.add_collection(LineCollection(lines, color=color))

    def plot_center(self, ax):
        from matplotlib.axes import Axes
        ax: Axes = ax
        ax.scatter(self.meshgrid[0], self.meshgrid[1], c="black")
    
    def plot_corners(self, ax):
        from matplotlib.axes import Axes
        from matplotlib.collections import LineCollection
        ax: Axes = ax
        
        x1 = self.meshgrid[0] + self.dim_cell_size[0]/2
        x2 = self.meshgrid[0] - self.dim_cell_size[0]/2
        
        y1 = (self.meshgrid[1] + self.dim_cell_size[1].reshape(-1, 1)/2)
        y2 = (self.meshgrid[1] - self.dim_cell_size[1].reshape(-1, 1)/2)
        
        ax.scatter(x1, y1, c="black")
        ax.scatter(x1, y2, c="black")
        ax.scatter(x2, y1, c="black")
        ax.scatter(x2, y2, c="black")

    def random_points(self, n):
        """
        Generate `n` random points uniformly distributed inside the grid.
        Returns an array of shape (n, 2).
        """
        x_min, x_max = self.edges[0][0], self.edges[0][-1]
        y_min, y_max = self.edges[1][0], self.edges[1][-1]
        xs = np.random.uniform(x_min, x_max, n)
        ys = np.random.uniform(y_min, y_max, n)
        return np.column_stack((xs, ys))


class RegularRectGridCfg(GridCfg):
        """
        Initializes a rectangular grid centered at `center`, with sides of `length` x `height`.

        Parameters
        -----------
            length: 
                The length of the grid along the x-axis.
            
            height: 
                The height of the grid along the y-axis.
            
            num_cols: 
                Number of columns in the grid.
            
            num_rows: 
                Number of rows in the grid.
            
            center: 
                The (x, y) coordinates of the grid center. Defaults to (0, 0).
        """
        def __init__(self, length: float, height: float, num_cols: int, num_rows: int, center=(0, 0)) -> None:
            self.center = center
            self.length = length
            self.height = height
            self.num_cols = num_cols
            self.num_rows = num_rows
            self.center = center

        def get_retangular_cfg(self):
            edges = (
                np.linspace(-self.length/2 + self.center[0], self.length/2 + self.center[0], self.num_cols+1),
                np.linspace(-self.height/2 + self.center[1], self.height/2 + self.center[1], self.num_rows+1),
            )
            return RectangularGridCfg(edges)

class RegularRectGrid(RectangularGrid):
    def __init__(self, configs: RegularRectGridCfg) -> None:
        self.center = configs.center
        self.length = configs.length
        self.height = configs.height
        self.num_cols = configs.num_cols
        self.num_rows = configs.num_rows
        self.center = configs.center

        self.parent_configs = configs
        super().__init__(configs.get_retangular_cfg())

        # Cell size in each dimension.
        self.cell_size = (
            self.edges[0][1] - self.edges[0][0],
            self.edges[1][1] - self.edges[1][0],
        )
    
    @classmethod
    def from_edges(Cls, edges):
        x_max, x_min = edges[0][-1], edges[0][0] 
        y_max, y_min = edges[1][-1], edges[1][0] 
        return Cls(
            length = x_max - x_min,
            height = y_max - y_min,
            num_cols = len(edges[0])-1, num_rows = len(edges[1])-1,
            center = ((x_max + x_min)/2, (y_max + y_min)/2),
        )

    def coords(self, points: np.ndarray, check_out_of_bounds=True, simplify_shape=False):
        num_indices = len(points.shape) 
        if num_indices == 2:
            points = points.reshape(1, *points.shape)
        elif num_indices != 3:
            raise Exception(f"`len(points.shape)` é {len(points.shape)}, mas deveria ser 2 ou 3.")

        x = points[:, :, 0] - self.center[0] + self.length/2
        y = points[:, :, 1] - self.center[1] + self.height/2

        col_pos = np.floor(x / self.cell_size[0]).astype(int)
        row_pos = np.floor(y / self.cell_size[1]).astype(int)

        if check_out_of_bounds:
            col_pos[col_pos >= self.shape[0]] = self.shape[0]
            row_pos[row_pos >= self.shape[1]] = self.shape[1]
            col_pos[col_pos < 0] = -1
            row_pos[row_pos < 0] = -1

        coords = np.empty(points.shape, int)

        coords[:, :, 0] = col_pos
        coords[:, :, 1] = row_pos

        if simplify_shape and coords.shape[0] == 1:
            coords = coords.reshape(*coords.shape[1:])

        return coords

    def get_save_configs(self):
        return self.parent_configs

class PolarGridCfg(GridCfg):
    def __init__(self, r_edges: np.ndarray, theta_edges: np.ndarray, center=(0, 0)):
        """
        Polar grid configuration.

        Parameters
        ----------
        r_edges : np.ndarray
            Array of radial edges (including extremes).
        
        theta_edges : np.ndarray
            Array of angular edges in radians (including extremes).
        
        center : tuple
            (x, y) coordinates of the grid center.
        """
        self.r_edges = np.array(r_edges)
        self.theta_edges = np.array(theta_edges)
        self.center = center

    def get_polar_cfg(self):
        return self

class PolarGrid(Grid):
    configs: PolarGridCfg

    def __init__(self, configs: PolarGridCfg):
        super().__init__(configs)

        self.r_edges = configs.r_edges
        self.theta_edges = configs.theta_edges
        self.center = configs.center

        self.r_centers = (self.r_edges[1:] + self.r_edges[:-1]) / 2
        self.theta_centers = (self.theta_edges[1:] + self.theta_edges[:-1]) / 2

        self.meshgrid = np.meshgrid(self.r_centers, self.theta_centers, indexing='ij')

    def compute_edges(self):
        self._edges = (self.configs.r_edges, self.configs.theta_edges)
    
    def coords(self, points: np.ndarray, check_out_of_bounds=True, simplify_shape=False):
        num_indices = len(points.shape)
        if num_indices == 2:
            points = points.reshape(1, *points.shape)
        elif num_indices != 3:
            raise Exception(f"`len(points.shape)` is {len(points.shape)}, should be 2 or 3.")

        # Shift points to grid center
        x = points[:, :, 0] - self.center[0]
        y = points[:, :, 1] - self.center[1]

        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        theta = np.mod(theta, 2 * np.pi)  # Ensure theta in [0, 2pi)

        r_idx = np.searchsorted(self.r_edges, r, side='right') - 1
        theta_idx = np.searchsorted(self.theta_edges, theta, side='right') - 1

        coords = np.empty(points.shape, dtype=int)
        coords[:, :, 0] = r_idx
        coords[:, :, 1] = theta_idx

        if check_out_of_bounds:
            coords[:, :, 0][coords[:, :, 0] < 0] = -1
            coords[:, :, 0][coords[:, :, 0] >= self.shape[0]] = self.shape[0]
            coords[:, :, 1][coords[:, :, 1] < 0] = -1
            coords[:, :, 1][coords[:, :, 1] >= self.shape[1]] = self.shape[1]

        if simplify_shape and coords.shape[0] == 1:
            coords = coords.reshape(*coords.shape[1:])

        return coords

    def random_points(self, n):
        "Generate `n` random points inside the grid."
        r = np.random.uniform(self.r_edges[0], self.r_edges[-1], n)
        theta = np.random.uniform(self.theta_edges[0], self.theta_edges[-1], n)
        x = r * np.cos(theta) + self.center[0]
        y = r * np.sin(theta) + self.center[1]
        return np.column_stack((x, y))

    def plot_grid(self, ax, adjust_lims=True, color="black"):
        from matplotlib.collections import LineCollection
        
        r_edges = self.r_edges
        theta_edges = self.theta_edges

        lines = []

        # Radial lines (constant theta)
        for theta in theta_edges:
            x = r_edges * np.cos(theta) + self.center[0]
            y = r_edges * np.sin(theta) + self.center[1]
            points = np.column_stack([x, y])
            lines.append(points)

        # Circular arcs (constant r)
        for r in r_edges:
            thetas = np.linspace(theta_edges[0], theta_edges[-1], 200)
            x = r * np.cos(thetas) + self.center[0]
            y = r * np.sin(thetas) + self.center[1]
            points = np.column_stack([x, y])
            lines.append(points)

        lc = LineCollection(lines, colors=color)
        ax.add_collection(lc)

        if adjust_lims:
            max_r = r_edges[-1]
            ax.set_xlim(self.center[0] - max_r, self.center[0] + max_r)
            ax.set_ylim(self.center[1] - max_r, self.center[1] + max_r)

cfg_to_class = {
    RectangularGridCfg: RectangularGrid,
    RegularRectGridCfg: RegularRectGrid,
    PolarGridCfg: PolarGrid,
}

def load_grid(path):
    with open(path, "r") as f:
        configs = yaml.unsafe_load(f)

    return cfg_to_class[type(configs)](configs)

if __name__ == "__main__":
    l, h = 10, 5
    grid = RegularRectGrid(l, h, 7, 5)

    n = 10
    xs = (np.random.random(n) - 0.5) * l
    ys = (np.random.random(n) - 0.5) * h
    ps = np.array([xs, ys]).T

    coords = grid.coords(ps)

    coords = np.array([
        [0, 0],
        [0, 0],
        [1, 2],
        [1, 2],
        [1, 2],
    ])

    values = np.array([
        [1 , 1 ],
        [-3, 4 ],
        [3 , -3],
        [6 , 2 ],
        [-1, 11],
    ], dtype=float)

    r_all = grid.mean_by_cell(values, coords) 

    check_x = grid.mean_by_cell(values[:, 0], coords) == r_all[:, :, 0]
    check_y = grid.mean_by_cell(values[:, 1], coords) == r_all[:, :, 1]
    print("Mean by cell Test:", check_x.all(), check_y.all())

    # import matplotlib.pyplot as plt
    # plt.scatter(xs, ys, c="black")
    # plt.show()