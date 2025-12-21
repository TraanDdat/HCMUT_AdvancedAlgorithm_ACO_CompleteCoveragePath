import numpy as np

def aco_algorithm(backend_grid, seg_grid, merge_field, start_point=None, start_point_direction=None):
    """ ACO algorithm to plan path to cover all field/ headland
    backend_grid: 2D numpy array, grid representation of field/ headland
    seg_grid: 2D numpy array, grid representation of segments
    merge_field: list of blocks to cover
    start_point: (x,y) tuple, starting point of the path
    start_point_direction: str, starting direction of the path
    return: path_final: list of (x,y) tuples representing the path
            direct_list: list of directions taken
            coverage_list: list of coverage percentages at each step
    """
    path_final = []
    direct_list = []
    coverage_list = []
    cost = 0
    
    # stub implementation of ACO algorithm

    return path_final, direct_list, coverage_list, cost