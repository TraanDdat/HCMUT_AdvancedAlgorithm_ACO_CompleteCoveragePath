import numpy as np
import math
import random
from scipy.interpolate import splprep, splev

# Constants from ui_graph (assumed mapping based on context)
OBSTACLE = 1
INVALID = 0
FIELD = 2
HEADLAND = 3
TRANSFER = 4

class ImprovedACO:
    def __init__(self, grid, start, end, max_iter=50, ant_count=30):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.start = start
        self.end = end
        self.max_iter = max_iter
        self.ant_count = ant_count
        
        # Parameters from PDF (Table 1 & Text)
        self.rho = 0.15  # Evaporation rate
        self.Q = 1.0     # Pheromone intensity
        self.alpha_base = 1.0 # Initial PHF
        self.beta_base = 2.0  # Initial EHF
        self.alpha = self.alpha_base
        self.beta = self.beta_base
        
        # Heuristic information (1/distance to target)
        self.eta = np.zeros(self.grid.shape)
        self._init_heuristic()
        
        # Pheromone matrix (Cone initialization)
        self.tau = np.zeros(self.grid.shape)
        self._init_cone_pheromone()
        
        self.best_path = []
        self.best_length = float('inf')

    def _init_heuristic(self):
        """Calculate heuristic value (inverse distance to target) for all cells"""
        target_x, target_y = self.end
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] != OBSTACLE and self.grid[r, c] != INVALID:
                    dist = math.sqrt((r - target_y)**2 + (c - target_x)**2)
                    self.eta[r, c] = 1.0 / (dist + 0.1) # Add epsilon to avoid div by 0

    def _init_cone_pheromone(self):
        """
        PDF Section: Cone pheromone initialization (Eq. 1)
        tau_0 = (0.09 * |x-y|) / len + 1/d
        Creates a cone shape guiding towards the target.
        """
        target_x, target_y = self.end
        start_x, start_y = self.start
        map_diag = math.sqrt(self.rows**2 + self.cols**2)
        
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] == OBSTACLE:
                    continue
                
                # Euclidean distance to target
                d = math.sqrt((r - target_y)**2 + (c - target_x)**2)
                if d == 0: d = 0.1
                
                # Distance from start line (simplified cone logic)
                # Here we use distance projection to line Start-End
                dist_to_start = math.sqrt((r - start_y)**2 + (c - start_x)**2)
                
                # Eq 1 implementation approximation
                term1 = (0.09 * dist_to_start) / map_diag
                term2 = 1.0 / d
                self.tau[r, c] = term1 + term2

    def _update_adaptive_factors(self, current_iter):
        """
        PDF Section: Adaptive heuristic factor (Eq. 2)
        Dynamically adjust alpha and beta based on iteration progress.
        Early stage: Exploratory (low alpha/beta).
        Late stage: Exploitation (high alpha/beta).
        """
        progress = current_iter / self.max_iter
        # Simplified implementation of the integral concept in Eq 2
        # Linearly increasing influence
        self.alpha = self.alpha_base + 2.0 * progress
        self.beta = self.beta_base + 3.0 * progress

    def _get_neighbors(self, pos, tabu_list):
        x, y = pos
        neighbors = []
        # 8-connected grid
        moves = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]
        
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.cols and 0 <= ny < self.rows:
                if self.grid[ny, nx] != OBSTACLE and self.grid[ny, nx] != INVALID:
                    if (nx, ny) not in tabu_list:
                        neighbors.append((nx, ny))
        return neighbors

    def _select_next_node(self, current_pos, tabu_list, is_king_ant):
        """
        Select next node based on Pheromone and Heuristic.
        PDF Section: Ant colony division of labor.
        King Ant: Deterministic (Max probability).
        Soldier Ant: Probabilistic (Roulette Wheel).
        """
        neighbors = self._get_neighbors(current_pos, tabu_list)
        
        if not neighbors:
            return None # Deadlock
            
        probabilities = []
        sum_prob = 0.0
        
        for nx, ny in neighbors:
            tau_val = self.tau[ny, nx]
            eta_val = self.eta[ny, nx]
            
            # Transition probability formula
            prob = (tau_val ** self.alpha) * (eta_val ** self.beta)
            probabilities.append(prob)
            sum_prob += prob
            
        if sum_prob == 0:
            return random.choice(neighbors)
            
        probabilities = [p / sum_prob for p in probabilities]
        
        # Decision Strategy based on Ant Type
        if is_king_ant:
            # King ant chooses the best path (Exploitation)
            best_idx = np.argmax(probabilities)
            return neighbors[best_idx]
        else:
            # Soldier ant chooses probabilistically (Exploration)
            # Use random.choices (Python 3.6+) or numpy choice
            idx = np.random.choice(len(neighbors), p=probabilities)
            return neighbors[idx]

    def _solve_deadlock(self, tabu_list):
        """
        PDF Section: Deadlock problem.
        If stuck, backtrack and mark current node as effectively 'visited' (tabu)
        to force exploration elsewhere.
        """
        if len(tabu_list) > 1:
            bad_node = tabu_list.pop() # Remove current stuck node
            # In a full implementation, we might mark this node as temporarily invalid 
            # for this specific ant, but here popping allows backtracking.
            return tabu_list[-1] # Return to previous node
        return None

    def run(self):
        for iteration in range(self.max_iter):
            self._update_adaptive_factors(iteration)
            
            # Division of Labor: Calculate ratio of King vs Soldier
            # Eq 3 & 4 in PDF describe dynamic Lambda. 
            # Simplified: More kings as iterations increase.
            king_ratio = 0.1 + (0.8 * (iteration / self.max_iter))
            
            paths = []
            lengths = []
            
            for k in range(self.ant_count):
                is_king = random.random() < king_ratio
                
                # Ant initialization
                current_pos = self.start
                path = [self.start]
                tabu_set = set([self.start]) # Faster lookup
                path_len = 0
                steps = 0
                max_steps = self.rows * self.cols # Safety break
                
                reached_target = False
                
                while steps < max_steps:
                    if current_pos == self.end:
                        reached_target = True
                        break
                        
                    next_node = self._select_next_node(current_pos, tabu_set, is_king)
                    
                    if next_node:
                        # Move forward
                        dist = math.sqrt((next_node[0]-current_pos[0])**2 + (next_node[1]-current_pos[1])**2)
                        path_len += dist
                        current_pos = next_node
                        path.append(current_pos)
                        tabu_set.add(current_pos)
                    else:
                        # Deadlock detected
                        backtrack_node = self._solve_deadlock(path)
                        if backtrack_node:
                            current_pos = backtrack_node
                            # Note: In strict PDF logic, we mark the deadlock area in a Tabu table
                            # Here we simply backtrack and retry other neighbors in next loop implicitly
                        else:
                            break # Cannot backtrack
                    steps += 1
                
                if reached_target:
                    paths.append(path)
                    lengths.append(path_len)
                    
                    if path_len < self.best_length:
                        self.best_length = path_len
                        self.best_path = path
            
            # Pheromone Update (Global)
            # Evaporation
            self.tau *= (1 - self.rho)
            
            # Reinforcement
            for i, path in enumerate(paths):
                L_k = lengths[i]
                deposit = self.Q / L_k
                for x, y in path:
                    self.tau[y, x] += deposit

        return self.best_path

    def smooth_path(self, path):
        """
        PDF Section: Path smoothing optimization.
        Uses B-spline to reduce turning points and smooth the trajectory.
        """
        if len(path) < 3:
            return path
            
        try:
            # Extract x and y coordinates
            x = [p[0] for p in path]
            y = [p[1] for p in path]
            
            # B-spline interpolation (requires scipy)
            # k=3 for cubic B-spline as mentioned in PDF
            tck, u = splprep([x, y], s=2.0, k=2) # s is smoothing factor
            u_new = np.linspace(0, 1, num=len(path)*2) # Increase resolution
            x_new, y_new = splev(u_new, tck)
            
            smoothed_path = []
            for i in range(len(x_new)):
                # Snap to grid integers or keep float depending on requirement
                # Here we keep integers for grid consistency
                px, py = int(round(x_new[i])), int(round(y_new[i]))
                # Ensure we don't hit obstacles after smoothing (simplified check)
                if 0 <= py < self.rows and 0 <= px < self.cols:
                     if self.grid[py, px] != OBSTACLE and self.grid[py, px] != INVALID:
                         smoothed_path.append((px, py))
            
            # Remove duplicates
            final_smooth = []
            seen = set()
            for p in smoothed_path:
                if p not in seen:
                    final_smooth.append(p)
                    seen.add(p)
            
            # Ensure start and end are preserved
            if final_smooth[0] != self.start: final_smooth.insert(0, self.start)
            if final_smooth[-1] != self.end: final_smooth.append(self.end)
            
            return final_smooth
            
        except ImportError:
            print("Scipy not found, skipping smoothing")
            return path
        except Exception as e:
            print(f"Smoothing failed: {e}")
            return path

def aco_algorithm(backend_grid, seg_grid, merge_field, start_point=None, start_point_direction=None):
    """ 
    Improved ACO algorithm based on:
    'Global and local path planning of robots combining ACO and dynamic window algorithm'
    
    This implementation maps the point-to-point ACO logic from the paper to the 
    provided grid environment.
    """
    
    # 1. Setup Configuration
    h, w = backend_grid.shape
    
    # Determine Start Point
    if start_point is None:
        # Find first available FIELD cell
        found = False
        for y in range(h):
            for x in range(w):
                if backend_grid[y, x] == FIELD:
                    start_point = (x, y)
                    found = True
                    break
            if found: break
        if start_point is None: start_point = (0, 0) # Fallback

    # Determine Target Point (End)
    # Since the inputs are for coverage (merge_field), but the PDF algorithm is Point-to-Point,
    # we simulate a goal: The center of the last field block, or the furthest valid point.
    end_point = None
    if merge_field and len(merge_field) > 0:
        last_block = merge_field[-1]
        # Calculate centroid of the last block
        cx = (last_block['xmin'] + last_block['xmax']) // 2
        cy = (last_block['ymin'] + last_block['ymax']) // 2
        end_point = (cx, cy)
    else:
        # Default to bottom-rightmost field
        end_point = start_point # Fallback
        for y in range(h-1, -1, -1):
            for x in range(w-1, -1, -1):
                if backend_grid[y, x] == FIELD:
                    end_point = (x, y)
                    break
            if end_point != start_point: break

    print(f"ACO Planning: Start={start_point}, End={end_point}")

    # 2. Initialize Improved ACO Planner
    aco = ImprovedACO(
        grid=backend_grid, 
        start=start_point, 
        end=end_point, 
        max_iter=50, # As per PDF experiments for 20x20/30x30 maps
        ant_count=30
    )

    # 3. Run Optimization
    raw_path = aco.run()
    
    # 4. Path Smoothing (PDF optimization #5)
    path_final = aco.smooth_path(raw_path)
    
    # 5. Generate Output Formats
    direct_list = []
    coverage_list = []
    cost = 0
    
    # Calculate direction list and cost
    if len(path_final) > 1:
        for i in range(len(path_final) - 1):
            p1 = path_final[i]
            p2 = path_final[i+1]
            cost += math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            
            # Determine direction
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dx == 1: direction = "right"
            elif dx == -1: direction = "left"
            elif dy == 1: direction = "down"
            elif dy == -1: direction = "up"
            else: direction = "diagonal"
            direct_list.append(direction)
            
            # Mock coverage calculation (percentage of FIELD cells visited)
            # In a real coverage planner, this would track visited set vs total field set
            coverage_list.append(min(100, (i / len(path_final)) * 100)) # Placeholder

    if not path_final:
        print("ACO failed to find a path.")
        path_final = [start_point]

    return path_final, direct_list, coverage_list, cost