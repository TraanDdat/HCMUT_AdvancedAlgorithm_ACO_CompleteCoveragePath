import numpy as np
import math
import random
import heapq

# Các tham số cho thuật toán ACO
ACO_ALPHA = 1.0       # Tầm quan trọng của Pheromone
ACO_BETA = 2.0        # Tầm quan trọng của Heuristic (khoảng cách) - Giảm nhẹ để kiến khám phá tốt hơn
ACO_RHO = 0.5         # Tốc độ bay hơi
ACO_Q = 100.0         # Hằng số cập nhật Pheromone
ACO_ITERATIONS = 30   # Số lần lặp
ACO_NUM_ANTS_FACTOR = 1.0 

# Định nghĩa các giá trị trên Grid (khớp với ui_graph.py)
INVALID = 0
OBSTACLE_INDEPENDENT = 1 # Black
FIELD = 2                # Green
OBSTACLE_SMALL = 3       # Yellow
HEADLAND = 4             # Brown
TRANSFER = 5             # Gray

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_passable(cell_value):
    """
    Định nghĩa ô nào được phép đi qua.
    Chặn: OBSTACLE_INDEPENDENT (1) và TRANSFER (5 - màu xám)
    """
    if cell_value == OBSTACLE_INDEPENDENT or cell_value == TRANSFER:
        return False
    return True

def a_star_search(start, goal, grid):
    """
    Tìm đường đi ngắn nhất từ start đến goal trên grid tránh vật cản.
    start, goal: tuple (x, y)
    grid: numpy array 2D
    Trả về: (path, cost)
        - path: list các điểm [(x,y), ...]
        - cost: chiều dài đường đi (số bước)
    """
    h, w = grid.shape
    start_x, start_y = start
    goal_x, goal_y = goal
    
    # Nếu điểm đầu hoặc cuối nằm trong vật cản, cố gắng thoát ra hoặc chấp nhận rủi ro
    # Ở đây ta giả định điểm entry/exit của block luôn nằm ở vùng an toàn (Field/Headland)
    
    # Priority queue: (f_score, cost, x, y, path)
    pq = [(0, 0, start_x, start_y, [(start_x, start_y)])]
    visited = set()
    
    # Các hướng di chuyển (8 hướng: ngang, dọc, chéo)
    directions = [
        (0, 1, 1), (0, -1, 1), (1, 0, 1), (-1, 0, 1) # Ngang dọc (cost 1)
        #(1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414) # Chéo (cost sqrt(2))
    ]

    while pq:
        f, cost, x, y, path = heapq.heappop(pq)
        
        if (x, y) == (goal_x, goal_y):
            return path, cost
        
        if (x, y) in visited:
            continue
        visited.add((x, y))
        
        for dx, dy, move_cost in directions:
            nx, ny = x + dx, y + dy
            
            # Kiểm tra biên
            if 0 <= nx < w and 0 <= ny < h:
                # Kiểm tra vật cản
                # Lưu ý: grid trong numpy là [row, col] tuc la [y, x]
                cell_val = grid[ny, nx]
                
                # Logic: Chỉ đi qua nếu ô đó passable HOẶC ô đó là đích đến
                if is_passable(cell_val) or (nx == goal_x and ny == goal_y):
                    if (nx, ny) not in visited:
                        new_cost = cost + move_cost
                        # Heuristic: Euclidean distance
                        heuristic = math.sqrt((nx - goal_x)**2 + (ny - goal_y)**2)
                        heapq.heappush(pq, (new_cost + heuristic, new_cost, nx, ny, path + [(nx, ny)]))
    
    # Nếu không tìm thấy đường (bị vây kín), trả về đường thẳng nhưng cost rất lớn
    # để thuật toán ACO tránh chọn đường này.
    return [start, goal], 999999.0

def get_block_corners(block):
    """Lấy tọa độ 4 góc của block"""
    return [
        block['top_left'],      # 0
        block['top_right'],     # 1
        block['bottom_left'],   # 2
        block['bottom_right']   # 3
    ]

def get_corner_name(idx):
    names = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    return names[idx]

def get_forced_exit_idx(block, entry_idx):
    """
    Xác định điểm ra bắt buộc dựa trên điểm vào và tính chẵn lẻ của chiều rộng block.
    """
    xmin = block['xmin']
    xmax = block['xmax']
    is_odd_width = (xmax - xmin) % 2 == 0 

    # Mapping index góc: 0: TL, 1: TR, 2: BL, 3: BR
    if is_odd_width:
        mapping = {0: 3, 1: 2, 2: 1, 3: 0}
    else:
        mapping = {0: 2, 1: 3, 2: 0, 3: 1}
        
    return mapping[entry_idx]

def generate_path_inside_block(block, entry_idx):
    """Sinh tọa độ đường đi chi tiết bên trong block"""
    entry_name = get_corner_name(entry_idx)
    path = []
    
    xmin = block['xmin']
    xmax = block['xmax']
    ymin = block['ymin'] 
    ymax = block['ymax']

    if entry_name == 'top_left':
        start_x, end_x, start_y, end_y = xmin, xmax, ymin, ymax
        slice_dir = 'ltr_down'
    elif entry_name == 'top_right':
        start_x, end_x, start_y, end_y = xmax, xmin, ymin, ymax
        slice_dir = 'rtl_down'
    elif entry_name == 'bottom_left':
        start_x, end_x, start_y, end_y = xmin, xmax, ymax, ymin
        slice_dir = 'ltr_up'
    elif entry_name == 'bottom_right':
        start_x, end_x, start_y, end_y = xmax, xmin, ymax, ymin
        slice_dir = 'rtl_up'
    else:
        return []

    if slice_dir == 'ltr_down':
        for x in range(start_x, end_x + 1):
            if (x - xmin) % 2 == 0: 
                path.extend([(x, y) for y in range(ymin, ymax + 1)])
            else: 
                path.extend([(x, y) for y in range(ymax, ymin - 1, -1)])
    elif slice_dir == 'rtl_down':
        for x in range(start_x, end_x - 1, -1):
            if (x - xmin) % 2 == 0:
                path.extend([(x, y) for y in range(ymin, ymax + 1)])
            else:
                path.extend([(x, y) for y in range(ymax, ymin - 1, -1)])
    elif slice_dir == 'ltr_up':
        for x in range(start_x, end_x + 1):
            if (x - xmin) % 2 == 0: 
                path.extend([(x, y) for y in range(ymax, ymin - 1, -1)])
            else:
                path.extend([(x, y) for y in range(ymin, ymax + 1)])
    elif slice_dir == 'rtl_up':
        for x in range(start_x, end_x - 1, -1):
            if (x - xmin) % 2 == 0:
                path.extend([(x, y) for y in range(ymax, ymin - 1, -1)])
            else:
                path.extend([(x, y) for y in range(ymin, ymax + 1)])
                
    return path

def aco_algorithm(backend_grid, seg_grid, merge_field, start_point=None, start_point_direction=None):
    blocks = merge_field
    num_blocks = len(blocks)
    
    if num_blocks == 0:
        return [], [], [], 0

    num_nodes = num_blocks * 4
    
    # Ma trận Pheromone và Heuristic
    pheromones = np.ones((num_nodes, num_nodes)) * 0.1
    heuristics = np.zeros((num_nodes, num_nodes))
    
    # Lấy tất cả tọa độ góc
    all_corners = []
    for b in blocks:
        all_corners.extend(get_block_corners(b))
    
    # --- 1. Tính Heuristic bằng A* (tốn thời gian hơn nhưng chính xác với map) ---
    print("Pre-calculating heuristics using A*...")
    # Cache khoảng cách để tránh tính lại
    dist_cache = {} 

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j: continue
            
            # Chỉ tính heuristic giữa các node KHÁC block (vì trong block đã có forced path)
            block_i = i // 4
            block_j = j // 4
            if block_i == block_j:
                heuristics[i][j] = 0.0001 # Không khuyến khích đi nội bộ kiểu này
                continue

            # Key cache
            key = tuple(sorted((i, j)))
            if key in dist_cache:
                dist = dist_cache[key]
            else:
                # Dùng A* để tính khoảng cách thực tế tránh vùng Xám/Đen
                # Sử dụng backend_grid để check vật cản
                _, dist = a_star_search(all_corners[i], all_corners[j], backend_grid)
                dist_cache[key] = dist
            
            if dist < 0.001: heuristics[i][j] = 1000.0
            else: heuristics[i][j] = 1.0 / dist
    
    print("Heuristics calculated.")

    best_tour = None
    best_cost = float('inf')
    num_ants = int(num_nodes * ACO_NUM_ANTS_FACTOR)
    if num_ants < 1: num_ants = 1

    # --- 2. Vòng lặp ACO ---
    for iteration in range(ACO_ITERATIONS):
        ant_tours = []
        ant_costs = []
        
        for ant in range(num_ants):
            visited_blocks = set()
            current_tour = [] 
            
            # Xử lý điểm bắt đầu
            start_block_idx = -1
            start_entry_idx = -1
            
            if start_point and iteration == 0 and ant == 0:
                # Kiến đầu tiên ở vòng đầu tiên sẽ thử đi từ điểm user chọn
                # Tìm block gần start_point nhất để bắt đầu
                min_d = float('inf')
                for b_i, blk in enumerate(blocks):
                    corners = get_block_corners(blk)
                    for c_i, c in enumerate(corners):
                        d = euclidean_distance(start_point, c)
                        if d < min_d:
                            min_d = d
                            start_block_idx = b_i
                            start_entry_idx = c_i
            else:
                start_block_idx = random.randint(0, num_blocks - 1)
                start_entry_idx = random.randint(0, 3)

            start_exit_idx = get_forced_exit_idx(blocks[start_block_idx], start_entry_idx)
            current_tour.append((start_block_idx, start_entry_idx, start_exit_idx))
            visited_blocks.add(start_block_idx)
            total_dist = 0.0
            
            # Xây dựng tour
            while len(visited_blocks) < num_blocks:
                curr_block_idx, _, curr_exit_idx = current_tour[-1]
                curr_global_node = curr_block_idx * 4 + curr_exit_idx
                
                probs = []
                candidates = []
                sum_prob = 0.0
                
                for next_block_idx in range(num_blocks):
                    if next_block_idx in visited_blocks: continue
                    
                    for next_entry_idx in range(4):
                        next_global_node = next_block_idx * 4 + next_entry_idx
                        tau = pheromones[curr_global_node][next_global_node]
                        eta = heuristics[curr_global_node][next_global_node]
                        
                        # Nếu heuristic quá nhỏ (đường bị chặn), eta sẽ rất nhỏ -> xác suất thấp
                        p = (tau ** ACO_ALPHA) * (eta ** ACO_BETA)
                        probs.append(p)
                        candidates.append((next_block_idx, next_entry_idx))
                        sum_prob += p
                
                if sum_prob == 0:
                    # Nếu bế tắc (vây kín), chọn đại
                    remain = [b for b in range(num_blocks) if b not in visited_blocks]
                    if not remain: break
                    next_block_idx = random.choice(remain)
                    next_entry_idx = random.randint(0, 3)
                else:
                    r = random.uniform(0, sum_prob)
                    cumsum = 0
                    selected = False
                    for idx, p in enumerate(probs):
                        cumsum += p
                        if r <= cumsum:
                            next_block_idx, next_entry_idx = candidates[idx]
                            selected = True
                            break
                    if not selected:
                        next_block_idx, next_entry_idx = candidates[-1]

                next_exit_idx = get_forced_exit_idx(blocks[next_block_idx], next_entry_idx)
                
                # Tính chi phí di chuyển (Dùng A*)
                prev_exit_coord = all_corners[curr_global_node]
                new_entry_coord = all_corners[next_block_idx * 4 + next_entry_idx]
                
                # Ta có thể dùng lại cache hoặc gọi A* (cache tốt hơn)
                key_dist = tuple(sorted((curr_global_node, next_block_idx * 4 + next_entry_idx)))
                if key_dist in dist_cache:
                    dist = dist_cache[key_dist]
                else:
                    _, dist = a_star_search(prev_exit_coord, new_entry_coord, backend_grid)
                
                total_dist += dist
                current_tour.append((next_block_idx, next_entry_idx, next_exit_idx))
                visited_blocks.add(next_block_idx)
            
            ant_tours.append(current_tour)
            ant_costs.append(total_dist)
            
            if total_dist < best_cost:
                best_cost = total_dist
                best_tour = current_tour

        # Cập nhật Pheromone
        pheromones *= (1.0 - ACO_RHO)
        for t_idx, tour in enumerate(ant_tours):
            cost = ant_costs[t_idx]
            if cost == 0: cost = 0.001
            delta_tau = ACO_Q / cost
            for i in range(len(tour) - 1):
                b1_idx, _, exit1_idx = tour[i]
                b2_idx, entry2_idx, _ = tour[i+1]
                n1 = b1_idx * 4 + exit1_idx
                n2 = b2_idx * 4 + entry2_idx
                pheromones[n1][n2] += delta_tau

    # --- 3. Tái tạo đường đi cuối cùng (vẽ chi tiết đường cong A*) ---
    path_final = []
    
    if best_tour:
        # Nếu có điểm start_point từ user, vẽ đường từ start_point đến entry đầu tiên
        if start_point:
             b0_idx, ent0, _ = best_tour[0]
             p0 = all_corners[b0_idx * 4 + ent0]
             # Đường dẫn từ vị trí xe hiện tại đến điểm vào block đầu tiên
             path_to_start, _ = a_star_search(start_point, p0, backend_grid)
             path_final.extend(path_to_start)

        for i, step in enumerate(best_tour):
            b_idx, entry_idx, _ = step
            block = blocks[b_idx]
            
            # Vẽ đường nối từ Block trước đến Block này (Headland Path)
            entry_coord = all_corners[b_idx * 4 + entry_idx]
            
            if i > 0:
                prev_b_idx, _, prev_ex_idx = best_tour[i-1]
                prev_exit_coord = all_corners[prev_b_idx * 4 + prev_ex_idx]
                
                # Tìm đường chi tiết bằng A* thay vì nối thẳng
                connection_path, _ = a_star_search(prev_exit_coord, entry_coord, backend_grid)
                path_final.extend(connection_path)

            # Thêm đường đi bên trong block (Zic-zac)
            block_path = generate_path_inside_block(block, entry_idx)
            path_final.extend(block_path)

    return path_final, [], [], best_cost