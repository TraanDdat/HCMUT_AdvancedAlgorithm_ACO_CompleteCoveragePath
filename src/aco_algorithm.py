import numpy as np
import math
import random
import heapq

# Các tham số cho thuật toán ACO
ACO_ALPHA = 1.0       
ACO_BETA = 2.0        
ACO_RHO = 0.5         
ACO_Q = 100.0         
ACO_ITERATIONS = 30   
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
    CẬP NHẬT: Cho phép đi qua HEADLAND (4) và TRANSFER (5) để quay đầu.
    Chỉ chặn OBSTACLE_INDEPENDENT (1) và INVALID (0).
    """
    if cell_value == OBSTACLE_INDEPENDENT or cell_value == INVALID:
        return False
    return True

def get_neighbors(x, y, w, h):
    """Lấy danh sách các ô lân cận (4 hướng)"""
    # Chỉ đi 4 hướng vuông góc (Manhattan movement) để phù hợp máy móc
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    results = []
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            results.append((nx, ny, dx, dy))
    return results

def a_star_search(start, goal, grid, start_direction=None):
    """
    Directional A*: Tìm đường đi ưu tiên quán tính (đi thẳng).
    
    Quy tắc:
    1. Ưu tiên giữ nguyên hướng cũ.
    2. Nếu vật cản ở TRƯỚC MẶT -> Được phép rẽ.
    3. Nếu vật cản ở BÊN CẠNH (trái/phải) nhưng đường trước mặt thoáng -> BẮT BUỘC đi thẳng (phạt nặng nếu rẽ).
    4. Nếu đang ở vùng HEADLAND/TRANSFER -> Được phép rẽ thoải mái hơn.
    """
    h, w = grid.shape
    start_x, start_y = start
    goal_x, goal_y = goal
    
    # Priority queue lưu trữ: (f_score, g_score, x, y, last_dx, last_dy, path)
    # last_dx, last_dy: hướng di chuyển trước đó để tính quán tính
    
    # Khởi tạo hướng mặc định (ví dụ hướng dọc) nếu chưa có
    if start_direction is None:
        init_dx, init_dy = 0, 0 
    else:
        init_dx, init_dy = start_direction

    pq = [(0, 0, start_x, start_y, init_dx, init_dy, [(start_x, start_y)])]
    
    # Visited state phải bao gồm cả hướng đi đến ô đó: (x, y, dx, dy)
    # Vì đến ô (x,y) từ hướng Bắc có thể khác với đến từ hướng Tây
    visited = {} # Map (x, y, dx, dy) -> cost
    
    # Cost Config
    COST_MOVE = 1.0
    PENALTY_TURN_FIELD = 500.0   # Phạt rất nặng nếu rẽ lung tung trong ruộng
    PENALTY_TURN_OBSTACLE = 1.5  # Phạt nhẹ nếu rẽ vì bị chặn (bắt buộc)
    PENALTY_TURN_ZONE = 2.0      # Phạt nhẹ nếu rẽ trong vùng cho phép (Headland/Transfer)

    best_path = None
    min_cost = float('inf')

    while pq:
        f, g, x, y, last_dx, last_dy, path = heapq.heappop(pq)
        
        # Nếu đã tìm thấy đích, nhưng vẫn check xem có đường tối ưu hơn không
        if (x, y) == (goal_x, goal_y):
            if g < min_cost:
                min_cost = g
                best_path = path
            continue
            
        # Pruning
        state = (x, y, last_dx, last_dy)
        if state in visited and visited[state] <= g:
            continue
        visited[state] = g

        neighbors = get_neighbors(x, y, w, h)
        
        for nx, ny, ndx, ndy in neighbors:
            cell_val = grid[ny, nx]
            
            # Check cơ bản: Có đi được không?
            # Cho phép đi vào đích kể cả nếu nó bị đánh dấu là vật cản (để thoát ra)
            if not is_passable(cell_val) and (nx, ny) != (goal_x, goal_y):
                continue

            # --- TÍNH TOÁN CHI PHÍ DI CHUYỂN ---
            move_cost = COST_MOVE
            
            # Kiểm tra xem có đang rẽ không (thay đổi hướng)
            is_turn = (ndx != last_dx or ndy != last_dy) and (last_dx != 0 or last_dy != 0)
            
            if is_turn:
                current_cell_type = grid[y, x]
                
                # Check ô phía trước mặt (theo hướng cũ)
                front_x, front_y = x + last_dx, y + last_dy
                is_blocked_front = False
                
                if not (0 <= front_x < w and 0 <= front_y < h):
                    is_blocked_front = True # Hết bản đồ coi như bị chặn
                elif not is_passable(grid[front_y, front_x]):
                    is_blocked_front = True # Gặp vật cản trước mặt

                if is_blocked_front:
                    # TH1: Vật cản đối diện -> Được phép rẽ
                    move_cost += PENALTY_TURN_OBSTACLE
                elif current_cell_type in [HEADLAND, TRANSFER]:
                    # TH2: Đang ở vùng quay đầu -> Được phép rẽ
                    move_cost += PENALTY_TURN_ZONE
                else:
                    # TH3: Đường trước mặt thoáng VÀ đang ở trong Ruộng (Field)
                    # -> CẤM RẼ (hoặc phạt cực nặng) để ép đi thẳng
                    # Đây là logic "Không được quẹo khi vật cản nằm bên trái hoặc phải"
                    move_cost += PENALTY_TURN_FIELD
            
            new_g = g + move_cost
            
            # Heuristic Manhattan vì đi theo lưới vuông
            h_score = abs(nx - goal_x) + abs(ny - goal_y)
            
            heapq.heappush(pq, (new_g + h_score, new_g, nx, ny, ndx, ndy, path + [(nx, ny)]))
    
    if best_path:
        return best_path, min_cost
    
    # Fallback nếu không tìm thấy đường
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
    # Lưu ý: Cần kiểm tra kỹ logic chẵn lẻ dựa trên hệ tọa độ backend
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

    # Logic zic-zac
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
    
    all_corners = []
    for b in blocks:
        all_corners.extend(get_block_corners(b))
    
    # --- 1. Tính Heuristic ---
    print("Pre-calculating heuristics using Directional A*...")
    dist_cache = {} 

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j: continue
            
            block_i = i // 4
            block_j = j // 4
            
            # Không đi nội bộ giữa các góc của cùng 1 block (đã có đường zic-zac lo)
            if block_i == block_j:
                heuristics[i][j] = 0.0001
                continue

            key = tuple(sorted((i, j)))
            # Lưu ý: A* có hướng thì A->B khác B->A, nhưng để đơn giản cho heuristic
            # ta ước lượng 1 chiều hoặc dùng Euclidean cho nhanh phần khởi tạo
            # Ở đây dùng Euclidean cho heuristic matrix để nhẹ gánh tính toán ban đầu
            dist = euclidean_distance(all_corners[i], all_corners[j])
            
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
            
            # Giả lập hướng thoát ra của block đầu tiên (thường là dọc)
            # Dựa vào exit corner:
            # TL/TR -> hướng lên (-1 y), BL/BR -> hướng xuống (+1 y)
            # Để A* biết hướng nào mà đi tiếp
            exit_corner_name = get_corner_name(start_exit_idx)
            if 'top' in exit_corner_name: last_dir = (0, -1)
            else: last_dir = (0, 1)

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
                        
                        p = (tau ** ACO_ALPHA) * (eta ** ACO_BETA)
                        probs.append(p)
                        candidates.append((next_block_idx, next_entry_idx))
                        sum_prob += p
                
                if sum_prob == 0:
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
                
                # Tính chi phí thực tế bằng Directional A*
                prev_exit_coord = all_corners[curr_global_node]
                new_entry_coord = all_corners[next_block_idx * 4 + next_entry_idx]
                
                # Gọi A* với hướng thoát của block trước
                _, dist = a_star_search(prev_exit_coord, new_entry_coord, backend_grid, start_direction=last_dir)
                
                total_dist += dist
                current_tour.append((next_block_idx, next_entry_idx, next_exit_idx))
                visited_blocks.add(next_block_idx)
                
                # Cập nhật last_dir cho vòng lặp sau
                nxt_ex_name = get_corner_name(next_exit_idx)
                if 'top' in nxt_ex_name: last_dir = (0, -1)
                else: last_dir = (0, 1)
            
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

    # --- 3. Tái tạo đường đi cuối cùng ---
    path_final = []
    
    if best_tour:
        # Xử lý điểm Start của user
        first_dir = (0, 1) # Mặc định hướng xuống
        if start_point:
             b0_idx, ent0, _ = best_tour[0]
             p0 = all_corners[b0_idx * 4 + ent0]
             path_to_start, _ = a_star_search(start_point, p0, backend_grid, start_direction=(0,1))
             path_final.extend(path_to_start)
             # Xác định hướng khi đến p0 để dùng cho block tiếp theo
             if len(path_to_start) > 1:
                 first_dir = (path_to_start[-1][0] - path_to_start[-2][0], path_to_start[-1][1] - path_to_start[-2][1])

        last_dir = first_dir

        for i, step in enumerate(best_tour):
            b_idx, entry_idx, _ = step
            block = blocks[b_idx]
            
            entry_coord = all_corners[b_idx * 4 + entry_idx]
            
            if i > 0:
                prev_b_idx, _, prev_ex_idx = best_tour[i-1]
                prev_exit_coord = all_corners[prev_b_idx * 4 + prev_ex_idx]
                
                # Nối block bằng Directional A*
                connection_path, _ = a_star_search(prev_exit_coord, entry_coord, backend_grid, start_direction=last_dir)
                path_final.extend(connection_path)
                
                # Cập nhật hướng cuối cùng của đoạn nối
                if len(connection_path) > 1:
                    last_dir = (connection_path[-1][0] - connection_path[-2][0], connection_path[-1][1] - connection_path[-2][1])

            # Đường trong block
            block_path = generate_path_inside_block(block, entry_idx)
            path_final.extend(block_path)
            
            # Cập nhật hướng thoát ra khỏi block (để dùng cho đoạn nối tiếp theo)
            if len(block_path) > 1:
                 last_dir = (block_path[-1][0] - block_path[-2][0], block_path[-1][1] - block_path[-2][1])

    return path_final, [], [], best_cost