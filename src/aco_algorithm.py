import numpy as np
import math
import random

# Các tham số cho thuật toán ACO
ACO_ALPHA = 1.0       # Tầm quan trọng của Pheromone
ACO_BETA = 5.0        # Tầm quan trọng của Heuristic (khoảng cách)
ACO_RHO = 0.5         # Tốc độ bay hơi
ACO_Q = 100.0         # Hằng số cập nhật Pheromone
ACO_ITERATIONS = 50   # Số lần lặp
ACO_NUM_ANTS_FACTOR = 1.0 

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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
    Logic này mô phỏng đường đi zíc-zắc (Boustrophedon).
    """
    xmin = block['xmin']
    xmax = block['xmax']
    # Nếu (xmax - xmin) là số chẵn -> Số lượng đường chạy là LẺ (ví dụ: x=0 đến x=0 là độ rộng 1 cell)
    is_odd_width = (xmax - xmin) % 2 == 0 

    # Mapping index góc: 0: TL, 1: TR, 2: BL, 3: BR
    if is_odd_width:
        # Số đường lẻ: Vào góc này -> Ra góc đối diện chéo
        mapping = {0: 3, 1: 2, 2: 1, 3: 0}
    else:
        # Số đường chẵn: Vào góc này -> Ra góc cùng phía theo trục dọc
        mapping = {0: 2, 1: 3, 2: 0, 3: 1}
        
    return mapping[entry_idx]

def generate_path_inside_block(block, entry_idx):
    """Sinh tọa độ đường đi chi tiết bên trong block"""
    entry_name = get_corner_name(entry_idx)
    path = []
    
    xmin = block['xmin']
    xmax = block['xmax']
    ymin = block['ymin'] # Lưu ý: hệ tọa độ trong block đã được chuẩn hóa ymin < ymax
    ymax = block['ymax']

    # Xác định hướng quét và điểm bắt đầu
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

    # Sinh đường đi zíc-zắc
    if slice_dir == 'ltr_down':
        for x in range(start_x, end_x + 1):
            if (x - xmin) % 2 == 0: # Đi xuống
                path.extend([(x, y) for y in range(ymin, ymax + 1)])
            else: # Đi lên
                path.extend([(x, y) for y in range(ymax, ymin - 1, -1)])
    elif slice_dir == 'rtl_down':
        for x in range(start_x, end_x - 1, -1):
            if (x - xmin) % 2 == 0:
                path.extend([(x, y) for y in range(ymin, ymax + 1)])
            else:
                path.extend([(x, y) for y in range(ymax, ymin - 1, -1)])
    elif slice_dir == 'ltr_up':
        for x in range(start_x, end_x + 1):
            if (x - xmin) % 2 == 0: # Đi lên (vì bắt đầu từ dưới)
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
    
    # Nếu không có block nào thì trả về rỗng
    if num_blocks == 0:
        return [], [], [], 0

    # Mỗi block có 4 node (4 góc). Tổng số node = 4 * num_blocks
    num_nodes = num_blocks * 4
    
    # Ma trận Pheromone và Heuristic
    pheromones = np.ones((num_nodes, num_nodes)) * 0.1
    heuristics = np.zeros((num_nodes, num_nodes))
    
    # Lấy tất cả tọa độ góc để tính khoảng cách
    all_corners = []
    for b in blocks:
        all_corners.extend(get_block_corners(b))
        
    # Tính Heuristic (nghịch đảo khoảng cách)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j: continue
            dist = euclidean_distance(all_corners[i], all_corners[j])
            if dist < 0.001: heuristics[i][j] = 1000.0
            else: heuristics[i][j] = 1.0 / dist

    best_tour = None
    best_cost = float('inf')
    num_ants = int(num_nodes * ACO_NUM_ANTS_FACTOR)
    if num_ants < 1: num_ants = 1

    # --- Vòng lặp ACO ---
    for iteration in range(ACO_ITERATIONS):
        ant_tours = []
        ant_costs = []
        
        for ant in range(num_ants):
            visited_blocks = set()
            current_tour = [] # List of (block_idx, entry_idx, exit_idx)
            
            # 1. Chọn điểm xuất phát ngẫu nhiên
            start_block_idx = random.randint(0, num_blocks - 1)
            start_entry_idx = random.randint(0, 3)
            start_exit_idx = get_forced_exit_idx(blocks[start_block_idx], start_entry_idx)
            
            current_tour.append((start_block_idx, start_entry_idx, start_exit_idx))
            visited_blocks.add(start_block_idx)
            total_dist = 0.0
            
            # 2. Xây dựng tour
            while len(visited_blocks) < num_blocks:
                curr_block_idx, _, curr_exit_idx = current_tour[-1]
                curr_global_node = curr_block_idx * 4 + curr_exit_idx
                
                probs = []
                candidates = []
                sum_prob = 0.0
                
                # Tìm block tiếp theo
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
                
                # Chọn theo xác suất (Roulette wheel)
                if sum_prob == 0:
                    next_block_idx = random.choice([b for b in range(num_blocks) if b not in visited_blocks])
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
                
                # Tính chi phí di chuyển (Headland cost)
                prev_exit_coord = all_corners[curr_global_node]
                new_entry_coord = all_corners[next_block_idx * 4 + next_entry_idx]
                dist = euclidean_distance(prev_exit_coord, new_entry_coord)
                
                total_dist += dist
                current_tour.append((next_block_idx, next_entry_idx, next_exit_idx))
                visited_blocks.add(next_block_idx)
            
            ant_tours.append(current_tour)
            ant_costs.append(total_dist)
            
            if total_dist < best_cost:
                best_cost = total_dist
                best_tour = current_tour

        # 3. Cập nhật Pheromone
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

    # --- Tái tạo đường đi cuối cùng ---
    path_final = []
    
    if best_tour:
        for i, step in enumerate(best_tour):
            b_idx, entry_idx, _ = step
            block = blocks[b_idx]
            
            # Thêm đường nối từ block trước đến block này (Headland path)
            entry_coord = get_block_corners(block)[entry_idx]
            if i > 0:
                path_final.append(entry_coord) 

            # Thêm đường đi bên trong block
            block_path = generate_path_inside_block(block, entry_idx)
            path_final.extend(block_path)

    return path_final, [], [], best_cost