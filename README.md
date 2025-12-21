# HCMUT_AdvancedAlgorithm_ACO_CompleteCoveragePath
Complete Coverage Path Using ACO Algorithm

# update by LAN (21 Dec 2025)
# input for aco_algorithm.py\aco_algorithm
# 
# # backend_grid: visible in figure 2 left side
# # # contains objects: INVALID, OBSTACLE, FIELD, HEADLAND, TRANSFER
# # # # INVALID: will be consider as FIELD in init
# # # # OBSTACLE: non coverage area
# # # # FIELD: need coverage area
# # # # HEADLAND: need coverage area
# # # # TRANSFER: for methology to split and merge segment only, from ACO alogirthm, there is not valid any more
# 
# # seg_grid: only container FIELD (FIELD + HEADLAND) AND OBSTACLE
# 
# # merge_field: list of block field area (corner)
# # {
# # ['xmin'], ['xmax'], ['ymin'], ['ymax']
# # ['top_left'], ['top_right'], ['bottom_left'], ['bottom_right']
# # ['visited']
# # } can append
#
# # start_point(x, y)
#
# # start_point_direction = 'top_left' or ...
#
# output for further processing
# 
# # path_final = point to point
# # direct_list = list of start point end point of block (option, can be stub)
# # coverage_list = list of cell coverage
# # cost: for compare
