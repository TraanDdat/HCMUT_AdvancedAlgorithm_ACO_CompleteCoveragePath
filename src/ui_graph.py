"""Map, obstacles, and coverage grid utilities"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.gridspec as mgridspec
from matplotlib.widgets import Button
from aco_algorithm import aco_algorithm

MOUSE_LEFT_CLICK = 1
MOUSE_MIDDLE_CLICK = 2
MOUSE_RIGHT_CLICK = 3
FIGURE_WIDTH_DEFAULT = 8
FIGURE_HEIGHT_DEFAULT = 8
WIDTH_MACHINE = 1.0  # machine width in meters (default value = 1.0, do not change this value)
MIN_WIDTH_OBSTACLE = 0.8  # minimum width of obstacle in frontend grid (default value = 0.5 meter)
CELL_BACKEND_IN_UNIT = 0.5 # 1 cell in backend grid = CELL_BACKEND_IN_UNIT meters 
WIDTH_FRONTEND_IN_UNIT = 0.5 # 1 cell in frontend grid = WIDTH_FRONTEND_IN_UNIT meters
HEIGHT_FRONTEND_IN_UNIT = 1.0 # 1 cell in backend grid = HEIGHT_FRONTEND_IN_UNIT meters (default min value = WIDTH_MACHINE)
WIDTH_SCALE_FRONTEND = int(WIDTH_MACHINE / WIDTH_FRONTEND_IN_UNIT)
WIDTH_SCALE_BACKEND = int(WIDTH_MACHINE / CELL_BACKEND_IN_UNIT)
HEIGHT_SCALE_FRONTEND = int(WIDTH_MACHINE / HEIGHT_FRONTEND_IN_UNIT)
HEIGHT_SCALE_BACKEND = int(WIDTH_MACHINE / CELL_BACKEND_IN_UNIT)

class FieldEnvironment:
    def __init__(self, width, height, grid = None):
        self.org_width = width
        self.org_height = height
        self.grid = grid if grid is not None else np.zeros((int(height*HEIGHT_SCALE_FRONTEND), int(width*WIDTH_SCALE_FRONTEND)), dtype=int)
        self.width = self.grid.shape[1]
        self.height = self.grid.shape[0]
        self.INVALID = 0
        self.OBSTACLE_INDEPENDENT = 1
        self.FIELD = 2
        self.OBSTACLE_SMALL = 3
        self.HEADLAND = 4
        self.TRANSFER = 5
        self.objcolor = {
            self.INVALID: "white",
            self.OBSTACLE_INDEPENDENT: "black",
            self.FIELD: "green",
            self.OBSTACLE_SMALL: "yellow",
            self.HEADLAND: "brown",
            self.TRANSFER: "gray"
            }
        self.objlabel = {
            self.INVALID: "Invalid",
            self.OBSTACLE_INDEPENDENT: "Independent Obstacle",
            self.FIELD: "Field",
            self.OBSTACLE_SMALL: "Small Obstacle",
            self.HEADLAND: "Headland",
            self.TRANSFER: "Transfer"
            }
        self.gridcolor = {
            self.INVALID: "white", 
            self.OBSTACLE_INDEPENDENT: "white", 
            self.FIELD: "#BCEF79", 
            self.HEADLAND: "#BCEF79", 
            self.TRANSFER: "gray"
            }
        self.gridlabel = {
            self.INVALID: "Invalid", 
            self.OBSTACLE_INDEPENDENT: "Obstacle", 
            self.FIELD: "Field", 
            self.HEADLAND: "Headland", 
            self.TRANSFER: "Covered"
            }
        self.funccolor = {
            0: "white", 
            1: "brown",
            2: "black", 
            3: "yellow"
            } # 0-invisited, 1-visited, 2-duplicate, 3-previouspath
        self.left_x = None
        self.left_y = None
        self.left_color = None
        self.fig_1 = None
        self.gs_left = None
        self.gs_1 = None
        self.gs_2 = None
        self.addin_text = None
        self.start_pt = None  # (x, y) frontend
        self.end_pt = None    # (x, y) frontend
        self.input_mode = 'normal' # 'normal', 'set_start', 'set_end'


    def _predata_check(self):
        self.is_obstacle_bypass = False
        if isinstance(MIN_WIDTH_OBSTACLE, float) is False and isinstance(MIN_WIDTH_OBSTACLE, int) is False:
            SystemExit('Error: MIN_WIDTH_OBSTACLE must be a number in meter')
        elif MIN_WIDTH_OBSTACLE < 0:
            SystemExit('Error: MIN_WIDTH_OBSTACLE must be >= 0')
        elif MIN_WIDTH_OBSTACLE >= WIDTH_MACHINE:
            print('Warning: MIN_WIDTH_OBSTACLE >= WIDTH_MACHINE. However, for easy to simulate, pls combine n-cell obstacle to one single obstacle cell in input')
        elif MIN_WIDTH_OBSTACLE >= CELL_BACKEND_IN_UNIT:
            print('Warning: MIN_WIDTH_OBSTACLE is enough to cause an independent obstacle in backend grid. So, assume that obstacle is not too small to bypass the machine')
        elif MIN_WIDTH_OBSTACLE == 0:
            print('Warning: MIN_WIDTH_OBSTACLE is zero, there is no required minimum width for obstacle. So, assume that obstacle is not too small to bypass the machine')
        elif MIN_WIDTH_OBSTACLE < CELL_BACKEND_IN_UNIT and MIN_WIDTH_OBSTACLE > 0:
            print('Info: Machine can by pass the obstacle with MIN_WIDTH_OBSTACLE meter, only consider an independent obstacle if its width >= 0.5 meter (width_machine/2)')
            self.is_obstacle_bypass = True
        else:
            SystemExit('Pls select MIN_WIDTH_OBSTACLE to be one of [0, 0.25, 0.5, 1.0] meter')

    def add_obstacle(self, cell_a, cell_b=None, multiple_grid=False):
        x1, y1 = cell_a

        if multiple_grid:
            if cell_b is None:
                raise ValueError("cell_b required when multiple_grid=True")
            x2, y2 = cell_b
            x_min, x_max = sorted((self.clamp(x1, self.width), self.clamp(x2, self.width)))
            y_min, y_max = sorted((self.clamp(y1, self.height), self.clamp(y2, self.height)))
            self.grid[y_min:y_max+1, x_min:x_max+1] = self.OBSTACLE_INDEPENDENT
        else:
            x = self.clamp(x1, self.width)
            y = self.clamp(y1, self.height)
            self.grid[y, x] = self.OBSTACLE_INDEPENDENT

    def plot_grid(self, grid=None, seg_grid=None, blocks=None, bboxes=None, show=True, scale=True, show_indices=False, path=None):
        """Draw the field on the left axis (`gs_left`) and optional segmentation/blocks on the
        right axis (`self.gs_2`) using a single reusable Figure (`self.fig_1`).

        - `grid`: array to draw on the left (defaults to `self.grid`).
        - `seg_grid`: optional backend/segmentation grid to draw on the right.
        - `blocks`: optional list of block dicts to overlay on the right axis.
        """

        grid_to_draw = grid if grid is not None else self.grid

        # ensure we have a single Figure to reuse and use constrained layout for nicer spacing
        if getattr(self, "fig_1", None) is None or not isinstance(self.fig_1, plt.Figure):
            self.fig_1 = plt.figure(figsize=(FIGURE_WIDTH_DEFAULT, FIGURE_HEIGHT_DEFAULT), constrained_layout=True)
        else:
            # clear existing figure contents but keep same Figure object
            self.fig_1.clf()

        # create fresh gridspec/axes attached to the reused figure using add_gridspec
        gs = self.fig_1.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.08)
        self.gs_1 = self.fig_1.add_subplot(gs[0, 0])
        self.gs_2 = self.fig_1.add_subplot(gs[0, 1])

        # draw grid on the left
        cmap = ListedColormap([self.objcolor[k] for k in sorted(self.objcolor.keys())])
        h, w = grid_to_draw.shape
        # draw with top-left origin so (0,0) maps to top-left cell and ticks align to cells

        self.gs_1.imshow(
            grid_to_draw,
            cmap=cmap,
            vmin=0,
            vmax=len(cmap.colors)-1,
            interpolation="nearest",
            extent=[0, w, h, 0], origin='upper')
        # minor ticks on cell boundaries
        self.gs_1.set_xticks(np.arange(0, w + 1, 1), minor=True)
        self.gs_1.set_yticks(np.arange(0, h + 1, 1), minor=True)
        self.gs_1.grid(which="minor", color="lightgray", linewidth=1)
        
        # redraw path is not None
        if path is not None and len(path) > 0:
            # middle of cell
            xs = [p[0] + 0.5 for p in path]
            ys = [p[1] + 0.5 for p in path]
            self.gs_1.plot(xs, ys, color='red', linewidth=1.5, marker='o', markersize=3, label='ACO Path')
            self.gs_1.legend(loc='upper right', fontsize=6)

        # optionally show axis indices 0..w-1 and 0..h-1
        if show_indices:
            # x indices along bottom
            self.gs_1.set_xticks(np.arange(0.0 + 0.5, w + 0.5, 1.0))
            self.gs_1.set_xticklabels([str(i) for i in range(w)], fontsize=6)
            # y indices top-to-bottom: we invert y to match image origin
            self.gs_1.set_yticks(np.arange(0.0 + 0.5, h + 0.5, 1.0))
            # show labels from 0..h-1 top->bottom
            self.gs_1.set_yticklabels([str(i) for i in range(h)], fontsize=6)
        else:
            self.gs_1.set_xticks([])
            self.gs_1.set_yticks([])
        self.gs_1.set_title("Field landscape", fontsize=6)
        self.gs_1.set_aspect("equal")

        # draw seg_grid/blocks on the right if provided, otherwise show information
        if seg_grid is not None:
            try:
                cmap_blocks = ListedColormap([self.gridcolor[k] for k in sorted(self.gridcolor.keys())])  # INVALID, OBSTACLE_INDEPENDENT, FIELD, HEADLAND, TRANSFER
                # show seg_grid with origin upper so row 0 is top
                h2, w2 = seg_grid.shape
                # avoid singular transforms when width/height are zero
                if w2 <= 0 or h2 <= 0:
                    self.gs_2.clear()
                    self.gs_2.text(0.5, 0.5, "Empty seg_grid", ha='center', va='center', transform=self.gs_2.transAxes)
                else:
                    extent = [0, max(1, w2), max(1, h2), 0]
                    self.gs_2.imshow(seg_grid, cmap=cmap_blocks, interpolation="nearest", extent=extent,vmax=len(cmap_blocks.colors) - 1, vmin=0, origin='upper')
                    # self.gs_2.set_ylim(max(1, h2), 0)
                    self.gs_2.set_aspect("equal")
                self.gs_2.set_title("Seg Grid with Blocks", fontsize=8)
                self.gs_2.set_xticks([])
                self.gs_2.set_yticks([])

            except Exception as _err:
                print("Warning: failed to draw blocks on self.gs_2:", _err)
        else:
            # show textual information on the right axis
            self.gs_2.clear()
            self.gs_2.set_title("Description", fontsize=6)
            self._show_text(ax=self.gs_2, text=None, mode='information',addin=self.addin_text)
            self.gs_2.axis('off')

        # store axes if needed elsewhere
        self.axes = [self.gs_1, self.gs_2]

        # draw/update display
        self.fig_1.canvas.draw_idle()
        if show:
            plt.draw()
            plt.show(block=False)

        # define on motion event (define first, then connect)
        def _on_motion(event):
            if event.inaxes is not self.gs_1 and event.inaxes is not self.gs_2:
                return
            if event.xdata is None or event.ydata is None:
                return

            x = int(event.xdata)
            y = int(event.ydata)

            if 0 <= x < self.width and 0 <= y < self.height:
                # clear old text is existing
                if hasattr(self, "_cursor_text") and self._cursor_text is not None:
                    self._cursor_text.remove()

            # display text
            self._cursor_text = self.gs_1.text(
                event.xdata, event.ydata,
                f"({x},{y})",
                color="blue", fontsize=10,
                ha="center", va="center"
            )
            event.canvas.draw_idle()

        # connect events once (avoid duplicate handlers on repeated redraws)
        if getattr(self, "_motion_cid", None) is None:
            self._motion_cid = self.fig_1.canvas.mpl_connect('motion_notify_event', _on_motion)

    def interactive_grid(self):
        # pre-data check for whether bypass small obstacle
        self._predata_check()

        # user can interactively set obstacles on the grid

        # create subplots
        fig = plt.figure(figsize = (FIGURE_WIDTH_DEFAULT, FIGURE_HEIGHT_DEFAULT))

        # create gridspec
        gs = mgridspec.GridSpec(
            nrows=3, \
            ncols=2, \
            width_ratios=[3, 1], \
            height_ratios=[3, 1, 3])
        self.gs_left = fig.add_subplot(gs[:, 0]) # all left side
        self.gs_right_top = fig.add_subplot(gs[0, 1]) # top right side
        tmp_gs_right_middle = fig.add_subplot(gs[1, 1]) # middle right side
        gs_right_bottom = fig.add_subplot(gs[2, 1]) # bottom right side

        # from gs_right_middle, break down to create button area
        tmp_gs_right_middle.axis('off')
        tmp_gs_right_middle = mgridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1, 1], wspace=0.1, hspace=0.1)
        gs_bt_reset = fig.add_subplot(tmp_gs_right_middle[0, 0])
        gs_bt_start = fig.add_subplot(tmp_gs_right_middle[0, 1])
        gs_bt_set_s = fig.add_subplot(tmp_gs_right_middle[1, 0])
        gs_bt_set_e = fig.add_subplot(tmp_gs_right_middle[1, 1])

        # create colormap
        cmap = ListedColormap([self.objcolor[k] for k in sorted(self.objcolor.keys())])

        # create grid show in the left and description show in the top right
        self._draw_colored_grid(ax_grid=self.gs_left,ax_desc=self.gs_right_top,mode='information')

        # create button
        # # to reset grid
        btn_reset = self.create_button_reset_grid(
            ax_button=gs_bt_reset,
            ax_grid=self.gs_left,
            ax_desc=self.gs_right_top)
        # # to start planner
        btn_start = self.create_button_start_planner(
            ax_button=gs_bt_start)
        
        # Set start point
        btn_set_s = Button(gs_bt_set_s, 'Set\nStart\Point', color='lightblue', hovercolor='blue')
        def _set_start_mode(event):
            self.input_mode = 'set_start'
            print("Mode: Click a cell to set START point")
        btn_set_s.on_clicked(_set_start_mode)

        # Reserved button
        btn_set_e = Button(gs_bt_set_e, 'Unuse', color='lightpink', hovercolor='red')
        def _set_end_mode(event):
            self.input_mode = 'set_end'
            print("Mode: Unuse button, user can modify here")
        btn_set_e.on_clicked(_set_end_mode)

        # create table show in the bottom right
        ax_obj_table = gs_right_bottom.table(\
        cellText=[[self.objlabel[k]] for k in sorted(self.objcolor.keys())], \
        rowLabels=[self.objcolor[k] for k in sorted(self.objcolor.keys())], \
        loc='right', \
        cellLoc='right')
        ax_obj_table.auto_set_font_size(False) 
        ax_obj_table.auto_set_column_width(col=list(range(len(self.objcolor))))
        ax_obj_table.set_fontsize(8) 
        ax_obj_table.scale(1, 1)
        gs_right_bottom.set_title("Legend", fontsize=6)
        gs_right_bottom.axis('off')

        # define onclick event
        def _onclick_cell(event):
            if event.inaxes is not self.gs_left:
                return
            if event.xdata is None or event.ydata is None:
                return

            cell_width = HEIGHT_SCALE_FRONTEND
            cell_height = WIDTH_SCALE_FRONTEND

            x = int(event.xdata // cell_width)
            y = int(event.ydata // cell_height)
			
			# after predata check, if min width of obstacle is too big to bypass by machine => assume that obstacle is an independent obstacle. 
			# Else, users can define small obstacle then it will bypass by program
			# # factor = 4 (INVALID, OBSTACLE_INDEPENDENT, FIELD, OBSTACLE_SMALL)
			# # factor = 3 (INVALID, OBSTACLE_INDEPENDENT, FIELD)
            factor = 4 if self.is_obstacle_bypass else 3

            if 0 <= x < self.width and 0 <= y < self.height:

                if self.input_mode == 'set_start':
                    self.start_pt = (x, y)
                    self.input_mode = 'normal' # Quay về bình thường sau khi chọn
                    self._draw_colored_grid(ax_grid=self.gs_left, ax_desc=self.gs_right_top, mode='information')
                    return # Kết thúc, không đổi màu ô

                if self.input_mode == 'set_end':
                    self.end_pt = (x, y)
                    self.input_mode = 'normal'
                    # self._draw_colored_grid(ax_grid=self.gs_left, ax_desc=self.gs_right_top, mode='information')
                    return
					
                # if left click, set cell color
                if event.button == MOUSE_LEFT_CLICK:
                    self.grid[y, x] = (self.grid[y, x] + 1) % factor
                    self._draw_colored_grid(ax_grid=self.gs_left, ax_desc=self.gs_right_top, mode='information')
                    self.left_x, self.left_y = x, y
                    self.left_color = self.grid[y, x] # color value after left click

                # if right click, fill rectangle area from previous left click
                elif event.button == MOUSE_RIGHT_CLICK:
                            if self.left_x is None or self.left_y is None or self.left_color is None:
                                return
                            x_min, x_max = sorted((self.clamp(x, self.width), self.clamp(self.left_x, self.width)))
                            y_min, y_max = sorted((self.clamp(y, self.height), self.clamp(self.left_y, self.height)))
                            self.grid[y_min:y_max+1, x_min:x_max+1] = self.left_color
                            self._draw_colored_grid(ax_grid=self.gs_left, ax_desc=self.gs_right_top, mode='information')
                event.canvas.draw_idle()

        # define on motion event
        def _on_motion(event):
            if event.inaxes is not self.gs_left:
                return
            if event.xdata is None or event.ydata is None:
                return

            cell_width = HEIGHT_SCALE_FRONTEND
            cell_height = WIDTH_SCALE_FRONTEND

            x_f = int(event.xdata // cell_width)
            y_f = int(event.ydata // cell_height)

            if 0 <= x_f < self.width and 0 <= y_f < self.height:
                # clear old text is existing
                if hasattr(self, "_cursor_text") and self._cursor_text is not None:
                    self._cursor_text.remove()

            # display text
            self._cursor_text = self.gs_left.text(
                event.xdata, event.ydata,
                f"({x_f},{y_f})",
                color="blue", fontsize=10,
                ha="center", va="center"
            )

            # redraw idle
            event.canvas.draw_idle()

        # connect events
        fig.canvas.mpl_connect('button_press_event', _onclick_cell)
        fig.canvas.mpl_connect('motion_notify_event', _on_motion)

        # show plot
        plt.tight_layout()
        plt.show()


    def _draw_colored_grid(self, ax_grid, ax_desc, mode='information'):
        # create colormap
        cmap = ListedColormap([self.objcolor[k] for k in sorted(self.objcolor.keys())])

        # clear and redraw grid
        ax_grid.clear()
        ax_desc.clear()

        # create grid image
        h, w = self.grid.shape
        # FIX: Sửa extent để h nằm dưới, 0 nằm trên (khớp với axes)
        ax_grid.imshow(self.grid, cmap=cmap, vmin=0, vmax=len(cmap.colors)-1,interpolation="nearest",extent=[0, w*HEIGHT_SCALE_FRONTEND, h*WIDTH_SCALE_FRONTEND, 0], origin='upper')

        # create grid
        ax_grid.set_xticks(np.arange(0, w*HEIGHT_SCALE_FRONTEND, HEIGHT_SCALE_FRONTEND), minor=True)
        ax_grid.set_yticks(np.arange(0, h*WIDTH_SCALE_FRONTEND, WIDTH_SCALE_FRONTEND), minor=True)
        ax_grid.grid(which="minor", color="lightgray", linewidth=1)
        ax_grid.set_xticks([])
        ax_grid.set_yticks([])
        ax_grid.set_aspect("equal")
        # ax_grid.set_ylim(h*WIDTH_SCALE_FRONTEND, 0)  # set origin to top-left

        # Redraw start point
        if self.start_pt:
            sx, sy = self.start_pt
            # middle of cell
            ax_grid.text(sx * HEIGHT_SCALE_FRONTEND + 0.5, sy * WIDTH_SCALE_FRONTEND + 0.5, 
                         'S', ha='center', va='center', color='blue', fontweight='bold', fontsize=12)
        
        # Unuse, modify later
        # if self.end_pt:
        #    ex, ey = self.end_pt
        #    ax_grid.text(ex * HEIGHT_SCALE_FRONTEND + 0.5, ey * WIDTH_SCALE_FRONTEND + 0.5, 
        #                 'E', ha='center', va='center', color='red', fontweight='bold', fontsize=12)
        # ---------------------

        # create description show in the top right
        ax_desc.set_title("Description", fontsize=6)
        self._show_text(\
            ax=ax_desc, \
            text=None, \
            mode='information', \
            addin=self.addin_text \
            )
        ax_desc.axis('off')

        # redraw plot 
        plt.draw()

    def _show_text(self, ax, text, position=(0, 0.5), font_size=8, halign='left', valign='center', wrap=True, mode='information',addin=None):
        # show information of field input grid
        if mode == 'information':
            text_infor = [\
                f"Field size: {self.org_width} (m) x {self.org_height} (m)", \
                f"Machine width: {WIDTH_MACHINE} (m)", \
                f"Cell scale(frontend): 1 cell = {WIDTH_MACHINE / WIDTH_SCALE_FRONTEND} (m)", \
                f"Column of field grid: {self.width} (pixels)", \
                f"Row of field grid: {self.height} (pixels)", \
                f"There is {np.sum(self.grid == self.OBSTACLE_INDEPENDENT)} obstacle object", \
                f"There is {np.sum(self.grid == self.FIELD)} field object", \
                f"There is {np.sum(self.grid == self.HEADLAND)} headland object", \
                f" Additional info: {addin}" if addin is not None else ""
                ]
            text_to_show = "\n".join(text_infor)
        
        ax.text(position[0], position[1], text_to_show, fontsize=8,
            ha=halign, va=valign, wrap=wrap, transform=ax.transAxes)

    def shape(self):
        return self.grid.shape

    def is_obstacle(self, x, y):
        return self.grid[y, x] == self.OBSTACLE_INDEPENDENT

    # create button to reset grid to Invalid grid
    def create_button_reset_grid(self, ax_button, ax_grid, ax_desc):
        # create button
        btn = Button(ax_button, f'Reset\nGrid', color='lightgray', hovercolor='gray')

        # callback event for button
        def _reset_grid(event):
            self.grid[:, :] = self.INVALID
            self.start_pt = None
            self.end_pt = None
            self._draw_colored_grid(ax_grid=ax_grid, ax_desc=ax_desc, mode='information')

        btn.on_clicked(_reset_grid)
        return btn

    # create button to start planner
    def create_button_start_planner(self, ax_button):

        btn = Button(ax_button, f'Start\nPlanner', color='red', hovercolor='gray')

        def _start_program(event):
            # print("Start planner button clicked\nIf cell is INVALID, it will be automaticlly set to FIELD and headland will be calculated")
            self._free_to_field()
            self._draw_colored_grid(ax_grid=self.gs_left, ax_desc=self.gs_right_top, mode='information')
			
            # Scale factor from Frontend to Backend
            scale_x = int(WIDTH_SCALE_BACKEND / WIDTH_SCALE_FRONTEND)
            scale_y = int(HEIGHT_SCALE_BACKEND / HEIGHT_SCALE_FRONTEND)
            
            # Init default
            algo_start = None
            algo_end = None

            if self.start_pt:
                # backend = frontend * scale
                # safe check middle of cell to avoid boundary threshold
                # (x*scale, y*scale)
                s_x = int(self.start_pt[0] * scale_x)
                s_y = int(self.start_pt[1] * scale_y)
                
                # boundary safe check
                s_x = min(s_x, self.width * scale_x - 1)
                s_y = min(s_y, self.height * scale_y - 1)
                
                algo_start = (s_x, s_y)
            
            if self.end_pt:
                e_x = int(self.end_pt[0] * scale_x)
                e_y = int(self.end_pt[1] * scale_y)
                
                # boundary safe check
                e_x = min(e_x, self.width * scale_x - 1)
                e_y = min(e_y, self.height * scale_y - 1)
                
                algo_end = (e_x, e_y)
            
            print(f"User selected inputs -> Start: {algo_start}, End: {algo_end}")

			# call program to execute
            processed_backend, seg_grid, blocks, bboxes, path_final, valid = compute_headland(
                self.grid, hl_pass=1, user_start=algo_start, user_end=algo_end, is_obstacle_bypass=False
            )
            # if invalid, show original grid (explicit)
            grid_to_show = processed_backend if valid else self.grid

            # reuse existing backend env/figure if open
            prev_env = getattr(self, "_backend_env", None)
            if prev_env is not None:
                prev_fig = getattr(prev_env, "fig_1", None)
                if prev_fig is not None and plt.fignum_exists(prev_fig.number):
                    # redraw the chosen grid into the existing backend figure (show both left and right)
                    prev_env.plot_grid(grid=grid_to_show, seg_grid=seg_grid, blocks=blocks, bboxes=bboxes, show=True, path=path_final)
                    self._backend_env = prev_env
                    return

            # otherwise create a new environment and open a new figure showing both self.gs_1 and self.gs_2
            backend_env = FieldEnvironment(width=grid_to_show.shape[1],
                                           height=grid_to_show.shape[0],
                                           grid=grid_to_show)
            backend_env.plot_grid(grid=grid_to_show, seg_grid=seg_grid, blocks=blocks, bboxes=bboxes, show=True, scale=True, path=path_final)
            self._backend_env = backend_env

        btn.on_clicked(_start_program)
        return btn

    # helper function: clamp index to avoid out of bound
    def clamp(self, ix, maxv):
        return max(0, min(int(ix), maxv - 1))
    
    def _free_to_field(self):
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[y, x] == self.INVALID:
                    self.grid[y, x] = self.FIELD

def compute_headland(grid, hl_pass, user_start=None, user_end=None, is_obstacle_bypass=False):
    """Compute headland on a duplicated-column backend grid.
    ----------------------> horizontal (x)
    |
    |
    |
    |
    |
    |
    v
    vertical (y)
    """
	# define object-color reference marco
    INVALID = 0
    OBSTACLE_INDEPENDENT = 1
    FIELD = 2
    OBSTACLE_SMALL = 3
    HEADLAND = 4
    TRANSFER = 5
	# define grid valid status marco
    GRID_VALID = 0
    GRID_VALID_MIN = 1
    GRID_INVALID = 2

    # duplicate grid in backend for processing (preserve original)
    # expand frontend grid to backend resolution (repeat rows and cols by integer scale factors)
    scale_y = HEIGHT_SCALE_BACKEND / float(HEIGHT_SCALE_FRONTEND)
    scale_x = WIDTH_SCALE_BACKEND / float(WIDTH_SCALE_FRONTEND)
    if not scale_x.is_integer() or not scale_y.is_integer():
        raise ValueError(
            f"Incompatible frontend/backend scales: \n"
            f"WIDTH_SCALE_BACKEND={WIDTH_SCALE_BACKEND}, WIDTH_SCALE_FRONTEND={WIDTH_SCALE_FRONTEND}, \n"
            f"HEIGHT_SCALE_BACKEND={HEIGHT_SCALE_BACKEND}, HEIGHT_SCALE_FRONTEND={HEIGHT_SCALE_FRONTEND}. \n"
            "Require backend scale to be an integer multiple of frontend scale."
        )
    repeat_rows = int(scale_y)
    repeat_cols = int(scale_x)
    # safe repeat (repeat_rows/repeat_cols >= 1)
    repeat_rows = max(1, repeat_rows)
    repeat_cols = max(1, repeat_cols)
    backend_grid = np.repeat(np.repeat(grid, repeat_rows, axis=0), repeat_cols, axis=1)

    h_backend, w_backend = backend_grid.shape
    print(f"h_backend = {h_backend}, w_backend = {w_backend}")
    hl_pass_in_backend = hl_pass * WIDTH_SCALE_FRONTEND
    min_hl2obstacle = hl_pass_in_backend + 2 # minimum headland to obstacle distance in backend cell unit
    visited = np.zeros((h_backend, w_backend), dtype=bool)
    # define backend table for headland calculation
    # # table_1 
    # # # check object_1, headland_pass*WIDTH_SCALE_FRONTEND, object_2
    # # # directory: (object_1, object_2) -> act_1
    table_1 = {
        (FIELD, FIELD) : (0xFF),
        (FIELD, OBSTACLE_INDEPENDENT) : (0xF0),
        (OBSTACLE_INDEPENDENT, FIELD) : (0x0F),
        (OBSTACLE_INDEPENDENT, OBSTACLE_INDEPENDENT) : (0x00)
    }
    # # table_2 (top to bottom)
    # # # check (object_1, object_2 => return value from table_1), headland_pass*WIDTH_SCALE_FRONTEND
    # # # directory: (action_value_1, if_all_field_in_headland_pass) -> final_action_value
    table_2 = {
        (0xFF, True) : [TRANSFER, HEADLAND] + [HEADLAND for _ in range(hl_pass_in_backend - 1)] + [FIELD] + [FIELD, FIELD],
        (0xF0, True)  : [TRANSFER, HEADLAND] + [HEADLAND for _ in range(hl_pass_in_backend - 1)] + [TRANSFER] + [OBSTACLE_INDEPENDENT, OBSTACLE_INDEPENDENT],
        (0x0F, True)  : [OBSTACLE_INDEPENDENT, OBSTACLE_INDEPENDENT, TRANSFER] + [HEADLAND for _ in range(hl_pass_in_backend)] + [FIELD],
        (0x00, True) : [OBSTACLE_INDEPENDENT, OBSTACLE_INDEPENDENT] + [INVALID for _ in range(hl_pass_in_backend )] + [OBSTACLE_INDEPENDENT, OBSTACLE_INDEPENDENT],
        (0xFF, False) : [INVALID, INVALID] + [INVALID for _ in range(hl_pass_in_backend )] + [FIELD, FIELD],
        (0xF0, False) : [INVALID, INVALID] + [INVALID for _ in range(hl_pass_in_backend)] + [OBSTACLE_INDEPENDENT, OBSTACLE_INDEPENDENT],
        (0x0F, False) : [OBSTACLE_INDEPENDENT, OBSTACLE_INDEPENDENT] + [INVALID for _ in range(hl_pass_in_backend)] + [FIELD, FIELD],
        (0x00, False) : [OBSTACLE_INDEPENDENT, OBSTACLE_INDEPENDENT] + [INVALID for _ in range(hl_pass_in_backend)] + [OBSTACLE_INDEPENDENT, OBSTACLE_INDEPENDENT]
    }
    # # table_3 (bottom to top)
    # # # check (object_1, object_2 => return value from table_1), headland_pass*WIDTH_SCALE_FRONTEND
    # # # directory: (action_value_1, if_all_field_in_headland_pass) -> final_action_value
    table_3 = {
        (0xFF, True) : [FIELD, FIELD] + [FIELD for _ in range(hl_pass_in_backend)] + [FIELD, FIELD],
        (0xF0, True)  : [FIELD] + [HEADLAND for _ in range(hl_pass_in_backend)] + [TRANSFER] + [OBSTACLE_INDEPENDENT, OBSTACLE_INDEPENDENT],
        (0x0F, True)  : [OBSTACLE_INDEPENDENT, OBSTACLE_INDEPENDENT] + [TRANSFER] + [HEADLAND for _ in range(hl_pass_in_backend)] + [FIELD],
        (0x00, True) : [OBSTACLE_INDEPENDENT, OBSTACLE_INDEPENDENT] + [INVALID, INVALID] + [OBSTACLE_INDEPENDENT, OBSTACLE_INDEPENDENT],
        (0xFF, False) : [INVALID, INVALID] + [INVALID for _ in range(hl_pass_in_backend )] + [FIELD, FIELD],
        (0xF0, False) : [INVALID, INVALID] + [INVALID for _ in range(hl_pass_in_backend)] + [OBSTACLE_INDEPENDENT, OBSTACLE_INDEPENDENT],
        (0x0F, False) : [OBSTACLE_INDEPENDENT, OBSTACLE_INDEPENDENT] + [INVALID for _ in range(hl_pass_in_backend)] + [FIELD, FIELD],
        (0x00, False) : [OBSTACLE_INDEPENDENT, OBSTACLE_INDEPENDENT] + [INVALID for _ in range(hl_pass_in_backend)] + [OBSTACLE_INDEPENDENT, OBSTACLE_INDEPENDENT]
    }
    # if grid has range is minimum field height valid
    table_1_min = {
        (True) : [TRANSFER] + [HEADLAND for _ in range(hl_pass_in_backend)] + [TRANSFER],
        (False) : [INVALID, INVALID] + [INVALID for _ in range(hl_pass_in_backend)]
    }

    def _calib_cell_length():
        w = int(WIDTH_MACHINE * WIDTH_SCALE_FRONTEND) # = 2
        return w

    def _is_neighbor_field(x, y, grid=None):
        g = backend_grid if grid is None else grid
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w_backend and 0 <= ny < h_backend:
                    if g[ny, nx] == FIELD:
                        return True
        return False
    
    def find_neighbors_hortical(grid, point, object, extension_value=1):
        if not isinstance(grid, np.ndarray):
            SystemError('grid must be numpy array')
        if object is None:
            SystemError('object must be defined')
        if point is None:
            SystemError('point must be defined')
        if extension_value < 1 or isinstance(extension_value, int) is False:
            SystemError('extension_value must be >= 1')

        rows, cols = grid.shape
        r, c = point
        neighbors = []
        
        # Check left
        for offset in range(1, extension_value + 1):
            if c - offset >= 0 and grid[r, c - offset] == object:
                neighbors.append((r, c - offset))
            else:
                pass
        
        # Check right
        for offset in range(1, extension_value + 1):
            if c + offset < cols and grid[r, c + offset] == object:
                neighbors.append((r, c + offset))
            else:
                pass
        
        return neighbors

    def _is_neighbor_obstacle_v(y): # vertical check
        y = y + 1 if y%2 == 0 else y # offset to the duplicate cell in vertical side
        for y in (0, 1, 2, 3):
            if grid[y, x] == OBSTACLE_INDEPENDENT:
                return True

        return False
    
    def _is_neighbor_obstacle_h(x): # horizontal check
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w_backend and 0 <= ny < h_backend:
                    if backend_grid[ny, nx] == OBSTACLE_INDEPENDENT:
                        return True
        return False

    def _is_obstacle_independent(x, y):
        for x in range(w_backend): # column
            for y in range(h_backend): # row => can change to marco later
                if backend_grid[y, x] == OBSTACLE_INDEPENDENT:
                    if _is_neighbor_field(x, y):
                        for hp in range(hl_pass_in_backend * 2):
                            ty = y - 1 - hp
                            if ty >= 0:
                                backend_grid[ty, x] = HEADLAND
                        if(grid[ty+1,x]) == FIELD:
                            grid[ty+1,x] = TRANSFER

    def _is_field_height_valid(col):
        # operate on backend column (already duplicated)
        # return np.sum(col == FIELD) >= hl_pass_in_backend * 2 + 2
        val = int(np.sum(col != INVALID))
        threshold = int(hl_pass_in_backend + 2)

        if val > threshold:
            return GRID_VALID
        elif val == threshold:
            return GRID_VALID_MIN
        else:
            return GRID_INVALID

    def _is_all_field_in_headland_pass(x, y_start, direction, grid=None):
        g = backend_grid if grid is None else grid
        # check all field in headland pass area
        for hp in range(hl_pass_in_backend):
            ty = y_start + hp if direction == 'down' else y_start - hp
            if ty < 0 or ty >= h_backend:
                return False
            if g[ty, x] != FIELD:
                return False
        return True

    def _merge_obstacle(grid, obs_y):
        """
        Merge obstacles in rows specified by obs_y if horizontal distance <= 2 backend cells
        Args:
            grid: np.ndarray, backend grid
            obs_y: int, list of int, or slice
        Returns:
            grid after merging obstacles
        """

        # convert obs_y to list of rows
        if isinstance(obs_y, slice):
            rows = list(range(grid.shape[0]))[obs_y]
        elif isinstance(obs_y, int):
            rows = [obs_y]
        else:
            rows = obs_y  # list or array

        for y in rows:
            row = grid[y, :]
            obstacle_indices = np.where(row == OBSTACLE_INDEPENDENT)[0]

            if len(obstacle_indices) < 2:
                continue

            for i in range(len(obstacle_indices)-1):
                x1 = obstacle_indices[i]
                x2 = obstacle_indices[i+1]

                # merge if distance <= 2
                if x2 - x1 <= 2:
                    # fill all intermediate cells
                    row[x1+1:x2] = OBSTACLE_INDEPENDENT

            grid[y, :] = row

        return grid

    def _cal_top_to_bottom(x, y, grid, target_table):
        y_offset = y + 1 if y%2 == 0 else y # offset to the duplicate cell in vertical side
        y_start_hlpass = y_offset + 1
        is_last_obstacle = False
        if grid[y_start_hlpass + hl_pass_in_backend - 1,x] == OBSTACLE_INDEPENDENT:
            is_last_obstacle = True

        # boundary check
        if y_start_hlpass - hl_pass_in_backend < 0 or y_start_hlpass + hl_pass_in_backend >= grid.shape[0]:
            return []
        
        obj_1, obj_2 = grid[y_offset, x], grid[y_start_hlpass+hl_pass_in_backend, x]

        if (obj_1, obj_2) not in table_1:
            return []
        
        act_1 = table_1[(obj_1, obj_2)]
        all_field = _is_all_field_in_headland_pass(x, y_start_hlpass, 'down', grid)

        if (act_1, all_field) not in target_table:
            return []
        
        final_act = target_table[(act_1, all_field)]

        if is_last_obstacle:
            final_act[hl_pass_in_backend] = OBSTACLE_INDEPENDENT
            final_act[hl_pass_in_backend + 1] = OBSTACLE_INDEPENDENT
        return final_act

    def _cal_bottom_to_top(x, y, grid, target_table):
        """
        Compute headland pattern from bottom boundary upward.
        final_act is a LOCAL WINDOW, indices must be RELATIVE.
        """

        # offset for duplicated vertical cell (mirror of top logic)
        y_offset = y - 1 if y % 2 == 1 else y
        y_start_hlpass = y_offset - 1
        
        # boundary check
        if y_start_hlpass - hl_pass_in_backend < 0 or y_start_hlpass + hl_pass_in_backend >= grid.shape[0]:
            return []

        # check obstacle just ABOVE the headland pass
        check_idx = y_start_hlpass - (hl_pass_in_backend - 1)
        is_last_obstacle = (0 <= check_idx < grid.shape[0] and
                            grid[check_idx, x] == OBSTACLE_INDEPENDENT)

        # IMPORTANT:
        # table_1 is defined as (obj_1, obj_2) = (before HL, after HL)
        # bottom-to-top => swap semantic roles
        obj_2 = grid[y_start_hlpass - hl_pass_in_backend, x]  # before headland
        obj_1 = grid[y_offset, x]                              # boundary side

        if (obj_1, obj_2) not in table_1:
            return []

        act_1 = table_1[(obj_1, obj_2)]

        # check full-field in headland pass (direction UP)
        all_field = _is_all_field_in_headland_pass(x, y_start_hlpass, 'up', grid)

        if (act_1, all_field) not in target_table:
            return []

        # table_2 pattern is TOP to BOTTOM, must reverse for bottom to top
        final_act = list(reversed(target_table[(act_1, all_field)]))
        # FIXED: write obstacle using RELATIVE indices only
        # obstacle always sits at the end of headland window
        if is_last_obstacle:
            final_act[hl_pass_in_backend] = OBSTACLE_INDEPENDENT
            final_act[hl_pass_in_backend + 1] = OBSTACLE_INDEPENDENT

        return final_act

    def _get_value_segment_from_xy(sgm,vx,vy):
        y_s, y_e = None, None
        for s in sgm:
            if s['visited'] is False:
                if s['x'] == vx:
                    if s['y_start'] <= vy <= s['y_end']:
                        y_s, y_e = s['y_start'], s['y_end']
                        s['visited'] = True
                        print(f"Found segment at x={vx}, y_start={y_s}, y_end={y_e}, visited={s['visited']}")
                        break
        return y_s, y_e

    def merge_right_to_current(segment, grid, segments):
            # create a recursive function to merge right segments into current segment
            cur_x = segment['x']
            cur_y_start = segment['y_start']
            cur_y_end = segment['y_end']
            # init block
            block = {
                'top_left': (cur_x, cur_y_start),
                'top_right': (cur_x, cur_y_start),
                'bottom_left': (cur_x, cur_y_end),
                'bottom_right': (cur_x, cur_y_end),
                'list_xy': [(cur_x, y) for y in range(cur_y_start, cur_y_end+1)],
                'list_seg_id': [segment.get('seg_id', None)]
                }
            # iterate through all y in current segment
            for idy in range(cur_y_start, cur_y_end+1):
                # check right neighbor
                if cur_x+1 < grid.shape[1] and grid[idy, cur_x+1] in (OBSTACLE_INDEPENDENT, TRANSFER, INVALID) and grid[idy, cur_x] in (OBSTACLE_INDEPENDENT, TRANSFER, INVALID):
                # lookup using the full segments list
                    y_s_right, y_e_right = _get_value_segment_from_xy(segments, cur_x+1, idy)
                    if y_s_right is not None and y_e_right is not None:
                        # update block corners
                        block['top_right'] = (cur_x+1, y_s_right)
                        block['bottom_right'] = (cur_x+1, y_e_right)
                        # update current segment to include right segment
                        segment['y_start'] = min(segment['y_start'], y_s_right)
                        segment['y_end'] = max(segment['y_end'], y_e_right)
                        cur_y_start, cur_y_end = segment['y_start'], segment['y_end']
                        idy = cur_y_start - 1 # reset idy to start of new segment
                        cur_x += 1 # move to next column
                        # recursive call to merge further right segments
                        merge_right_to_current(segment, grid, segments)
            return block

    def rectangles_from_grid_vertical(grid,object = 'obstacle'):
        h, w = grid.shape
        merged = []

        prev_col_rects = []

        if object == 'obstacle':
            obj = (OBSTACLE_INDEPENDENT,INVALID,TRANSFER)
        elif object == 'field':
            obj = (FIELD,HEADLAND)
        else:
            pass

        for x in range(w):
            y = 0
            curr_col_rects = []
            while y < h:
                if grid[y, x] in obj:
                    y_start = y
                    while y < h and grid[y, x] in obj:
                        y += 1
                    y_end = y - 1
                    curr_col_rects.append({'xmin': x, 'xmax': x, 'ymin': y_start, 'ymax': y_end})
                else:
                    y += 1

            # merge with prev_col_rects
            new_prev = []
            used_prev = set()
            for r in curr_col_rects:
                merged_flag = False
                for i, p in enumerate(prev_col_rects):
                    if p['ymin'] == r['ymin'] and p['ymax'] == r['ymax']:
                        # merge with X extension
                        p['xmax'] = r['xmax']
                        new_prev.append(p)
                        used_prev.add(i)
                        merged_flag = True
                        break
                if not merged_flag:
                    new_prev.append(r)
            # remaining rectangles in prev_col_rects that were not merged
            for i, p in enumerate(prev_col_rects):
                if i not in used_prev:
                    merged.append(p)
            prev_col_rects = new_prev

        # push remaining
        merged.extend(prev_col_rects)

        return merged

    def path_inside_block(block,entry_point_dicrect='top_left', entry_point_coord=None):
        """ a block contains 4 corners: 
        top_left (xmin, ymin)
        top_right (xmax, ymin)
        bottom_left (xmin, ymax)
        bottom_right (xmax, ymax)
        (A) ################## (B)
            ##################
            ##################
            ##################
            ##################
        (C) ################## (D)
        (A) : top_left, (B): top_right, (C): bottom_left, (D): bottom_right
        return a list of (x,y) path that cover all field/ headland inside block
        """
        path = []
        exit_point = (None,None)
        exit_point_direction = None

        if block[entry_point_dicrect][0] != entry_point_coord[0] or block[entry_point_dicrect][1] != entry_point_coord[1]:
            raise ValueError("Entry point coordinate does not match the specified entry point direction.")
            return path, None  # invalid entry point

        # extract block corners
        xmin = block['top_left'][0]
        xmax = block['top_right'][0]
        ymin = block['top_left'][1]
        ymax = block['bottom_left'][1]
        
        # init direction
        slice = None

        if xmax < xmin or ymax < ymin:
            return path, None  # invalid block

        # init start point
        if entry_point_dicrect == 'top_left':
            start_x, start_y = xmin, ymin
            end_x = xmax
            end_y = ymax
            slice = 'ltr_down'
            exit_point, exit_point_direction = (xmax, ymax), 'bottom_right' if (xmax - xmin) %2 ==0 else (xmax, ymin), 'top_right'
        elif entry_point_dicrect == 'top_right':
            start_x, start_y = xmax, ymin
            end_x = xmin
            end_y = ymax
            slice = 'rtl_down'
            exit_point, exit_point_direction = (xmin, ymax), 'bottom_left' if (xmax - xmin) %2 ==0 else (xmin, ymin), 'top_left'
        elif entry_point_dicrect == 'bottom_left':
            start_x, start_y = xmin, ymax
            end_x = xmax
            end_y = ymin
            slice = 'ltr_up'
            exit_point, exit_point_direction = (xmax, ymin), 'bottom_right' if (xmax - xmin) %2 ==0 else (xmax, ymax), 'top_right'
        elif entry_point_dicrect == 'bottom_right':
            start_x, start_y = xmax, ymax
            end_x = xmin
            end_y = ymin
            slice = 'rtl_up'
            exit_point, exit_point_direction = (xmin, ymin), 'bottom_left' if (xmax - xmin) %2 ==0 else (xmin, ymax), 'top_left'
        else:
            return path, None  # invalid entry point

        if slice == 'ltr_down':
            for x_ltr_d in range(start_x, end_x+1):
                if (x_ltr_d - xmin) % 2 == 0:
                    path.extend([(x_ltr_d, y) for y in range(start_y, end_y+1)])
                else:
                    path.extend([(x_ltr_d, y) for y in reversed(range(start_y, end_y+1))])
        elif slice == 'rtl_down':
            for x_rtl_d in range(start_x, end_x-1, -1):
                if (x_rtl_d - xmin) % 2 == 0:
                    path.extend([(x_rtl_d, y) for y in range(start_y, end_y+1)])
                else:
                    path.extend([(x_rtl_d, y) for y in reversed(range(start_y, end_y+1))])
        elif slice == 'ltr_up':
            for x_ltr_u in range(start_x, end_x+1):
                if (x_ltr_u - xmin) % 2 == 0:
                    path.extend([(x_ltr_u, y) for y in reversed(range(end_y, start_y+1))])
                else:
                    path.extend([(x_ltr_u, y) for y in range(end_y, start_y+1)])
        elif slice == 'rtl_up':
            for x_rtl_u in range(start_x, end_x-1, -1):
                if (x_rtl_u - xmin) % 2 == 0:
                    path.extend([(x_rtl_u, y) for y in reversed(range(end_y, start_y+1))])
                else:
                    path.extend([(x_rtl_u, y) for y in range(end_y, start_y+1)])
        else:
            return path, None  # invalid slice

        return path, exit_point, exit_point_direction
        
    """ helper functions for main program """
    # debug prints (optional)
    def _print_grid(g):
        for row in g:
            print(" ".join(str(x) for x in row))
            pass
        print("==============================")

    # replace all 'a' in grid to 'b'
    def _from_a_to_b(grid,a,b):
        for x in range(w_backend):
            for y in range(h_backend):
                if grid[y,x] == a:
                    grid[y,x] = b
        return grid

	# replace a single cell from a to b
    def _from_a_to_b_single_cell(grid, x, y, a, b):
        if grid[y,x] == a:
            grid[y,x] = b
        return grid

    """ ================================= main program of calculating headland ================================= """
    # debugging prints
    # _print_grid(backend_grid)

    # stub for further processing
    w_machine_backend = _calib_cell_length() # calculate that width of machine in backend cell unit
    
    # merge obstacle that are close to each other horizontally
    backend_grid = _merge_obstacle(backend_grid, obs_y=slice(0, h_backend))
	
	# treat small obstacle as field for headland calculation
    backend_grid = _from_a_to_b(backend_grid, OBSTACLE_SMALL, FIELD)  

	# from front-end to back-end view
	# back-end: 1 cell = machine width for further path planing
    for mx in range(w_backend):
        for my in range(h_backend):
            if mx % 2 == 0 and (mx + 1 < w_backend): # even column
                if backend_grid[my, mx] == OBSTACLE_INDEPENDENT and backend_grid[my, mx + 1] == FIELD:
                    backend_grid[my, mx + 1] = OBSTACLE_INDEPENDENT
            elif mx % 2 == 1 and (mx - 1 >= 0): # odd column
                if backend_grid[my, mx] == OBSTACLE_INDEPENDENT and backend_grid[my, mx - 1] == FIELD:
                    backend_grid[my, mx - 1] = OBSTACLE_INDEPENDENT

    # calculate headland column by column
    for x in range(w_backend):
        backup_grid = backend_grid.copy()
        grid_state = _is_field_height_valid(backend_grid[:, x])
		# invalid grid
        if grid_state == GRID_INVALID:
            return []
		# valid grid, process calculate headland
		# # top to bottom
		# # bottom to top
		# # top to bottom again to avoid mismatch transfer object
        elif grid_state == GRID_VALID:
            # from top to bottom
            top_hl = _cal_top_to_bottom(x, 0,backend_grid, table_2)
            if top_hl:
                for y in range(len(top_hl)):
                    backend_grid[y, x] = top_hl[y]
            backend_grid = _from_a_to_b(backend_grid, INVALID, OBSTACLE_INDEPENDENT)        
            # headland for obstacle in top to bottom direction
            y_max = h_backend - hl_pass_in_backend - WIDTH_SCALE_FRONTEND - 1
            if y_max < 0:
                y_max = -1
            for y in range(0, y_max + 1, 2):
                y1 = y + 1
                # ensure we can safely inspect y and y1
                if not (0 <= y < h_backend and 0 <= y1 < h_backend):
                    continue

                # skip if either row already HEADLAND/TRANSFER
                if backend_grid[y, x] in (HEADLAND, TRANSFER) or backend_grid[y1, x] in (HEADLAND, TRANSFER):
                    continue

                # compute pattern using current grid (or pass backup_grid if you want original-state decisions)
                top_hl = _cal_top_to_bottom(x, y, backend_grid, table_3)
                if not top_hl:
                    continue

                # write pattern with bounds checks (stop if we would go OOB)
                for i, val in enumerate(top_hl):
                    idy = y + i
                    if idy >= h_backend:
                        break
                    # keep existing HEADLAND/TRANSFER if present
                    if backend_grid[idy, x] in (HEADLAND, TRANSFER):
                        continue
                    backend_grid[idy, x] = val

                # normalize temporary INVALID markers to OBSTACLE_INDEPENDENT (if intended)
                backend_grid = _from_a_to_b(backend_grid, INVALID, OBSTACLE_INDEPENDENT)

            # from bottom to top
            bottom_hl = _cal_bottom_to_top(x, h_backend - 1, backup_grid, table_2)
            bottom_hl = list(reversed(bottom_hl))
            if bottom_hl:
                for k in range(len(bottom_hl)):
                    idy = h_backend - 1 - k
                    if backend_grid[idy, x] in [HEADLAND, TRANSFER]:
                        backend_grid[idy, x] = backend_grid[idy, x]
                        continue
                    backend_grid[idy, x] = bottom_hl[k]
            backend_grid = _from_a_to_b(backend_grid, INVALID, OBSTACLE_INDEPENDENT)      
            # headland for obstacle in bottom to top direction
            for y in range(0,h_backend - hl_pass_in_backend - 1*WIDTH_SCALE_FRONTEND - 1,2):
                # backup_grid = backend_grid.copy()
                # compute indices safely
                idx1 = h_backend - 1 - y
                idx2 = h_backend - 2 - y
                v1 = backend_grid[idx1, x] if 0 <= idx1 < h_backend else None
                v2 = backend_grid[idx2, x] if 0 <= idx2 < h_backend else None
                if (v1 in (HEADLAND, TRANSFER)) or (v2 in (HEADLAND, TRANSFER)):
                    # print(f"y = {y}, skip cell ({x},{idx1}) and ({x},{idx2}) as it is HEADLAND or TRANSFER")
                    continue
                else:
                    # from bottom to top
                    bottom_hl = _cal_bottom_to_top(x, h_backend - 1 - y, backup_grid, table_3)
                    bottom_hl = list(reversed(bottom_hl))
                    # print(f"y = {y}, bottom_hl for column {x} at y={h_backend - 1 - y}: {bottom_hl}")
                    if bottom_hl:
                        for j in range(len(bottom_hl)):
                            idy = h_backend - 1 - j - y
                            if backend_grid[idy, x] in [HEADLAND, TRANSFER]:
                                backend_grid[idy, x] = backend_grid[idy, x]
                                continue
                            backend_grid[idy, x] = bottom_hl[j]

                backend_grid = _from_a_to_b(backend_grid, INVALID, OBSTACLE_INDEPENDENT)
            top_hl = _cal_top_to_bottom(x, 0,backend_grid, table_2)
            if top_hl:
                for y in range(len(top_hl)):
                    backend_grid[y, x] = top_hl[y]
            backend_grid = _from_a_to_b(backend_grid, INVALID, OBSTACLE_INDEPENDENT)
        # minimum field height  
        elif grid_state == GRID_VALID_MIN:
            all_field = _is_all_field_in_headland_pass(x, 1, 'down')
            if(all_field not in table_1_min):
                return []
            else:
                final_act = table_1_min[all_field]
                if final_act:
                    for y in range(len(final_act)):
                        backend_grid[y, x] = final_act[y]
    

    # after fill headland boundary, convert all INVALID to OBSTACLE_INDEPENDENT    
    # current, backend grid contain: FIELD, HEADLAND, OBSTACLE_INDEPENDENT, TRANSFER (replace INVALID to OBSTACLE_INDEPENDENT)
    backend_grid = _from_a_to_b(backend_grid, INVALID, OBSTACLE_INDEPENDENT)

    # scale backend grid and seg_grid to frontend resolution for path planning
    backend_grid = backend_grid[:, ::2]
    h_backend, w_backend = backend_grid.shape

    # make a copy of seg_grid for further processing
    # mark OBSTACLE_INDEPENDENT and TRANSFER as OBSTACLE_INDEPENDENT in seg_grid => TRANSFER use for turn back only
    # mark FIELD and HEADLAND as FIELD in seg_grid => area need coverage
    seg_grid = backend_grid.copy()
    seg_grid = np.where((seg_grid == TRANSFER), OBSTACLE_INDEPENDENT, seg_grid)
    seg_grid = np.where((seg_grid == HEADLAND), FIELD, seg_grid)

    # extract 4 corners of field area from seg_grid for further apply ACO algorithm
    merge_field = rectangles_from_grid_vertical(seg_grid, object='field')

	# store corner of field as direction
    for mf in merge_field:
        mf['top_left'] = (mf['xmin'], mf['ymin'])
        mf['top_right'] = (mf['xmax'], mf['ymin'])
        mf['bottom_left'] = (mf['xmin'], mf['ymax'])
        mf['bottom_right'] = (mf['xmax'], mf['ymax'])

    # add visited flag for further path planning
    for mf in merge_field:
        mf['visited'] = False  

    # print field block (debug only)
    for i in range(len(merge_field)):
        print(f"Field Block {i}: [{merge_field[i]['xmin']},{merge_field[i]['ymin']}] - [{merge_field[i]['xmax']},{merge_field[i]['ymax']}]\n++ top_left: {merge_field[i]['top_left']}\n++ top_right: {merge_field[i]['top_right']}\n++ bottom_left: {merge_field[i]['bottom_left']}\n++ bottom_right: {merge_field[i]['bottom_right']}\n")

    # path planning inside each block
    # path_cover, exit_point, exit_point_direction = path_inside_block(merge_field[0], entry_point='top_left') # trial with first block only

    # demo for only first block (debug only)
    # print(f"Path cover inside block: {path_cover}\nLength of path = {len(path_cover)}\nexit point: {exit_point}\nexit_point_direction: {exit_point_direction}")

    # 
    path_final = []
    direct_list = [] # debug only
    coverage_list = [] # debug only
    cost = 0
    
    # get final path, list of direction from ACO algorithm
    path_final, direct_list, coverage_list, cost = aco_algorithm(
        backend_grid, 
        seg_grid, 
        merge_field, 
        start_point=user_start, 
        end_point=user_end
    )

    
    # draw path on backend grid for visualization
    # # mark path on backend grid

    # stub fo further use
    blocks = []
    bboxes = [] # mark obstacle box or field area (not yet implemented)

    # # return backend grid, segmentation, blocks and obstacle bounding boxes for caller to display
    # #return backend_grid, seg_grid, blocks, bboxes, True
    return backend_grid, seg_grid, blocks, bboxes, path_final, True

#######################################################################################################

if __name__ == "__main__":
    # trial with headland calculation & ACO path planning
    hl_pass = 1
    h = 12 # row
    w = 8 # column
    test_env = FieldEnvironment(width=w, height=h)
    test_env.interactive_grid()