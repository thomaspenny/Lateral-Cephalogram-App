import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tkinter as tk
from tkinter import simpledialog, filedialog, Menu, messagebox
from tabulate import tabulate
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import math
import csv
import os

class CoordinateUtils:
    """
    Utility class containing static methods for coordinate calculations and analysis
    """
    @staticmethod
    def calculate_perpendicular_intersection(p1, p2, p3):
        (x1, y1) = p1
        (x2, y2) = p2
        (x3, y3) = p3

        # Calculate direction vector of the first line
        dx1 = x2 - x1
        dy1 = y2 - y1
        
        # Parameter t for the projection of p3 onto the line defined by p1 and p2
        t = ((x3 - x1) * dx1 + (y3 - y1) * dy1) / (dx1**2 + dy1**2)
        
        # Closest point on the line to (x3, y3)
        x_closest = x1 + t * dx1
        y_closest = y1 + t * dy1

        return (x_closest, y_closest)
    
    @staticmethod
    def create_variables(named_coords):
        for name, coord in named_coords.items():
            globals()[name] = coord

    @staticmethod
    def calculate_angle_of_3(A, B, C, outer_angle=False):
        # Unpack the coordinates
        Ax, Ay = A
        Bx, By = B
        Cx, Cy = C
        
        # Calculate vectors AB and BC
        AB = (Ax - Bx, Ay - By)
        BC = (Cx - Bx, Cy - By)
        
        # Calculate dot product and magnitudes of AB and BC
        dot_product = AB[0] * BC[0] + AB[1] * BC[1]
        magnitude_AB = math.sqrt(AB[0]**2 + AB[1]**2)
        magnitude_BC = math.sqrt(BC[0]**2 + BC[1]**2)
        
        # Calculate the angle in radians
        angle_radians = math.acos(dot_product / (magnitude_AB * magnitude_BC))
        
        # Convert to degrees
        angle_degrees = round(math.degrees(angle_radians),1)

        plt.plot([A[0],B[0]],[A[1],B[1]], color='y')
        plt.plot([B[0],C[0]],[B[1],C[1]], color='y')
        
        # If outer_angle is True, return the outer angle (180 - inner angle)
        if outer_angle:
            return 180 - angle_degrees
        else:
            return angle_degrees

    @staticmethod
    def calculate_angle_and_intersection(A, B, C, D, outer_angle=False):
        # Unpack the coordinates
        Ax, Ay = A
        Bx, By = B
        Cx, Cy = C
        Dx, Dy = D
        
        # Calculate vectors AB and CD
        AB = (Bx - Ax, By - Ay)
        CD = (Dx - Cx, Dy - Cy)
        
        # Calculate dot product and magnitudes of AB and CD
        dot_product = AB[0] * CD[0] + AB[1] * CD[1]
        magnitude_AB = math.sqrt(AB[0]**2 + AB[1]**2)
        magnitude_CD = math.sqrt(CD[0]**2 + CD[1]**2)
        
        # Calculate the angle in radians
        angle_radians = math.acos(dot_product / (magnitude_AB * magnitude_CD))
        
        # Convert to degrees
        angle_degrees = round(math.degrees(angle_radians),1)
        
        # Calculate the outer angle if requested
        if outer_angle:
            angle_degrees = 180 - angle_degrees
        
        # To find the intersection point, solve the line equations:
        # Line AB: y = m1 * x + b1
        # Line CD: y = m2 * x + b2
        
        # Calculate slopes (m1, m2) and intercepts (b1, b2)
        denominator1 = (Bx - Ax)
        denominator2 = (Dx - Cx)
        
        # Handle vertical lines by assigning infinite slope
        if denominator1 != 0:
            m1 = (By - Ay) / denominator1
            b1 = Ay - m1 * Ax
        else:
            m1 = float('inf')
            b1 = float('inf')

        if denominator2 != 0:
            m2 = (Dy - Cy) / denominator2
            b2 = Cy - m2 * Cx
        else:
            m2 = float('inf')
            b2 = float('inf')

        if m1 == m2:  # Parallel lines case
            intersection = None
        elif m1 == float('inf'):  # Line AB is vertical
            x_intersect = Ax
            y_intersect = m2 * x_intersect + b2
            intersection = (x_intersect, y_intersect)
        elif m2 == float('inf'):  # Line CD is vertical
            x_intersect = Cx
            y_intersect = m1 * x_intersect + b1
            intersection = (x_intersect, y_intersect)
        else:
            # Calculate the intersection point
            x_intersect = (b2 - b1) / (m1 - m2)
            y_intersect = m1 * x_intersect + b1
            intersection = (x_intersect, y_intersect)
        
        return angle_degrees, intersection

    @staticmethod
    def distance(p1, p2):        
        #Calculate the Euclidean distance between two points p1 and p2.
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

class CoordinateCalculations:
    @staticmethod
    def calculate_all(named_coordinates, scale):
        # Pairing names with the first x value and the second y value
        CoordinateUtils.create_variables(named_coordinates)

        # Draw straight reference lines
        straight_lines = [(S, N, "SNL"), (Or, Po, "FH"), (Go, Me, "ML"),
                          (SnA, SnP, "NL"), (Occ1, Occ2, "OccL"), (Ar, Go, "RL"), (pg, n, "EL")]

        for p1, p2, name in straight_lines:
            plt.axline(p1, p2, color='r')

        # Draw shorter segments
        short_segments = [(U1_tip, U1_apex, "U1 inc"), (L1_apex, L1_tip, "L1 inc"),
                          (N, Ba, "Facial angle part1"), (Pt, Gn, "Facial angle part2")]

        for p1, p2, name in short_segments:
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='r')

        # Perp. intersections
        A_dot = CoordinateUtils.calculate_perpendicular_intersection(SnA, SnP, A)
        Pg_dot = CoordinateUtils.calculate_perpendicular_intersection(Go, Me, Pg)
        U1_dot = CoordinateUtils.calculate_perpendicular_intersection(SnA, SnP, U1_tip)
        L1_dot = CoordinateUtils.calculate_perpendicular_intersection(Go, Me, L1_tip)
        U6_dot = CoordinateUtils.calculate_perpendicular_intersection(SnA, SnP, U6_tip)
        L6_dot = CoordinateUtils.calculate_perpendicular_intersection(Go, Me, L6_tip)
        A_dotdot = CoordinateUtils.calculate_perpendicular_intersection(Occ1, Occ2, A)
        B_dotdot = CoordinateUtils.calculate_perpendicular_intersection(Occ1, Occ2, B)

        # 3-point angles
        SNA_angle = CoordinateUtils.calculate_angle_of_3(S, N, A, outer_angle=False)
        SNB_angle = CoordinateUtils.calculate_angle_of_3(S, N, B, outer_angle=False)
        ArGoMe_angle = CoordinateUtils.calculate_angle_of_3(Ar, Go, Me, outer_angle=False)
        ArGoN_angle = CoordinateUtils.calculate_angle_of_3(Ar, Go, N, outer_angle=False)
        NGoMe_angle = CoordinateUtils.calculate_angle_of_3(N, Go, Me, outer_angle=False)
        NSAr_angle = CoordinateUtils.calculate_angle_of_3(N, S, Ar, outer_angle=False)
        SArGo_angle = CoordinateUtils.calculate_angle_of_3(S, Ar, Go, outer_angle=False)

        # 4-point angles
        U1_NL_angle = CoordinateUtils.calculate_angle_and_intersection(U1_tip, U1_apex, SnA, SnP, outer_angle=False)
        L1_ML_angle = CoordinateUtils.calculate_angle_and_intersection(L1_tip, L1_apex, Go, Me, outer_angle=False)
        U1_L1_angle = CoordinateUtils.calculate_angle_and_intersection(U1_tip, U1_apex, L1_tip, L1_apex, outer_angle=False)
        NL_SN_angle = CoordinateUtils.calculate_angle_and_intersection(SnA, SnP, S, N, outer_angle=True)
        ML_SN_angle = CoordinateUtils.calculate_angle_and_intersection(Go, Me, S, N, outer_angle=False)
        NL_ML_angle = CoordinateUtils.calculate_angle_and_intersection(SnA, SnP, Go, Me, outer_angle=True)
        FH_SN_angle = CoordinateUtils.calculate_angle_and_intersection(Or, Po, S, N, outer_angle=True)
        NBa_PtGn_angle = CoordinateUtils.calculate_angle_and_intersection(N, Ba, Pt, Gn, outer_angle=False)

        # Perpendicular intersections
        plt.plot([A_dot[0], A[0]], [A_dot[1], A[1]], color='c')
        plt.text(A_dot[0], A_dot[1], "A`", color='c')

        plt.plot([Pg_dot[0], Pg[0]], [Pg_dot[1], Pg[1]], color='c')
        plt.text(Pg_dot[0], Pg_dot[1], "Pg`", color='c')

        plt.plot([U1_dot[0], U1_tip[0]], [U1_dot[1], U1_tip[1]], color='c')
        plt.text(U1_dot[0], U1_dot[1], "U1`", color='c')

        plt.plot([L1_dot[0], L1_tip[0]], [L1_dot[1], L1_tip[1]], color='c')
        plt.text(L1_dot[0], L1_dot[1], "L1`", color='c')

        plt.plot([U6_dot[0], U6_tip[0]], [U6_dot[1], U6_tip[1]], color='c')
        plt.text(U6_dot[0], U6_dot[1], "U6`", color='c')

        plt.plot([L6_dot[0], L6_tip[0]], [L6_dot[1], L6_tip[1]], color='c')
        plt.text(L6_dot[0], L6_dot[1], "L6`", color='c')

        plt.plot([A_dotdot[0], A[0]], [A_dotdot[1], A[1]], color='c')
        plt.text(A_dotdot[0], A_dotdot[1], "A``", color='c')

        plt.plot([B_dotdot[0], B[0]], [B_dotdot[1], B[1]], color='c')
        plt.text(B_dotdot[0], B_dotdot[1], "B``", color='c')

        # 3-point angle text
        plt.text(N[0], N[1], "SNA: " + str(SNA_angle) + "°", color='c', va='top', fontsize='x-small')
        plt.text(N[0], N[1], "SNB: " + str(SNB_angle) + "°", color='c', va='bottom', fontsize='x-small')
        plt.text(Go[0], Go[1], "ArGoMe: " + str(ArGoMe_angle) + "°", color='c', va='top', ha='left', fontsize='x-small')
        plt.text(Go[0], Go[1], "ArGoN: " + str(ArGoN_angle) + "°", color='c', va='center', ha='right', fontsize='x-small')
        plt.text(Go[0], Go[1], "NGoMe: " + str(NGoMe_angle) + "°", color='c', va='bottom', ha='left', fontsize='x-small')
        plt.text(S[0], S[1], "NSAr: " + str(NSAr_angle) + "°", color='c', va='center', fontsize='x-small')
        plt.text(Ar[0], Ar[1], "SArGo: " + str(SArGo_angle) + "°", color='c', va='center', fontsize='x-small')

        # 4-point angle text
        four_point_format = {'color': 'm', 'va': 'center', 'fontsize': 'x-small'}
        plt.text(U1_NL_angle[1][0], U1_NL_angle[1][1], "U1/NL: " + str(U1_NL_angle[0]) + "°", four_point_format)
        plt.text(L1_ML_angle[1][0], L1_ML_angle[1][1], "L1/ML: " + str(L1_ML_angle[0]) + "°", four_point_format)
        plt.text(U1_L1_angle[1][0], U1_L1_angle[1][1], "U1/L1: " + str(U1_L1_angle[0]) + "°", four_point_format)
        plt.text(NL_SN_angle[1][0], NL_SN_angle[1][1], "NL/SN: " + str(NL_SN_angle[0]) + "°", four_point_format)
        plt.text(ML_SN_angle[1][0], ML_SN_angle[1][1], "ML/SN: " + str(ML_SN_angle[0]) + "°", four_point_format)
        plt.text(NL_ML_angle[1][0], NL_ML_angle[1][1], "NL/ML: " + str(NL_ML_angle[0]) + "°", four_point_format)
        plt.text(FH_SN_angle[1][0], FH_SN_angle[1][1], "FH/SN: " + str(round(FH_SN_angle[0], 4)) + "°", four_point_format)
        plt.text(NBa_PtGn_angle[1][0], NBa_PtGn_angle[1][1], "NBa/PtGn: " + str(NBa_PtGn_angle[0]) + "°", four_point_format)

        # Calculate scale ratio and distances
        scale_distance = CoordinateUtils.distance(Scale_Min, Scale_Max)
        scale_ratio = round(int(scale) / scale_distance, 4)

        # Calculate all distances
        N_S_dist = round(CoordinateUtils.distance(N, S) * scale_ratio, 4)
        SnA_SnP_dist = round(CoordinateUtils.distance(SnA, SnP) * scale_ratio, 4)
        Go_Me_dist = round(CoordinateUtils.distance(Go, Me) * scale_ratio, 4)
        Ar_Go_dist = round(CoordinateUtils.distance(Ar, Go) * scale_ratio, 4)
        Co_Go_dist = round(CoordinateUtils.distance(Co, Go) * scale_ratio, 4)
        N_Me_dist = round(CoordinateUtils.distance(N, Me) * scale_ratio, 4)
        U1_NL_dist = round(CoordinateUtils.distance(U1_tip, U1_dot) * scale_ratio, 4)
        L1_ML_dist = round(CoordinateUtils.distance(L1_tip, L1_dot) * scale_ratio, 4)
        U1_length_dist = round(CoordinateUtils.distance(U1_tip, U1_apex) * scale_ratio, 4)
        L1_length_dist = round(CoordinateUtils.distance(L1_tip, L1_apex) * scale_ratio, 4)
        U6_NL_dist = round(CoordinateUtils.distance(U6_tip, U6_dot) * scale_ratio, 4)
        L6_ML_dist = round(CoordinateUtils.distance(L6_tip, L6_dot) * scale_ratio, 4)
        Witts_dist = round(CoordinateUtils.distance(A_dotdot, B_dotdot) * scale_ratio, 4)

        # Calculate aesthetic line perps
        ul_dot = CoordinateUtils.calculate_perpendicular_intersection(pg, n, ul)
        ul_E_dist = round(CoordinateUtils.distance(ul, ul_dot) * scale_ratio, 4)
        ll_dot = CoordinateUtils.calculate_perpendicular_intersection(pg, n, ll)
        ll_E_dist = round(CoordinateUtils.distance(ll, ll_dot) * scale_ratio, 4)

        # Further calculations
        Max_Norm = round(N_S_dist * 0.7, 3)
        Mand_Norm = round(N_S_dist * 1.01, 3)
        ArGo_NMe_ratio = round(Ar_Go_dist / N_Me_dist, 4)
        CoGo_GoMe_ratio = round(Co_Go_dist / Go_Me_dist, 4)
        U1NL_L1ML_ratio = round(U1_NL_dist / L1_ML_dist, 4)
        U1NL_U6NL_ratio = round(U1_NL_dist / U6_NL_dist, 4)
        L1ML_L6ML_ratio = round(L1_ML_dist / L6_ML_dist, 4)
        U6NL_L6ML_ratio = round(U6_NL_dist / L6_ML_dist, 4)

        # Collect all measurements into appropriate tables
        table_angles = [
            ("SNA angle (deg):", SNA_angle), ("SNB angle (deg):", SNB_angle), ("ArGoMe angle (deg):", ArGoMe_angle),
            ("ArGoN angle (deg):", ArGoN_angle), ("NGoMe angle (deg):", NGoMe_angle), ("NSAr angle (deg):", NSAr_angle),
            ("SArGo angle (deg):", SArGo_angle), ("U1 NL angle (deg):", U1_NL_angle[0]), ("L1 ML angle (deg):", L1_ML_angle[0]),
            ("U1 L1 angle (deg):", U1_L1_angle[0]), ("NL SN angle (deg):", NL_SN_angle[0]), ("ML SN angle (deg):", ML_SN_angle[0]),
            ("NL ML angle (deg):", NL_ML_angle[0]), ("FH SN angle (deg):", round(FH_SN_angle[0], 4)), ("NBa PtGn angle (deg):", NBa_PtGn_angle[0]),
        ]

        table_distances = [
            ("Scale (mm):", scale), ("Scale ratio (mm/pixel):", scale_ratio), ("N-S distance (mm):", N_S_dist),
            ("SnA-SnP distance (mm):", SnA_SnP_dist), ("Go-Me distance (mm):", Go_Me_dist), ("Ar-Go distance (mm):", Ar_Go_dist),
            ("Co-Go distance (mm):", Co_Go_dist), ("N-Me distance (mm):", N_Me_dist), ("U1-NL distance (mm):", U1_NL_dist),
            ("L1-ML distance (mm):", L1_ML_dist), ("U1-length distance (mm):", U1_length_dist),
            ("L1-length distance (mm):", L1_length_dist), ("U6-NL distance (mm):", U6_NL_dist), ("L6-ML distance (mm):", L6_ML_dist), 
            ("Witts-distance (mm):", Witts_dist), ("ul-E distance (mm):", ul_E_dist), ("ll-E distance (mm):", ll_E_dist),
        ]

        table_further_calculations = [
            ("Max Norm (mm):", Max_Norm), ("Mand Norm (mm):", Mand_Norm), ("ArGo:NMe ratio:", ArGo_NMe_ratio),
            ("CoGo:GoMe ratio:", CoGo_GoMe_ratio), ("U1NL:L1NL ratio:", U1NL_L1ML_ratio), ("U1NL:U6NL ratio:", U1NL_U6NL_ratio),
            ("L1ML:L6ML ratio:", L1ML_L6ML_ratio), ("U6NL:L6ML ratio:", U6NL_L6ML_ratio),
        ]

        # Combine all tables
        combined_table = {**dict(table_angles), **dict(table_distances), **dict(table_further_calculations)}

        return combined_table

class ImageCoordinateTool:
    def __init__(self, root):
        #Main init of program
        self.root = root
        self.root.title("Image Coordinate Tool")
        self.coordinates = []
        self.all_coordinates = []
        self.coordinate_names = [
            "Scale_Min", "Scale_Max", "S", "Se", "N", "A", "B", "Or", "Pg", "Gn", "Me", "Go", "AntGo", 
            "Ar", "Ba", "Po", "Co", "SnA", "SnP", "Pt", "U1_tip", "U1_apex", "L1_tip", "L1_apex",
            "U6_tip", "U6_apex", "L6_tip", "L6_apex", "Occ1", "Occ2", "ul", "ll", "sto", "n", "pg"
        ]
        self.current_coordinate = None
        self.scale = None
        self.img = None
        self.fig = None
        self.ax = None
        self.table = None
        self.plot_points = []
        self.text_annotations = []
        self.img_width = None
        self.img_height = None
        self.combined_table = {}

        # Create menu bar instead of buttons
        self.create_menu()

    def create_menu(self):
        
        self.csv_directory = None  # Stores the selected CSV directory
        self.img_directory = None  # Stores the selected image directory
        self.load_img_directory = None  # Stores the selected image directory

        # Create a menu bar
        menu_bar = Menu(self.root)

        # 'File' menu with 'Load Image' option
        file_menu = Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Load Image", command=self.load_image)
        file_menu.add_command(label="Save Plot", command=self.save_both)  # Add save plot command
        menu_bar.add_cascade(label="File", menu=file_menu)

        # 'Options' menu with 'Set Scale' option
        options_menu = Menu(menu_bar, tearoff=0)
        options_menu.add_command(label="Set Scale", command=self.get_scale)
        options_menu.add_command(label="Set 'Load Image' Directory", command=self.set_load_img_dir)        
        options_menu.add_command(label="Set CSV directory", command=self.set_csv_dir)
        options_menu.add_command(label="Set Image Directory", command=self.set_img_dir)
        menu_bar.add_cascade(label="Options", menu=options_menu)

        # Add the menu bar to the root window
        self.root.config(menu=menu_bar)
       
       
    # Function to set the CSV directory
    def set_load_img_dir(self):
        directory = filedialog.askdirectory(title="Select 'Load Image' Directory")
        if directory:
            self.load_img_directory = directory
            print(f"CSV directory set to: {self.load_img_directory}")
       
    # Function to set the CSV directory
    def set_csv_dir(self):
        directory = filedialog.askdirectory(title="Select CSV Directory")
        if directory:
            self.csv_directory = directory
            print(f"CSV directory set to: {self.csv_directory}")

    # Function to set the Image directory
    def set_img_dir(self):
        directory = filedialog.askdirectory(title="Select Image Directory")
        if directory:
            self.img_directory = directory
            print(f"Image directory set to: {self.img_directory}")

    def load_image(self):
        #Handle image selection and start the tool
        try:
            # Reset coordinates when loading a new image
            self.all_coordinates.clear()  # Clears previously stored coordinates
            self.current_coordinate = None  # Resets the currently selected coordinate
            
            selected_image_path = self.select_image_file()
            self.img = mpimg.imread(selected_image_path)
            self.img_height, self.img_width = self.img.shape[:2]
            
            self.fig, self.ax = plt.subplots()
            self.ax.imshow(self.img)
            self.ax.set_title("Select first coordinate by pressing the spacebar.")
            plt.subplots_adjust(right=0.7)

            self.scale = 0

            # Embed the Matplotlib figure in Tkinter
            if hasattr(self, 'canvas') and self.canvas is not None:
                self.canvas.get_tk_widget().pack_forget()
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            # Add the navigation toolbar below the canvas
            if hasattr(self, 'toolbar') and self.toolbar is not None:
                self.toolbar.pack_forget()
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
            self.toolbar.update()
            self.toolbar.pack(side=tk.TOP, fill=tk.X)

            self.canvas.draw()

            self.initialize_table()
            self.get_coordinates()

            # Set plot limits based on image size
            self.ax.set_xlim(0, self.img_width)
            self.ax.set_ylim(self.img_height, 0)

        except ValueError as e:
            print(e)

    def select_image_file(self):
        # Open a file dialog to select an image file.
        if self.load_img_directory:
        # If a directory has been set, start the dialog from that directory
            initial_dir = self.load_img_directory
        else:
        # If no directory is set, start from the current working directory
            initial_dir = None
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Select an Image File",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All Files", "*.*")]
        )
        if file_path:
            return file_path
        else:
            raise ValueError("No image file selected.")

    def get_scale(self):
        #Open a dialog box to ask the user for an integer scale value.
        scale = simpledialog.askinteger("Set Scale", "Enter the scale (must be an integer):", parent=self.root)
        if scale is not None:
            self.scale = scale
            print(f"Scale set to: {self.scale} mm")
        else:
            print("No scale entered.")

    def initialize_table(self):
        #Initialize the table with the scale size and placeholder for coordinates.
        cell_text = [["Scale Size", "Options > Scale Size"]]
        cell_text += [[name, "( , )"] for name in self.coordinate_names]

        self.table = self.ax.table(
            cellText=cell_text, colLabels=['Name', 'Coordinates'], loc='center', cellLoc='center',
            bbox=[1.05, 0.0, 0.3, 1.0]
        )

        self.update_table_fontsize()

    def update_table(self):
        #Update the coordinates in the table.
        for i, coord in enumerate(self.all_coordinates):
            self.table[(i + 2, 1)].get_text().set_text(f"({coord[0]:.4g}, {coord[1]:.4g})")
        self.canvas.draw()

    def update_table_fontsize(self):
        #Update the font size of the table dynamically based on the figure size.
        width, height = self.fig.get_size_inches()
        fontsize = max(10, int(min(width, height) * 1.5))
        for key, cell in self.table.get_celld().items():
            cell.set_text_props(fontsize=fontsize)
        self.canvas.draw()

    def update_plot_points(self):
        #Update plot points and labels.
        for point in self.plot_points:
            point.remove()
        for text in self.text_annotations:
            text.remove()

        self.plot_points.clear()
        self.text_annotations.clear()

        for i, (x, y) in enumerate(self.all_coordinates):
            self.plot_points.append(self.ax.plot(x, y, 'x', color='lime', markersize=6, markeredgewidth=1)[0])
            self.text_annotations.append(self.ax.text(x, y, self.coordinate_names[i], color='lime', fontsize=10, ha='right', va='bottom'))

        if self.current_coordinate:
            x, y = self.current_coordinate
            self.plot_points.append(self.ax.plot(x, y, 'x', color='cyan', markersize=6, markeredgewidth=1)[0])

        self.canvas.draw()

    def save_both(self):
        #Save both the plot and the CSV file.
        # Prompt the user to enter a filename for the plot
        filename = simpledialog.askstring("Save Plot", "Enter the filename (without extension):")
        if filename:
            # Save the plot
            self.save_plot(filename)

            # Save the CSV using the same filename
            self.save_csv(filename)
        else:
            print("Saving plot and CSV canceled.")
            
    def save_plot(self, filename):
        #Automatically save the plot with selected data points to a pre-configured directory.
        
        # Check if the image directory has been set
        if not self.img_directory:
            messagebox.showerror("Error", "Image directory is not set! Please configure the directory first.")
            return

        # Save the plot using the provided filename
        file_path = os.path.join(self.img_directory, f"{filename}.png")
        self.fig.savefig(file_path)  # Save the figure as a PNG file
        print(f"Plot saved as: {file_path}")

    def save_csv(self, filename):
        #Save CSV file using the pre-configured CSV directory and the same filename as the image.
        
        # Check if the CSV directory has been set
        if not self.csv_directory:
            messagebox.showerror("Error", "CSV directory is not set! Please configure the directory first.")
            return
        
        # Create the full file path for the CSV file
        csv_file_path = os.path.join(self.csv_directory, f"{filename}.csv")
        
        # Save the CSV file in the configured directory
        with open(csv_file_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([filename] + list(self.combined_table.keys()))  # Assuming combined_table is a dict
            writer.writerow([filename] + list(self.combined_table.values()))
        
        print(f"CSV file saved as: {csv_file_path}")        

    def on_key(self, event):
        #Handle key press events for coordinate selection.
        if event.key == ' ':
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                self.current_coordinate = (x, y)
                self.update_plot_points()

        elif event.key == 'enter':
            if self.scale <= 0:
                # Show a pop-up informing the user that scale needs to be set
                tk.messagebox.showwarning("Scale Not Set", "Scale must be greater than 0. Please enter a positive integer for the scale.")
                
                # Open a dialog to prompt the user for a valid scale value
                self.get_scale()
                
                # After setting the scale, check again
                if self.scale <= 0:
                    # If the user still hasn't entered a valid scale, do not proceed
                    self.ax.set_title("No valid scale provided. Unable to continue.")
                    return

            if self.current_coordinate and len(self.all_coordinates) < len(self.coordinate_names):
                self.all_coordinates.append(self.current_coordinate)
                self.current_coordinate = None
                self.update_plot_points()
                self.update_table()

                next_field_number = len(self.all_coordinates)
                if next_field_number < len(self.coordinate_names):
                    self.ax.set_title(f"Select coordinate {next_field_number + 1} by pressing the spacebar.")
                else:
                    self.ax.set_title("All coordinates selected.")
                    self.print_table()
                    named_coordinates = {name: coord for name, coord in zip(self.coordinate_names, self.all_coordinates)}
                    self.combined_table = CoordinateCalculations.calculate_all(named_coordinates, self.scale)

    def print_table(self):
        #Print the final table of coordinates.
        table_data = [[name, f"({coord[0]:.4g}, {coord[1]:.4g})"] for name, coord in zip(self.coordinate_names, self.all_coordinates)]
        print("\nFinal Coordinates Table:\n",tabulate(table_data, headers=["Name", "Coordinates"], tablefmt="grid"))

    def on_close(self, event):
        #Handle window close event.
        plt.close(self.fig)

    def on_resize(self, event):
        #Handle window resize event to dynamically adjust font size.
        self.update_table_fontsize()

    def get_coordinates(self):
        #Capture coordinates from user input.#
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.fig.canvas.mpl_connect('resize_event', self.on_resize)

def launch_application():
    """Launch the Image Coordinate Tool application"""
    try:
        root = tk.Tk()
        tool = ImageCoordinateTool(root)
        root.mainloop()
    except Exception as e:
        print(f"Error launching application: {e}")
        # Could add logging here
        # Could show an error dialog
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    launch_application()