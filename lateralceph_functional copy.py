import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tkinter as tk
from tkinter import simpledialog, filedialog, Menu, messagebox, Text, Scrollbar, Frame, Button
from tabulate import tabulate
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import math
import csv
import os
from collections import OrderedDict

# Dictionary to store coordinates with their names in order
coordinates = OrderedDict([
    ("Scale_Min", None), ("Scale_Max", None), ("S", None), ("Se", None), ("N", None), 
    ("A", None), ("B", None), ("Or", None), ("Pg", None), ("Gn", None), ("Me", None), 
    ("Go", None), ("AntGo", None), ("Ar", None), ("Ba", None), ("Po", None), ("Co", None), 
    ("SnA", None), ("SnP", None), ("Pt", None), ("U1_tip", None), ("U1_apex", None), 
    ("L1_tip", None), ("L1_apex", None), ("U6_tip", None), ("U6_apex", None), 
    ("L6_tip", None), ("L6_apex", None), ("Occ1", None), ("Occ2", None), 
    ("ul", None), ("ll", None), ("sto", None), ("n", None), ("pg", None)
])
scale = None
img = None
fig = None
ax = None
table_text = None
text_frame = None
plot_points = []
text_annotations = []
img_width = None
img_height = None
combined_table = {}
canvas = None
toolbar = None
undo_button = None
scale_button = None
submit_button = None
root = None

# Directory settings
csv_directory = None
img_directory = None
load_img_directory = None

# Calculate the perpendicular intersection point from p3 to the line defined by p1-p2
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

# Create global variables from coordinate dictionary for use in calculations
def create_variables(coord_dict):
    for name, coord in coord_dict.items():
        globals()[name] = coord

# Calculate the angle at point B formed by points A-B-C (3-point angle)
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

# Calculate the angle between two lines AB and CD and their intersection point
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

# Calculate the Euclidean distance between two points
def distance(p1, p2):        
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Perform all cephalometric calculations and draw analysis lines on the plot
def calculate_all(coord_dict, scale):
    global combined_table
    
    # Create variables from coordinate dictionary
    create_variables(coord_dict)

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
    A_dot    = calculate_perpendicular_intersection(SnA, SnP, A)
    Pg_dot   = calculate_perpendicular_intersection(Go, Me, Pg)
    U1_dot   = calculate_perpendicular_intersection(SnA, SnP, U1_tip)
    L1_dot   = calculate_perpendicular_intersection(Go, Me, L1_tip)
    U6_dot   = calculate_perpendicular_intersection(SnA, SnP, U6_tip)
    L6_dot   = calculate_perpendicular_intersection(Go, Me, L6_tip)
    A_dotdot = calculate_perpendicular_intersection(Occ1, Occ2, A)
    B_dotdot = calculate_perpendicular_intersection(Occ1, Occ2, B)

    # 3-point angles
    SNA_angle    = calculate_angle_of_3(S, N, A, outer_angle=False)
    SNB_angle    = calculate_angle_of_3(S, N, B, outer_angle=False)
    ArGoMe_angle = calculate_angle_of_3(Ar, Go, Me, outer_angle=False)
    ArGoN_angle  = calculate_angle_of_3(Ar, Go, N, outer_angle=False)
    NGoMe_angle  = calculate_angle_of_3(N, Go, Me, outer_angle=False)
    NSAr_angle   = calculate_angle_of_3(N, S, Ar, outer_angle=False)
    SArGo_angle  = calculate_angle_of_3(S, Ar, Go, outer_angle=False)

    # 4-point angles
    U1_NL_angle    = calculate_angle_and_intersection(U1_tip, U1_apex, SnA, SnP, outer_angle=False)
    L1_ML_angle    = calculate_angle_and_intersection(L1_tip, L1_apex, Go, Me, outer_angle=False)
    U1_L1_angle    = calculate_angle_and_intersection(U1_tip, U1_apex, L1_tip, L1_apex, outer_angle=False)
    NL_SN_angle    = calculate_angle_and_intersection(SnA, SnP, S, N, outer_angle=True)
    ML_SN_angle    = calculate_angle_and_intersection(Go, Me, S, N, outer_angle=False)
    NL_ML_angle    = calculate_angle_and_intersection(SnA, SnP, Go, Me, outer_angle=True)
    FH_SN_angle    = calculate_angle_and_intersection(Or, Po, S, N, outer_angle=True)
    NBa_PtGn_angle = calculate_angle_and_intersection(N, Ba, Pt, Gn, outer_angle=False)

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
    plt.text(U1_NL_angle[1][0], U1_NL_angle[1][1], "U1/NL: " + str(U1_NL_angle[0]) + "°", **four_point_format)
    plt.text(L1_ML_angle[1][0], L1_ML_angle[1][1], "L1/ML: " + str(L1_ML_angle[0]) + "°", **four_point_format)
    plt.text(U1_L1_angle[1][0], U1_L1_angle[1][1], "U1/L1: " + str(U1_L1_angle[0]) + "°", **four_point_format)
    plt.text(NL_SN_angle[1][0], NL_SN_angle[1][1], "NL/SN: " + str(NL_SN_angle[0]) + "°", **four_point_format)
    plt.text(ML_SN_angle[1][0], ML_SN_angle[1][1], "ML/SN: " + str(ML_SN_angle[0]) + "°", **four_point_format)
    plt.text(NL_ML_angle[1][0], NL_ML_angle[1][1], "NL/ML: " + str(NL_ML_angle[0]) + "°", **four_point_format)
    plt.text(FH_SN_angle[1][0], FH_SN_angle[1][1], "FH/SN: " + str(round(FH_SN_angle[0], 4)) + "°", **four_point_format)
    plt.text(NBa_PtGn_angle[1][0], NBa_PtGn_angle[1][1], "NBa/PtGn: " + str(NBa_PtGn_angle[0]) + "°", **four_point_format)

    # Calculate scale ratio and distances
    scale_distance = distance(Scale_Min, Scale_Max)
    scale_ratio = round(int(scale) / scale_distance, 4)

    # Calculate all distances
    N_S_dist       = round(distance(N, S) * scale_ratio, 4)
    SnA_SnP_dist   = round(distance(SnA, SnP) * scale_ratio, 4)
    Go_Me_dist     = round(distance(Go, Me) * scale_ratio, 4)
    Ar_Go_dist     = round(distance(Ar, Go) * scale_ratio, 4)
    Co_Go_dist     = round(distance(Co, Go) * scale_ratio, 4)
    N_Me_dist      = round(distance(N, Me) * scale_ratio, 4)
    U1_NL_dist     = round(distance(U1_tip, U1_dot) * scale_ratio, 4)
    L1_ML_dist     = round(distance(L1_tip, L1_dot) * scale_ratio, 4)
    U1_length_dist = round(distance(U1_tip, U1_apex) * scale_ratio, 4)
    L1_length_dist = round(distance(L1_tip, L1_apex) * scale_ratio, 4)
    U6_NL_dist     = round(distance(U6_tip, U6_dot) * scale_ratio, 4)
    L6_ML_dist     = round(distance(L6_tip, L6_dot) * scale_ratio, 4)
    Witts_dist     = round(distance(A_dotdot, B_dotdot) * scale_ratio, 4)

    # Calculate aesthetic line perps
    ul_dot    = calculate_perpendicular_intersection(pg, n, ul)
    ul_E_dist = round(distance(ul, ul_dot) * scale_ratio, 4)
    ll_dot    = calculate_perpendicular_intersection(pg, n, ll)
    ll_E_dist = round(distance(ll, ll_dot) * scale_ratio, 4)

    # Further calculations
    Max_Norm        = round(N_S_dist * 0.7, 3)
    Mand_Norm       = round(N_S_dist * 1.01, 3)
    ArGo_NMe_ratio  = round(Ar_Go_dist / N_Me_dist, 4)
    CoGo_GoMe_ratio = round(Co_Go_dist / Go_Me_dist, 4)
    U1NL_L1ML_ratio = round(U1_NL_dist / L1_ML_dist, 4)
    U1NL_U6NL_ratio = round(U1_NL_dist / U6_NL_dist, 4)
    L1ML_L6ML_ratio = round(L1_ML_dist / L6_ML_dist, 4)
    U6NL_L6ML_ratio = round(U6_NL_dist / L6_ML_dist, 4)

    # Collect all measurements into appropriate tables
    table_angles = [
        ("SNA (deg)", SNA_angle), ("SNB (deg)", SNB_angle), ("ArGoMe (deg)", ArGoMe_angle),
        ("ArGoN (deg)", ArGoN_angle), ("NGoMe (deg)", NGoMe_angle), ("NSAr (deg)", NSAr_angle),
        ("SArGo (deg)", SArGo_angle), ("U1 NL (deg)", U1_NL_angle[0]), ("L1 ML (deg)", L1_ML_angle[0]),
        ("U1 L1 (deg)", U1_L1_angle[0]), ("NL SN (deg)", NL_SN_angle[0]), ("ML SN (deg)", ML_SN_angle[0]),
        ("NL ML (deg)", NL_ML_angle[0]), ("FH SN (deg)", round(FH_SN_angle[0], 4)), ("NBa PtGn (deg)", NBa_PtGn_angle[0]),
    ]

    table_distances = [
        ("Scale (mm)", scale), ("Scale (mm/pixel)", scale_ratio), ("N-S (mm)", N_S_dist),
        ("SnA-SnP (mm)", SnA_SnP_dist), ("Go-Me (mm)", Go_Me_dist), ("Ar-Go (mm)", Ar_Go_dist),
        ("Co-Go (mm)", Co_Go_dist), ("N-Me (mm)", N_Me_dist), ("U1-NL (mm)", U1_NL_dist),
        ("L1-ML (mm)", L1_ML_dist), ("U1-length (mm)", U1_length_dist),
        ("L1-length (mm)", L1_length_dist), ("U6-NL (mm)", U6_NL_dist), ("L6-ML (mm)", L6_ML_dist), 
        ("Witts-(mm)", Witts_dist), ("ul-E (mm)", ul_E_dist), ("ll-E (mm)", ll_E_dist),
    ]

    table_further_calculations = [
        ("Max Norm (mm)", Max_Norm), ("Mand Norm (mm)", Mand_Norm), ("ArGo:NMe", ArGo_NMe_ratio),
        ("CoGo:GoMe", CoGo_GoMe_ratio), ("U1NL:L1NL", U1NL_L1ML_ratio), ("U1NL:U6NL", U1NL_U6NL_ratio),
        ("L1ML:L6ML", L1ML_L6ML_ratio), ("U6NL:L6ML", U6NL_L6ML_ratio),
    ]

    # Combine all tables
    combined_table = {**dict(table_angles), **dict(table_distances), **dict(table_further_calculations)}

    return combined_table

# Create the application menu bar with File and Options menus
def create_menu():
    global root

    # Create a menu bar
    menu_bar = Menu(root)

    # 'File' menu with 'Load Image' option
    file_menu = Menu(menu_bar, tearoff=0)
    file_menu.add_command(label="Load Image", command=load_image)
    file_menu.add_command(label="Save Plot", command=save_both)
    menu_bar.add_cascade(label="File", menu=file_menu)

    # 'Options' menu
    options_menu = Menu(menu_bar, tearoff=0)
    options_menu.add_command(label="Set 'Load Image' Directory", command=set_load_img_dir)        
    options_menu.add_command(label="Set CSV directory", command=set_csv_dir)
    options_menu.add_command(label="Set Image Directory", command=set_img_dir)
    menu_bar.add_cascade(label="Options", menu=options_menu)

    # Add the menu bar to the root window
    root.config(menu=menu_bar)

# Set the default directory for loading images
def set_load_img_dir():
    global load_img_directory
    directory = filedialog.askdirectory(title="Select 'Load Image' Directory")
    if directory:
        load_img_directory = directory

# Set the default directory for saving CSV files
def set_csv_dir():
    global csv_directory
    directory = filedialog.askdirectory(title="Select CSV Directory")
    if directory:
        csv_directory = directory

# Set the default directory for saving plot images
def set_img_dir():
    global img_directory
    directory = filedialog.askdirectory(title="Select Image Directory")
    if directory:
        img_directory = directory

# Load and display an image file, set up the plotting interface and coordinate collection
def load_image():
    global coordinates, img, img_height, img_width, fig, ax, canvas, toolbar, text_frame, scale
    
    try:
        # Reset coordinates when loading a new image
        for key in coordinates:
            coordinates[key] = None
        
        selected_image_path = select_image_file()
        img = mpimg.imread(selected_image_path)
        img_height, img_width = img.shape[:2]
        
        fig, ax = plt.subplots()
        ax.imshow(img)
        
        # Get the first coordinate name that needs to be filled
        first_coord_name = next(iter(coordinates.keys()))
        ax.set_title(f"Select {first_coord_name} by pressing the spacebar.")
        
        # Create main frame for layout
        main_frame = Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create left frame for image and toolbar
        left_frame = Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create button frame above the canvas
        button_frame = Frame(left_frame)
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Add Undo, Set Scale, and Submit buttons
        global undo_button, scale_button, submit_button
        if undo_button:
            undo_button.destroy()
        if scale_button:
            scale_button.destroy()
        if submit_button:
            submit_button.destroy()
            
        undo_button = tk.Button(button_frame, text="Undo Last Point", command=undo_last_point, 
                               bg="gray", fg="darkgray", font=("Arial", 10, "bold"), state=tk.DISABLED)
        undo_button.pack(side=tk.LEFT, padx=5, expand=True)

        scale_button = tk.Button(button_frame, text="Set Scale", command=get_scale, 
                                bg="lightblue", fg="black", font=("Arial", 10, "bold"))
        scale_button.pack(side=tk.LEFT, padx=5, expand=True)

        submit_button = tk.Button(button_frame, text="Submit Points", command=submit_points,
                                 bg="gray", fg="darkgray", font=("Arial", 10, "bold"), state=tk.DISABLED)
        submit_button.pack(side=tk.LEFT, padx=5, expand=True)

        scale = 0

        # Embed the Matplotlib figure in Tkinter
        if canvas is not None:
            canvas.get_tk_widget().pack_forget()
        canvas = FigureCanvasTkAgg(fig, master=left_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add the navigation toolbar below the canvas (in the left frame)
        if toolbar is not None:
            toolbar.pack_forget()
        toolbar = NavigationToolbar2Tk(canvas, left_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Create text frame for coordinates table
        if text_frame is not None:
            text_frame.pack_forget()
        text_frame = Frame(main_frame, width=360)
        text_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        text_frame.pack_propagate(False)

        canvas.draw()

        initialize_table()
        get_coordinates()

        # Set plot limits based on image size
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)

    except ValueError as e:
        messagebox.showerror("Error", f"Failed to load image: {str(e)}")

# Open file dialog to select an image file for analysis
def select_image_file():
    global load_img_directory
    if load_img_directory:
        initial_dir = load_img_directory
    else:
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

# Prompt user to enter the measurement scale in millimeters
def get_scale():
    global scale
    scale_value = simpledialog.askinteger("Set Scale", "Enter the scale (must be an integer):", parent=root)
    if scale_value is not None:
        scale = scale_value

# Initialize the coordinate display text widget with headers and empty coordinate slots
def initialize_table():
    global table_text
    # Create scrollbar
    scrollbar = Scrollbar(text_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Create text widget with monospaced font
    table_text = Text(text_frame, 
                     font=("Courier New", 12),
                     wrap=tk.WORD,
                     yscrollcommand=scrollbar.set,
                     state=tk.DISABLED)
    table_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Configure scrollbar
    scrollbar.config(command=table_text.yview)
    
    # Create initial table content
    table_content = "   Name".ljust(10) + "| Coordinates\n"
    table_content += "-" * 27 + "\n"
    
    for name in coordinates.keys():
        table_content += name.ljust(10) + "|    ( , )\n"

    # Insert content
    table_text.config(state=tk.NORMAL)
    table_text.insert(tk.END, table_content)
    table_text.config(state=tk.DISABLED)

# Update the coordinate display with current coordinate values
def update_table():
    global table_text
    # Build the table content
    table_content = "   Name".ljust(10) + "| Coordinates\n"
    table_content += "-" * 27 + "\n"
    
    for name, coord in coordinates.items():
        if coord is not None:
            coord_text = f"({coord[0]:.4g}, {coord[1]:.4g})"
        else:
            coord_text = "    ( , )"
        table_content += name.ljust(10) + "|" + coord_text + "\n"
    
    # Update text widget
    table_text.config(state=tk.NORMAL)
    table_text.delete(1.0, tk.END)
    table_text.insert(tk.END, table_content)
    table_text.config(state=tk.DISABLED)

# Update the visual plot points and labels on the image display
def update_plot_points():
    global plot_points, text_annotations, ax, canvas
    
    for point in plot_points:
        point.remove()
    for text in text_annotations:
        text.remove()

    plot_points.clear()
    text_annotations.clear()

    for name, coord in coordinates.items():
        if coord is not None:
            x, y = coord
            plot_points.append(ax.plot(x, y, 'x', color='lime', markersize=6, markeredgewidth=1)[0])
            text_annotations.append(ax.text(x, y, name, color='lime', fontsize=10, ha='right', va='bottom'))

    canvas.draw()

# Save both the plot image and CSV data with a user-specified filename
def save_both():
    filename = simpledialog.askstring("Save Plot", "Enter the filename (without extension):")
    if filename:
        save_plot(filename)
        save_csv(filename)

# Save the current plot as a PNG image file
def save_plot(filename):
    global img_directory, fig
    if not img_directory:
        messagebox.showerror("Error", "Image directory is not set! Please configure the directory first.")
        return

    file_path = os.path.join(img_directory, f"{filename}.png")
    fig.savefig(file_path)

# Save the calculation results as a CSV file
def save_csv(filename):
    global csv_directory, combined_table
    if not csv_directory:
        messagebox.showerror("Error", "CSV directory is not set! Please configure the directory first.")
        return
    
    csv_file_path = os.path.join(csv_directory, f"{filename}.csv")
    
    with open(csv_file_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([filename] + list(combined_table.keys()))
        writer.writerow([filename] + list(combined_table.values()))        

# Update the visual state of the Undo and Submit buttons based on current progress
def update_button_states():
    global submit_button, undo_button, coordinates
    
    # Count filled coordinates
    filled_count = sum(1 for coord in coordinates.values() if coord is not None)
    total_count = len(coordinates)
    
    if submit_button:
        if filled_count == total_count:
            submit_button.config(state=tk.NORMAL, bg="lightgreen", fg="black")
        else:
            submit_button.config(state=tk.DISABLED, bg="gray", fg="darkgray")
    
    if undo_button:
        if filled_count > 0:
            undo_button.config(state=tk.NORMAL, bg="lightcoral", fg="white")
        else:
            undo_button.config(state=tk.DISABLED, bg="gray", fg="darkgray")

# Remove the last clicked coordinate point and update the display
def undo_last_point():
    global coordinates, ax, canvas
    
    # Find the last filled coordinate
    last_filled_name = None
    for name, coord in reversed(coordinates.items()):
        if coord is not None:
            last_filled_name = name
            break
    
    if last_filled_name:
        removed_coord = coordinates[last_filled_name]
        coordinates[last_filled_name] = None
        
        update_plot_points()
        update_table()
        update_button_states()
        
        # Find next coordinate to fill
        next_coord_name = None
        for name, coord in coordinates.items():
            if coord is None:
                next_coord_name = name
                break
        
        if next_coord_name:
            ax.set_title(f"Select {next_coord_name} by pressing the spacebar.")
        
        canvas.draw()
    else:
        messagebox.showinfo("Undo", "No points to undo!")

# Display the calculation results in a formatted table within the text widget
def display_results_in_textbox():
    global table_text, combined_table
    
    if not combined_table or not table_text:
        return
    
    # Create formatted table content for results
    results_content = "\nCALCULATION RESULTS\n\n"
    
    # Format the combined_table as a nice table
    table_data = []
    for key, value in combined_table.items():
        # Format the value to handle different data types
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)
        table_data.append([key, formatted_value])
    
    # Use tabulate to create a nicely formatted table
    formatted_table = tabulate(table_data, headers=["Measurement", "Value"], tablefmt="grid")
    results_content += formatted_table
    
    # Update text widget with results
    table_text.config(state=tk.NORMAL)
    table_text.delete(1.0, tk.END)  # Clear existing content
    table_text.insert(tk.END, results_content)
    table_text.config(state=tk.DISABLED)

# Submit all coordinates for calculation and display results
def submit_points():
    global scale, coordinates, ax, canvas, combined_table
    
    if scale <= 0:
        messagebox.showwarning("Scale Not Set", "Please set a valid scale before submitting.")
        get_scale()
        if scale <= 0:
            return
    
    ax.set_title("Calculations completed! Points submitted.")
    # Create a copy of coordinates with only filled values for calculations
    filled_coordinates = {name: coord for name, coord in coordinates.items() if coord is not None}
    combined_table = calculate_all(filled_coordinates, scale)
    display_results_in_textbox()  # Display results in the text box
    canvas.draw()
    
    messagebox.showinfo("Success", "Points submitted and calculations completed!")

# Handle the plot window close event
def on_close(event):
    global fig
    plt.close(fig)

# Handle keyboard input events for coordinate selection
def on_key(event):
    global coordinates, scale, ax, canvas
    
    if event.key == ' ':
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            # Find the next coordinate to fill
            next_coord_name = None
            for name, coord in coordinates.items():
                if coord is None:
                    next_coord_name = name
                    break
            
            if next_coord_name:
                # Add the coordinate to the dictionary
                coordinates[next_coord_name] = (x, y)
                
                update_plot_points()
                update_table()
                update_button_states()

                # Find next unfilled coordinate
                next_unfilled = None
                for name, coord in coordinates.items():
                    if coord is None:
                        next_unfilled = name
                        break
                
                if next_unfilled:
                    ax.set_title(f"Select {next_unfilled} by pressing the spacebar.")
                else:
                    ax.set_title("All coordinates selected. Use Submit Points button to calculate measurements.")
                
                canvas.draw()

# Set up event handlers for coordinate input and plot interaction
def get_coordinates():
    global fig
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('close_event', on_close)

# Initialize and launch the main application window
def launch_application():
    global root
    try:
        root = tk.Tk()
        root.title("Image Coordinate Tool")
        
        # Set initial window size and center it on screen
        window_width = 800
        window_height = 600
        
        # Get screen dimensions
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Calculate position to center window
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        
        # Set window size and position
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # Set minimum window size so it can't be made too small
        root.minsize(800, 600)
        
        create_menu()
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Application Error", f"Failed to launch application: {str(e)}")

if __name__ == "__main__":
    launch_application()