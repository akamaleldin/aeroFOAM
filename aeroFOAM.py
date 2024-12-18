#!/usr/bin/env python
# coding: utf-8

# All the stuff that needs to be imported, we will import it in the next cell

# In[188]:


import sys
import numpy as np
import math
import tkinter as tk
import os
import shutil
import subprocess
import ctypes

from PyQt5.QtWidgets import QApplication, QFileDialog
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
from tkinter import filedialog
from PIL import Image, ImageTk

from expansionRatioCalc import calculate_total_expansion_ratio


# First things first, import the airfoil .dat file (later, we want to accomodate other file types like .csv .xlsx .txt ...etc)

# Then, we have to extract the information for the airfoil .dat file. x-coordinate of the maximum thickness, y-coordinate, ...etc

# In[191]:


def interpolate(y1, y2, x1, x2, x):
    """
    Perform linear interpolation to calculate the y value at a given x.
    """
    return y1 + ((x - x1) * (y2 - y1) / (x2 - x1))

def analyze_airfoil(file_path):
    """
    Analyze an airfoil .dat file to determine key points and split coordinates into groups.

    Parameters:
        file_path (str): Path to the airfoil .dat file.

    Returns:
        tuple: Coordinates of maximum y (x_max, y_max), interpolated minimum y, and grouped coordinates.
    """
    # Load the airfoil data from the file
    data = np.loadtxt(file_path, skiprows=1)  # Assuming the first row is a header
    x_coords = data[:, 0]
    y_coords = data[:, 1]

    # Change the very first and last y-coordinates to 0
    y_coords[0] = 0.0
    y_coords[-1] = 0.0

    # Identify the maximum y-coordinate and its corresponding x-coordinate
    max_index = np.argmax(y_coords)
    x_max = x_coords[max_index]
    y_max = y_coords[max_index]

    # Find the x-coordinates around x_max for interpolation of minimum y
    lower_indices = np.where((x_coords < x_max) & (y_coords < 0))[0]
    upper_indices = np.where((x_coords > x_max) & (y_coords < 0))[0]

    if len(lower_indices) > 0 and len(upper_indices) > 0:
        # Get the closest points to x_max for interpolation
        x1_idx = lower_indices[-1]
        x2_idx = upper_indices[0]

        x1 = x_coords[x1_idx]
        y1 = y_coords[x1_idx]

        x2 = x_coords[x2_idx]
        y2 = y_coords[x2_idx]

        # Interpolate the minimum y-coordinate at x_max
        y_interpolated_min = np.interp(x_max, [x1, x2], [y1, y2])
    else:
        y_interpolated_min = np.nan  # Fallback if interpolation is not possible

    # Prepare formatted data and split into groups
    formatted_data = [f"({x:.6f} {y:.6f} 0.000000)" for x, y in zip(x_coords, y_coords)]
    formatted_data_1 = [f"({x:.6f} {y:.6f} 1)" for x, y in zip(x_coords, y_coords)]

    groups = {
        "group_1": [],
        "group_2": [],
        "group_3": [],
        "group_4": [],
        "group_5": [],
        "group_6": [],
        "group_7": [],
        "group_8": []
    }

    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        if x_max < x < 1 and y > 0:
            groups["group_1"].append(formatted_data[i])
            groups["group_5"].append(formatted_data_1[i])
        elif 0 < x < x_max and y > 0:
            groups["group_2"].append(formatted_data[i])
            groups["group_6"].append(formatted_data_1[i])
        elif 0 < x < x_max and y < 0:
            groups["group_3"].append(formatted_data[i])
            groups["group_7"].append(formatted_data_1[i])
        elif x_max < x < 1 and y < 0:
            groups["group_4"].append(formatted_data[i])
            groups["group_8"].append(formatted_data_1[i])

    # Prepare key points
    key_points = {
        "max_point": {"x": x_max, "y": y_max},
        "min_point": {"x": x_max, "y": y_interpolated_min}
    }

    return key_points, groups


# Then, we will simply calculate the first layer thickness given the Reynolds Number

# In[193]:


def calculate_delta_s(Re):
    L = 1  # Characteristic length (m)
    rho = 1.0  # Density of air (kg/m^3)
    mu = 1e-3  # Dynamic viscosity (kg/m.s)
    y_plus = 0.1  # Dimensionless wall distance
    U_inf = (Re * mu) / (L * rho)  # Free stream velocity (m/s)
    
    # Calculate C_f from Reynolds number
    C_f = 0.026 / (Re ** (1/7))
    
    # Calculate tau_wall
    tau_wall = (C_f * rho * U_inf**2) / 2
    
    # Calculate U_fric
    U_fric = math.sqrt(tau_wall / rho)
    
    # Calculate Delta_s
    delta_s = (y_plus * mu) / (U_fric * rho)

    if delta_s >= 1e-3:
        delta_s = 1e-3
    
    return delta_s


# Next, we will calculate the thickness of the of each layer of the 50 layers!

# In[195]:


def calculate_layer_thickness(upstream_length, first_layer_thickness):
    """
    Calculate the thickness of layers incremented by 1.2 until the total thickness
    is less than 10% of the upstream length.

    Parameters:
        upstream_length (float): Length of the upstream region.
        first_layer_thickness (float): Initial thickness of the first layer.

    Returns:
        tuple: A tuple containing:
            - int: The number of layers generated.
            - float: The total thickness of the layers.
    """
    # Initialize variables
    total_thickness = 0  # Total sum of layers' thickness
    layer_thickness = first_layer_thickness  # Initial first layer thickness
    layers = []  # List to store the thickness of each layer

     # Loop to calculate the layers until the total thickness is less than 10% of the upstream length
    while len(layers) < 50:
        layers.append(layer_thickness)
        total_thickness += layer_thickness
        layer_thickness *= 1.05  # Increase the thickness of the next layer by 1.2

    return total_thickness, layers


# Next, calculate the number of cells in each block

# In[197]:


def calculate_spline_length(coordinates):
    """
    Calculates the length of the spline defined by a list of coordinates.

    Parameters:
        coordinates (list of str): List of coordinates in the format '(x y z)'.

    Returns:
        float: The total length of the spline.
    """
    # Parse the coordinates into a list of tuples (x, y, z)
    points = []
    for coord in coordinates:
        # Remove parentheses and split into x, y, z
        x, y, z = map(float, coord.strip('()').split())
        points.append((x, y, z))
    
    # Calculate the total length by summing distances between consecutive points
    total_length = 0.0
    for i in range(len(points) - 1):
        x1, y1, z1 = points[i]
        x2, y2, z2 = points[i + 1]
        # Calculate the Euclidean distance between points
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        total_length += distance

    return total_length

def calculate_cell_length(first_layer_thickness):
    #thresholds = [
        #(1e-3, first_layer_thickness * 10),         # (Threshold, Cell Length)
        #(1e-4, first_layer_thickness * 100),
        #(1e-5, first_layer_thickness * 200),
        #(1e-6, first_layer_thickness * 300),
        #(1e-7, first_layer_thickness * 500)
    #]
    
    #for threshold, cell_value in thresholds:
        #if first_layer_thickness >= threshold:
            #return cell_value

    # Default case for first_layer_thickness < 1e-7
    return 0.0013 * math.log(first_layer_thickness) + 0.0187


def calculate_cells(upstream_length, downstream_length, boundary_layer, key_points, first_layer_thickness, groups):
    """
    Calculate the number of cells in the x1 and x2 directions.

    Parameters:
        upstream_length (float): Length of the upstream region (x-axis).
        downstream_length (float): Length of the downstream region (x-axis).

    Returns:
        tuple: A tuple containing:
            - int: Number of cells in the x1 direction.
            - int: Number of cells in the x2 direction.
    """
    # Calculate the length of the each cell on the airfoil
    
    airfoil_cell_length = calculate_cell_length(first_layer_thickness)

    downstream_list = [airfoil_cell_length]
    
    while sum(downstream_list) < (downstream_length - 1):

        if downstream_list[-1]*1.05 / first_layer_thickness <= 10000:
            downstream_list.append(downstream_list[-1]*1.05)

        else:
            downstream_list.append(first_layer_thickness * 10000)

    num_downstream_cells = len(downstream_list)
    

    # Prepare the groups and their respective start and end points
    groups = {
        "group_1": {"data": groups["group_1"], "start": "(1.0000 0.0000 0.0000)", "end": f"({key_points['max_point']['x']} {key_points['max_point']['y']} 0.0000)"},
        "group_2": {"data": groups["group_2"], "start": f"({key_points['max_point']['x']} {key_points['max_point']['y']} 0.0000)", "end": "(0.0000 0.0000 0.0000)"},
        "group_3": {"data": groups["group_3"], "start": "(0.0000 0.0000 0.0000)", "end": f"({key_points['min_point']['x']} {key_points['min_point']['y']} 0.0000)"},
        "group_4": {"data": groups["group_4"], "start": f"({key_points['min_point']['x']} {key_points['min_point']['y']} 0.0000)", "end": "(1.0000 0.0000 0.0000)"},
    }
    
    # Initialize results dictionary
    group_distances = {}
    group_num_cells = {}
    
    # Process each group dynamically
    for group_name, group_info in groups.items():
        # Make a local copy of the group data
        group_local = group_info["data"].copy()
        
        # Append start and end points
        group_local.insert(0, group_info["start"])
        group_local.append(group_info["end"])
        
        # Calculate distance
        group_distance = calculate_spline_length(group_local)
        group_distances[group_name] = group_distance
        
        # Calculate number of cells
        group_num_cells[group_name] = int(group_distance / airfoil_cell_length)

    
    total_upstream_list =  boundary_layer.copy()

    while sum(total_upstream_list) < upstream_length:

        if total_upstream_list[-1]*1.05 / airfoil_cell_length <= 50:
            total_upstream_list.append(total_upstream_list[-1]*1.05)

        else:
            total_upstream_list.append(airfoil_cell_length * 50)
            

    num_upstream_cells = len(total_upstream_list)

    return num_upstream_cells, num_downstream_cells, group_num_cells, airfoil_cell_length


# Next is to calculate the grading of the cells along the each direction

# In[199]:


def calculate_grading(upstream_length, downstream_length, num_downstream_cells, num_upstream_cells, boundary_layer, total_thickness, airfoil_cell_length):
    """
    Calculate grading values for blockMesh.

    Parameters:
        upstream_length (float): Total upstream length of the domain.
        total_cells_x (int): Total number of cells in the upstream direction.
        layers_list (list): List of layer thicknesses.
        total_thickness (float): Total thickness of the boundary layer.

    Returns:
        dict: Grading values including far_boundary_ratio, close_boundary_ratio, outer_layer_ratio, inner_layer_ratio, and first_to_last_layer_ratio.
    """ 
    
    # Boundary layer thickness ratios
    close_boundary_ratio = total_thickness / upstream_length
    far_boundary_ratio = 1 - close_boundary_ratio

    # Layer distribution ratios
    number_of_layers = len(boundary_layer)
    inner_layer_ratio = number_of_layers / num_upstream_cells
    outer_layer_ratio = 1 - inner_layer_ratio

    # First-to-last layer ratio
    first_layer_thickness = boundary_layer[0]
    last_layer_thickness = boundary_layer[-1]
    first_to_last_layer_ratio = first_layer_thickness / last_layer_thickness

    outer_first_layer_thickness = 1.05 * last_layer_thickness
    outer_num_cells = outer_layer_ratio * num_upstream_cells
    outer_length = far_boundary_ratio * upstream_length

    outer_expansion_ratio = calculate_total_expansion_ratio(outer_num_cells, outer_first_layer_thickness, outer_length)

    if outer_expansion_ratio > 1:
        outer_expansion_ratio = 1 / outer_expansion_ratio
    else:
        outer_expansion_ratio = outer_expansion_ratio

    downstream_expansion_ratio = calculate_total_expansion_ratio(num_downstream_cells, airfoil_cell_length, (downstream_length - 1))

    return {
        "far_boundary_ratio": far_boundary_ratio,
        "close_boundary_ratio": close_boundary_ratio,
        "outer_layer_ratio": outer_layer_ratio,
        "inner_layer_ratio": inner_layer_ratio,
        "first_to_last_layer_ratio": first_to_last_layer_ratio,
        "outer_expansion_ratio": outer_expansion_ratio,
        "downstream_expansion_ratio": downstream_expansion_ratio
    }


# Next, we will build the blockMeshDict

# In[201]:


def generate_blockMeshDict(upstream_length, downstream_length, num_upstream_cells, num_downstream_cells, key_points, groups, group_num_cells, grading):
    """
    Generate the FoamFile, vertices, and blocks sections of the blockMeshDict file.

    Parameters:
        upstream_length (float): Length of the domain upstream of the airfoil.
        downstream_length (float): Length of the domain downstream of the airfoil.
        reynolds_number (float): Reynolds number for flow.

    Returns:
        str: The FoamFile, vertices, and blocks section as a string.
    """
    radius  =  upstream_length + key_points['max_point']['x']
    
    # FoamFile header
    foam_file_header = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
"""

    # Vertices definition
    vertices = f"""
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

scale 1;

vertices
(
    // back
    name v0  ({downstream_length:.6f}    {radius:.6f}         0)
    name v1  (1                {radius:.6f}         0)
    name v2  ({key_points['max_point']['x']:.6f}    {radius:.6f}         0)
    name v3  ({-upstream_length:.6f}  0                     0)

    name v4  ({key_points['max_point']['x']:.6f}    {-radius:.6f}     0)
    name v5  (1                {-radius:.6f}     0)
    name v6  ({downstream_length:.6f}    {-radius:.6f}     0)
    name v7  ({downstream_length:.6f}    0                     0)

    name v8  (1                0                     0)
    name v9  ({key_points['max_point']['x']:.6f}    {key_points['max_point']['y']:.6f}    0)
    name v10 ({0:.6f}    0                     0)
    name v11 ({key_points['max_point']['x']:.6f}    {key_points['min_point']['y']:.6f} 0)

    // front
    name v12 ({downstream_length:.6f}    {radius:.6f}         1)
    name v13 (1                {radius:.6f}         1)
    name v14 ({key_points['max_point']['x']:.6f}    {radius:.6f}         1)
    name v15 ({-upstream_length:.6f}  0                     1)

    name v16 ({key_points['max_point']['x']:.6f}    {-radius:.6f}     1)
    name v17 (1                {-radius:.6f}     1)
    name v18 ({downstream_length:.6f}    {-radius:.6f}     1)
    name v19 ({downstream_length:.6f}    0                     1)

    name v20 (1                0                     1)
    name v21 ({key_points['max_point']['x']:.6f}    {key_points['max_point']['y']:.6f}    1)
    name v22 ({0:.6f}    0                     1)
    name v23 ({key_points['max_point']['x']:.6f}    {key_points['min_point']['y']:.6f} 1)
);
"""


    # Blocks definition
    blocks = f"""
blocks
(
    hex ( v0 v1 v8 v7 v12 v13 v20 v19 ) ({num_downstream_cells}  {num_upstream_cells} {1})
    simpleGrading  ({1/grading["downstream_expansion_ratio"]}
                    (
                        ({grading["far_boundary_ratio"]} {grading["outer_layer_ratio"]} {grading["outer_expansion_ratio"]}) // far to airfoil
                        ({grading["close_boundary_ratio"]} {grading["inner_layer_ratio"]} {grading["first_to_last_layer_ratio"]}) // close to airfoil
                    )    
                    1)

    hex ( v1 v2 v9 v8 v13 v14 v21 v20 ) ({group_num_cells["group_1"]} {num_upstream_cells} {1})
    simpleGrading  (1    
                    (
                        ({grading["far_boundary_ratio"]} {grading["outer_layer_ratio"]} {grading["outer_expansion_ratio"]}) // far to airfoil
                        ({grading["close_boundary_ratio"]} {grading["inner_layer_ratio"]} {grading["first_to_last_layer_ratio"]}) // close to airfoil
                    )     
                    1)

    hex ( v2 v3 v10 v9 v14 v15 v22 v21 ) ({group_num_cells["group_2"]} {num_upstream_cells} {1})
    simpleGrading  (1    
                    (
                        ({grading["far_boundary_ratio"]} {grading["outer_layer_ratio"]} {grading["outer_expansion_ratio"]}) // far to airfoil
                        ({grading["close_boundary_ratio"]} {grading["inner_layer_ratio"]} {grading["first_to_last_layer_ratio"]}) // close to airfoil
                    )     
                    1)

    hex ( v3 v4 v11 v10 v15 v16 v23 v22) ({group_num_cells["group_3"]} {num_upstream_cells} {1})
    simpleGrading  (1    
                    (
                        ({grading["far_boundary_ratio"]} {grading["outer_layer_ratio"]} {grading["outer_expansion_ratio"]}) // far to airfoil
                        ({grading["close_boundary_ratio"]} {grading["inner_layer_ratio"]} {grading["first_to_last_layer_ratio"]}) // close to airfoil
                    )     
                    1)

    hex ( v4 v5 v8 v11 v16 v17 v20 v23) ({group_num_cells["group_4"]} {num_upstream_cells} {1})
    simpleGrading  (1    
                    (
                        ({grading["far_boundary_ratio"]} {grading["outer_layer_ratio"]} {grading["outer_expansion_ratio"]}) // far to airfoil
                        ({grading["close_boundary_ratio"]} {grading["inner_layer_ratio"]} {grading["first_to_last_layer_ratio"]}) // close to airfoil
                    )    
                    1)

    hex ( v5 v6 v7 v8 v17 v18 v19 v20) ({num_downstream_cells} {num_upstream_cells} {1})
    simpleGrading  ({grading["downstream_expansion_ratio"]}    
                    (
                        ({grading["far_boundary_ratio"]} {grading["outer_layer_ratio"]} {grading["outer_expansion_ratio"]}) // far to airfoil
                        ({grading["close_boundary_ratio"]} {grading["inner_layer_ratio"]} {grading["first_to_last_layer_ratio"]}) // close to airfoil
                    )    
                    1)
);
"""


    #Edges Definistion:
    Edges = f"""
edges
(

    arc v2  v3  ({-upstream_length * 0.70710678118 }	{upstream_length * 0.70710678118 }   0)
    arc v14 v15 ({-upstream_length * 0.70710678118 }   {upstream_length * 0.70710678118 }   1)
    
    arc v3  v4  ({-upstream_length * 0.70710678118 }	{-upstream_length * 0.70710678118 }  0)
    arc v15 v16 ({-upstream_length * 0.70710678118 }	{-upstream_length * 0.70710678118 }  1)

    polyLine v8 v9
    (
"""

    # Add tuples from group_1 as arcs under the initial arcs
    for item in groups["group_1"]:
        # Format each tuple as arc vX vY (x_point y_point z)
        Edges += f"       {item}\n"
    Edges += f"""    )
    polyLine v9 v10
    (
    """


    for item in groups["group_2"]:
        # Format each tuple as arc vX vY (x_point y_point z)
        Edges += f"       {item}\n"
    Edges += f"""    )
    polyLine v10 v11
    (
    """



    for item in groups["group_3"]:
        # Format each tuple as arc vX vY (x_point y_point z)
        Edges += f"       {item}\n"
    Edges += f"""    )
    polyLine v11 v8
    (
    """


    for item in groups["group_4"]:
        # Format each tuple as arc vX vY (x_point y_point z)
        Edges += f"       {item}\n"
    Edges += f"""    )
    polyLine v20 v21
    (
    """



    for item in groups["group_5"]:
        # Format each tuple as arc vX vY (x_point y_point z)
        Edges += f"       {item}\n"
    Edges += f"""    )
    polyLine v21 v22
    (
    """




    for item in groups["group_6"]:
        # Format each tuple as arc vX vY (x_point y_point z)
        Edges += f"       {item}\n"
    Edges += f"""    )
    polyLine v22 v23
    (
    """



    for item in groups["group_7"]:
        # Format each tuple as arc vX vY (x_point y_point z)
        Edges += f"       {item}\n"
    Edges += f"""    )
    polyLine v23 v20
    (
    """


    for item in groups["group_8"]:
        # Format each tuple as arc vX vY (x_point y_point z)
        Edges += f"       {item}\n"
    Edges += f"""    )
    """
    # Close the edges string with a closing parenthesis
    Edges += """
);
"""

    # Boundary definition
    Boundary = f"""
boundary
(
    inlet
    {{
        type patch;
        faces
        (
            ( v12 v0 v1 v13)
            ( v13 v1 v2 v14)
            ( v14 v2 v3 v15)
            ( v15 v3 v4 v16)
            ( v17 v5 v4 v16)
            ( v17 v18 v6 v5)
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
            ( v18 v6 v7 v19)
            ( v19 v7 v0 v12)
        );
    }}
    frontandback
    {{
        type empty;
        faces
        (
            (v12 v13 v20 v19)
            (v20 v13 v14 v21)
            (v21 v14 v15 v22)
            (v15 v16 v23 v22)
            (v16 v17 v20 v23)
            (v17 v18 v19 v20)
            
            (v0 v1 v8 v7)
            (v1 v2 v9 v8)
            (v2 v3 v10 v9)
            (v3 v4 v11 v10)
            (v4 v5 v8 v11)
            (v5 v6 v7 v8)

        );
    }}
    airfoil
    {{
        type wall;
        faces
        (
            ( v23 v20 v8 v11)
            ( v20 v8 v9 v21)
            ( v21 v9 v10 v22)
            ( v23 v11 v10 v22)
        );
    }}
);

mergePatchPairs
(
);

// ************************************************************************* // 
"""


    # Combine all sections
    blockMeshDict_content = foam_file_header + vertices + blocks + Edges + Boundary
    
    return blockMeshDict_content


# This is the main part where the program launches

# In[203]:


def check_wsl():
    if not shutil.which("wsl"):
        raise EnvironmentError("WSL is not installed. Please install WSL before proceeding.")

def run_blockMesh_in_wsl(test_dir):
    try:
        # WSL command to source OpenFOAM environment and execute blockMesh
        wsl_command = f"source /usr/lib/openfoam/openfoam2406/etc/bashrc && cd {test_dir} && blockMesh"
        # Replace 'openfoamX' with the actual version number (e.g., openfoam10)
        result = subprocess.run(
            ["wsl", "bash", "-c", wsl_command],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        return result.stdout  # Return the success output
    except subprocess.CalledProcessError as e:
        print(f"Error occurred:\n{e.stderr}")
        raise RuntimeError(f"blockMesh failed with error: {e.stderr or e}")


# In[204]:


# GUI Class
class AirfoilMeshGUI:
    def __init__(self, root):
        self.root = root  # Proper initialization of root
        self.root.title("Airfoil Mesh Generator")
        self.root.geometry("600x300")  # Set smaller window size

        icon_path = "C:/Users/akama/1-50e0a881.ico"
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)

        # Frame for inputs (left side)
        input_frame = ttk.Frame(root)
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        # Add input fields to the left
        ttk.Label(input_frame, text="Upstream Length:").grid(row=0, column=0, sticky="w", pady=5)
        self.upstream_length_entry = ttk.Entry(input_frame)
        self.upstream_length_entry.grid(row=0, column=1, pady=5)

        ttk.Label(input_frame, text="Downstream Length:").grid(row=1, column=0, sticky="w", pady=5)
        self.downstream_length_entry = ttk.Entry(input_frame)
        self.downstream_length_entry.grid(row=1, column=1, pady=5)

        ttk.Label(input_frame, text="Reynolds Number:").grid(row=2, column=0, sticky="w", pady=5)
        self.re_entry = ttk.Entry(input_frame)
        self.re_entry.grid(row=2, column=1, pady=5)

        ttk.Label(input_frame, text="Select Airfoil Data File:").grid(row=3, column=0, sticky="w", pady=5)
        self.airfoil_file_entry = ttk.Entry(input_frame, width=30)
        self.airfoil_file_entry.grid(row=3, column=1, pady=5)

        self.browse_button = ttk.Button(input_frame, text="Browse", command=self.browse_airfoil_file)
        self.browse_button.grid(row=4, column=1, pady=5, sticky="w")

        # Generate mesh button
        self.generate_button = ttk.Button(input_frame, text="Generate Mesh", command=self.generate_mesh)
        self.generate_button.grid(row=5, column=1, pady=5, sticky="nsew")

        # Exit button
        exit_button = ttk.Button(input_frame, text="Exit", command=self.on_exit)
        exit_button.grid(row=6, column=1, pady=5, sticky="nsew")

        # Image display on the right
        image_frame = ttk.Frame(root)
        image_frame.grid(row=0, column=1, padx=10, pady=10, sticky="ne")

        image_path = "C:/Users/akama/aeroFOAM V0.1.png"  # Replace with your image path
        if os.path.exists(image_path):
            self.image = tk.PhotoImage(file=image_path)
            self.image = self.image.subsample(2, 2)  # Adjust the subsample factor (2, 2) to make the image smaller
            self.image_label = ttk.Label(image_frame, image=self.image)
            self.image_label.pack()
        else:
            ttk.Label(image_frame, text="Image not found").pack()


    def on_exit(self):
        """Method to handle exit button click."""
        self.root.quit()  # Gracefully close the Tkinter window.
        self.root.destroy()  # Ensure proper cleanup

    def browse_airfoil_file(self):
        """Method to browse and select the airfoil data file."""
        file_path = filedialog.askopenfilename(title="Select Airfoil Data File", filetypes=[("Data files", "*.dat"), ("All files", "*.*")])
        if file_path:
            self.airfoil_file_entry.delete(0, tk.END)  # Clear any existing text
            self.airfoil_file_entry.insert(0, file_path)  # Insert the selected file path
    
    def generate_mesh(self):
        try:
            # Get the selected airfoil file
            file_path = self.airfoil_file_entry.get().strip()
            if not file_path:
                messagebox.showerror("Error", "No airfoil file selected!")
                return
            
            # Get inputs
            upstream_length = float(self.upstream_length_entry.get())
            downstream_length = float(self.downstream_length_entry.get())
            Re = float(self.re_entry.get())
            
            # Main program logic
            key_points, groups = analyze_airfoil(file_path)
            first_layer_thickness = calculate_delta_s(Re)
            total_thickness, boundary_layer = calculate_layer_thickness(upstream_length, first_layer_thickness)
            num_upstream_cells, num_downstream_cells, group_num_cells, airfoil_cell_length = calculate_cells(
                upstream_length, downstream_length, boundary_layer, key_points, first_layer_thickness, groups
            )
            grading = calculate_grading(upstream_length, downstream_length, num_downstream_cells, num_upstream_cells,
                                        boundary_layer, total_thickness, airfoil_cell_length)
            blockMeshDict_content = generate_blockMeshDict(upstream_length, downstream_length, num_upstream_cells,
                                                           num_downstream_cells, key_points, groups, group_num_cells,
                                                           grading)
            # Store the content
            self.blockMeshDict_content = blockMeshDict_content

            # Folder setup
            test_dir = "test"
            system_dir = os.path.join(test_dir, "system")
            os.makedirs(system_dir, exist_ok=True)

            # Save blockMeshDict
            block_mesh_path = os.path.join(system_dir, "blockMeshDict")
            with open(block_mesh_path, "w") as file:
                file.write(blockMeshDict_content)
            
            # Create a basic controlDict file
            control_dict_content = """\
            FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    application     blockMesh;
    startFrom       startTime;
    startTime       0;
    stopAt          endTime;
    endTime         1;
    deltaT          1;
    writeControl    timeStep;
    writeInterval   1;
    purgeWrite      0;
    writeFormat     ascii;
    writePrecision  6;
    writeCompression off;
    timeFormat      general;
    timePrecision   6;
    runTimeModifiable yes;
    """
            control_dict_path = os.path.join(system_dir, "controlDict")
            if not os.path.exists(control_dict_path):
                with open(control_dict_path, "w") as file:
                    file.write(control_dict_content)

            # Run blockMesh in WSL and capture output
            result = run_blockMesh_in_wsl(test_dir)
            messagebox.showinfo("Success", f"Mesh generation successful!")

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values!")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Failed to generate mesh. Error: {e.stderr}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

# Main function
if __name__ == "__main__":
    root = tk.Tk()
    app = AirfoilMeshGUI(root)
    root.mainloop()


# In[ ]:




