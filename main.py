import streamlit as st
import numpy as np
import plotly.graph_objects as go

def create_simple_cubic(size=1.0):
    # Vertices
    vertices = np.array([
        [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],
        [0, 0, size], [size, 0, size], [size, size, size], [0, size, size]
    ])
    
    # Edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]
    
    return vertices, edges

def create_bcc(size=1.0):
    # Get simple cubic vertices and edges
    vertices, edges = create_simple_cubic(size)
    
    # Add center point
    center = np.array([[size/2, size/2, size/2]])
    vertices = np.vstack([vertices, center])
    
    # Add edges from center to corners
    center_idx = len(vertices) - 1
    new_edges = [[center_idx, i] for i in range(8)]
    edges.extend(new_edges)
    
    return vertices, edges

def create_octet(size=1.0):
    vertices = np.array([
        [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],
        [0, 0, size], [size, 0, size], [size, size, size], [0, size, size],
        [size/2, size/2, 0], [size/2, 0, size/2], [0, size/2, size/2],
        [size, size/2, size/2], [size/2, size, size/2]
    ])
    
    edges = [
        [0, 8], [8, 1], [1, 9], [9, 5], [5, 11], [11, 2],
        [2, 12], [12, 6], [6, 7], [7, 10], [10, 3], [3, 8],
        [0, 9], [9, 4], [4, 10], [10, 0],
        [1, 11], [11, 5], [5, 12], [12, 2]
    ]
    
    return vertices, edges

def create_lattice(unit_cell_type, grid_size, cell_size=1.0):
    # Get unit cell function
    cell_functions = {
        'Simple Cubic': create_simple_cubic,
        'Body-Centered Cubic': create_bcc,
        'Octet Truss': create_octet
    }
    
    cell_func = cell_functions[unit_cell_type]
    
    all_vertices = []
    all_edges = []
    vertex_count = 0
    
    # Create grid of unit cells
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                # Get base unit cell
                vertices, edges = cell_func(cell_size)
                
                # Translate vertices
                translated_vertices = vertices + np.array([x, y, z]) * cell_size
                all_vertices.extend(translated_vertices)
                
                # Update edges with new vertex indices
                translated_edges = [[e[0] + vertex_count, e[1] + vertex_count] for e in edges]
                all_edges.extend(translated_edges)
                
                vertex_count += len(vertices)
    
    return np.array(all_vertices), all_edges

def plot_lattice(vertices, edges, strut_thickness):
    # Create lines for edges
    x_lines = []
    y_lines = []
    z_lines = []
    
    for edge in edges:
        start, end = edge
        x_lines.extend([vertices[start, 0], vertices[end, 0], None])
        y_lines.extend([vertices[start, 1], vertices[end, 1], None])
        z_lines.extend([vertices[start, 2], vertices[end, 2], None])
    
    # Create color array based on position
    colors = np.zeros(len(x_lines))
    idx = 0
    for edge in edges:
        start, end = edge
        pos = vertices[start] + vertices[end]
        colors[idx:idx+3] = np.sum(pos) % 8  # Create color variation
        idx += 3
    
    # Create the 3D plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode='lines',
        line=dict(
            color=colors,
            width=strut_thickness,
            colorscale='Viridis'
        )
    )])
    
    # Update layout
    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig

# Streamlit app
st.title("Beam-based Lattice Structure Visualizer")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    
    unit_cell = st.selectbox(
        "Select Unit Cell Type",
        ["Simple Cubic", "Body-Centered Cubic", "Octet Truss"]
    )
    
    grid_size = st.selectbox(
        "Grid Size",
        [2, 4],
        format_func=lambda x: f"{x}x{x}x{x}"
    )
    
    strut_thickness = st.slider(
        "Strut Thickness",
        min_value=1,
        max_value=10,
        value=5
    )

# Generate and display lattice
vertices, edges = create_lattice(unit_cell, grid_size)
fig = plot_lattice(vertices, edges, strut_thickness)
st.plotly_chart(fig, use_container_width=True)

# Add information
st.markdown("""
### Instructions
1. Use the sidebar to select the unit cell type
2. Choose the grid size (2x2x2 or 4x4x4)
3. Adjust the strut thickness
4. Interact with the 3D plot:
   - Rotate: Click and drag
   - Zoom: Scroll
   - Pan: Right-click and drag
""")
