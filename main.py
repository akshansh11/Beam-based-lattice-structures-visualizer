import streamlit as st
import numpy as np
import plotly.graph_objects as go

def create_simple_cubic(size=1.0):
    """Create Simple Cubic unit cell"""
    vertices = np.array([
        [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],
        [0, 0, size], [size, 0, size], [size, size, size], [0, size, size]
    ])
    
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]
    
    return vertices, edges

def create_bcc(size=1.0):
    """Create Body-Centered Cubic unit cell"""
    vertices, edges = create_simple_cubic(size)
    
    center = np.array([[size/2, size/2, size/2]])
    vertices = np.vstack([vertices, center])
    
    center_idx = len(vertices) - 1
    new_edges = [[center_idx, i] for i in range(8)]
    edges.extend(new_edges)
    
    return vertices, edges

def create_fcc(size=1.0):
    """Create Face-Centered Cubic unit cell"""
    # Corner vertices
    vertices = np.array([
        [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],
        [0, 0, size], [size, 0, size], [size, size, size], [0, size, size]
    ])
    
    # Face center vertices
    face_centers = np.array([
        [size/2, size/2, 0],    # bottom
        [size/2, size/2, size], # top
        [size/2, 0, size/2],    # front
        [size/2, size, size/2], # back
        [0, size/2, size/2],    # left
        [size, size/2, size/2]  # right
    ])
    
    vertices = np.vstack([vertices, face_centers])
    
    edges = []
    # Connect face centers to corners
    for i in range(8):
        for j in range(8, 14):
            dist = np.linalg.norm(vertices[i] - vertices[j])
            if np.isclose(dist, size/np.sqrt(2)):
                edges.append([i, j])
    
    return vertices, edges

def create_octet(size=1.0):
    """Create Octet Truss unit cell"""
    f = size / 2.0
    
    vertices = np.array([
        # Cube vertices
        [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],  # bottom
        [0, 0, size], [size, 0, size], [size, size, size], [0, size, size],  # top
        
        # Octahedron centers
        [f, f, 0],     # bottom center
        [f, f, size],  # top center
        [f, 0, f],     # front center
        [f, size, f],  # back center
        [0, f, f],     # left center
        [size, f, f],  # right center
        [f, f, f]      # middle center
    ])
    
    edges = [
        # Octahedron edges
        [14, 8], [14, 9], [14, 10], [14, 11], [14, 12], [14, 13],
        
        # Tetrahedron edges
        [0, 8], [1, 8], [2, 8], [3, 8],  # bottom
        [4, 9], [5, 9], [6, 9], [7, 9],  # top
        [0, 10], [1, 10], [4, 10], [5, 10],  # front
        [2, 11], [3, 11], [6, 11], [7, 11],  # back
        [0, 12], [3, 12], [4, 12], [7, 12],  # left
        [1, 13], [2, 13], [5, 13], [6, 13]   # right
    ]
    
    return vertices, edges

def create_diamond(size=1.0):
    """Create Diamond unit cell"""
    f = size/4
    
    vertices = np.array([
        [0, 0, 0],
        [size, size, 0],
        [size, 0, size],
        [0, size, size],
        # Second tetrahedral group
        [size/2, size/2, size/2],
        [0, 0, size],
        [size, 0, 0],
        [0, size, 0],
        [size, size, size]
    ])
    
    edges = [
        # First tetrahedron
        [0, 1], [0, 2], [0, 3],
        [1, 2], [2, 3], [3, 1],
        # Second tetrahedron
        [4, 5], [4, 6], [4, 7], [4, 8],
        # Inter-tetrahedral connections
        [0, 4], [1, 4], [2, 4], [3, 4]
    ]
    
    return vertices, edges

def create_tetrahedral(size=1.0):
    """Create Tetrahedral unit cell"""
    h = size * np.sqrt(3)/2
    vertices = np.array([
        [0, 0, 0],
        [size, 0, 0],
        [size/2, h, 0],
        [size/2, h/3, h]
    ])
    
    edges = [
        [0, 1], [1, 2], [2, 0],  # base
        [0, 3], [1, 3], [2, 3]   # to apex
    ]
    
    return vertices, edges

def create_kagome(size=1.0):
    """Create Kagome lattice"""
    f = size/2
    
    vertices = np.array([
        # Base hexagon
        [f, 0, 0], [f+f*np.cos(np.pi/3), f*np.sin(np.pi/3), 0],
        [f+f*np.cos(2*np.pi/3), f*np.sin(2*np.pi/3), 0],
        [f, size, 0], [f-f*np.cos(2*np.pi/3), f*np.sin(2*np.pi/3), 0],
        [f-f*np.cos(np.pi/3), f*np.sin(np.pi/3), 0],
        # Centers
        [f, f, 0],
        [f, f, size],
        # Top hexagon
        [f, 0, size], [f+f*np.cos(np.pi/3), f*np.sin(np.pi/3), size],
        [f+f*np.cos(2*np.pi/3), f*np.sin(2*np.pi/3), size],
        [f, size, size], [f-f*np.cos(2*np.pi/3), f*np.sin(2*np.pi/3), size],
        [f-f*np.cos(np.pi/3), f*np.sin(np.pi/3), size]
    ])
    
    edges = []
    # Connect hexagon vertices
    for i in range(6):
        edges.append([i, (i+1)%6])
        edges.append([i+8, ((i+1)%6)+8])
    # Connect to centers
    for i in range(6):
        edges.append([i, 6])
        edges.append([i+8, 7])
    # Connect layers
    for i in range(6):
        edges.append([i, i+8])
    
    return vertices, edges

def create_lattice(unit_cell_type, grid_size, cell_size=1.0):
    """Create complete lattice structure"""
    cell_functions = {
        'Simple Cubic': create_simple_cubic,
        'Body-Centered Cubic': create_bcc,
        'Face-Centered Cubic': create_fcc,
        'Octet Truss': create_octet,
        'Diamond': create_diamond,
        'Tetrahedral': create_tetrahedral,
        'Kagome': create_kagome
    }
    
    cell_func = cell_functions[unit_cell_type]
    
    all_vertices = []
    all_edges = []
    vertex_count = 0
    
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                vertices, edges = cell_func(cell_size)
                translated_vertices = vertices + np.array([x, y, z]) * cell_size
                all_vertices.extend(translated_vertices)
                translated_edges = [[e[0] + vertex_count, e[1] + vertex_count] for e in edges]
                all_edges.extend(translated_edges)
                vertex_count += len(vertices)
    
    return np.array(all_vertices), all_edges

def calculate_strut_lengths(vertices, edges):
    """Calculate lengths of all struts"""
    lengths = []
    for edge in edges:
        start, end = edge
        vec = vertices[end] - vertices[start]
        length = np.sqrt(np.sum(vec**2))
        lengths.append(length)
    return np.array(lengths)

def plot_lattice(vertices, edges, strut_thickness, colorscale='Viridis'):
    """Create interactive 3D plot"""
    x_lines = []
    y_lines = []
    z_lines = []
    
    for edge in edges:
        start, end = edge
        x_lines.extend([vertices[start, 0], vertices[end, 0], None])
        y_lines.extend([vertices[start, 1], vertices[end, 1], None])
        z_lines.extend([vertices[start, 2], vertices[end, 2], None])
    
    colors = np.zeros(len(x_lines))
    idx = 0
    for edge in edges:
        start, end = edge
        pos = vertices[start] + vertices[end]
        colors[idx:idx+3] = np.sum(pos) % 8
        idx += 3
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode='lines',
        line=dict(
            color=colors,
            width=strut_thickness,
            colorscale=colorscale
        )
    )])
    
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
st.set_page_config(layout="wide", page_title="Lattice Structure Visualizer")
st.title("Advanced Beam-based Lattice Structure Visualizer")

# Create two columns
col1, col2 = st.columns([1, 3])

# Control panel
with col1:
    st.header("Settings")
    
    with st.expander("Unit Cell Settings", expanded=True):
        unit_cell = st.selectbox(
            "Select Unit Cell Type",
            ["Simple Cubic", "Body-Centered Cubic", "Face-Centered Cubic", 
             "Octet Truss", "Diamond", "Tetrahedral", "Kagome"]
        )
        
        st.write("""
        **Unit Cell Types:**
        - Simple Cubic: Basic cubic structure
        - Body-Centered Cubic: Added central node
        - Face-Centered Cubic: Nodes at face centers
        - Octet Truss: Complex triangulated structure
        - Diamond: Based on diamond crystal structure
        - Tetrahedral: Basic tetrahedral arrangement
        - Kagome: Hexagonal-prismatic structure
        """)
    
    with st.expander("Visualization Settings", expanded=True):
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
        
        colorscale = st.selectbox(
            "Color Scheme",
            ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Rainbow']
        )
    
    with st.expander("Structure Analysis", expanded=True):
        vertices, edges = create_lattice(unit_cell, grid_size)
        strut_lengths = calculate_strut_lengths(vertices, edges)
        
        st.write(f"Total struts: {len(edges)}")
        st.write(f"Total nodes: {len(vertices)}")
        st.write(f"Average strut length: {strut_lengths.mean():.2f}")
        st.write(f"Strut length std dev: {strut_lengths.std():.2f}")

# Visualization area
with col2:
    vertices, edges = create_lattice(unit_cell, grid_size)
    fig = plot_lattice(vertices, edges, strut_thickness, colorscale)
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Instructions", expanded=False):
        st.markdown("""
        ### How to Use
        1. Select unit cell type from the sidebar
        2. Choose grid size (2x2x2 or 4x4x4)
        3. Adjust visualization settings:
           - Strut thickness
           - Color scheme
        
        ### Interaction
        - Rotate: Click and drag
        - Zoom: Scroll
        - Pan: Right-click and drag
        
        ### Tips
        - Start with 2x2x2 grid for faster visualization
        - Use 4x4x4 grid to see more detailed patterns
        - Adjust strut thickness for better visibility
        - Try different color schemes for better contrast
        """)

# Footer
st.markdown("---")
st.markdown("Created for visualization of advanced beam-based lattice structures")
