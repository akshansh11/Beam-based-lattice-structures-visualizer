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
    # Factor for proper octahedron spacing
    f = size / 2.0
    
    vertices = np.array([
        # Cube vertices
        [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],  # bottom
        [0, 0, size], [size, 0, size], [size, size, size], [0, size, size],  # top
        
        # Octahedron center points
        [f, f, 0],     # bottom center
        [f, f, size],  # top center
        [f, 0, f],     # front center
        [f, size, f],  # back center
        [0, f, f],     # left center
        [size, f, f],  # right center
        [f, f, f]      # middle center
    ])
    
    # Create edges to form proper octahedron-tetrahedron pattern
    edges = [
        # Octahedron edges
        [14, 8], [14, 9], [14, 10], [14, 11], [14, 12], [14, 13],  # center to face centers
        
        # Tetrahedron edges
        [0, 8], [1, 8], [2, 8], [3, 8],  # bottom pyramid
        [4, 9], [5, 9], [6, 9], [7, 9],  # top pyramid
        [0, 10], [1, 10], [4, 10], [5, 10],  # front pyramid
        [2, 11], [3, 11], [6, 11], [7, 11],  # back pyramid
        [0, 12], [3, 12], [4, 12], [7, 12],  # left pyramid
        [1, 13], [2, 13], [5, 13], [6, 13]   # right pyramid
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

def calculate_strut_lengths(vertices, edges):
    lengths = []
    for edge in edges:
        start, end = edge
        vec = vertices[end] - vertices[start]
        length = np.sqrt(np.sum(vec**2))
        lengths.append(length)
    return np.array(lengths)

def plot_lattice(vertices, edges, strut_thickness, colorscale='Viridis'):
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
            colorscale=colorscale
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
st.set_page_config(layout="wide", page_title="Lattice Structure Visualizer")
st.title("Beam-based Lattice Structure Visualizer")

# Create two columns for layout
col1, col2 = st.columns([1, 3])

# Sidebar controls
with col1:
    st.header("Settings")
    
    with st.expander("Unit Cell Settings", expanded=True):
        unit_cell = st.selectbox(
            "Select Unit Cell Type",
            ["Simple Cubic", "Body-Centered Cubic", "Octet Truss"]
        )
        
        st.write("""
        **Unit Cell Types:**
        - Simple Cubic: Basic cubic structure
        - Body-Centered Cubic: Added central node
        - Octet Truss: Complex triangulated structure
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
    
    # Add analysis section
    with st.expander("Structure Analysis", expanded=True):
        # Generate lattice for analysis
        vertices, edges = create_lattice(unit_cell, grid_size)
        strut_lengths = calculate_strut_lengths(vertices, edges)
        
        st.write(f"Total struts: {len(edges)}")
        st.write(f"Total nodes: {len(vertices)}")
        st.write(f"Average strut length: {strut_lengths.mean():.2f}")
        st.write(f"Strut length std dev: {strut_lengths.std():.2f}")

# Main visualization area
with col2:
    # Generate and display lattice
    vertices, edges = create_lattice(unit_cell, grid_size)
    fig = plot_lattice(vertices, edges, strut_thickness, colorscale)
    st.plotly_chart(fig, use_container_width=True)
    
    # Add instructions
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
st.markdown("Created for visualization of beam-based lattice structures")
