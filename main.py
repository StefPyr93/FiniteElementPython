# main.py

import numpy as np
from meshing.structured_mesh import StructuredMesh
from materials.linear_elastic_material import LinearElasticMaterial
from solvers.linear_solver import LinearSolver

# Define material properties
E = 210e9  # Young's Modulus in Pascals
nu = 0.3   # Poisson's Ratio
material = LinearElasticMaterial(E=E, nu=nu)

# Define beam dimensions and create a structured mesh
length_x = 4.0  # Length of the beam (m)
height_y = 1.0  # Height of the beam (m)
num_x = 8       # Number of elements along the length
num_y = 2       # Number of elements along the height

mesh = StructuredMesh(length_x, height_y, num_x, num_y, element_type='quad', material=material)

# Define external force vector
external_forces = np.zeros(mesh.total_dof)

# Apply a point load at the center node (midpoint of the beam)
center_node_index = (num_x + 1) * num_y + num_x // 2  # Center node index
center_dof_y = mesh.nodes[center_node_index].get_dof_indices()[1]  # Vertical DOF of center node
external_forces[center_dof_y] = -1000.0  # Apply load (negative for downward force)

# Define boundary conditions (fix bottom nodes at both ends)
boundary_conditions = {}
left_node = mesh.nodes[0]
right_node = mesh.nodes[num_x]

# Fix horizontal and vertical displacement at both ends
for dof in left_node.get_dof_indices():
    boundary_conditions[dof] = 0.0
for dof in right_node.get_dof_indices():
    boundary_conditions[dof] = 0.0

# Initialize and run the linear solver
solver = LinearSolver(mesh)
displacements = solver.solve(external_forces, boundary_conditions)

# Print final displacements
print("Final Displacements at Nodes:")
for i, node in enumerate(mesh.nodes):
    u, v = displacements[node.get_dof_indices()]
    print(f"Node {i + 1}: u = {u:.6e} m, v = {v:.6e} m")
