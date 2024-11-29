# node.py

import numpy as np

class Node:
    def __init__(self, x, y, dof_start_index, num_dof):
        """
        Initialize a node with its ID, coordinates, and degrees of freedom.

        Parameters:
            node_id (int): Unique identifier for the node.
            x (float): X-coordinate of the node.
            y (float): Y-coordinate of the node.
            num_dof (int): Number of degrees of freedom per node (default is 2 for 2D).
        """
        #self.id = node_id
        self.x = x
        self.y = y
        self.dofs = [dof_start_index, dof_start_index + 1]  # [ux, uy]
        self.num_dof = num_dof
        
        # Initialize DOF values (displacements and forces)
        self.displacements = np.zeros(num_dof)  # e.g., [u, v] in 2D
        self.forces = np.zeros(num_dof)         # e.g., [Fx, Fy] in 2D
        
        # Boundary condition flags (True if fixed, False if free)
        self.fixed_dof = [False] * num_dof

    def get_dof_indices(self):
        """
        Get the global degrees of freedom indices for this node.

        Returns:
            list[int]: List of global DOF indices [ux, uy].
        """
        return self.dofs

    def set_displacement(self, dof, value):
        """Set a specific displacement for a degree of freedom."""
        if 0 <= dof < self.num_dof:
            self.displacements[dof] = value
        else:
            raise IndexError("DOF index out of range")

    def set_force(self, dof, value):
        """Set a specific force for a degree of freedom."""
        if 0 <= dof < self.num_dof:
            self.forces[dof] = value
        else:
            raise IndexError("DOF index out of range")

    def apply_boundary_condition(self, dof):
        """Fix a specific degree of freedom."""
        if 0 <= dof < self.num_dof:
            self.fixed_dof[dof] = True
        else:
            raise IndexError("DOF index out of range")

    def is_fixed(self, dof):
        """Check if a degree of freedom is fixed."""
        if 0 <= dof < self.num_dof:
            return self.fixed_dof[dof]
        else:
            raise IndexError("DOF index out of range")

    def __repr__(self):
        """String representation of the node."""
        return (f"Node(ID={self.id}, x={self.x:.3f}, y={self.y:.3f}, "
                f"Displacements={self.displacements}, Forces={self.forces})")
