import numpy as np

class Element:
    def __init__(self, nodes, material):
        self.nodes = nodes  # List of Node objects
        self.material = material

    def compute_stiffness_matrix(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def compute_internal_forces(self, displacements):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_dof_indices(self):
        """
        Retrieve the global DOF indices for the element's nodes.

        Returns:
            list[int]: List of global DOF indices.
        """
        dof_indices = []
        for node in self.nodes:
            dof_indices.extend(node.dofs)  # Assumes each node stores its DOFs in a `dofs` attribute
        return dof_indices
