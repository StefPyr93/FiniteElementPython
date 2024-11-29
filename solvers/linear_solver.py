# linear_solver.py

import numpy as np

class LinearSolver:
    def __init__(self, mesh):
        """
        Initialize the Linear Solver.

        Parameters:
            mesh: The finite element mesh object containing nodes and elements.
        """
        self.mesh = mesh

    def assemble_global_stiffness_matrix(self):
        """
        Assemble the global stiffness matrix from all elements in the mesh.
        """
        num_dof = self.mesh.total_dof  # Total degrees of freedom
        global_stiffness = np.zeros((num_dof, num_dof))

        for element in self.mesh.elements:
            local_stiffness = element.compute_stiffness_matrix()
            element_dof_indices = element.get_dof_indices()

            # Map local stiffness to the global stiffness matrix
            for i, global_i in enumerate(element_dof_indices):
                for j, global_j in enumerate(element_dof_indices):
                    global_stiffness[global_i, global_j] += local_stiffness[i, j]

        return global_stiffness

    def apply_boundary_conditions(self, stiffness_matrix, force_vector, boundary_conditions):
        """
        Apply boundary conditions by modifying the stiffness matrix and force vector.

        Parameters:
            stiffness_matrix (np.ndarray): Global stiffness matrix.
            force_vector (np.ndarray): Global force vector.
            boundary_conditions (dict): Dictionary with DOF indices as keys and prescribed values as values.

        Returns:
            Modified stiffness matrix and force vector.
        """
        for dof, value in boundary_conditions.items():
            # Set the row and column corresponding to the fixed DOF to zero
            stiffness_matrix[dof, :] = 0
            stiffness_matrix[:, dof] = 0
            stiffness_matrix[dof, dof] = 1  # Set diagonal to 1 to avoid singularity

            # Adjust the force vector to reflect the prescribed displacement
            force_vector[dof] = value

        return stiffness_matrix, force_vector

    def solve(self, external_forces, boundary_conditions):
        """
        Solve the linear system K*u = F.

        Parameters:
            external_forces (np.ndarray): Global force vector.
            boundary_conditions (dict): Prescribed displacements (fixed DOFs).

        Returns:
            displacements (np.ndarray): Solved displacement vector.
        """
        # Assemble the global stiffness matrix
        global_stiffness = self.assemble_global_stiffness_matrix()

        # Apply boundary conditions
        global_stiffness, external_forces = self.apply_boundary_conditions(
            global_stiffness, external_forces, boundary_conditions
        )

        # Solve the linear system using numpy's linear solver
        displacements = np.linalg.solve(global_stiffness, external_forces)

        return displacements
