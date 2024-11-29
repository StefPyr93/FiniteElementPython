# nonlinear_solver.py

import numpy as np

class NewtonRaphsonSolver:
    def __init__(self, mesh, tol=1e-6, max_iter=20):
        """
        Initialize the Newton-Raphson solver.

        Parameters:
            mesh: The finite element mesh object containing nodes and elements.
            tol (float): Convergence tolerance for the residual norm.
            max_iter (int): Maximum number of iterations.
        """
        self.mesh = mesh
        self.tol = tol
        self.max_iter = max_iter

    def assemble_global_residual(self, displacements):
        """
        Assemble the global residual vector based on current displacements.
        """
        residual = np.zeros_like(displacements)
        for element in self.mesh.elements:
            local_residual = element.compute_internal_forces(displacements)
            # Map local residual to the global residual vector
            element_dof_indices = element.get_dof_indices()
            residual[element_dof_indices] += local_residual
        return residual

    def assemble_global_tangent_stiffness(self, displacements):
        """
        Assemble the global tangent stiffness matrix based on current displacements.
        """
        num_dof = len(displacements)
        global_stiffness = np.zeros((num_dof, num_dof))
        for element in self.mesh.elements:
            local_stiffness = element.compute_tangent_stiffness(displacements)
            element_dof_indices = element.get_dof_indices()
            for i, global_i in enumerate(element_dof_indices):
                for j, global_j in enumerate(element_dof_indices):
                    global_stiffness[global_i, global_j] += local_stiffness[i, j]
        return global_stiffness

    def solve(self, external_forces):
        """
        Perform the Newton-Raphson iteration to solve the nonlinear system.

        Parameters:
            external_forces: External force vector applied to the system.
        
        Returns:
            displacements: Solved displacement vector.
        """
        num_dof = len(external_forces)
        displacements = np.zeros(num_dof)  # Initial guess: zero displacement

        for iteration in range(self.max_iter):
            residual = self.assemble_global_residual(displacements) - external_forces
            residual_norm = np.linalg.norm(residual)

            print(f"Iteration {iteration + 1}: Residual Norm = {residual_norm:.6e}")

            if residual_norm < self.tol:
                print("Convergence achieved!")
                return displacements

            tangent_stiffness = self.assemble_global_tangent_stiffness(displacements)

            # Solve for the displacement increment: Δu = K⁻¹ * (-R)
            delta_displacements = np.linalg.solve(tangent_stiffness, -residual)
            displacements += delta_displacements

        print("Warning: Newton-Raphson did not converge within the maximum number of iterations.")
        return displacements
