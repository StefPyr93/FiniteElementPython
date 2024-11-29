import numpy as np
from elements.elements import Element

class TriElement(Element):
    def __init__(self, nodes, material):
        """
        Initialize a triangular element.

        Parameters:
        nodes (list): List of 3 Node objects for the triangular element.
        material (Material): The material model (Elastic, Plastic, etc.)
        """
        super().__init__(nodes, material)
        self.num_nodes = 3  # A triangular element has 3 nodes

    def get_dof_indices(self):
        """
        Override to retrieve the global DOF indices for a quadrilateral element.
        
        Returns:
            list[int]: List of global DOF indices.
        """
        return super().get_dof_indices()    

    def shape_functions(self, xi, eta):
        """
        Compute the shape functions for a triangular element in natural coordinates.
        This is a linear triangle, so the shape functions are linear.

        Parameters:
        xi (float): Natural coordinate xi.
        eta (float): Natural coordinate eta.

        Returns:
        N (numpy array): Shape functions evaluated at (xi, eta).
        """
        N1 = 1 - xi - eta
        N2 = xi
        N3 = eta
        return np.array([N1, N2, N3])

    def compute_jacobian(self, xi, eta):
        """
        Compute the Jacobian matrix for the triangular element at the given natural coordinates.

        Parameters:
        xi (float): Natural coordinate xi.
        eta (float): Natural coordinate eta.

        Returns:
        J (numpy array): Jacobian matrix.
        """
        # Derivatives of the shape functions with respect to xi and eta
        dN_dxi = np.array([[-1,  1,  0],  # dN1/dxi, dN2/dxi, dN3/dxi
                           [-1,  0,  1]]) # dN1/deta, dN2/deta, dN3/deta

        # Node coordinates
        node_coords = np.array([[node.x, node.y] for node in self.nodes])
        
        # Jacobian matrix: J = dN/dxi * [x1, x2, x3; y1, y2, y3]
        J = np.dot(dN_dxi, node_coords)
        return J

    def compute_stiffness_matrix(self):
        """
        Compute the local stiffness matrix for the triangular element.

        Returns:
        K (numpy array): Local stiffness matrix (3x3 for a 2D triangular element).
        """
        gauss_points = [(1/3, 1/3)]  # Using a single Gauss point for simplicity (center of mass)
        element_stiffness_matrix = np.zeros((6, 6))  # 3 nodes, 2 DOFs per node (6x6 matrix)

        # Loop over the Gauss points (single point for simplicity here)
        for xi, eta in gauss_points:
            # Compute the Jacobian matrix at the Gauss point
            J = self.compute_jacobian(xi, eta)
            
            # Compute the derivatives of the shape functions in global coordinates
            dN_dx = np.linalg.inv(J) @ np.array([[-1, 1, 0], [-1, 0, 1]])  # Derivative in global coordinates

            # Compute the strain-displacement matrix B
            B = np.zeros((3, 6))
            for i in range(3):
                B[0, 2 * i]     = dN_dx[0, i]
                B[1, 2 * i + 1] = dN_dx[1, i]
                B[2, 2 * i]     = dN_dx[1, i]
                B[2, 2 * i + 1] = dN_dx[0, i]

            # Compute the material matrix (D is the constitutive matrix of the material)
            D = self.material.D  # Material matrix (e.g., elastic matrix for linear elasticity)
            
            # Local stiffness matrix: K = B^T * D * B * det(J)
            local_stiffness = B.T @ D @ B * np.linalg.det(J)

            # Add to the element's stiffness matrix
            element_stiffness_matrix += local_stiffness

        return element_stiffness_matrix
