import numpy as np
from elements.elements import Element

class QuadElement(Element):
    def __init__(self, nodes, material):
        super().__init__(nodes, material)
        self.num_nodes = 4  # Quadrilateral has 4 nodes

    def get_dof_indices(self):
        """
        Override to retrieve the global DOF indices for a quadrilateral element.
        
        Returns:
            list[int]: List of global DOF indices.
        """
        return super().get_dof_indices()
    
    def shape_functions(self, xi, eta):
        # Shape functions N1, N2, N3, N4
        N1 = 0.25 * (1 - xi) * (1 - eta)
        N2 = 0.25 * (1 + xi) * (1 - eta)
        N3 = 0.25 * (1 + xi) * (1 + eta)
        N4 = 0.25 * (1 - xi) * (1 + eta)
        return np.array([N1, N2, N3, N4])

    def compute_jacobian(self, xi, eta):
        # Compute the Jacobian matrix
        dN_dxi = np.array([
            [-0.25 * (1 - eta),  0.25 * (1 - eta),  0.25 * (1 + eta), -0.25 * (1 + eta)],
            [-0.25 * (1 - xi),  -0.25 * (1 + xi),   0.25 * (1 + xi),   0.25 * (1 - xi)]
        ])
        node_coords = np.array([[node.x, node.y] for node in self.nodes])
        J = dN_dxi @ node_coords  # Jacobian matrix

        return J

    def compute_stiffness_matrix(self):
        # Quadrature points and weights for Gaussian integration (2x2 Gauss)
        gauss_points = [-1/np.sqrt(3), 1/np.sqrt(3)]
        gauss_weights = [1, 1]
        
        # Initialize element stiffness matrix
        ke = np.zeros((self.num_nodes * 2, self.num_nodes * 2))
        
        for xi in gauss_points:
            for eta in gauss_points:
                N = self.shape_functions(xi, eta)
                J = self.compute_jacobian(xi, eta)
                detJ = np.linalg.det(J)
                
                # Calculate the strain-displacement matrix B
                dN_dxi = np.array([
                    [-0.25 * (1 - eta),  0.25 * (1 - eta),  0.25 * (1 + eta), -0.25 * (1 + eta)],
                    [-0.25 * (1 - xi),  -0.25 * (1 + xi),   0.25 * (1 + xi),   0.25 * (1 - xi)]
                ])
                dN_dx = np.linalg.inv(J) @ dN_dxi  # Derivatives in global coordinates

                B = np.zeros((3, 8))  # For plane stress/strain
                for i in range(4):
                    B[0, 2 * i]     = dN_dx[0, i]
                    B[1, 2 * i + 1] = dN_dx[1, i]
                    B[2, 2 * i]     = dN_dx[1, i]
                    B[2, 2 * i + 1] = dN_dx[0, i]
                
                # Constitutive matrix D (plane stress example)
                E = self.material.properties["E"]
                nu = self.material.properties["nu"]
                D = (E / (1 - nu**2)) * np.array([
                    [1, nu, 0],
                    [nu, 1, 0],
                    [0, 0, (1 - nu) / 2]
                ])
                
                # Integrate the stiffness matrix
                ke += B.T @ D @ B * detJ * gauss_weights[0] * gauss_weights[1]
        
        return ke

    def compute_internal_forces(self, displacements):
        gauss_points = [-1/np.sqrt(3), 1/np.sqrt(3)]  # 2x2 Gauss integration points
        
        # Initialize internal force vector
        internal_forces = np.zeros(self.num_nodes * 2)

        for xi in gauss_points:
            for eta in gauss_points:
                # Compute shape function derivatives
                J = self.compute_jacobian(xi, eta)
                dN_dxi = np.array([
                    [-0.25 * (1 - eta),  0.25 * (1 - eta),  0.25 * (1 + eta), -0.25 * (1 + eta)],
                    [-0.25 * (1 - xi),  -0.25 * (1 + xi),   0.25 * (1 + xi),   0.25 * (1 - xi)]
                ])
                dN_dx = np.linalg.inv(J) @ dN_dxi  # Derivatives in global coordinates
                
                # Strain-displacement matrix B
                B = np.zeros((3, 8))
                for i in range(4):
                    B[0, 2 * i]     = dN_dx[0, i]
                    B[1, 2 * i + 1] = dN_dx[1, i]
                    B[2, 2 * i]     = dN_dx[1, i]
                    B[2, 2 * i + 1] = dN_dx[0, i]
                
                # Compute strain from nodal displacements
                strain = B @ displacements
                
                # Compute stress using the material constitutive law
                E = self.material.properties["E"]
                nu = self.material.properties["nu"]
                D = (E / (1 - nu**2)) * np.array([
                    [1, nu, 0],
                    [nu, 1, 0],
                    [0, 0, (1 - nu) / 2]
                ])
                stress = D @ strain
                
                # Internal force vector contribution at this Gauss point
                detJ = np.linalg.det(J)
                internal_forces += B.T @ stress * detJ * 1 * 1  # Gauss weights = 1
        
        return internal_forces
