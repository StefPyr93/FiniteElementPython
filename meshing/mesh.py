import numpy as np

class Mesh:
    def __init__(self, nodes, elements):
        self.nodes = nodes  # List of Node objects
        self.elements = elements  # List of Element objects
        self.num_nodes = len(nodes)
        self.global_stiffness_matrix = np.zeros((self.num_nodes * 2, self.num_nodes * 2))  # For 2 DOFs per node

    def assemble_global_stiffness_matrix(self):
        # Loop over all elements
        for element in self.elements:
            local_stiffness = element.compute_stiffness_matrix()
            # Get the global node indices for this element
            global_dofs = []
            for node in element.nodes:
                global_dofs.append(2 * node.id)     # x displacement DOF
                global_dofs.append(2 * node.id + 1) # y displacement DOF

            # Add the local stiffness matrix into the global stiffness matrix
            for i in range(len(global_dofs)):
                for j in range(len(global_dofs)):
                    self.global_stiffness_matrix[global_dofs[i], global_dofs[j]] += local_stiffness[i, j]
