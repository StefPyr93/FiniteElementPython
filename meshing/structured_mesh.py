import numpy as np
import matplotlib.pyplot as plt
from nodes.node import Node
from elements.quad_element import QuadElement
from elements.tri_element import TriElement

class StructuredMesh:
    def __init__(self, length_x, length_y, num_x, num_y, element_type='quad', material=None):
        """
        Initialize the structured mesh.
        
        Parameters:
        length_x (float): Length of the mesh in the x-direction.
        length_y (float): Length of the mesh in the y-direction.
        num_x (int): Number of divisions in the x-direction (number of elements in x).
        num_y (int): Number of divisions in the y-direction (number of elements in y).
        element_type (str): Type of elements to generate ('quad' or 'tri').
        material (Material): Material object to assign to elements (optional).
        """
        self.length_x = length_x
        self.length_y = length_y
        self.num_x = num_x
        self.num_y = num_y
        self.total_dof = 2 * (self.num_x + 1) * (self.num_y + 1)
        self.element_type = element_type
        self.material = material
        
        # Initialize nodes and elements lists
        self.nodes = []
        self.elements = []
        
        # Generate nodes and elements
        self.generate_nodes()
        self.generate_elements()

    def generate_nodes(self):
        """
        Generate nodes for the structured mesh.
        Nodes are spaced evenly along the x and y directions.
        """
        dx = self.length_x / self.num_x  # Spacing between nodes in x-direction
        dy = self.length_y / self.num_y  # Spacing between nodes in y-direction
        
        node_id = 0
        dof_counter = 0
        for j in range(self.num_y + 1):  # Including the last row
            for i in range(self.num_x + 1):  # Including the last column
                # Node coordinates (x, y)
                x = i * dx
                y = j * dy
                # Create node object and append it to the list of nodes
                self.nodes.append(Node(x, y, dof_counter, num_dof=2))
                dof_counter += 2

    def generate_elements(self):
        """
        Generate elements for the structured mesh.
        For quadrilateral elements, the element is defined by four adjacent nodes.
        For triangular elements, the element is defined by three adjacent nodes.
        """
        if self.element_type == 'quad':
            self.generate_quad_elements()
        elif self.element_type == 'tri':
            self.generate_tri_elements()
        else:
            raise ValueError(f"Unsupported element type: {self.element_type}")

    def generate_quad_elements(self):
        """
        Generate quadrilateral elements based on the structured grid of nodes.
        Each quadrilateral is defined by four adjacent nodes in the grid.
        """
        for j in range(self.num_y):
            for i in range(self.num_x):
                # Get the node indices for the four nodes that form a quadrilateral
                n1 = i + j * (self.num_x + 1)  # Node at (i, j)
                n2 = n1 + 1  # Node at (i+1, j)
                n3 = n2 + self.num_x + 1  # Node at (i+1, j+1)
                n4 = n1 + self.num_x + 1  # Node at (i, j+1)
                
                # Create a quadrilateral element with the nodes
                nodes = [self.nodes[n1], self.nodes[n2], self.nodes[n3], self.nodes[n4]]
                element = QuadElement(nodes, self.material)
                self.elements.append(element)

    def generate_tri_elements(self):
        """
        Generate triangular elements based on the structured grid of nodes.
        Each quadrilateral is split into two triangles.
        """
        for j in range(self.num_y):
            for i in range(self.num_x):
                # Get the node indices for the four nodes that form a quadrilateral
                n1 = i + j * (self.num_x + 1)  # Node at (i, j)
                n2 = n1 + 1  # Node at (i+1, j)
                n3 = n2 + self.num_x + 1  # Node at (i+1, j+1)
                n4 = n1 + self.num_x + 1  # Node at (i, j+1)
                
                # Split the quadrilateral into two triangles:
                # Triangle 1: (n1, n2, n4)
                # Triangle 2: (n2, n3, n4)
                
                # Create two triangular elements
                nodes1 = [self.nodes[n1], self.nodes[n2], self.nodes[n4]]
                nodes2 = [self.nodes[n2], self.nodes[n3], self.nodes[n4]]
                
                element1 = TriElement(nodes1, self.material)
                element2 = TriElement(nodes2, self.material)
                
                self.elements.append(element1)
                self.elements.append(element2)

    def visualize_mesh(self):
        """
        Visualize the mesh using matplotlib.
        """
        fig, ax = plt.subplots()
        
        # Plot nodes
        x_coords = [node.x for node in self.nodes]
        y_coords = [node.y for node in self.nodes]
        ax.scatter(x_coords, y_coords, color='blue', label='Nodes')
        
        # Plot elements (edges)
        for element in self.elements:
            if isinstance(element, QuadElement):
                # Get the node coordinates for the quadrilateral element
                x_coords = [node.x for node in element.nodes]
                y_coords = [node.y for node in element.nodes]
                x_coords.append(x_coords[0])  # Close the loop
                y_coords.append(y_coords[0])  # Close the loop
                ax.plot(x_coords, y_coords, 'r-', linewidth=1)
            elif isinstance(element, TriElement):
                # Get the node coordinates for the triangular element
                x_coords = [node.x for node in element.nodes]
                y_coords = [node.y for node in element.nodes]
                x_coords.append(x_coords[0])  # Close the loop
                y_coords.append(y_coords[0])  # Close the loop
                ax.plot(x_coords, y_coords, 'g-', linewidth=1)

        ax.set_aspect('equal', 'box')
        ax.set_title("Structured Mesh")
        ax.legend()
        plt.show()
