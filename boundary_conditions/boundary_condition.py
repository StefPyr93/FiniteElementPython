class BoundaryCondition:
    def __init__(self, constrained_nodes, applied_forces):
        self.constrained_nodes = constrained_nodes
        self.applied_forces = applied_forces

    def apply(self, global_stiffness_matrix, global_force_vector):
        # Modify stiffness matrix and force vector according to constraints
        pass