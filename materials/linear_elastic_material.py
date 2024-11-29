import numpy as np
from materials.materials import Material

class LinearElasticMaterial(Material):
    def __init__(self, E, nu, plane_stress=True):
        super().__init__({"E": E, "nu": nu, "plane_stress": plane_stress})
        self.E = E                # Young's modulus
        self.nu = nu              # Poisson's ratio
        self.plane_stress = plane_stress  # True for plane stress, False for plane strain
        self.D = self.compute_constitutive_matrix()

    def compute_constitutive_matrix(self):
        # Construct the constitutive matrix D for plane stress or plane strain
        if self.plane_stress:
            factor = self.E / (1 - self.nu**2)
            D = factor * np.array([
                [1, self.nu, 0],
                [self.nu, 1, 0],
                [0, 0, (1 - self.nu) / 2]
            ])
        else:  # Plane strain
            factor = self.E / ((1 + self.nu) * (1 - 2 * self.nu))
            D = factor * np.array([
                [1 - self.nu, self.nu, 0],
                [self.nu, 1 - self.nu, 0],
                [0, 0, (1 - 2 * self.nu) / 2]
            ])
        return D

    def stress_strain_relation(self, strain):
        # Calculate stress from strain using Hooke's law: sigma = D * strain
        stress = self.D @ strain
        return stress
