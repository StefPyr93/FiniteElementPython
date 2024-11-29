import numpy as np
from materials.materials import Material

class PlasticMaterial(Material):
    def __init__(self, E, nu, yield_stress, plane_stress=True):
        super().__init__({"E": E, "nu": nu, "yield_stress": yield_stress, "plane_stress": plane_stress})
        self.E = E                # Young's modulus
        self.nu = nu              # Poisson's ratio
        self.yield_stress = yield_stress  # Yield stress
        self.plane_stress = plane_stress  # True for plane stress, False for plane strain
        self.D = self.compute_constitutive_matrix()

    def compute_constitutive_matrix(self):
        # Construct the elastic constitutive matrix D for plane stress or plane strain
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
        # First, calculate the elastic stress using Hooke's law
        elastic_stress = self.D @ strain

        # Calculate the deviatoric stress (von Mises stress) and check for yielding
        J2 = 0.5 * np.dot(elastic_stress, elastic_stress)  # Deviatoric stress measure
        yield_condition = np.sqrt(3 * J2) > self.yield_stress

        if yield_condition:
            # If the stress exceeds the yield stress, apply the yield stress
            # In this simple example, we use perfect plasticity where the stress remains constant
            # after yielding, i.e., the stress is capped at the yield stress
            stress_magnitude = np.sqrt(3 * J2)
            scaling_factor = self.yield_stress / stress_magnitude
            plastic_stress = elastic_stress * scaling_factor
            return plastic_stress
        else:
            # If stress is within elastic limits, return the elastic stress
            return elastic_stress
