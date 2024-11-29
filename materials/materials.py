class Material:
    def __init__(self, properties):
        self.properties = properties  # Dictionary of material properties

    def stress_strain_relation(self, strain):
        raise NotImplementedError("Subclasses should implement this method.")