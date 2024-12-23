from abc import ABC, abstractmethod

class _Methods(ABC):
    """Abstract Base Class for all methods."""

    @abstractmethod
    def q(self):
        pass

    @abstractmethod
    def u(self):
        pass

    @abstractmethod
    def bodies(self):
        pass

    @abstractmethod
    def loads(self):
        pass

    @abstractmethod
    def mass_matrix(self):
        pass

    @abstractmethod
    def forcing(self):
        pass

    @abstractmethod
    def mass_matrix_full(self):
        pass

    @abstractmethod
    def forcing_full(self):
        pass

    def _form_eoms(self):
        raise NotImplementedError("Subclasses must implement this.")
