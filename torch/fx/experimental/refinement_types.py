class Equality:
    def __init__(self, lhs: str, rhs: str):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.lhs} = {self.rhs}"

    def __repr__(self) -> str:
        return f"{self.lhs} = {self.rhs}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Equality):
            return self.lhs == other.lhs and self.rhs == other.rhs
        else:
            return False
