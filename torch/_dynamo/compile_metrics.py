from typing import Optional
# Global information for the purposes of adding to CompilationMetrics

# From Inductor's reinplacing pass: the number of Tensors that we failed to reinplace.
possibly_missed_reinplacing_opportunities: Optional[int] = 0


def reset() -> None:
    global possibly_missed_reinplacing_opportunities
    possibly_missed_reinplacing_opportunities = 0
