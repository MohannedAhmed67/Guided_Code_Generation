"""All the responses structures that the generalist model can return."""
from pydantic import BaseModel, Field

class DecomposedProblem(BaseModel):
    """The response structure of the generalist model."""

    decomposed_subproblems: list[str] = Field(description="The list of the generated subproblems of the generalist model if they exist")
