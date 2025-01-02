from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field

class subproblems(BaseModel):
    subproblems: Optional[List[str]] = Field(description="The List that contains the subproblems as List[str]")
    divisible: Optional[str] = Field(description="If the main problem can be divided into subproblems then set the value of this string to 'True', otherwise, set it to 'False")

class coderForTheProblem(BaseModel):
    code_with_doc: Optional[str] = Field(description="The code and its docstring for the given problem as string and this must be a code and only the code and docstring without a single line of explanation")

class code_extractor(BaseModel):
    function_header: Optional[str] = Field(description="The header of the function like add(a: int, b: int) -> int")
    docstring: Optional[str] = Field(description="The doc string of the function")