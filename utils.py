from typing import Union
from ast import parse, FunctionDef, Module


def str_to_python_func(code: str) -> Union['function', BaseException]:
    """
    Extracts one function definition from text.

    If a syntax error found, more than one sentence found, or the single sentence is not a function,
    an exception is returned. Otherwise, it return a function object.
    """
    tree = parse(code, type_comments=True)
    if not isinstance(tree, Module):
        return ValueError("Code has Syntax Error") 
    elif len(tree.body) != 1:
        return("Code has more than just a `FunctionDef`")
    elif not isinstance(tree.body[0], FunctionDef):
        return ValueError("Parsed code is not a `FunctionDef`")
    
    code_object = compile(tree, "<string>", "exec")
    exec(code_object)

    return locals()[tree.body[0].name]

