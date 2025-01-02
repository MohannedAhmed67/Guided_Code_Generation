from typing import List, Tuple, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage



class Problem(BaseMessage):
    """
    `Solution` is a `BaseMessage` object type that has all the information about the solution of some problem
    (Code, Doc, debug and reviews)
    """

    def __init__(self,
                 code: str, 
                 documentation: str, 
                 review: str,
                 debug: str,
                 **kwargs: Any
                 ):
        """
        """
        super().__init__(content={"code" : code, "documentation": documentation, "review":review, "debug": debug})

        