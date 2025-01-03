from typing import List, Tuple, Any, Optional
from langchain_core.messages import BaseMessage

class Problem(BaseMessage):
    """
    `Problem` is a `BaseMessage` object type that has all the information about a given problem.
    """

    def __init__(self,
                 description: str, 
                 candidate_solutions_number: Optional[int] = None,
                 problem_type: Optional[str] = None,  # Add the type field
                 problem_solution_examples: List[Tuple[str, str]] = None,
                 **kwargs: Any
                 ):
        """
        Initialize a new `Problem` instance.
        
        Args:
            description (str): A brief description of the problem.
            candidate_solutions_number (int): The number of candidate solutions to generate. Defaults to 1.
            problem_type (str): The type or category of the problem.
            problem_solution_examples (List[Tuple[str, str]], optional): Example problem-solution pairs for few-shot learning.
            **kwargs (Any): Additional keyword arguments passed to the base class.
        """
        # Initialize the parent `BaseMessage` with the provided description and problem type.
        super().__init__(content=description, type=problem_type, **kwargs)

        # Problem description provided by the user.
        self.description: str = description
        
        # Verbose description generated by an LLM (optional).
        self.verbose_description: str = ''
        
        # List of subproblems that constitute the main problem (optional).
        self.subproblems: List['Problem'] = []
        
        # List of candidate solutions generated for the problem (optional).
        self.candidate_solutions: List[str] = []
        
        # The chosen solution selected from the candidate solutions (optional).
        self.chosen_solution: str = ''
        
        # Test cases generated by an LLM to validate the chosen solution (optional).
        self.test_cases: List[Tuple[str, bool]] = []
        
        # Example problem-solution pairs used for few-shot learning (optional).
        self.problem_solution_examples: List[Tuple[str, str]] = problem_solution_examples
        
        # The number of candidate solutions to generate.
        self.num_of_candidate_solutions = candidate_solutions_number
        
        # The type or category of the problem.
        self.type = problem_type
