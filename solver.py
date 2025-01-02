from typing import Optional, List, Tuple
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_ollama import ChatOllama
from problem import Problem
from generalist import Generalist
from critic import Critic
from coder import Coder
from utils import str_to_python_func
import json
from openai import OpenAI
from structures_of_prompts import *

class Solver(RunnableSerializable):
    """
    Solver is a `Runnable` that implements the entire pipeline of the prompting technique.

    It should be initialized with the parameters of the pipelines 
    (e.g., number of candidate solutions, number of reviews, etc.).
    """
    
    def generate_subproblems(self, problem: Problem, generalist_model: Generalist) -> List[str]:
        """Prompts `generalist_model` to figure out practical subproblems."""
        # Initialize subproblems list
        problem.subproblems = []
        
        solution_suproblems_analysis_generalist_template = PromptTemplate.from_template(            
            """You are an expert programmer with dozens of years of experience, so help me with the following task:
            Problem: {problem_verbose_description}
            Given the problem,,can it be divided into smaller practical 
            subproblems to be implmented and then combined again the solve the main problem ?
            
            For example: 
            I want the subproblems to be very descriptive and relating to the main problem
            like implementing some function `A`, it requires `B` and `C`, so the subproblems 
            of `A` will be ["implementing B", "implementing C"]
            
                
            Don't provide the subproblems that are too easy, and if the solution
            can be implemented all at once, just do it immediately without
            having to divide the problem into subproblems
                
            You're obligated to use the tools provided for you to get the answer
            
            If the solution can be divided into subproblems:
                Provide the subproblems as List[str] and set divisible to 'True'
            
            otherwise:
                Leave the subproblems as an empty List[str] and set divisible to 'False'  
            """
        )
        
        # Invoke the generalist model to get potential subproblems
        generalist_model_with_tools = generalist_model.bind_tools([subproblems], strict = True)
        ret = generalist_model_with_tools.invoke(
            solution_suproblems_analysis_generalist_template.format(
                problem_verbose_description=problem.verbose_description, 
                problem_verbal_solution=problem.chosen_solution,
            )
        )
        print(ret.tool_calls)
        try:
            if ret.tool_calls[0]['args']['divisible'] == 'False':
                return None
        except:
            return None
        
        print(ret.tool_calls[0]['args']['subproblems'])
        problem.subproblems = [Problem(subproblem, problem.num_of_candidate_solutions , problem.type) for subproblem in ret.tool_calls[0]['args']['subproblems']]
        
        return problem.subproblems
    
    def invoke(
            self, 
            problem: Problem, 
            generalist: Generalist,
            critic: Critic,
            coder: Coder,
            current_depth: int,
            depth_limit: int,
            config: Optional[RunnableConfig] = None
            ) -> Tuple[Problem, dict]:

        problem_understading_generalist_template = PromptTemplate.from_template(
            """You are an expert programmer with dozens of years of experience, so help me with the following task:
            Analyze the following problem and break it down to pieces 
            while considering the possible edge cases without providing code
            and make your analysis smart and straight to the point and don't be lengthy:
            Problem: {problem_description}""")
        
        problem_understanding_critic_template = PromptTemplate.from_template(
            """You are an expert programmer with dozens of years of experience, so help me with the following task:
            Problem: {problem_description}
            Analysis: {problem_verbose_description}
            Review the break down for the given problem without providing code.
            Does it cover all factors and parameters in the problem and don't be lengthy? 
            How can this analysis be improved? Explain briefly.
            """
        )

        problem_understanding_feedback_generalist_template = PromptTemplate.from_template(
            """You are an expert programmer with dozens of years of experience, so help me with the following task:
            Problem: {problem_description}
            Analysis: {problem_verbose_description}
            Feedback: {critic_analysis_feedback}
            Based on the feedback and critique of the problem analysis, give a more comprehensive and brief analysis to the problem without providing code and don't be lengthy.
            """
        )

        candidate_verbal_solution_generalist_template = PromptTemplate.from_template(
            """You are an expert programmer with dozens of years of experience, so help me with the following task:
            Problem: {problem_verbose_description}
            Available functions to use: {returned_functions}
            Without providing code, how can this problem be solved? Go through the solution step by step, and
            the solution must be verbal and different from the following solutions {previous}            
            """
        )

        candidate_verbal_solution_critic_template = PromptTemplate.from_template(
            '''You are an expert programmer with dozens of years of experience, so help me with the following task:
            Problem: {problem_verbose_description}
            Available functions to use: {returned_functions}
            Solution: {problem_verbal_solution}
            Without providing code, does this solution solve the problem correctly while covering the edge cases? Why?
            How can it be improved? Explain your answer and break it down thoroughly.
            '''
        )

        candidate_verbal_solution_feedback_generalist_template = PromptTemplate.from_template(
            '''You are an expert programmer with dozens of years of experience, so help me with the following task:
            Problem: {problem_verbose_description}
            Solution: {problem_verbal_solution}
            Available functions to use: {returned_functions}
            Feedback: {problem_verbal_solution_feedback}
            Without providing code, rewrite the verbal solution to the given problem based on this feedback.
            '''

        )
        
        best_candidate_verbal_solution_critic_template = PromptTemplate.from_template(
            """You are an expert programmer with dozens of years of experience, so help me with the following task:
            Problem: {problem_verbose_description}
            Verbal_candidate_solutions: {candidate_solutions}
            Available functions to use: {returned_functions}
            Without providing code, choose the best verbal solution among them that addresses 
            the problem and covers all its aspects. You must only return the best verbal 
            solution among them without adding anything else or explaining.

            For example:
                Problem: Sort an array
                Verbal_candidate_solutions: ["Bubble Sort", "Merge Sort"]

                Your output must be "Merge Sort" because it runs in O(n log(n)) and the other
                in O(n ^ 2).
            """
        )
        
        code_provider_for_verbal_solution_coder_template = PromptTemplate.from_template(
            """You are an expert programmer with dozens of years of experience, so help me with the following task:
            Problem: {problem_verbose_description}
            Available functions to use: {returned_functions}
            Verbal Solution: {problem_verbal_solution}
            Programming Langauge: {programming_language}
                        
            Don't provide the answer without using the given tool
            You are obligated to use tools here
            
            
            You must use the tool even if it's so simple and not needed, use it
            using coderForTheProblem tool and ensure that you used the tool
            the key of the tool is `code_with_doc`
            your output, which is the final code with docstring will be saved in `code_with_doc`, using
            the provided tool for you
            The name of the tool is `coderForTheProblem`, and the args are `code_with_doc`
            You must provide the implementation of the specified solution in the specified programming language
            """
        )
        
        code_extractor_coder_template = PromptTemplate.from_template(
            """You are an expert programmer with dozens of years of experience, so help me with the following task:
            Code: {code}
            
            Given the previous code, your response must be only the function header
            of the code and its docstring, only both of them, and don't explain anything 
            else or add anything else
            
            For example:
                Code: def add(a: int, b: int) -> int:
                    '''This is a code to add two numbers and return their result
                    
                    a: the first number 
                    b: the second number
                    
                    output: is an int that resembles the sum of both of the numbers
                    '''
                    return a + b

                your response must be only the following:
                    function_header = "add(a: int, b: int) -> int"
                    docstring = '''This is a code to add two numbers and return their result
                    
                    a: the first number 
                    b: the second number
                    
                    output: is an int that resembles the sum of both of the numbers
                    '''
            
            Another example:
                Code: def div(a: int, b: int) -> int:
                    '''This is a code to divide a by b return the result
                    
                    a: the first number 
                    b: the second number
                    
                    output: is an int that resembles the result of dividing a by b
                    '''
                    return a / b

                your response must be only the following:
                    function_header = "div(a: int, b: int) -> int"
                    docstring = '''This is a code to divide a by b return the result
                    
                    a: the first number 
                    b: the second number
                    
                    output: is an int that resembles the result of dividing a by b
                    '''
                
            
            
            You must use the tool even if it's so simple and not needed, use it
            use the tool provided for you the provide the response in the required format, and it's
            not optional, it's a must that you follow it
            The name of the tool is `code_extractor`, and the args are `function_header` and `docstring`
            """
        )
        
        """
        1. Analyze the problem at hand and break it down into pieces.
        """
        
        # Use the generalist model to generate to analyse the problem further and generate verbose description
        problem_understanding_prompt = problem_understading_generalist_template.format(
            problem_description=problem.description
        )
        problem_understanding = generalist.invoke([problem_understanding_prompt])
        problem_verbose_description = problem_understanding.content
        
        # Review the problem's description and verbose description and provide enhancements for the analysis
        feedback_prompt = problem_understanding_critic_template.format(
            problem_description=problem.description,
            problem_verbose_description=problem_verbose_description
        )
        feedback = critic.invoke([feedback_prompt])
        feedback_text = feedback.content
        
        # Use the problem's description, verbose description and feedback from the critic to generate the final problem's verbose description
        revised_analysis_prompt = problem_understanding_feedback_generalist_template.format(
            problem_description=problem.description,
            problem_verbose_description=problem_verbose_description,
            critic_analysis_feedback=feedback_text
        )
        revised_analysis = generalist.invoke([revised_analysis_prompt])
        problem.verbose_description = revised_analysis.content
        problem.verbose_description = problem.description
        
        problem.verbose_description = problem.description
        
        print(problem.verbose_description)

        # Generate subproblems
        if current_depth < depth_limit:
            problem.subproblems = self.generate_subproblems(problem, generalist)
        
        returned_functions_and_docs = []
        Tree = {
            "Problem": problem.description,
            "code": None,
            "subproblems": []
        }
        # Process each subproblem recursively
        if problem.subproblems != None:
            for subproblem in problem.subproblems:
                returned_problem_and_tree = self.invoke(subproblem, generalist, critic, coder, current_depth + 1, depth_limit)
                returned_functions_and_docs.append(returned_problem_and_tree[0].chosen_solution)
                Tree["subproblems"].append(returned_problem_and_tree[1])
        """
        2. Generate candidate verbal solutions.
        """

        for _ in range(problem.num_of_candidate_solutions):
            #Generate candididate solution for the problem that's different from the previous candidate solutions
            candidate_solution_prompt = candidate_verbal_solution_generalist_template.format(
                problem_verbose_description=problem.verbose_description,
                returned_functions=returned_functions_and_docs if len(returned_functions_and_docs) > 0 else "There are no functions that can used",
                previous= problem.candidate_solutions if len(problem.candidate_solutions) > 0 else "No previous solutions"
            )
            candidate_solution = generalist.invoke([candidate_solution_prompt])
            problem.candidate_solutions.append(candidate_solution.content)
        

        """
        3. Analyze candidate verbal solutions via Critic.
        """
        # Generate a review for each candidate solution to fix it and ehance it
        candidate_reviews = []
        for solution in problem.candidate_solutions:
            candidate_review_prompt = candidate_verbal_solution_critic_template.format(
                problem_verbose_description=problem.verbose_description,
                returned_functions=returned_functions_and_docs if len(returned_functions_and_docs) > 0 else "There are no functions that can used",
                problem_verbal_solution=solution
            )
            review = critic.invoke([candidate_review_prompt]).content
            candidate_reviews.append(review)

        """
        4. Feedback those critiques to the Generalist.
        """

        # Given the description of the problem, candidate solution and its review, this part works on improving the candidate solutions
        enhanced_candidate_solutions = []
        for i in range(problem.num_of_candidate_solutions):
            feedback_prompt = candidate_verbal_solution_feedback_generalist_template.format(
                problem_verbose_description=problem.verbose_description,
                problem_verbal_solution=problem.candidate_solutions[i],
                returned_functions=returned_functions_and_docs if len(returned_functions_and_docs) > 0 else "There are no functions that can used",
                problem_verbal_solution_feedback=candidate_reviews[i]
            )
            revised_solution = generalist.invoke([feedback_prompt])
            enhanced_candidate_solutions.append(revised_solution.content)

        """
        5. Choose the solution that best addresses the problem without missing any of its aspects
        """
        problem.chosen_solution = critic.invoke([
            best_candidate_verbal_solution_critic_template.format(
                problem_verbose_description=problem.verbose_description,
                candidate_solutions=enhanced_candidate_solutions,
                returned_functions=returned_functions_and_docs
            )
        ])
        
        problem.chosen_solution = problem.content
        
        
        """
        6. Generate the code for the chosen solution.
        """
        
        while True:
            temp_result = coder.bind_tools([coderForTheProblem]).invoke([
                code_provider_for_verbal_solution_coder_template.format(
                    problem_verbose_description=problem.verbose_description,
                    returned_functions=returned_functions_and_docs if len(returned_functions_and_docs) > 0 else "There are no functions that can used",
                    problem_verbal_solution=problem.chosen_solution, 
                    programming_language=problem.type
                )
            ]).tool_calls
            
            print(temp_result)
            if len(temp_result) > 0:
                if 'args' in temp_result[0].keys():
                    if 'code_with_doc' in temp_result[0]['args'].keys():
                        if 'def' in temp_result[0]['args']['code_with_doc']:
                            problem.chosen_solution = temp_result[0]['args']['code_with_doc']
                            break
                    
        Tree["code"] = problem.chosen_solution

        
        """
        7. Extract the function header of the code and its docstring 
        """
        print(Tree['code'])
        while True:
            temp_result = coder.bind_tools([code_extractor]).invoke([
                code_extractor_coder_template.format(
                    code=problem.chosen_solution
                )
            ]).tool_calls
            
            print(temp_result)
            if len(temp_result) > 0:
                if 'args' in temp_result[0].keys():
                    if 'function_header' in temp_result[0]['args'].keys() and 'docstring' in temp_result[0]['args'].keys():
                        problem.chosen_solution = f"[{temp_result[0]['args']['function_header']}, {temp_result[0]['args']['docstring']}]" 
                        break
        
        
        return (problem, Tree)

if __name__ == '__main__':
    client = ChatOllama(
        model="llama3.1",
        temperature=0.7,
    )
    
    solve = Solver()    
    ret = solve.invoke(Problem('''
    """The Chef likes to stay in touch with his staff. So, the Chef, the head server, and the sous-chef all carry two-way transceivers so they can stay in constant contact. Of course, these transceivers have a limited range so if two are too far apart, they cannot communicate directly. The Chef invested in top-of-the-line transceivers which have a few advanced features. One is that even if two people cannot talk directly because they are out of range, if there is another transceiver that is close enough to both, then the two transceivers can still communicate with each other using the third transceiver as an intermediate device. There has been a minor emergency in the Chef's restaurant and he needs to communicate with both the head server and the sous-chef right away. Help the Chef determine if it is possible for all three people to communicate with each other, even if two must communicate through the third because they are too far apart. Input The first line contains a single positive integer T ≤ 100 indicating the number of test cases to follow. The first line of each test case contains a positive integer R ≤ 1,000 indicating that two transceivers can communicate directly without an intermediate transceiver if they are at most R meters away from each other. The remaining three lines of the test case describe the current locations of the Chef, the head server, and the sous-chef, respectively. Each such line contains two integers X,Y (at most 10,000 in absolute value) indicating that the respective person is located at position X,Y. Output For each test case you are to output a single line containing a single string. If it is possible for all three to communicate then you should output "yes". Otherwise, you should output "no". To be clear, we say that two transceivers are close enough to communicate directly if the length of the straight line connecting their X,Y coordinates is at most R. Example Input: 3 1 0 1 0 0 1 0 2 0 1 0 0 1 0 2 0 0 0 2 2 1 Output: yes yes no"""
''', 1, "Python"), client, client, client, 0, 1)
    Project_Graph = json.dumps(ret[1], indent=4)
    
    with open("output.json", "w") as outfile:
        outfile.write(Project_Graph)
