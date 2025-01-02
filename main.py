import openai
from node import Node
from langchain.llms import LLaMA
from langchain import PromptTemplate, LLMChain, Agent
from langchain.schema import BaseOutputParser
from typing import List, Tuple, Any
from critic import Critic

openai.api_key = ''

class JSONOutputParser(BaseOutputParser):
    def parse(self, output: str) -> Any:
        import json
        return json.loads(output)


code_and_doc_prompt = PromptTemplate(
    template="Generate Python code and documentation for the following task:\n\n{task}"
)

subtask_prompt = PromptTemplate(
    template="Divide the following task into smaller subtasks:\n\n{task}"
)

combine_code_prompt = PromptTemplate(
    template="Given the following list of functions with their documentation, generate the complete solution for the task:\n\nTask: {task}\n\nFunctions:\n{functions}"
)

review_code_prompt = PromptTemplate(
    template="Review the following code and documentation for any issues or improvements:\n\nCode:\n{code}\n\nDocumentation:\n{documentation}"
)

generate_tests_prompt = PromptTemplate(
    template="Generate test cases for the following Python function:\n\nFunction:\n{code}"
)

test_code_prompt = PromptTemplate(
    template="Given the following code and test cases, run the tests and provide the results:\n\nCode:\n{code}\n\nTest Cases:\n{test_cases}"
)

llm = LLaMA(api_key=openai.api_key, model_name='llama2')
code_and_doc_chain = LLMChain(llm=llm, prompt=code_and_doc_prompt, output_parser=JSONOutputParser())
subtask_chain = LLMChain(llm=llm, prompt=subtask_prompt, output_parser=JSONOutputParser())
combine_code_chain = LLMChain(llm=llm, prompt=combine_code_prompt, output_parser=JSONOutputParser())
review_code_chain = LLMChain(llm=llm, prompt=review_code_prompt, output_parser=JSONOutputParser())
generate_tests_chain = LLMChain(llm=llm, prompt=generate_tests_prompt, output_parser=JSONOutputParser())
test_code_chain = LLMChain(llm=llm, prompt=test_code_prompt, output_parser=JSONOutputParser())


critic = Critic(
    review_code_chain=review_code_chain,
    generate_tests_chain=generate_tests_chain,
    test_code_chain=test_code_chain
)





class ProjectAgent(Agent):
    def __init__(self, code_and_doc_chain: LLMChain, subtask_chain: LLMChain, combine_code_chain: LLMChain, review_code_chain: LLMChain, generate_tests_chain: LLMChain, test_code_chain: LLMChain):
        self.code_and_doc_chain: LLMChain = code_and_doc_chain
        self.subtask_chain: LLMChain = subtask_chain
        self.combine_code_chain: LLMChain = combine_code_chain
        self.review_code_chain: LLMChain = review_code_chain
        self.generate_tests_chain: LLMChain = generate_tests_chain
        self.test_code_chain: LLMChain = test_code_chain

    def run(self, task_description: str) -> Tuple[str, str]:
        root_node = Node(task_description)

        self.divide_task(root_node)

        self.recursive_generate_and_assemble(root_node)

        project_code, project_doc = self.assemble_solution(root_node)

        return project_code, project_doc

    def divide_task(self, node: Node) -> None:
        node.divide_task()
        for child in node.children:
            self.divide_task(child)

    def recursive_generate_and_assemble(self, node: Node) -> None:
        node.generate_code_and_doc()
        
        for child in node.children:
            self.recursive_generate_and_assemble(child)

        node.review_and_test_code()

    def assemble_solution(self, node: Node) -> Tuple[str, str]:
        if not node.children:
            return node.code, node.documentation

        functions_doc: str = ""
        for child in node.children:
            child_code, child_doc = self.assemble_solution(child)
            functions_doc += f"Function: {child_code}\nDocumentation: {child_doc}\n\n"

        response: str = self.combine_code_chain.run(task=node.description, functions=functions_doc)
        node.code = response.strip()

        return node.code, node.documentation

def main() -> None:
    agent = ProjectAgent(
        code_and_doc_chain=code_and_doc_chain,
        subtask_chain=subtask_chain,
        combine_code_chain=combine_code_chain,
        review_code_chain=review_code_chain,
        generate_tests_chain=generate_tests_chain,
        test_code_chain=test_code_chain
    )

    root_task: str = "Create a web application with authentication, database, and API endpoints"

    project_code, project_doc = agent.run(root_task)

    print("Generated Project Code:\n")
    print(project_code)
    print("\nGenerated Project Documentation:\n")
    print(project_doc)

if __name__ == "__main__":
    main()