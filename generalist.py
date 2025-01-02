from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage 
from langchain_core.prompts.chat import HumanMessagePromptTemplate

from problem import Problem


#TODO: Add the ability to receive feedback (reviews, debug messages, etc) from other models by
#       writing a custom `invoke` function—or by creating another `Solution` type with reviews,
#       and debug messages as attributes. 

class Generalist(BaseChatModel):    
    """
    `Generalist` is a BaseChatModel to solve analyze the problem description and propose verbal solutions.
    
    It is intended to be used with 70+B chat/instruct LLMs (for chain-of-thought to have effect).
    """

    def __init__(
            self, 
            model: BaseChatModel, 
            system_message: SystemMessage, 
            human_message_prompt_template: HumanMessagePromptTemplate, 
            **kwargs: Any
            ):
        
        self.model: BaseChatModel = model
        """`model` is the `BaseChatModel` that runs the inference"""
        self.system_message: SystemMessage = system_message
        """The system message to steer the model to the space of tokens appropriate for the problem domain."""
        self.human_message_prompt_template = human_message_prompt_template
        """The template for the final prompt"""
        super().__init__(**kwargs)

        
    
    def _generate(
            self, 
            messages: List[Problem],
            stop: Optional[List[str]]=None,
            run_manager: Optional[CallbackManagerForLLMRun]=None
            ) -> ChatResult:
        
        #Currently, only the last BaseMessage is prompted.
        problem: Problem = messages[-1]

        prompt: HumanMessage = HumanMessage(
            content = self.human_message_prompt_template.format(problem = Problem.content).content
            )
        
        messages: List[BaseMessage] = [prompt]

        solution_ai_message: AIMessage = self.model.invoke(messages)

        generation = ChatGeneration(message=solution_ai_message)
        return ChatResult(generations=[generation])
        
    @property
    def _llm_type(self) -> str:
        return "Generalist: " + self.model._llm_type
