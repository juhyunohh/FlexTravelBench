from typing import Callable, List, Optional, Union

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
# from transformers import AutoTokenizer, AutoModelForCausalLM
from together import Together

# Third-party imports
from pydantic import BaseModel

AgentFunction = Callable[[], Union[str, "Agent", dict]]


class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o"
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    functions: List[AgentFunction] = []
    tool_choice: str = None
    parallel_tool_calls: bool = True
    client: Union[OpenAI, Together]

    # Allow arbitrary attributes
    class Config:
        arbitrary_types_allowed = True  # Needed for OpenAI client
        extra = "allow"  # Allows additional attributes to be set



# class HuggingFaceClient:
#     def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", cache_dir: str = "/mnt/nas2/juhyun/FlexibleReasoningBench/model/checkpoints"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             torch_dtype=torch.float16,
#             device_map="auto",
#             cache_dir=cache_dir
#         )
        
#     def chat(self, messages: List[dict], **kwargs) -> str:
#         # Convert chat messages to prompt format
#         prompt = self._format_messages(messages)
        
#         # Tokenize and generate
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
#         outputs = self.model.generate(
#             **inputs,
#             max_new_tokens=32768,
#             temperature=0.6,
#             do_sample=True,
#             pad_token_id=self.tokenizer.eos_token_id
#         )
        
#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         # Extract only the model's response (after the prompt)
#         response_text = response[len(prompt):].strip()
        
#         # Return format similar to OpenAI's chat completion
#         return {
#             "choices": [{
#                 "message": {
#                     "content": response_text,
#                     "role": "assistant"
#                 }
#             }]
#         }
    
#     def _format_messages(self, messages: List[dict]) -> str:
#         """Convert chat messages to prompt format."""
#         formatted_prompt = []
#         for message in messages:
#             role = message["role"]
#             content = message["content"]
#             if role == "system":
#                 formatted_prompt.append(f"System: {content}")
#             elif role == "user":
#                 formatted_prompt.append(f"Human: {content}")
#             elif role == "assistant":
#                 formatted_prompt.append(f"Assistant: {content}")
#         return "\n".join(formatted_prompt) + "\nAssistant:"

# class Agent(BaseModel):
#     name: str = "Agent"
#     model: str
#     instructions: Union[str, Callable[[], str]]
#     client: Union[OpenAI, HuggingFaceClient]
#     functions: List[AgentFunction] = []
#     tool_choice: Optional[str] = None
#     parallel_tool_calls: bool = True

#     class Config:
#         arbitrary_types_allowed = True
#         extra = "allow"

class Response(BaseModel):
    messages: List = []
    agent: Optional[Agent] = None
    context_variables: dict = {}


class Result(BaseModel):
    """
    Encapsulates the possible return values for an agent function.

    Attributes:
        value (str): The result value as a string.
        agent (Agent): The agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    value: str = ""
    agent: Optional[Agent] = None
    context_variables: dict = {}
