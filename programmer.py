import openai
from prompt_engineering.prompts import SYSTEM_PROMPT
from knw_in import retrieval_knowledge
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Programmer:

    def __init__(self, api_key, model="gpt-3.5-turbo-1106", base_url=None):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.messages = []
        self.function_repository = {}
        self.last_snaps = None

    def add_functions(self, function_lib: dict) -> None:
        self.function_repository = function_lib

    def _call_chat_model(self, functions=None, include_functions=False, retrieval=True):
        if retrieval:
            snaps = retrieval_knowledge(self.messages[-1]["content"])
            if snaps:
                self.last_snaps = snaps
                self.messages[-1]["content"] += snaps
            else:
                self.last_snaps = None

        params = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": 4000,
            "temperature": 0.4,
        }

        if include_functions:
            params['functions'] = functions
            params['function_call'] = "auto"

        try:
            return self.client.chat.completions.create(**params)
        except Exception as e:
            print(f"Error calling chat model: {e}")
            return None

    def _call_chat_model_streaming(self, functions=None, include_functions=False, retrieval=False, kernel=None):
        temp = self.messages[-1]["content"]
        if retrieval:
            snaps = retrieval_knowledge(self.messages[-1]["content"], kernel=kernel)
            if snaps:
                for chunk in snaps:
                    yield chunk
                self.last_snaps = snaps
                self.messages[-1]["content"] += snaps  # already add retrieval code to chat history
            else:
                self.last_snaps = None

        params = {
            "model": self.model,
            "messages": self.messages,
            "stream": True
        }

        if include_functions:
            params['functions'] = functions
            params['function_call'] = "auto"

        try:
            stream = self.client.chat.completions.create(**params)
            self.messages[-1]["content"] = temp
            for chunk in stream:
                if hasattr(chunk, 'choices') and chunk.choices[0].delta.content is not None:
                    chunk_message = chunk.choices[0].delta.content
                    yield chunk_message
        except Exception as e:
            print(f"Error calling chat model: {e}")
            return None

    def run(self, function_lib=None):
        try:
            if function_lib is None:
                response = self._call_chat_model()
                final_response = response.choices[0].message.content
                return final_response

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def clear(self):
        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]
        self.function_repository = {}




