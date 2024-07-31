from typing import Any, Dict, Iterator, List, Mapping, Optional
from langchain_core.language_models.llms import LLM
from huggingface_hub import InferenceClient


class CustomLLM(LLM):

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        response = ""
        client = InferenceClient(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            token="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",
        )

        for message in client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            stream=True,
        ):
            response = response + str(message.choices[0].delta.content)
            
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return response

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"

llm = CustomLLM()
print(llm.invoke("capital of uganda"))