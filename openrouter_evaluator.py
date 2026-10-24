"""Direct OpenRouter judge for DeepEval — no LiteLLM.

Put this file beside your project .env file and use:
    from openrouter_evaluator import router
    TaskCompletionMetric(threshold=0.7, model=router)

Required .env entry:
    OPENROUTER_API_KEY=sk-or-v1-...

Install:
    pip install openai python-dotenv deepeval
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional, Type, Union

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from deepeval.models import DeepEvalBaseLLM

DEFAULT_MODEL = "openai/gpt-oss-20b:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterJudge(DeepEvalBaseLLM):
    """DeepEval judge using OpenRouter's OpenAI-compatible endpoint directly."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing. Add it to .env or your environment."
            )

        self.model_id = model or os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.async_client: Optional[AsyncOpenAI] = None
        super().__init__(model=self.model_id)
        self.async_client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=self.api_key,
        )

    def load_model(self) -> OpenAI:
        return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=self.api_key)

    def get_model_name(self) -> str:
        return f"OpenRouter {self.model_id}"

    def supports_log_probs(self) -> bool:
        return False

    @staticmethod
    def _schema_json(schema: Type[BaseModel]) -> dict[str, Any]:
        if hasattr(schema, "model_json_schema"):
            return schema.model_json_schema()
        return schema.schema()

    def _params(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        user_prompt = prompt
        params: dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if schema is not None:
            schema_text = json.dumps(self._schema_json(schema), ensure_ascii=False)
            user_prompt += (
                "\n\nReturn only one valid JSON object matching this schema. "
                "Do not include markdown fences or text outside the JSON object.\n"
                f"Schema: {schema_text}"
            )
            params["messages"] = [{"role": "user", "content": user_prompt}]
            params["response_format"] = {"type": "json_object"}

        for key in ("top_p", "seed", "stop"):
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]
        return params

    @staticmethod
    def _parse(content: Optional[str], schema: Optional[Type[BaseModel]]) -> Union[str, BaseModel]:
        if not content or not content.strip():
            raise ValueError("OpenRouter returned an empty judge response.")
        if schema is None:
            return content.strip()

        text = content.strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)
        data = json.loads(text)
        if hasattr(schema, "model_validate"):
            return schema.model_validate(data)
        return schema.parse_obj(data)

    def generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Union[str, BaseModel]:
        response = self.model.chat.completions.create(**self._params(prompt, schema, **kwargs))
        return self._parse(response.choices[0].message.content, schema)

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Union[str, BaseModel]:
        if self.async_client is None:
            self.async_client = AsyncOpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=self.api_key,
            )
        response = await self.async_client.chat.completions.create(
            **self._params(prompt, schema, **kwargs)
        )
        return self._parse(response.choices[0].message.content, schema)

    def generate_raw_response(self, prompt: str, **kwargs: Any) -> Any:
        raise AttributeError("This OpenRouter judge does not provide log probabilities.")

    async def a_generate_raw_response(self, prompt: str, **kwargs: Any) -> Any:
        raise AttributeError("This OpenRouter judge does not provide log probabilities.")


router = OpenRouterJudge()
