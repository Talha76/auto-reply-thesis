from pydantic import BaseModel
from typing import Literal
import json


sentiment_types = Literal["positive", "negative", "neutral"]


class Aspect(BaseModel):
    aspect: str
    polarity: sentiment_types


class Output(BaseModel):
    translation: str = ""
    overall_sentiment: sentiment_types
    aspects: list[Aspect]


output_schema = Output.model_json_schema()
print(json.dumps(output_schema, indent=2))
