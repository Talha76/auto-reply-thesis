from pydantic import BaseModel
from typing import Literal
import json


sentiment_types = Literal["Positive", "Negative", "Neutral"]


# class Aspect(BaseModel):
#     aspect: str
#     polarity: sentiment_types


# class Output(BaseModel):
#     translation: str = ""
#     overall_sentiment: sentiment_types
#     aspects: list[Aspect]


class Aspect(BaseModel):
    text: str
    aspect: str
    polarity: sentiment_types


class Output(BaseModel):
    aspects: list[Aspect] = []


output_schema = Output.model_json_schema()

if __name__ == "__main__":
    print("Output Schema:")
    print(json.dumps(output_schema, indent=2))
