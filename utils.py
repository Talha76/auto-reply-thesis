from cerebras.cloud.sdk import Cerebras
import json
from schema import output_schema


def get_sentiment(client: Cerebras, review: str) -> dict:
    chat_completion = client.chat.completions.create(
        model="llama3.3-70b",
        messages=[
            {
                "role": "system",
                "content": "You are good at aspect based sentiment analysis. You can extract the aspects and their polarity from the review text. The polarity can be positive, negative, or neutral.",
            },
            {
                "role": "user",
                "content": f"Extract the aspects and their polarity from the following review text enclosed in three backticks. Also, detect the overall sentiment of the review. If the review is not in English, then translate the review in English and give me the translation. Give me a json output as the response:\n```\n{review}\n```",
            }
        ],
        temperature=0,
        response_format={
            "type": "json_schema", 
            "json_schema": {
                "name": "output_schema",
                "strict": True,
                "schema": output_schema
            }
        }
    )

    output = chat_completion.choices[0].message.content
    return json.loads(output)
