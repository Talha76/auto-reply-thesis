from cerebras.cloud.sdk import Cerebras
import json
from schema import output_schema
import re


def get_sentiment(client: Cerebras, review: str) -> dict:
    """
    Get the sentiment and aspects from a review using the Cerebras LLM.

    Args:
        client (Cerebras): The Cerebras client to use for the request.
        review (str): The review text to analyze.

    Returns:
        dict: A dictionary containing the aspects, their polarity, and the overall sentiment of the review.
    """

    prompts = [
        f"""Given examples of a dataset for some reviews:
Review: the design is good but the performance is bad
Output:
```json
{{
    "aspects": [
        {{
            "text": "the $T$ is good but the performance is bad",
            "aspect": "design",
            "polarity": "Positive"
        }},
        {{
            "text": "the design is good but the $T$ is bad",
            "aspect": "performance",
            "polarity": "Negative"
        }}
    ]
}}
```
Review: thanks to those who made it
Output:
```json
{{
    "aspects": []
}}
```
Review: picture quality really is good.
Output:
```json
{{
    "aspects": [
        {{
            "text": "$T$ really is good",
            "aspect": "picture quality",
            "polarity": "Positive"
        }}
    ]
}}
```
Review: {review.lower()}
Output:""",
    f"""You are given a review inside three backticks. You will do the following:
1. Identify the aspects mentioned in the review.
2. Determine the sentiment polarity of each aspect (Positive, Negative, Neutral).
3. For each of the aspects, return the text of the review with the aspect replaced by $T$.
4. Return the aspects and their polarity in the specified JSON format.

Below is a example of the output format:
```json
{{
    "aspects": [
        {{
            "text": "the $T$ is good but the performance is bad",
            "aspect": "design",
            "polarity": "Positive"
        }},
        {{
            "text": "the design is good but the $T$ is bad",
            "aspect": "performance",
            "polarity": "Negative"
        }}
    ]
}}
```
You will return the output in the same format as above.
Review:
```{review}```
Output:""",
    ]

    chat_completion = client.chat.completions.create(
        model="llama3.3-70b",
        messages=[
            {
                "role": "system",
                "content": "You are good at aspect based sentiment analysis. You can extract the aspects and their polarity from the review text. The polarity can be positive, negative, or neutral.",
            },
            {
                "role": "user",
                "content": prompts[1],
            }
        ],
        temperature=0.1,
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


def clean_text(text: str) -> str:
    """
    Clean the input text by lowercasing and removing non-alphanumeric characters.

    Args:
        text (str): The input text to clean.
    Returns:
        str: The cleaned text.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()
