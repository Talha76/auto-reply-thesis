from logger import logger
from cerebras.cloud.sdk import Cerebras
import os, pandas as pd
from tqdm import tqdm
from utils import get_sentiment
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()


DATASET_PATH = f"./datasets/merged_reviews_with_sentiments.csv"

merged_df = pd.read_csv("./datasets/merged_reviews.csv")

dataset_df = pd.DataFrame(columns=["id", "content", "replyContent", "score", "app_name", "translation", "overall_sentiment", "aspects"]) if not os.path.exists(DATASET_PATH) else pd.read_csv(DATASET_PATH)

not_processed_ids = sorted(set(merged_df["id"]) - set(dataset_df["id"]))

api_keys = [os.getenv(f"CEREBRAS_API_KEY_{i}") for i in range(6)]
current_api_key_index = 0
client = Cerebras(api_key=api_keys[current_api_key_index])
for id in tqdm(not_processed_ids, desc="Processing reviews"):
    _, content, replyContent, score, app_name = merged_df[merged_df["id"] == id].iloc[0]
    if content == "":
        continue
    
    while True:
        try:
            sentiment = get_sentiment(client, content)
            dataset_df.loc[len(dataset_df)] = [
                id,
                content,
                replyContent,
                score,
                app_name,
                sentiment.get("translation", ""),
                sentiment["overall_sentiment"],
                [
                    {
                        "aspect": aspect["aspect"],
                        "polarity": aspect["polarity"]
                    }
                    for aspect in sentiment["aspects"]
                ]
            ]
            dataset_df.to_csv(DATASET_PATH, index=False)
            logger.info(f"Processed id: {id}")
            break
        except Exception as e:
            dataset_df.to_csv(DATASET_PATH, index=False)
            error_msg = e.__str__()
            if "request_quota_exceeded" in error_msg:
                logger.error(f"Request quota exceeded for key serial {current_api_key_index}.")
                current_api_key_index += 1
                logger.info(f"Switching to API key serial {current_api_key_index}.")
            elif "token_quota_exceeded" in error_msg:
                logger.error(f"Token quota exceeded for key serial {current_api_key_index}. Please check your API key limits.")
                api_keys.pop(current_api_key_index)
            else:
                logger.error(f"An error occurred: {error_msg}")
                raise e

            if current_api_key_index >= len(api_keys):
                current_api_key_index = 0
            if len(api_keys) == 0:
                break
            client = Cerebras(api_key=api_keys[current_api_key_index])
    if len(api_keys) == 0:
        logger.error("No more API keys available. Exiting.")
        logger.info(f"Total reviews processed: {len(dataset_df)}")
        break
