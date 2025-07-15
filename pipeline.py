from logger import logger
from cerebras.cloud.sdk import Cerebras
import os, pandas as pd
from tqdm import tqdm
from utils import get_sentiment
import dotenv

dotenv.load_dotenv()


OUTPUT_PATH = f"./out/apc_dataset.csv"
INPUT_PATH = "./datasets/clean_contents.csv"

df = pd.read_csv(INPUT_PATH)
out_df = pd.read_csv(OUTPUT_PATH) if os.path.exists(OUTPUT_PATH) else pd.DataFrame(columns=[
    "id", "content", "llm_output"
])

not_processed_ids = sorted(set(df["id"]) - set(out_df["id"]))

api_keys = [os.getenv(f"CEREBRAS_API_KEY_{i}") for i in range(6)]
current_api_key_index = 0
client = Cerebras(api_key=api_keys[current_api_key_index])
for id in tqdm(not_processed_ids, desc="Processing reviews"):
    _, content = df[df["id"] == id].iloc[0]
    while True:
        try:
            replyContent = get_sentiment(client, content)
            out_df.loc[len(out_df)] = [
                id,
                content,
                replyContent,
            ]
            out_df.to_csv(OUTPUT_PATH, index=False)
            logger.info(f"Processed id: {id}")
            break
        except Exception as e:
            out_df.to_csv(OUTPUT_PATH, index=False)
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
        logger.info(f"Total reviews processed: {len(out_df)}")
        break
