import os
from dotenv import load_dotenv

load_dotenv()

MAX_RETRIES = 3

DB_CONNECTION_STRING = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)