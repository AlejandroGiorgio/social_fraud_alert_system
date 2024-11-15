import os


class Settings:
    def __init__(self):
        self.OPENAI_LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)