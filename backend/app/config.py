from pydantic_settings import BaseSettings
from typing import Optional, Set

class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    
    # Pinecone
    pinecone_api_key: str
    pinecone_environment: str = "us-east-1-aws"
    pinecone_index_name: str = "research-papers"

    # Neo4j
    neo4j_uri: str = ""
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    
    # Rate limiting & protection
    rate_limit_per_ip_hourly: int = 20       # max requests per IP per hour
    rate_limit_per_ip_daily: int = 50        # max requests per IP per day
    global_daily_query_limit: int = 200      # max total queries per day (all users)
    require_api_key: bool = False            # set True to lock down access
    valid_api_keys: Set[str] = set()         # comma-separated in .env: VALID_API_KEYS=key1,key2

    # Application
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()