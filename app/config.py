"""Environment-driven configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AI_SERVICE_")

    app_name: str = "AI Service Template"
    api_keys: str = "dev-key,local-test-key"
    rate_limit_per_minute: int = 60
    embedding_dim: int = 256

    @property
    def key_set(self) -> set[str]:
        return {key.strip() for key in self.api_keys.split(",") if key.strip()}
