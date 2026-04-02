from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    bot_token: str

    # Режим тестирования RPS
    rps_easy: int = 5
    rps_medium: int = 15
    rps_hard: int = 50

    # Настройки для логирования в Google Sheets
    google_creds_json: str = ''  # Путь или json строка для gspread
    google_sheet_url: str = ''

    database_path: str = 'data/recsys_bot.db'

    model_config = SettingsConfigDict(
        env_file='.env', env_file_encoding='utf-8', extra='ignore'
    )


settings = Settings()
