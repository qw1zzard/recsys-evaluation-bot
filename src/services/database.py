import logging
import os
from datetime import datetime

import aiosqlite

from src.core.config import settings

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def init_db(self):
        """Initializes the database and creates tables if they don't exist."""
        # Ensure the directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            # Table for user configurations
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_configs (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    model_name TEXT,
                    base_url TEXT,
                    token TEXT,
                    target_rps INTEGER,
                    last_state TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Table for test results
            await db.execute("""
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    model_name TEXT,
                    target_rps INTEGER,
                    actual_rps REAL,
                    success_rate REAL,
                    duration REAL,
                    latency_p50 REAL,
                    latency_p95 REAL,
                    latency_p99 REAL,
                    precision_at_10 REAL,
                    recall_at_10 REAL,
                    f1_at_10 REAL,
                    ndcg_at_10 REAL,
                    map_at_10 REAL,
                    mrr REAL,
                    hitrate_at_10 REAL,
                    status TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_configs (user_id)
                )
            """)
            await db.commit()
            logger.info('Database initialized successfully.')

    async def upsert_user_config(
        self,
        user_id: int,
        username: str = None,
        model_name: str = None,
        base_url: str = None,
        token: str = None,
        target_rps: int = None,
    ):
        """Updates or inserts user configuration."""
        async with aiosqlite.connect(self.db_path) as db:
            # Check if user exists
            async with db.execute(
                'SELECT user_id FROM user_configs WHERE user_id = ?', (user_id,)
            ) as cursor:
                exists = await cursor.fetchone()

            if exists:
                # Build update query dynamically based on provided fields
                fields = []
                values = []
                if username is not None:
                    fields.append('username = ?')
                    values.append(username)
                if model_name is not None:
                    fields.append('model_name = ?')
                    values.append(model_name)
                if base_url is not None:
                    fields.append('base_url = ?')
                    values.append(base_url)
                if token is not None:
                    fields.append('token = ?')
                    values.append(token)
                if target_rps is not None:
                    fields.append('target_rps = ?')
                    values.append(target_rps)

                fields.append('updated_at = ?')
                values.append(datetime.now())

                if fields:
                    query = (
                        f'UPDATE user_configs SET {", ".join(fields)} WHERE user_id = ?'
                    )
                    values.append(user_id)
                    await db.execute(query, tuple(values))
            else:
                # Insert new record
                query = 'INSERT INTO user_configs (user_id, username, model_name, base_url, token, target_rps) VALUES (?, ?, ?, ?, ?, ?)'
                await db.execute(
                    query, (user_id, username, model_name, base_url, token, target_rps)
                )

            await db.commit()

    async def get_user_config(self, user_id: int) -> dict | None:
        """Retrieves user configuration."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                'SELECT * FROM user_configs WHERE user_id = ?', (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def save_test_result(self, result_data: dict):
        """Saves a test result summary."""
        async with aiosqlite.connect(self.db_path) as db:
            keys = list(result_data.keys())
            placeholders = ', '.join(['?'] * len(keys))
            columns = ', '.join(keys)
            query = f'INSERT INTO test_results ({columns}) VALUES ({placeholders})'
            try:
                await db.execute(query, tuple(result_data.values()))
                await db.commit()
                logger.info(f'Test result saved for user {result_data.get("user_id")}')
            except Exception as e:
                logger.error(f'Error saving test result: {e}')
                await db.rollback()


# Global instance
db = Database(settings.database_path)
