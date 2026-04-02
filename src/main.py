import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram_sqlite_storage.sqlitestore import SQLStorage

from src.core.config import settings
from src.handlers import dialogs
from src.services.database import db

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    logger.info('Starting bot...')

    # Initialize database
    await db.init_db()

    bot = Bot(token=settings.bot_token)
    storage = SQLStorage(settings.database_path)
    dp = Dispatcher(storage=storage)

    dp.include_router(dialogs.router)

    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info('Bot stopped.')
