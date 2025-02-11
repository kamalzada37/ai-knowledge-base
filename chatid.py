from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext
# for checking chat id
# Replace with your bot token
TELEGRAM_BOT_TOKEN = '7929207317:AAHEmS2wQ6msVMx7ZujXqfAK8kzqhvAyUhw'

async def start(update: Update, context: CallbackContext) -> None:
    chat_id = update.message.chat_id
    await update.message.reply_text(f'Your chat ID is: {chat_id}')

def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))

    application.run_polling()

if __name__ == '__main__':
    main()