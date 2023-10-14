import os

from PIL import Image
import urllib.request as urllib
import io

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, CallbackContext, filters


#delet this import after you integrated nn
import random

load_dotenv()
API_KEY = os.getenv("API_KEY")

messages = {
    '0': 'Correct',
    '1': 'Not on the brake stand!',
    '2': 'The image has been taken from the screen',
    '3': 'The image has been taken from the screen and been photoshoped!',
    '4': 'The image was photoshoped!'
}


# This is the function that evaluates the image, feed your function with PIL image object
def get_prediction(image) :
    ### YOUR INFERENCE FUNCTION HERE ###
    print(image)
    res = random.randint(0, 1)
    if res == 1 :
        res = get_sub_prediction(image)
    return res

def get_sub_prediction(image):
    return random.randint(1, 4)



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None :
    await update.message.reply_text(f'Hello, {update.message.from_user.first_name}! Just send me the photo of your car and I will tell you if it\'s good to send it for your technical service')

async def handle_image(update: Update, context: CallbackContext) :
    file_id = update.message.photo[-1].file_id

    file = await context.bot.get_file(file_id)

    fd = urllib.urlopen(file.file_path)
    image_file = io.BytesIO(fd.read())
    im = Image.open(image_file)

    await update.message.reply_text(f"{messages[get_prediction(image)]}")


#=============================================================================================================

app = ApplicationBuilder().token(API_KEY).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.ALL, handle_image))

app.run_polling()