import asyncio
import telebot
from telebot.async_telebot import AsyncTeleBot
from telebot import types

api = ""
with open("api.txt", "r") as myfile:
    for text in myfile:
        api += text

bot = AsyncTeleBot(api)

@bot.message_handler(commands=['help', 'start'])
async def url(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton('Add plate number')
    markup.add(btn1)
    await bot.send_message(message.from_user.id, "By pressing the button you can add the plate number into the database", reply_markup = markup)

@bot.message_handler(func=lambda message: True)
async def get_text_messages(message):

    if message.text == 'Add plate number':
        await bot.send_message(message.from_user.id, 'Please enter the plate number in the format: "Plate: ..."') #ответ бота

    elif message.text.split(" ")[0] == 'Plate:':
        number = ""
        for i in range(1, len(message.text.split(" "))):
            number += message.text.split(" ")[i]
        number = number.upper()
        reply = 'The plate number is added into the database!'
        with open("database.txt", "a") as myfile:
            if number in open('database.txt').read():
                reply = 'The plate number is already in the database'
            else:
                number += "\n"
                myfile.write(number)
        await bot.send_message(message.from_user.id, reply)
    
    else:
        await bot.send_message(message.from_user.id, 'Please follow the prompt "Plate: ..."')

asyncio.run(bot.polling(none_stop=True, interval=0)) #обязательная для работы бота часть