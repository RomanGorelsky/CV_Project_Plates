import telebot
from telebot import types

bot = telebot.TeleBot('7741285251:AAG1r-ujkIO89lgNWnBM0KwKNmxr1zIy7DI')

@bot.message_handler(commands = ['start'])
def url(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton('Add plate number')
    markup.add(btn1)
    bot.send_message(message.from_user.id, "By pressing the button you can add the plate number into the database", reply_markup = markup)

@bot.message_handler(commands=['hello'])
def hello(message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("👋 Hi")
    markup.add(btn1)
    bot.send_message(message.from_user.id, "👋 Hello!", reply_markup=markup)

@bot.message_handler(content_types=['text'])
def get_text_messages(message):

    if message.text == 'Add plate number':
        bot.send_message(message.from_user.id, 'Please enter the plate number in the format: "Plate: ..."') #ответ бота

    elif message.text.split(" ")[0] == 'Plate:':
        number = ""
        for i in range(1, len(message.text.split(" "))):
            number += message.text.split(" ")[i]
        number += "\n"
        with open("database.txt", "a") as myfile:
            myfile.write(number)
        bot.send_message(message.from_user.id, 'The plate number is added into the database!')

bot.polling(none_stop=True, interval=0) #обязательная для работы бота часть