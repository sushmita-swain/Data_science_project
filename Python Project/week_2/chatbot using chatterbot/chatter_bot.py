from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from tkinter import *
import speech_recognition as s
import pyttsx3 as pp
import threading


engine = pp.init()

voices = engine.getProperty('voices')
print(voices)

engine.setProperty('voice', voices[1].id)


def speak(word):
    engine.say(word)
    engine.runAndWait()



bot = ChatBot("My Bot")


convo = [
    'hi'
    'hi, how i can help you'
    'hello',
    'hi there !',
    'what is your name ?',
    'My name is Bot ',
    'how are you ?',
    'I am doing great these days',
    'thank you',
    'In which city you live ?',
    'I live in Bangalore',
    'In which language you talk?',
    ' I mostly talk in english'
]

trainer = ListTrainer(bot)

# now training the bot with the help of trainer


trainer.train(convo)

# answer = bot.get_response("what is your name?")
# print(answer)

# print("Talk to bot ")
# while True:
#     query = input('You : ')
#     if query == 'exit':
#         break
#     answer = bot.get_response(query)
#     print("bot : ", answer)    


main = Tk()

main.geometry("500x650")

main.title("Chatter Bot")

img = PhotoImage(file="bot1.png")

photoL = Label(main, image=img)

photoL.pack(pady=5)



# take query  : it takes audio as a input from a user and convert it  to string .... 


def takeQuery():
    sr = s.Recognizer()
    sr.pause_threshold = 1
    print("Your bot is listening try to speak")
    with s.Microphone() as m:
        try:
            audio = sr.listen(m)
            query = sr.recognize_google(audio, language=' eng-in')
            print(query)
            textF.delete(0,END)
            textF.insert(0,query)
            ask_from_bot()

        except Exception as e:
            print(e)  
            print("Not recognized")  


def ask_from_bot():
    query = textF.get()
    answer_from_bot = bot.get_response(query)
    msgs.insert(END, "you : " + query)
    print(type(answer_from_bot))
    msgs.insert(END, "bot : " + str(answer_from_bot))
    speak(answer_from_bot)
    textF.delete(0, END)
    msgs.yview(END) 


frame = Frame(main)

sc = Scrollbar(frame)

msgs = Listbox(frame, height=20, width=80, yscrollcommand=sc.set)

sc.pack(side=RIGHT, fill= Y)

msgs.pack(side=LEFT, fill= BOTH, pady = 10)

frame.pack()

# creating text field 

textF = Entry(main, font=("Verdana", 20))
textF.pack(fill=X, pady=10)

btn = Button(main, text="Ask from bot", font=("Verdana", 20), command=ask_from_bot)
btn.pack()


# creating a function

def enter_function(event):
    btn.invoke()

# going to bind main window with enter key ....

main.bind('<Return>', enter_function)

def repeatL():
    while True:
        takeQuery()


t = threading.Thread(target=repeatL)

t.start()

main.mainloop()