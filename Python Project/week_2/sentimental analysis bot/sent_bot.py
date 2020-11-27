
from collections import OrderedDict
import speech_recognition as sr
import pyttsx3
import tkinter as tk
from textblob import TextBlob


engine = pyttsx3.init()

# Defining the Function to get the voice command

def check():

    global order
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        order = r.recognize_google(audio,language = 'en-in')

        # insert in texture
    sen_analy()

def sen_analy():
    obj = TextBlob(order)
    sentiment = obj.sentiment.polarity

    if sentiment > 0 :
         engine.say('Your sentence is Postive')
         engine.runAndWait()

         l2.configure(text='Positive\nSentences\n:)')

    elif sentiment == 0:
         engine.say('Your sentence is Neutral')
         engine.runAndWait()

         l2.configure(text='Neutral\nSentences\n:(-_-)')
 
    else:
        engine.say('Your sentence is Negetive')
        engine.runAndWait()

        l2.configure(text='Negetive\nSentences\n:(')


# creating GUI


root = tk.Tk()
root.geometry('500x300')
root.title("SENTBOT")
root.configure(bg = 'black')
font = ('verdana', 15, 'bold')
font2 = ('verdana', 30, 'bold')

l1 = tk.Label(root, text="Click on the Button to speak", bg="black",fg="white", font=font)
l1.place(x=100,y=10)

l2 = tk.Label(root, text= ":)", bg="black", fg="white", font=font2)
l2.place(x=120,y=50)

b1 = tk.Button(root,text="Speak", bg="red", fg="white", command=check)
b1.place(x=50,y=220,height=50,width=400)

root.mainloop()
