
from tkinter import *
from tkinter import filedialog
import pyttsx3

root = Tk()
root.title("Text to speech.app")
# root.iconbitmap("sound.ico")
root.geometry("676x400")
root.resizable(False, False)

engine = pyttsx3.init()

def speak():
    engine.say(text.get())
    engine.runAndWait()

def adjust_rate(rate):
    engine.setProperty('rate', int(rate))

def save_to_file():
    filename = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("Wave files", "*.wav"), ("All Files", "*.*")])
    if filename:
        engine.save_to_file(text.get(), filename)
        engine.runAndWait()

frame = LabelFrame(root, text="Text to speech", font=20, bd=1)
frame.pack(fill="both", expand="yes", padx=10, pady=10)

Label(frame, text="Type text", font=12, bd=1).place(x=255, y=70)

text = Entry(frame, font=12)
text.place(x=190, y=30)

btn_speak = Button(frame, text="Speak", font=12, bd=1, command=speak)
btn_speak.place(x=262, y=230)

btn_save = Button(frame, text="Save", font=12, bd=1, command=save_to_file)
btn_save.place(x=352, y=230)

rate_scale = Scale(frame, from_=50, to=200, orient=HORIZONTAL, label="Speech Rate", command=adjust_rate)
rate_scale.place(x=250, y=130)

root.mainloop()