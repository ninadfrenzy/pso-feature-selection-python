from tkinter import *
import logic
import seaborn as sns
from PIL import ImageTk, Image
import pandas as pd
from functools import partial
from pandastable import Table, TableModel
import os
def load_dataset():
    dataset_window = Tk()
    dataset_window.geometry('1000x800')
    dataset_window.title('Dataset')
    f = Frame(dataset_window)
    tb = Table(f)
    f.pack(fill=BOTH, expand=True)
    tb.importCSV('train.csv')
    tb.show()
    dataset_window.mainloop()
def perform_pso(wndow, ap, nump, its):
    alpha = float(ap.get())
    num_particles = int(nump.get())
    iters = int(its.get())
    score_label = Label(wndow)
    score_label.grid(row=4, column=0, columnspan=2)
    print(alpha, num_particles, iters)
    features, score = logic.PSO(num_particles, alpha, iters)
    score_label.config(text=score)
    feat_label = Label(wndow, text=features)
    feat_label.grid(row=5, column=0)
def PSO_window():
    pso_window = Tk()
    pso_window.geometry('400x400')
    pso_window.title('PSO')
    L1 = Label(pso_window, text="alpha")
    L2 = Label(pso_window, text="num of particles")
    L3 = Label(pso_window, text="iterations")
    alpha = Entry(pso_window)
    num_particles = Entry(pso_window)
    iters = Entry(pso_window)
    L1.grid(row=0,column=0)
    L2.grid(row=1,column=0)
    L3.grid(row=2,column=0)
    alpha.grid(row=0,column=1)
    num_particles.grid(row=1,column=1)
    iters.grid(row=2,column=1)
    pso_btn = Button(pso_window, text = "PSO run", command=partial(perform_pso, pso_window, alpha, num_particles, iters))
    pso_btn.grid(row=3, column=0, columnspan=2)
    pso_window.mainloop()
def classify_normal(wndow):
    acc = logic.classification()
    acc_score_label = Label(wndow,text="basic classification accuracy is " + str(acc*100), font=("Courier", 15))
    acc_score_label.place(x=250,y= 500)
    print(acc)

def gui_plot():
    wndo = Tk()
    data = pd.read_csv('train.csv')
    if not(os.path.exists('plot.png')):
        sns_plot = sns.pairplot(data,hue='price_range',size=1.5)
        sns_plot.savefig('plot.png')
    img = ImageTk.PhotoImage(Image.open('plot.png'), master=wndo)
    vis_frame = LabelFrame(wndo)
    vis_frame.pack()
    vis = Label(vis_frame,image=img)
    vis.image = img
    vis.pack()
    wndo.mainloop()
main_window  = Tk()
main_window.configure(background='black')
main_window.geometry('800x600')
main_window.title('Mobile price Classifier')
label = Label(main_window,text='PSO based Feature Subset selection for classification', font=("Courier", 20), fg='white', bg="black")
label.place(x=50,y=0)
load_dataset_btn = Button(main_window, text ="Load Dataset", command = load_dataset, bg='white', font=("Courier", 12))
PSO_btn = Button(main_window, text="PSO", command = PSO_window, bg='white',font=("Courier", 12))
normal_classification_btn = Button(main_window, text="classify", command = partial(classify_normal, main_window), bg='white', font=("Courier", 12))
create_plot_btn = Button(main_window, text ="Pairplot", command = gui_plot, bg='white', font=("Courier", 12))
load_dataset_btn.place(x=350,y=100, height=50, width=100)
PSO_btn.place(x=350,y=200, height=50, width=100)
normal_classification_btn.place(x=350,y=300, height=50, width=100)
create_plot_btn.place(x=350,y=400, height=50, width=100)
main_window.mainloop()
