import tkinter as tk
import seaborn as sns

class Application(object):
    def __init__(self,parent,**kwargs):
        self.parent = parent
        self.parent.geometry("900x600")
        super().__init__(**kwargs)
        self.vis_frame = tk.LabelFrame(self.parent)
        self.vis_frame.grid(column=1,row=5,sticky='WE')
        self.gui_button()

    def gui_button(self):
        df = sns.load_dataset('iris')
        columns = df.columns


        for i in range(len(columns)):
                    button = tk.Button(self.vis_frame,text=columns[i],command = lambda c=columns[i]: self.gui_plot(df,c))
                    button.grid(row=i+1,column=0,sticky='W')

    def gui_plot(self,data,column):
        from PIL import ImageTk, Image
        self.sns_plot = sns.pairplot(data,hue=column,size=1.5)
        self.sns_plot.savefig('plot.png')
        img = ImageTk.PhotoImage(Image.open('plot.png'))
        self.vis = tk.Label(self.vis_frame,image=img)
        self.vis.image = img
        self.vis.grid(row=0,column=1)


if __name__ == '__main__':
    root = tk.Tk()
    app = Application(root)
    root.mainloop()