'''
@Description: Checkout DataSet
@Author: Lai
@Date: 2019-08-25 14:04:34
@LastEditTime: 2019-08-25 14:47:52
@LastEditors: Lai
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg   # NavigationToolbar2TkAgg
import tkinter as tk


# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
mpl.rcParams['axes.unicode_minus'] = False  # 负号显示


class CheckoutData(object):
    def __init__(self):
        self.root = tk.Tk()  # 创建主窗体
        self.root.title('checkout DataSet')
        self.canvas = tk.Canvas()  # 创建一块显示图形的画布
        self.raw = np.load('raw000000.npz')['arr_0']
        self.im_num = 0
        self.create_form()  # 将figure显示在tkinter窗体上面
        self.add_button()
        self.add_image(self.im_num)
        self.root.mainloop()

    def draw_image(self, d_im, g_2d, q):
        im = self.raw[i * 100]
        plt.clf()
        plt.imshow(im, cmap='gray')
        plt.colorbar()
        plt.title(f'{i}')
        self.canvas.draw()
        return True

    def net_image(self):
        self.im_num += 1
        if self.im_num >= 10:
            self.im_num = 0
        self.add_image(self.im_num)

    def create_form(self):
        f = plt.figure(num=2, figsize=(9, 8), dpi=80)
        # 把绘制的图形显示到tkinter窗口上
        self.canvas = FigureCanvasTkAgg(f, self.root)
        self.canvas.draw()  # 以前的版本使用show()方法，matplotlib 2.2之后不再推荐show（）用draw代替，但是用show不会报错，会显示警告
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def add_button(self):
        """ 增加三个按键 """
        button_frame = tk.Frame(self.root)
        button_frame.pack()
        b1 = tk.Button(button_frame, text='True', font=('Arial', 12),
                       width=12, height=3, command=self.net_image)
        b1.pack(side=tk.LEFT)
        b2 = tk.Button(button_frame, text='False', font=('Arial', 12),
                       width=12, height=3, command=self.net_image)
        b2.pack(side=tk.LEFT)
        b3 = tk.Button(button_frame, text='Exit', font=('Arial', 12),
                       width=12, height=3, command=self.root.quit)
        b3.pack(side=tk.LEFT)

if __name__ == "__main__":
    form = From()
