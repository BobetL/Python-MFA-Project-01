from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import multiprocessing as mp

def figuretest(name):


    img = Image.open("path/file%s.jpg" % name)  # I change the format when I saved the plots.
    x = img.size[0]
    z = img.size[1]


    for name in range(name + 1):
        fig = plt.figure()
        ax = Axes3D(fig)
        for i in range(x):
            for j in range(z):
                r = hex(img.getpixel((i, j))[0])[2:]
                b = hex(img.getpixel((i, j))[1])[2:]
                g = hex(img.getpixel((i, j))[2])[2:]

                if len(r) == 1:
                    r = '0' + r
                if len(b) == 1:
                    b = '0' + b
                if len(g) == 1:
                    g = '0' + g
                color = '#' + r + b + g
                ax.scatter(i, name * 500, -j, c=color) # Here 500 is because the size of my figure is 700*500.
    plt.show()

def multi():
    pool = mp.Pool()
    pool.map(figuretest, range(26))

if __name__ == '__main__':
    multi()
