import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd 
import plotly.express as px
from scipy.interpolate import make_interp_spline


data = pd.read_csv(r'2d_spectrum.csv')
#print(data)

# sorting according to multiple columns
data.sort_values("distance",axis=0, ascending=True,inplace=True,na_position='first')

# #print(data[1][1])

# fig = data.scatter(x = 'distance', y = 'det_count')
# fig.set_xlabel("distance")
# fig.set_ylabel("# detections")

# data.sort_values("rel_vel",axis=0, ascending=True,inplace=True,na_position='first')

# fig1 = data.scatter(x = 'rel_vel', y = 'det_count')
# fig1.set_xlabel("rel_vel")
# fig1.set_ylabel("# detections")

# plt.show()
count = 0
x = []
y = []

def draw_graph(i):
    global count
    count +=1
    if count:
        pass
    x.append(data['distance'][count])
    y.append(data['det_count'][count])

    plt.xlabel('rel_vel')
    plt.ylabel('det_count')

    plt.cla()
    plt.scatter(x,y)

anima = animation.FuncAnimation(plt.gcf(), draw_graph, interval=500)

plt.show()

# import csv 

# header = ['Name', 'M1 Score', 'M2 Score']
# data = [['Alex', 62, 80], ['Brad', 45, 56], ['Joey', 85, 98]]

# filename = 'Students_Data.csv'
# with open(filename, 'w', newline="") as file:
#     csvwriter = csv.writer(file) # 2. create a csvwriter object
#     csvwriter.writerow(header) # 4. write the header
#     csvwriter.writerows(data) # 5. write the rest of the data