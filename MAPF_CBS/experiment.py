import matplotlib.pyplot as plt
import numpy as np


labels = ['2 agents', '3 agents', '4 agents', '5 agents', '6 agents']
men_means = [20, 34, 60, 120, 2000]
women_means = [25, 32, 34, 20, 25]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='TAMAA')
rects2 = ax.bar(x + width/2, women_means, width, label='MCRL')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of explored states (thousands)')
ax.set_title('The number of explored states for checking Queryagents')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()


# from matplotlib.pyplot import figure, show, cm
# from numpy import arange
# from numpy.random import rand


# def gbar(ax, x, y, width=0.5, bottom=0):
#     X = [[.6, .6], [.7, .7]]
#     for left, top in zip(x, y):
#         right = left + width
#         ax.imshow(X, interpolation='bicubic', cmap=cm.Blues,
#                   extent=(left, right, bottom, top), alpha=1)

# fig = figure()

# xmin, xmax = xlim = 0, 10
# ymin, ymax = ylim = 0, 1
# ax = fig.add_subplot(111, xlim=xlim, ylim=ylim,
#                      autoscale_on=False)
# X = [[.6, .6], [.7, .7]]

# ax.imshow(X, interpolation='bicubic', cmap=cm.copper,
#           extent=(xmin, xmax, ymin, ymax), alpha=1)

# N = 10
# x = arange(N) + 0.25
# y = rand(N)
# gbar(ax, x, y, width=0.7)
# ax.set_aspect('auto')
# show()

# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure

# def gbar(ax, x, y, width=0.5, bottom=0):
#     X = np.arange(100)[:, np.newaxis]
#     for left, top in zip(x, y):
#         right = left + width
#         mask = X > top
#         ax.imshow(np.ma.masked_array(X, mask), origin="lower", interpolation='nearest', cmap="RdYlGn", vmin=0, vmax=100,
#                   extent=(left, right, bottom, 100), alpha=1)

# A = [5, 30, 45, 80]
# x = [i + 0.5 for i in range(4)]

# fig = figure()

# xmin, xmax = xlim = 0.25, 4.5
# ymin, ymax = ylim = 0, 100
# ax = fig.add_subplot(111, xlim=xlim, ylim=ylim,
#                      autoscale_on=False)

# gbar(ax, x, A, width=0.7)
# ax.set_aspect('auto')
# ax.set_yticks([i for i in range(0, 100, 10)])
# ax.set_yticklabels([str(i) + " %" for i in range(0, 100, 10)])
# plt.show()