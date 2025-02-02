import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
# x = [3, 3, 3.5, 4]
# y = [0, 10, 25, 0]
# z = [10, 20, 25, 40]
# plt.plot(x, y)
# plt.show()



# Define X and Y coordinates
# x = [0, 0, -2, -2, -4, -4, -6, -6, -8, -8]  # Moving left and straight
# y = [0, 3, 3, 6, 6, 3, 3, 6, 6, 3]  # Moving up and down

# # Plot the path
# plt.plot(x, y, marker='o', linestyle='-', color='b')

# # Labels and title
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Zig-Zag Vertical & Left Pattern")

# # Show the grid for better visualization
# plt.grid(True)

# # Show the plot
# plt.show()


# secont attempt
# Define the path coordinates
x_coords = [0, 0, -2, -2, -4, -4, -6,-6,0]  # X moves
y_coords = [0, 3, 3, 6, 6, 3, 3, 0, 0, 3]  # Y moves

fig, ax = plt.subplots()
ax.set_xlim(-10, 2)  # Adjusting limits for better view
ax.set_ylim(-1, 8)

line, = ax.plot([], [], marker='o', linestyle='-', color='b')

# Function to update the plot frame by frame
def update(frame):
    line.set_data(x_coords[:frame], y_coords[:frame])  # Show partial path
    return line,

# animation
ani = animation.FuncAnimation(fig, update, frames=len(x_coords)+1, interval=500, repeat=True)

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("curator path")
plt.grid(True)

plt.show()

# third attempt

# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# Initial path coordinates
# x_coords = [0, 0, -2, -2, -4, -4, -6, -6, -8, -8]
# y_coords = [0, 3, 3, 6, 6, 3, 3, 6, 6, 3]

# fig, ax = plt.subplots()
# plt.subplots_adjust(bottom=0.3)  # Space for sliders

# ax.set_xlim(-10, 2)
# ax.set_ylim(-1, 8)
# line, = ax.plot([], [], marker='o', linestyle='-', color='b')

# # Sliders for tweaking
# ax_x_shift = plt.axes([0.25, 0.15, 0.5, 0.03])
# ax_y_shift = plt.axes([0.25, 0.1, 0.5, 0.03])

# slider_x = Slider(ax_x_shift, 'Shift X', -5, 5, valinit=0)
# slider_y = Slider(ax_y_shift, 'Shift Y', -5, 5, valinit=0)

# def update(val):
#     shift_x = slider_x.val
#     shift_y = slider_y.val
#     new_x = [x + shift_x for x in x_coords]
#     new_y = [y + shift_y for y in y_coords]
#     line.set_data(new_x, new_y)
#     fig.canvas.draw_idle()

# slider_x.on_changed(update)
# slider_y.on_changed(update)

# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Interactive Zig-Zag Path")
# plt.grid(True)

# plt.show()
