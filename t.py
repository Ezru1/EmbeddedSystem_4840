import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import time

# Create initial data for the surface plot
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the initial surface
surf = ax.plot_surface(x, y, z, cmap='viridis')

# Function to update the surface plot with new data
def update_surface():
    for i in range(100):
        # Generate new data
        new_z = np.sin(np.sqrt((x+i)**2 + (y-i)**2))
        surf = ax.plot_surface(x, y, new_z, cmap='viridis')
        # Update the data of the surface plot
        # surf.set_array(new_z.ravel())  # Update the color data (z-values)
        
        # Redraw the plot
        fig.canvas.draw()
        plt.pause(0.5)  # Pause to display the updated plot
        # Optionally, clear the previous plot for a smoother animation
        ax.clear()
        surf.remove()

# Call the update function
update_surface()

# Show the plot
plt.show()
