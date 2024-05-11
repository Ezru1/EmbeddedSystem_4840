import pyaudio
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
# np.set_printoptions(threshold=np.inf)

# Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 0.3




i = 0
ydata = np.zeros((512, 512))
for i in tqdm(range(50)):
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    # Open stream
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    # print("Recording...")

    # Initialize array to store audio data
    audio_data = []

    # Record audio data
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        audio_data.append(np.frombuffer(data, dtype=np.int16))

    # print("Finished recording.")

    # Close stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Convert audio_data list to NumPy array
    audio_array = np.concatenate(audio_data)

    # Print array information
    # print("Shape of audio array:", audio_array.shape)
    # print("Data type of audio array:", audio_array)


    N = 1024
    T = 1/800

    xf = fftfreq(N, T)
    yf = fft(audio_array)
    ydata[i] = 2.0/N * np.abs(yf[:N//2])



# Generate example data
x = np.linspace(0, 1.5, 50)
y = np.linspace(0, 512, 512)
lx,ly = len(x),len(y)
x, y = np.meshgrid(x, y)
z = np.zeros((ly,lx))
for i in range(ly):
    for j in range(lx):
        z[i][j] = ydata[j][i]


# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(x, y, z, cmap='viridis')

# Add labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.title('3D Plot')

# Show the plot
plt.show()
