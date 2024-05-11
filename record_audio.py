import pyaudio
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import threading
from queue import Queue
from collections import deque
import time
# np.set_printoptions(threshold=np.inf)

# Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 0.3

def rec_aud(aud_q):
    c = 1
    while 1:
        c += 1
        print(c)
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

        aud_q.put(audio_array)
    return 




if __name__ == "__main__":
    aud_q = Queue()
    thread = threading.Thread(target=rec_aud, args = (aud_q,))
    thread.start()

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(0, 1.5, 50)
    y = np.linspace(0, 512, 512)
    lx,ly = len(x),len(y)
    x, y = np.meshgrid(x, y)
    z = np.zeros((512,50))
    surf = ax.plot_surface(x, y, z, cmap='viridis')
    # Add labels and title
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_zlabel('Z-axis')
    
    # plt.title('3D Plot')

    
    ydata = deque()
    while aud_q:
        aud_a = aud_q.get()
        if len(ydata) == 50:ydata.popleft()
        N = 1024
        T = 1/800
        xf = fftfreq(N, T)
        yf = fft(aud_a)
        ydata.append(2.0/N * np.abs(yf[:N//2]))



        # Generate example data
        if len(ydata) < 50:continue
        newz = np.zeros((ly,lx))
        for i in range(ly):
            for j in range(lx):
                newz[i][j] = ydata[j][i]
        print(newz)
        # Plot the surface
        surf = ax.plot_surface(x, y, newz, cmap='viridis')
        # Show the plot
        fig.canvas.draw()
        plt.pause(0.1)
        ax.clear()
        surf.remove()
    
    # plt.show()
    thread.join()

    print("Thread finished")
