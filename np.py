import numpy as np
import pprint
# Open .npz file using NpzFile class
with np.load('mnist.npz') as npz_file:
    # Get list of compressed file names in .npz file
    compressed_files = npz_file.files

    # Save each compressed file as a text file
    for file_name in compressed_files:
        # Load compressed numpy array from .npz file
        data = npz_file[file_name]
        flat_data = data.reshape(data.shape[0], -1)
        if file_name.startswith("x"):
            np.savetxt(f'{file_name}.txt', flat_data, fmt="%d")
        else:
            np.savetxt(f'{file_name}.txt', data, fmt="%d")


