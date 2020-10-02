# Save and restore data

This directory provides an easy solution to save data into a file and also a way to restore the data using The [pickle](https://docs.python.org/3/library/pickle.html) module, which implements binary protocols for serializing and de-serializing a Python object structure.

[save_load_pickle](save_load_pickle.py) program contains two basic functions:
- save_binary: Input (data, file_name, directory)
- load_binary: Input (file_name, directory) -- Output (data)

[save_load_data](save_load_data.ipynb) contains a jupyter notebook with an example of how to use the functions. 
