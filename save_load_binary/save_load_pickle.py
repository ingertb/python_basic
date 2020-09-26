#*********************************************************
# Libraries declaration 
#*********************************************************
import pickle as pk
from pathlib import Path

#*********************************************************
# Function to store data in a file 
#*********************************************************
def save_binary (data, file_name, directory):
	# Create directory if it doesn't exist
	Path(directory).mkdir(parents=True, exist_ok=True)
	# Set the file name alongside the path
	file_name_ = directory + '/' + file_name + '.dat'
	# Save the data into a binary file
	try:
		pk_out = open(file_name_, 'wb')
		pk.dump(data, pk_out)
		pk_out.close()
		print ("The data was successfully stored in:", file_name_)
	except:
		print ("something was wrong")
#*********************************************************
# Function to load data stored in a file 
#*********************************************************
def load_binary (file_name, directory):
	# Set the file name alongside the path
	file_name_ = directory + '/' + file_name + '.dat'
	# Load data from a binary file
	try:
		pk_in = open(file_name_, 'rb')
		data = pk.load(pk_in)
		pk_in.close()
		print ("Data loaded correctly")
		return data
	except:
		print ("The file", file_name_, "does not exits")
		

	

