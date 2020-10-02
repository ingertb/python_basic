from matplotlib import pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np
import sys
sys.path.insert(0, '..')
from save_load_binary.save_load_pickle import load_binary
#plt.style.use('fivethirtyeight')

if __name__ == '__main__':

	training_data = load_binary('training_data', 'data')
	steps_h = training_data[:,-1]
	reward_h = training_data[:,-2]
	epsilon_h = training_data[:,-3]
	ass_time_h = training_data[:,2]

	formatter_sec= EngFormatter(unit='s')

	#******************************************************************************
	# Plotting one graph on a figure
	#******************************************************************************
	fig1, ax11 = plt.subplots()
	ax11.set_title ('Basic plot')
	ax11.set_xlabel('Number of steps')
	ax11.set_ylabel('Reward', color='#ed3215')
	ax11.plot(steps_h, reward_h, color='#ed3215', linestyle='-', marker='', \
	         linewidth=1, label='Reward')
	ax11.tick_params(axis='y', labelcolor='#ed3215')
	ax11.legend()		
	#******************************************************************************
	# Plotting two graphs on the same figure and the same chart
	#******************************************************************************
	fig2, ax21 = plt.subplots()
	ax21.set_title ('Two plots with different y-axis on the same chart')
	ax21.set_xlabel('Number of steps')
	#------------------------------------------------------------------------------	
	ax21.set_ylabel('Reward', color='#ed3215')
	ax21.plot(steps_h, reward_h, color='#ed3215', linestyle='-', marker='', \
	         linewidth=1, label='Reward')
	ax21.tick_params(axis='y', labelcolor='#ed3215')
	#------------------------------------------------------------------------------
	ax22 = ax21.twinx()  # instantiate a second axes that shares the same x-axis
	ax22.set_ylabel('Association time', color='#1544ed') 	
	ax22.yaxis.set_major_formatter(formatter_sec)
	ax22.plot(steps_h, ass_time_h, color='#1544ed', linestyle='-', marker='', \
	         linewidth=1, label='Association time')
	ax22.tick_params(axis='y', labelcolor='#1544ed')
	#------------------------------------------------------------------------------
	fig2.legend(loc="upper right")
	#******************************************************************************
	# Plotting three graphs on the same figure and the same chart
	#******************************************************************************
	fig3, ax31 = plt.subplots()
	print (fig3)
	ax31.set_title ('Three plots with different y-axis on the same chart')
	ax31.set_xlabel('Number of steps')
	#------------------------------------------------------------------------------	
	ax31.plot(steps_h, reward_h, color='#ed3215', linestyle='-', marker='', \
	         linewidth=1, label='Reward')
	ax31.tick_params(axis='y', labelcolor='#ed3215')
	#------------------------------------------------------------------------------
	ax32 = ax31.twinx()  # instantiate a second axes that shares the same x-axis
	ax32.plot(steps_h, ass_time_h, color='#1544ed', linestyle='-', marker='', \
	         linewidth=1, label='Association time (s)')
	ax32.tick_params(axis='y', labelcolor='#1544ed')
	#------------------------------------------------------------------------------
	ax33 = ax31.twinx()  # instantiate a second axes that shares the same x-axis
	ax33.plot(steps_h, epsilon_h, color='#066311', linestyle='-', marker='', \
	         linewidth=1, label='Epsilon')
	ax33.tick_params(axis='y', labelcolor='#066311')
	#------------------------------------------------------------------------------
	fig3.legend(bbox_to_anchor=(0.5,0.0), borderaxespad=0, loc="lower center",
	        ncol=3, fancybox=True, shadow=True)
	#******************************************************************************
	# Plotting two graphs on the same figure, but on different charts
	#******************************************************************************
	fig4, (ax41, ax42) = plt.subplots(nrows=2, ncols=1, sharex=True)
	ax41.set_title ('Two plots on different charts ')
	#------------------------------------------------------------------------------	
	ax41.set_ylabel('Reward', color='#ed3215')
	ax41.plot(steps_h, reward_h, color='#ed3215', linestyle='-', marker='', \
	         linewidth=1, label='Reward')
	ax41.tick_params(axis='y', labelcolor='#ed3215')
	ax41.legend(loc="upper right")
	#------------------------------------------------------------------------------
	ax42.set_ylabel('Association time', color='#1544ed') 	
	ax42.set_xlabel('Number of steps')
	ax42.yaxis.set_major_formatter(formatter_sec)
	ax42.plot(steps_h, ass_time_h, color='#1544ed', linestyle='-', marker='', \
	         linewidth=1, label='Association time')
	ax42.tick_params(axis='y', labelcolor='#1544ed')
	ax42.legend(loc="upper right")
	#------------------------------------------------------------------------------
	#******************************************************************************
	# Plotting a graph on a figure, and add a secondary x-axis
	#******************************************************************************
	fig5, ax51 = plt.subplots()
	ax51.set_title ('A plot with two x-axis')
	#------------------------------------------------------------------------------	
	ax51.set_ylabel('Reward', color='#ed3215')
	ax51.set_xlabel('Number of steps')
	ax51.plot(steps_h, reward_h, color='#ed3215', linestyle='-', marker='', \
	         linewidth=1, label='Reward')
	ax51.tick_params(axis='y', labelcolor='#ed3215')
	#------------------------------------------------------------------------------
	ax52 = ax51.twiny()  # instantiate a second axes that shares the same x-axis	
	ax52.plot(np.arange(len(steps_h)), np.ones(len(steps_h))) # dummy plot
	ax52.cla()
	ax52.set_xlabel('Number of episodes')
	#------------------------------------------------------------------------------
	fig5.legend(loc="upper right")	
	#******************************************************************************
	plt.tight_layout()
	#ax.grid(True)
	fig1.savefig('fig1.png')
	fig2.savefig('fig2.png')
	fig3.savefig('fig3.png')
	fig4.savefig('fig4.png')
	fig5.savefig('fig5.png')
	plt.show()
