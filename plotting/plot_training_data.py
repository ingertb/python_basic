from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '..')
from save_load_binary.save_load_pickle import load_binary
from pathlib import Path
#plt.style.use('fivethirtyeight')

#**************************************************************************
# Function to plot a training curve tracking the agent’s cumulative reward
# and the association time for a certain number of stations
# feature:
# 	- re (reward, epsilon)
# 	- rae (reward, association time, epsilon)
#**************************************************************************
def plot_fea_epsilon(n_sta, feature, avg_size):
	if len(feature) == 3:
		steps_h, episodes_h, data_h1, data_h2, data_h3 = get_data_n_sta(n_sta, feature)
	else:
		steps_h, episodes_h, data_h1, data_h3 = get_data_n_sta(n_sta, feature)
	fig, ax1 = plt.subplots()
	fig.set_size_inches(8, 6)
	ax1.set_xlabel('Number of training steps')
	#------------------------------------------------------------------------------
	if avg_size == 0: # plot a line 
		# Plotting total reward
		ax1.set_ylabel('Cumulative reward', color='#ff6600')
		ax1.plot(steps_h, data_h1, color='#ff6600', linestyle='-', marker='', \
		         linewidth=1, label='Cumulative reward')		
	else:	# Plotting average reward over x episodes
		N = len(data_h1)
		reward_avg_size = np.empty(N)
		for t in range(N):
			reward_avg_size[t] = np.mean(data_h1[max(0, t-avg_size):(t+1)])
		ax1.set_ylabel('Cumulative reward', color='#ff6600')
		ax1.scatter(steps_h, reward_avg_size, color="#ff6600", label='Cumulative reward', marker='^')
	ax1.tick_params(axis='y', labelcolor='#ff6600')	
	#------------------------------------------------------------------------------
	# Plotting epsilon
	ax2 = ax1.twinx() 	
	ax2.set_ylabel('Epsilon', color='#1544ed') 	
	ax2.plot(steps_h, data_h3, color='#1544ed', linestyle='-', marker='', \
	         linewidth=1, label='Epsilon')
	ax2.tick_params(axis='y', labelcolor='#1544ed')
	#------------------------------------------------------------------------------
	# Plotting Association time
	if len(feature) == 3:
		ax3 = ax1.twinx() 	
		ax3.invert_yaxis()
		if avg_size == 0: # plot a line 	
			ax3.set_ylabel('Association time (s)', color='#009933') 	
			ax3.plot(steps_h, data_h2, color='#009933', linestyle='--', marker='', \
			         linewidth=1, label='Association time (s)')
		else: 
			N = len(data_h2)
			ass_time_avg_size = np.empty(N)
			for t in range(N):
				ass_time_avg_size[t] = np.mean(data_h2[max(0, t-avg_size):(t+1)])
			ax3.set_ylabel('Association time (s)', color='#009933')
			ax3.scatter(steps_h, ass_time_avg_size, color="#009933", label='Association time (s)', marker='.')
		ax3.spines['right'].set_position(('outward', 60))  
		ax3.tick_params(axis='y', labelcolor='#009933')
	fig.legend(bbox_to_anchor=(0.5,0.95), borderaxespad=0, loc="lower center",
	        ncol=3, fancybox=True, shadow=True)
	#------------------------------------------------------------------------------
	# Plotting secondary x label (Number of training episodes)
	ax4 = ax1.twiny() 
	secondary_label_ = np.arange(0, len(steps_h), int(len(steps_h)/7))
	secondary_label = [int(episodes_h[i]) for i in secondary_label_]
	label_pos = [steps_h[i] for i in secondary_label_]
	ax4.set_xticks(label_pos)
	ax4.set_xticklabels(secondary_label)
	ax4.xaxis.set_ticks_position('bottom')
	ax4.xaxis.set_label_position('bottom')
	ax4.spines['bottom'].set_position(('outward', 36))
	ax4.set_xlabel('Number of training episodes')
	ax4.set_xlim(ax1.get_xlim())
	#ax3.grid(True)
	#------------------------------------------------------------------------------
	plt.tight_layout(pad=2.05)
	if avg_size == 0:	
		fig.savefig('plots/trn_curve_' + str(n_sta) + '_' + feature + '.png', dpi=400)
	else: 
		fig.savefig('plots/trn_curve_' + str(n_sta) + '_' + feature + '_avg.png', dpi=400)
	plt.show()
#**************************************************************************
# Function to plot training curves tracking:
# 	- Association time
#		- Queue occupancy
#		- Collision probability
#**************************************************************************
def plot_mult_curves(n_sta, color_list, feature, avg_size):
	fig, ax1 = plt.subplots()
	fig.set_size_inches(8, 6)
	ax1.set_xlabel('Number of training steps')	
	ax1.tick_params(axis='y', labelcolor='#000000')
	if feature == 'a':
		ax1.set_ylabel('Association time (s)', color='#000000')
	if feature == 'o':
		ax1.set_ylabel('Queue occupancy', color='#000000')
	if feature == 'c':
		ax1.set_ylabel('Collision probability', color='#000000')	

	for it in range(len(n_sta)):
		steps_h, episodes_h, data_h = get_data_n_sta(n_sta[it], feature)
		if avg_size == 0:			
			ax1.plot(steps_h, data_h, color=color_list[it], linestyle='-', marker='', \
			         linewidth=1, label=str(n_sta[it]))		
		else:	
			N = len(data_h)
			ass_time_avg_size = np.empty(N)
			for t in range(N):
				ass_time_avg_size[t] = np.mean(data_h[max(0, t-avg_size):(t+1)])
			ax1.scatter(steps_h, ass_time_avg_size, color=color_list[it], label=str(n_sta[it]), marker='.')
	fig.legend(bbox_to_anchor=(0.5,0.95), borderaxespad=0, loc="lower center",
	        ncol=7, fancybox=True, shadow=True)
	#------------------------------------------------------------------------------
	# Plotting secondary x label (Number of training episodes)
	ax2 = ax1.twiny()
	secondary_label_ = np.arange(0, len(steps_h), int(len(steps_h)/7))
	secondary_label = [int(episodes_h[i]) for i in secondary_label_]
	label_pos = [steps_h[i] for i in secondary_label_]
	ax2.set_xticks(label_pos)
	ax2.set_xticklabels(secondary_label)
	ax2.xaxis.set_ticks_position('bottom')
	ax2.xaxis.set_label_position('bottom')
	ax2.spines['bottom'].set_position(('outward', 36))
	ax2.set_xlabel('Number of training episodes')
	ax2.set_xlim(ax1.get_xlim())
	#ax3.grid(True)
	#------------------------------------------------------------------------------
	plt.tight_layout(pad=2.05)
	if avg_size == 0:	
		fig.savefig('plots/trn_curve_' + feature + '_comparison.png', dpi=400)
	else: 
		fig.savefig('plots/trn_curve_' + feature + '_comparison_avg.png', dpi=400)
	plt.show()
#**************************************************************************
# Funtion to obtain the data necessary to create a plot
# 'output' defines wich data will return the function 
#		re = reward and epsilon
#		rae = reward, association time, epsilon
#		a = association time 
#		o = queue occupancy 
#		c = collision probability
#**************************************************************************
def get_data_n_sta(n_sta, output):
	
	data = load_binary('training_data', 'data')

	if (n_sta == 0):
		episodes_h = np.arange(1, len(data[:,0]))
		new_data = data
	else:
		new_data = data[0,:]
		episodes_h = [0]
		for i in range (len(data[:,0])):
			if (data[i,0] == n_sta):
				new_data = np.vstack([new_data, data[i,:]])
				episodes_h = np.append(episodes_h,i)
	new_data = np.delete(new_data, 0, 0)
	steps_h = new_data[:,-1]
	if output == 're':
		reward_h = new_data[:,-2]
		epsilon_h = new_data[:,-3]
		return (steps_h, episodes_h, reward_h, epsilon_h)
	elif output == 'rae':
		reward_h = new_data[:,-2]
		ass_time_h = new_data[:,2]
		epsilon_h = new_data[:,-3]
		return (steps_h, episodes_h, reward_h, ass_time_h, epsilon_h)
	elif output == 'a':
		ass_time_h = new_data[:,2]
		return (steps_h, episodes_h, ass_time_h)
	elif output == 'o':
		queue_occ = (new_data[:,4] + new_data[:,8]) / (new_data[:,3] + new_data[:,7])
		return (steps_h, episodes_h, queue_occ)
	elif output == 'c':
		Col_prob = (1 - (new_data[:,6] + new_data[:,10]) / (new_data[:,5] + new_data[:,9]))
		return (steps_h, episodes_h, Col_prob)
#**************************************************************************
# main function
#**************************************************************************
if __name__ == '__main__':
	
	sta_list = [1500, 1100, 900, 700, 500, 300, 100]
	color_list = ['#ff0000', '#0040ff', '#ff8000', '#40ff00', \
						 '#8000ff', '#00ffff', '#808080']
	# Creating "plots directory" if it doesn't exist
	Path('plots').mkdir(parents=True, exist_ok=True)
	# Plotting training curves tracking the agent’s cumulative reward 
	plot_fea_epsilon(0, 're', 0)	
	plot_fea_epsilon(0, 're', 15)
	plot_fea_epsilon(500, 'rae', 0)
	plot_fea_epsilon(500, 'rae', 15)
	plot_fea_epsilon(1100, 'rae', 0)
	plot_fea_epsilon(1100, 'rae', 15)
	plot_mult_curves(sta_list, color_list, 'a', 0)
	plot_mult_curves(sta_list, color_list, 'a', 10)
