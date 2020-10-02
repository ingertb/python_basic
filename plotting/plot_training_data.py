from matplotlib import pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np
import sys
sys.path.insert(0, '..')
from save_load_binary.save_load_pickle import load_binary
#plt.style.use('fivethirtyeight')

def plot_reward_epsilon(num_episodes, steps_h, reward_h, epsilon_h, avg_size):

	fig, ax1 = plt.subplots()
	ax1.set_xlabel('Number of training steps')
	#------------------------------------------------------------------------------
	if avg_size == 0: # plot a line 
		# Plotting total reward
		ax1.set_ylabel('Total reward', color='#ed3215')
		ax1.plot(steps_h, reward_h, color='#ed3215', linestyle='-', marker='', \
		         linewidth=1, label='Reward')		
	else:	# Plotting average reward over (avg_size) episodes
		N = len(reward_h)
		reward_avg_size = np.empty(N)
		for t in range(N):
			reward_avg_size[t] = np.mean(reward_h[max(0, t-avg_size):(t+1)])
		ax1.set_ylabel('Average reward', color='#ed3215')
		ax1.scatter(steps_h, reward_avg_size, color="#ed3215", label='Reward', marker='^')
	ax1.tick_params(axis='y', labelcolor='#ed3215')	
	#------------------------------------------------------------------------------
	# Plotting epsilon
	ax2 = ax1.twinx() 	
	ax2.set_ylabel('Epsilon', color='#1544ed') 	
	ax2.plot(steps_h, epsilon_h, color='#1544ed', linestyle='-', marker='', \
	         linewidth=1, label='Epsilon')
	ax2.tick_params(axis='y', labelcolor='#1544ed')
	fig.legend(bbox_to_anchor=(0.5,0.95), borderaxespad=0, loc="lower center",
	        ncol=3, fancybox=True, shadow=True)
	#------------------------------------------------------------------------------
	# Plotting secondary x label (Number of training episodes)
	ax3 = ax1.twiny() 
	secondary_label_ = np.arange(0, len(steps_h), int(len(steps_h)/7))
	secondary_label = [int(num_episodes[i]) for i in secondary_label_]
	label_pos = [steps_h[i] for i in secondary_label_]
	ax3.set_xticks(label_pos)
	ax3.set_xticklabels(secondary_label)
	ax3.xaxis.set_ticks_position('bottom')
	ax3.xaxis.set_label_position('bottom')
	ax3.spines['bottom'].set_position(('outward', 36))
	ax3.set_xlabel('Number of training episodes')
	ax3.set_xlim(ax1.get_xlim())
	#ax3.grid(True)
	#------------------------------------------------------------------------------
	plt.tight_layout()	
	fig.savefig('reward_epsilon.png')
	plt.show()

def get_data_n_sta(n_sta, data):
	new_data = data[0,:]
	num_episodes = [0]
	for i in range (len(data[:,0])):
		if (data[i,0] == n_sta):
			new_data = np.vstack([new_data, data[i,:]])
			num_episodes = np.append(num_episodes,i)
	new_data = np.column_stack((new_data, num_episodes))
	new_data = np.delete(new_data, 0, 0)
	return (new_data)

if __name__ == '__main__':

	n_sta = 900
	avg_size = 10
	# Extracting necessary data
	#------------------------------------------------------------------------------
	training_data = load_binary('training_data_2', 'data')

	if (n_sta == 0):
		num_episodes = np.arange(1, len(training_data[:,0]))
		training_data_ =  np.delete(training_data, 0, 0)		
		steps_h = training_data_[:,-1]
		reward_h = training_data_[:,-2]
		epsilon_h = training_data_[:,-3]
	else:
		training_data_ = get_data_n_sta(n_sta, training_data)
		num_episodes = training_data_[:,-1]
		steps_h = training_data_[:,-2]
		reward_h = training_data_[:,-3]
		epsilon_h = training_data_[:,-4]
	#------------------------------------------------------------------------------
	# Plotting
	plot_reward_epsilon(num_episodes, steps_h, reward_h, epsilon_h, avg_size)
