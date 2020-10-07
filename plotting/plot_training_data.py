from matplotlib import pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np
import sys
sys.path.insert(0, '..')
from save_load_binary.save_load_pickle import load_binary
#plt.style.use('fivethirtyeight')

def plot_reward_epsilon(n_sta, training_data, avg_size):
	# Training curve tracking the agent’s cumulative reward.
	steps_h, num_episodes, reward_h, _, epsilon_h = get_data_n_sta(n_sta, training_data)
	fig, ax1 = plt.subplots()
	fig.set_size_inches(8, 6)
	ax1.set_xlabel('Number of training steps')
	#------------------------------------------------------------------------------
	if avg_size == 0: # plot a line 
		# Plotting total reward
		ax1.set_ylabel('Cumulative reward', color='#ff6600')
		ax1.plot(steps_h, reward_h, color='#ff6600', linestyle='-', marker='', \
		         linewidth=1, label='Cumulative reward')		
	else:	# Plotting average reward over x episodes
		N = len(reward_h)
		reward_avg_size = np.empty(N)
		for t in range(N):
			reward_avg_size[t] = np.mean(reward_h[max(0, t-avg_size):(t+1)])
		ax1.set_ylabel('Cumulative reward', color='#ff6600')
		ax1.scatter(steps_h, reward_avg_size, color="#ff6600", label='Cumulative reward', marker='^')
	ax1.tick_params(axis='y', labelcolor='#ff6600')	
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
	plt.tight_layout(pad=2.05)	
	if avg_size == 0:
		fig.savefig('training_curve_rwd_eps.png', dpi=400)
	else:
		fig.savefig('training_curve_rwd_eps_avg.png', dpi=400)
	plt.show()

def plot_reward_assTime_epsilon(n_sta, training_data, avg_size):

	steps_h, num_episodes, reward_h, ass_time_h, epsilon_h = get_data_n_sta(n_sta, training_data)
	fig, ax1 = plt.subplots()
	fig.set_size_inches(8, 6)
	ax1.set_xlabel('Number of training steps')
	#------------------------------------------------------------------------------
	if avg_size == 0: # plot a line 
		# Plotting total reward
		ax1.set_ylabel('Cumulative reward', color='#ff6600')
		ax1.plot(steps_h, reward_h, color='#ff6600', linestyle='-', marker='', \
		         linewidth=1, label='Cumulative reward')		
	else:	# Plotting average reward over x episodes
		N = len(reward_h)
		reward_avg_size = np.empty(N)
		for t in range(N):
			reward_avg_size[t] = np.mean(reward_h[max(0, t-avg_size):(t+1)])
		ax1.set_ylabel('Cumulative reward', color='#ff6600')
		ax1.scatter(steps_h, reward_avg_size, color="#ff6600", label='Cumulative reward', marker='^')
	ax1.tick_params(axis='y', labelcolor='#ff6600')	
	#------------------------------------------------------------------------------
	# Plotting epsilon
	ax2 = ax1.twinx() 	
	ax2.set_ylabel('Epsilon', color='#1544ed') 	
	ax2.plot(steps_h, epsilon_h, color='#1544ed', linestyle='-', marker='', \
	         linewidth=1, label='Epsilon')
	ax2.tick_params(axis='y', labelcolor='#1544ed')
	#------------------------------------------------------------------------------
	# Plotting Association time
	ax3 = ax1.twinx() 	
	ax3.invert_yaxis()
	if avg_size == 0: # plot a line 	
		ax3.set_ylabel('Association time (s)', color='#009933') 	
		ax3.plot(steps_h, ass_time_h, color='#009933', linestyle='--', marker='', \
		         linewidth=1, label='Association time (s)')
	else: 
		N = len(ass_time_h)
		ass_time_avg_size = np.empty(N)
		for t in range(N):
			ass_time_avg_size[t] = np.mean(ass_time_h[max(0, t-avg_size):(t+1)])
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
	secondary_label = [int(num_episodes[i]) for i in secondary_label_]
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
		fig.savefig('training_curve_rwd_ass_eps.png', dpi=400)
	else: 
		fig.savefig('training_curve_rwd_ass_eps_avg.png', dpi=400)
	plt.show()

def plot_assTime_comparison(training_data, avg_size):

	steps_h, num_episodes, _, ass_time_h, _ = get_data_n_sta(1100, training_data)
	fig, ax1 = plt.subplots()
	fig.set_size_inches(8, 6)
	ax1.set_xlabel('Number of training steps')
	#------------------------------------------------------------------------------
	if avg_size == 0: # plot a line 
		# Plotting Association time
		ax1.set_ylabel('Association time (s)', color='#000000')
		ax1.plot(steps_h, ass_time_h, color='#ff6600', linestyle='-', marker='', \
		         linewidth=1, label='1100')		
	else:	# Plotting average Association time x episodes
		N = len(ass_time_h)
		ass_time_avg_size = np.empty(N)
		for t in range(N):
			ass_time_avg_size[t] = np.mean(ass_time_h[max(0, t-avg_size):(t+1)])
		ax1.set_ylabel('Association time (s)', color='#000000')
		ax1.scatter(steps_h, ass_time_avg_size, color="#ff6600", label='1100', marker='.')
	ax1.tick_params(axis='y', labelcolor='#000000')	
	#------------------------------------------------------------------------------
	steps_h, num_episodes, _, ass_time_h, _ = get_data_n_sta(900, training_data)
	ax7 = ax1.twiny()	
	if avg_size == 0: # plot a line 
		ax7.plot(steps_h, ass_time_h, color='#1544ed', linestyle='-', marker='', \
		         linewidth=1, label='900')		
	else:	# Plotting average reward over x episodes
		N = len(ass_time_h)
		ass_time_avg_size = np.empty(N)
		for t in range(N):
			ass_time_avg_size[t] = np.mean(ass_time_h[max(0, t-avg_size):(t+1)])
		ax7.scatter(steps_h, ass_time_avg_size, color="#1544ed", label='900', marker='.')
	ax7.axes.xaxis.set_visible(False)
	#------------------------------------------------------------------------------
	steps_h, num_episodes, _, ass_time_h, _ = get_data_n_sta(700, training_data)
	ax2 = ax1.twiny()	
	if avg_size == 0: # plot a line 
		ax2.plot(steps_h, ass_time_h, color='#ff3300', linestyle='-', marker='', \
		         linewidth=1, label='700')		
	else:	# Plotting average reward over x episodes
		N = len(ass_time_h)
		ass_time_avg_size = np.empty(N)
		for t in range(N):
			ass_time_avg_size[t] = np.mean(ass_time_h[max(0, t-avg_size):(t+1)])
		ax2.scatter(steps_h, ass_time_avg_size, color="#ff3300", label='700', marker='.')
	ax2.axes.xaxis.set_visible(False)
	#------------------------------------------------------------------------------
	steps_h, num_episodes, _, ass_time_h, _ = get_data_n_sta(500, training_data)
	ax3 = ax1.twiny() 	
	if avg_size == 0: # plot a line 	
		ax3.plot(steps_h, ass_time_h, color='#009933', linestyle='-', marker='', \
		         linewidth=1, label='500')
	else: 
		N = len(ass_time_h)
		ass_time_avg_size = np.empty(N)
		for t in range(N):
			ass_time_avg_size[t] = np.mean(ass_time_h[max(0, t-avg_size):(t+1)])
		ax3.scatter(steps_h, ass_time_avg_size, color="#009933", label='500', marker='.')
	ax3.axes.xaxis.set_visible(False)
	#------------------------------------------------------------------------------
	steps_h, num_episodes, _, ass_time_h, _ = get_data_n_sta(300, training_data)
	ax4 = ax1.twiny() 	
	if avg_size == 0: # plot a line 	
		ax4.plot(steps_h, ass_time_h, color='#9900cc', linestyle='-', marker='', \
		         linewidth=1, label='300')
	else: 
		N = len(ass_time_h)
		ass_time_avg_size = np.empty(N)
		for t in range(N):
			ass_time_avg_size[t] = np.mean(ass_time_h[max(0, t-avg_size):(t+1)])
		ax4.scatter(steps_h, ass_time_avg_size, color="#9900cc", label='300', marker='.')
	ax4.axes.xaxis.set_visible(False)
	#------------------------------------------------------------------------------
	steps_h, num_episodes, _, ass_time_h, _ = get_data_n_sta(100, training_data)
	ax6 = ax1.twiny() 	
	if avg_size == 0: # plot a line 	
		ax6.plot(steps_h, ass_time_h, color='#663300', linestyle='-', marker='', \
		         linewidth=1, label='100')
	else: 
		N = len(ass_time_h)
		ass_time_avg_size = np.empty(N)
		for t in range(N):
			ass_time_avg_size[t] = np.mean(ass_time_h[max(0, t-avg_size):(t+1)])
		ax6.scatter(steps_h, ass_time_avg_size, color="#663300", label='100', marker='.')
	ax6.axes.xaxis.set_visible(False)
	fig.legend(bbox_to_anchor=(0.5,0.95), borderaxespad=0, loc="lower center",
	        ncol=6, fancybox=True, shadow=True)
	#------------------------------------------------------------------------------
	# Plotting secondary x label (Number of training episodes)
	ax5 = ax1.twiny() 
	secondary_label_ = np.arange(0, len(steps_h), int(len(steps_h)/7))
	secondary_label = [int(num_episodes[i]) for i in secondary_label_]
	label_pos = [steps_h[i] for i in secondary_label_]
	ax5.set_xticks(label_pos)
	ax5.set_xticklabels(secondary_label)
	ax5.xaxis.set_ticks_position('bottom')
	ax5.xaxis.set_label_position('bottom')
	ax5.spines['bottom'].set_position(('outward', 36))
	ax5.set_xlabel('Number of training episodes')
	ax5.set_xlim(ax1.get_xlim())
	#ax3.grid(True)
	#------------------------------------------------------------------------------
	plt.tight_layout(pad=2.05)
	if avg_size == 0:	
		fig.savefig('assTime_comparison.png', dpi=400)
	else: 
		fig.savefig('assTime_comparison_avg.png', dpi=400)
	plt.show()

def plot_rae_comparison(training_data, avg_size):

	steps_h, num_episodes, reward_h, ass_time_h, epsilon_h = get_data_n_sta(100, training_data)

	fig, (ax11, ax21, ax31) = plt.subplots(nrows=3, ncols=1, sharex=True)
	ax31.set_xlabel('Number of training steps')
	#------------------------------------------------------------------------------
	if avg_size == 0: # plot a line 
		# Plotting total reward
		ax11.set_ylabel('Cumulative reward', color='#ff6600')
		ax11.plot(steps_h, reward_h, color='#ff6600', linestyle='-', marker='', \
		         linewidth=1, label='Cumulative reward')		
	else:	# Plotting average reward over x episodes
		N = len(reward_h)
		reward_avg_size = np.empty(N)
		for t in range(N):
			reward_avg_size[t] = np.mean(reward_h[max(0, t-avg_size):(t+1)])
		ax11.set_ylabel('Cumulative reward', color='#ff6600')
		ax11.scatter(steps_h, reward_avg_size, color="#ff6600", label='Cumulative reward', marker='^')
	ax11.tick_params(axis='y', labelcolor='#ff6600')	
	#------------------------------------------------------------------------------
	# Plotting epsilon
	ax12 = ax11.twinx() 	
	ax12.set_ylabel('Epsilon', color='#1544ed') 	
	ax12.plot(steps_h, epsilon_h, color='#1544ed', linestyle='-', marker='', \
	         linewidth=1, label='Epsilon')
	ax12.tick_params(axis='y', labelcolor='#1544ed')
	#------------------------------------------------------------------------------
	# Plotting epsilon
	ax13 = ax11.twinx() 	
	ax13.invert_yaxis()
	if avg_size == 0: # plot a line 	
		ax13.set_ylabel('Association time (s)', color='#009933') 	
		ax13.plot(steps_h, ass_time_h, color='#009933', linestyle='--', marker='', \
		         linewidth=1, label='Association time (s)')
	else: 
		N = len(ass_time_h)
		ass_time_avg_size = np.empty(N)
		for t in range(N):
			ass_time_avg_size[t] = np.mean(ass_time_h[max(0, t-avg_size):(t+1)])
		ax13.set_ylabel('Association time', color='#009933')
		ax13.scatter(steps_h, ass_time_avg_size, color="#009933", label='Association time (s)', marker='.')
	ax13.spines['right'].set_position(('outward', 60))  
	ax13.tick_params(axis='y', labelcolor='#009933')
	fig.legend(bbox_to_anchor=(0.5,0.95), borderaxespad=0, loc="lower center",
	        ncol=3, fancybox=True, shadow=True)
	#------------------------------------------------------------------------------
	#------------------------------------------------------------------------------
	steps_h, num_episodes, reward_h, ass_time_h, epsilon_h = get_data_n_sta(700, training_data)
	if avg_size == 0: # plot a line 
		# Plotting total reward
		ax21.set_ylabel('Cumulative reward', color='#ff6600')
		ax21.plot(steps_h, reward_h, color='#ff6600', linestyle='-', marker='', \
		         linewidth=1, label='Cumulative reward')		
	else:	# Plotting average reward over x episodes
		N = len(reward_h)
		reward_avg_size = np.empty(N)
		for t in range(N):
			reward_avg_size[t] = np.mean(reward_h[max(0, t-avg_size):(t+1)])
		ax21.set_ylabel('Cumulative reward', color='#ff6600')
		ax21.scatter(steps_h, reward_avg_size, color="#ff6600", label='Cumulative reward', marker='^')
	ax21.tick_params(axis='y', labelcolor='#ff6600')	
	#------------------------------------------------------------------------------
	# Plotting epsilon
	ax22 = ax21.twinx() 	
	ax22.set_ylabel('Epsilon', color='#1544ed') 	
	ax22.plot(steps_h, epsilon_h, color='#1544ed', linestyle='-', marker='', \
	         linewidth=1, label='Epsilon')
	ax22.tick_params(axis='y', labelcolor='#1544ed')
	#------------------------------------------------------------------------------
	# Plotting epsilon
	ax23 = ax21.twinx() 	
	ax23.invert_yaxis()
	if avg_size == 0: # plot a line 	
		ax23.set_ylabel('Association time (s)', color='#009933') 	
		ax23.plot(steps_h, ass_time_h, color='#009933', linestyle='--', marker='', \
		         linewidth=1, label='Association time (s)')
	else: 
		N = len(ass_time_h)
		ass_time_avg_size = np.empty(N)
		for t in range(N):
			ass_time_avg_size[t] = np.mean(ass_time_h[max(0, t-avg_size):(t+1)])
		ax23.set_ylabel('Association time', color='#009933')
		ax23.scatter(steps_h, ass_time_avg_size, color="#009933", label='Association time (s)', marker='.')
	ax23.spines['right'].set_position(('outward', 60))  
	ax23.tick_params(axis='y', labelcolor='#009933')
	#------------------------------------------------------------------------------
	#------------------------------------------------------------------------------
	steps_h, num_episodes, reward_h, ass_time_h, epsilon_h = get_data_n_sta(1100, training_data)
	if avg_size == 0: # plot a line 
		# Plotting total reward
		ax31.set_ylabel('Cumulative reward', color='#ff6600')
		ax31.plot(steps_h, reward_h, color='#ff6600', linestyle='-', marker='', \
		         linewidth=1, label='Cumulative reward')		
	else:	# Plotting average reward over x episodes
		N = len(reward_h)
		reward_avg_size = np.empty(N)
		for t in range(N):
			reward_avg_size[t] = np.mean(reward_h[max(0, t-avg_size):(t+1)])
		ax31.set_ylabel('Average Cumulative reward', color='#ff6600')
		ax31.scatter(steps_h, reward_avg_size, color="#ff6600", label='Cumulative reward', marker='^')
	ax31.tick_params(axis='y', labelcolor='#ff6600')	
	#------------------------------------------------------------------------------
	# Plotting epsilon
	ax32 = ax31.twinx() 	
	ax32.set_ylabel('Epsilon', color='#1544ed') 	
	ax32.plot(steps_h, epsilon_h, color='#1544ed', linestyle='-', marker='', \
	         linewidth=1, label='Epsilon')
	ax32.tick_params(axis='y', labelcolor='#1544ed')
	#------------------------------------------------------------------------------
	# Plotting epsilon
	ax33 = ax31.twinx() 	
	ax33.invert_yaxis()
	if avg_size == 0: # plot a line 	
		ax33.set_ylabel('Association time (s)', color='#009933') 	
		ax33.plot(steps_h, ass_time_h, color='#009933', linestyle='--', marker='', \
		         linewidth=1, label='Association time (s)')
	else: 
		N = len(ass_time_h)
		ass_time_avg_size = np.empty(N)
		for t in range(N):
			ass_time_avg_size[t] = np.mean(ass_time_h[max(0, t-avg_size):(t+1)])
		ax33.set_ylabel('Association time', color='#009933')
		ax33.scatter(steps_h, ass_time_avg_size, color="#009933", label='Association time (s)', marker='.')
	ax33.spines['right'].set_position(('outward', 60))  
	ax33.tick_params(axis='y', labelcolor='#009933')
	#------------------------------------------------------------------------------
	# Plotting secondary x label (Number of training episodes)
	ax34 = ax31.twiny() 
	secondary_label_ = np.arange(0, len(steps_h), int(len(steps_h)/7))
	secondary_label = [int(num_episodes[i]) for i in secondary_label_]
	label_pos = [steps_h[i] for i in secondary_label_]
	ax34.set_xticks(label_pos)
	ax34.set_xticklabels(secondary_label)
	ax34.xaxis.set_ticks_position('bottom')
	ax34.xaxis.set_label_position('bottom')
	ax34.spines['bottom'].set_position(('outward', 36))
	ax34.set_xlabel('Number of training episodes')
	ax34.set_xlim(ax31.get_xlim())
	#ax3.grid(True)
	#------------------------------------------------------------------------------
	plt.tight_layout(pad=2.05, h_pad=0.2)
	if avg_size == 0:	
		fig.savefig('rae_comparison.png')
	else: 
		fig.savefig('rae_comparison_avg.png')
	plt.show()

def get_data_n_sta(n_sta, data):
	
	if (n_sta == 0):
		num_episodes = np.arange(1, len(data[:,0]))
		new_data = data
	else:
		new_data = data[0,:]
		num_episodes = [0]
		for i in range (len(data[:,0])):
			if (data[i,0] == n_sta):
				new_data = np.vstack([new_data, data[i,:]])
				num_episodes = np.append(num_episodes,i)
		#new_data = np.column_stack((new_data, num_episodes))
	new_data = np.delete(new_data, 0, 0)
	steps_h = new_data[:,-1]
	reward_h = new_data[:,-2]
	epsilon_h = new_data[:,-3]
	ass_time_h = new_data[:,2]

	return (steps_h, num_episodes, reward_h, ass_time_h, epsilon_h)

if __name__ == '__main__':
	
	# Extracting necessary data
	#------------------------------------------------------------------------------
	training_data = load_binary('training_data', 'data')
	#------------------------------------------------------------------------------
	# Plotting training curves tracking the agent’s cumulative reward 
	n_sta = 0	
	avg_size = 0
	plot_reward_epsilon(n_sta, training_data, avg_size)
	avg_size = 15
	plot_reward_epsilon(n_sta, training_data, avg_size)
	# Plotting training curves tracking the agent’s cumulative reward and 
	# the association time for a certain number of stations
	n_sta = 900	
	avg_size = 0
	plot_reward_assTime_epsilon(n_sta, training_data, avg_size)
	avg_size = 15
	plot_reward_assTime_epsilon(n_sta, training_data, avg_size)
	avg_size = 0
	# Plotting training curves tracking the association time
	plot_assTime_comparison(training_data, avg_size)
	avg_size = 15
	plot_assTime_comparison(training_data, avg_size)