#!/bin/python
## author: Ethan D. Evans
## date: 10/10/2020
import argparse, os
import pandas as pd
import numpy as np
import mdtraj as md

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', help='path to the folder containing replica folders and state csv file')
	parser.add_argument('--equil', help='length of the preproduction equilibration to subtract off from times', default=0, type=int)
	parser.add_argument('--remove', help='frames to start at...ie remove the first N frame if remove is N', default=0, type=int)
	parser.add_argument('--restart_num', help='specific restart number to run on, default is to run on all', default=None)
	return parser.parse_args()

def main():
	# python sort_replicas.py -p /media/ee/Seagate\ Portable\ Drive/MD/ala/ala_rest_1/ --equil 50000 --remove 1
	### normally you would not need the --remove above. 
	args = parse_args()
	### remove is basically only used for initial ala REST analysis, i changed code s.t. it should normally be 0
	remove = args.remove
	### 1) read in files listing replica and which state it was in at a given snapshot
	work_dir = os.listdir(args.p)
	state_files = [f for f in work_dir if 'csv' in f]
	restart_range = [f[-5] for f in state_files] ### assumes it ends with '#.csv', we want the #
	if args.restart_num != None:
		state_files = [f for f in state_files if 'restart_{}.csv'.format(args.restart_num) in f]
	state_files = [os.path.join(args.p, sf) for sf in state_files]
	### restart_states maps from the restart number to the csv file that hold the states + replica
	restart_states = {restart:pd.read_csv(state_file) for restart,state_file in zip(restart_range,state_files)}
	### replica_n will be 0...N for the different dirs, also the headers for the replicas in the csvs
	replica_n = [f for f in work_dir if 'csv' not in f]
	### get the temps and map to replicas with 0 mapping to lowest real / effective temp (ie highest REST fraction=1)
	temps_obtained = False
	for _,file in restart_states.items():
		### fix the start time, maybe later fixed the restart times if its important
		file['step'] = file['step'] - args.equil
		if not temps_obtained:
			headers = [h for h in list(file) if h != 'step']
			temps = sorted(file[headers].iloc[0,:], reverse=True)
			replica_n_to_temp = {r:i for r,i in zip(replica_n, temps)}
			temps_obtained = True

	### 2) read in DCD files one at a time and split frames into different new dcd files for each state 
	for restart in restart_states:
		state_corrected_reps = {s:[] for s in replica_n}
		sorted_f_names = {}
		for r in replica_n:
			rep_dir = os.path.join(args.p,r)
			rep_dir_files = os.listdir(rep_dir)
			f = [f for f in rep_dir_files if restart+'.dcd' in f][0]
			restart_dcd = os.path.join(rep_dir,f)
			sorted_f_names[r] = os.path.join(rep_dir, f[:-4]+'_sorted.dcd')
			top = os.path.join(rep_dir,[f for f in rep_dir_files if restart+'.pdb' in f][0])
			traj = md.load_dcd(restart_dcd, top=top)
			traj = traj[remove:]
			# print(traj.n_frames, traj.xyz.shape, traj.time, traj[0])
			### currently the traj has frames from different states	
			rep_i_states_and_restart_n = restart_states[restart][r]

			#### used the following for the trpcage sim to get only the 300k data
			# rep = '0'
			# temp = 1
			# temp_data = rep_i_states_and_restart_n[rep_i_states_and_restart_n==temp]
			# try:
			# 	state_corrected_reps[rep].append(traj[temp_data.index])
			# except:
			# 	indicies = temp_data.index
			# 	indicies = [i for i in indicies if i in range(traj.n_frames)]
			# 	state_corrected_reps[rep].append(traj[indicies])

			for rep,temp in replica_n_to_temp.items():
				temp_data = rep_i_states_and_restart_n[rep_i_states_and_restart_n==temp]
				try:
					state_corrected_reps[rep].append(traj[temp_data.index])
				except:
					indicies = temp_data.index
					indicies = [i for i in indicies if i in range(traj.n_frames)]
					state_corrected_reps[rep].append(traj[indicies])
		### 3) sort the new DCD files to give proper time ordering 
		for s, data in state_corrected_reps.items():
			if data == []:
				continue
			d_ = data[0]
			for d in data[1:]:
				d_ = d_.join(d)
			order = np.argsort(d_.time)
			d_ = d_[order]
			d_.save_dcd(sorted_f_names[s])
			
	



if __name__ == '__main__':
	main()
