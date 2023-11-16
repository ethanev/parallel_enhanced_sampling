import mdtraj as md
import os
import pyemma
import numpy as np
import matplotlib.pyplot as plt


def get_atom_pairs(res_df,lower,upper):
	'''
	lower and upper are residue NUMBERS STARTING AT 1...N, ie not 0 indexed
	'''
	bb_res = res_df[(res_df.resSeq>=lower)&(res_df.resSeq<=upper)]
	bb_res = bb_res[(bb_res.name=='N')|(bb_res.name=='C')|(bb_res.name=='O')|(bb_res.name=='CA')|(bb_res.name=='CB')]
	bb_res_ind = bb_res.index.tolist()
	pairs = []
	for i in bb_res_ind:
		for j in bb_res_ind:
			if j <= i:
				continue
			pairs.append([i,j])
	pairs = np.asarray(pairs)
	return pairs

def ideal_helix_distances():
	'''
	note this gives the distances in a VERY particular order...N,CA,C,O,CB and assumes the traj is in this order
	- look like mdtraj generally puts them in this order
	- for Gly original authors day to use a H on the gly for the where the C would be
	'''
	helix = md.load('/home/ee/Documents/Evans_lab/projects/MD/helix.pdb')
	res_df, conn = helix.top.to_dataframe()
	pairs = get_atom_pairs(res_df,2,7)
	N = np.unique(pairs.flatten()).shape[0]
	ideal_distances = md.compute_distances(helix, pairs)
	return ideal_distances, N

def helical_order_param(traj, start, stop):
	'''
	start and stop are residue values starting at 1
	Ntail-alpha MoRE = GSQDS RRSAD ALLRL QAMAG I
	start = 2, stop = 19 --> first window is [2..7], last is [15,20] so from S2 to A19
	Robustelli paper said there were 13 windows, im guessing they didnt use termini, but then also didnt 
	use window with the last G at the end
	***** NOTE: res 1 and 23 are ACE and NME respectively
	so Ntail-alpha MoRE = [ACE]-GSQD SRRSA DALLR LQAMA GI-[NME]
	so use start = 3, stop = 20
	'''
	s_alpha = np.zeros((traj.n_frames,))
	r_0 = 0.08 # nm
	ideal_dist, N = ideal_helix_distances()
	traj_df, traj_conn = traj.top.to_dataframe()
	for i in range(start, stop):
		j = i + 5
		if j > stop:
			return s_alpha
		real_pairs = get_atom_pairs(traj_df,i,j)
		real_dist = md.compute_distances(traj,real_pairs)
		rmsd_alpha = np.sqrt((1/(N*(N-1)))*((real_dist - ideal_dist)**2).sum(axis=1))
		order_param = (1-(rmsd_alpha/r_0)**8)/(1-(rmsd_alpha/r_0)**12)
		s_alpha += order_param 
	return s_alpha
##############################################

data_path = '/media/ee/Seagate Portable Drive/MD/IDPs/'
pdb = '/media/ee/Seagate Portable Drive/MD/IDPs/Ntail_collapsed_a99sb-disp_1_pre_sim.pdb'
helix = './helix.pdb'

topology = md.load(pdb).topology
num_traj = 30
files = [f for f in os.listdir(data_path) if '.dcd' in f]
indiv_trajs = {i+1:[] for i in range(num_traj)}
traj_rmsd = {i+1:None for i in range(num_traj)}
traj_rg = {i+1:None for i in range(num_traj)}
traj_ss = {i+1:None for i in range(num_traj)}
for f in files:
	key = int(f.split('_')[3])
	indiv_trajs[key].append(f)
for i, traj_list in indiv_trajs.items():
	start = True
	for traj in traj_list:
		if start:
			indiv_trajs[i] = md.load(os.path.join(data_path, traj), top=topology).remove_solvent()
			start = False
		else:
			t_ = md.load(os.path.join(data_path, traj), top=topology).remove_solvent()
			indiv_trajs[i] = indiv_trajs[i].join(t_)
	# traj_ss[i] = md.compute_dssp(indiv_trajs[i])
	traj_rg[i] = md.compute_rg(indiv_trajs[i])
	# traj_rmsd[i] = md.rmsd(indiv_trajs[i], indiv_trajs[i][0])

orders = []
for i, traj in indiv_trajs.items():
	if type(traj) == list:
		continue
	orders.append(helical_order_param(traj, 3, 20)[1500:])
helical_order = np.concatenate(orders)
fig = plt.figure(figsize=(5,5))
ax = plt.gca()
ax.violinplot(helical_order, showmeans=True)
plt.show()

################### NOTE STILL NEED TO DO THE RUNNING AVG SMOOTHING...should be avg of neighboring 100 frames



# avg_rg = 0
# avg_rmsd = 0
# ct = 0
rg_trajs = []
for i,traj_i in traj_rg.items():
	try:
		rg_trajs.append(traj_i[1500:])
	except:
		pass
# 	# print(t[1500:].shape, t[1500:].mean())
# 	# avg = t.mean()
# 	# avg_rg += avg
# 	# ct += 1
# trajs = np.hstack(trajs)
# print(trajs.mean(), trajs.std())
rgs = np.concatenate(rg_trajs)
fig = plt.figure(figsize=(5,5))
ax = plt.gca()
ax.violinplot(rgs, showmeans=True)
plt.show()

print(helical_order, rgs)
fig = plt.figure(figsize=(5,5))
ax = plt.gca()
ax.scatter(helical_order,rgs, c='c', s=5, alpha=0.5)
plt.show()


fig, ax = plt.subplots(1,1,figsize=(6,5))
fig, ax, misc = pyemma.plots.plot_free_energy(helical_order,rgs,ax=ax, legacy=False,kT=0.59249587,
											cbar_label='Free energy | kcal/mol', ncontours=100,nbins=100)
plt.show()

# for i, traj_i in traj_rmsd.items():
# 	print(traj_i.shape)
# 	fig = plt.figure(figsize=(5,5))
# 	ax = plt.gca()
# 	ax.plot([i for i in range(traj_i.shape[0])], traj_i)
# 	plt.show()

# ds = []
# for j,ss in traj_ss.items():
# 	ds.append(ss[1500:,:])
# ds = np.vstack(ds)
# pos = []
# # ds[ds!='H'] = 0
# # ds[ds=='H'] = 1
# for i in range(ds.shape[1]):
# 	pos.append(ds[:,i][ds[:,i]=='H'].shape[0]/ds[:,i].shape[0])
# ss_mean = [ds.mean(axis=0) for ele in pos]
# ss_std = [ds.std(axis=0) for ele in pos]

# fig = plt.figure(figsize=(5,5))
# ax = plt.gca()
# ax.errorplot([i for i in range(len(ss_mean))], ss_mean, yerr=ss_std)
# ax.plot([i for i in range(len(pos))], pos)
# plt.show()






# helicity, helicity_std = [], []
# temps = [300,317,336,356,376,398,422,446,472,500]
# for i,t in enumerate(temps):
# 	ss_m, ss_d = single_temp_ss(data_path, topology, i)
# 	print(t, ss_m, ss_d)
# 	helicity.append(ss_m)
# 	helicity_std.append(ss_d)


# fig,ax = plt.subplots(1,1,figsize=(6,5))   
# plt.xlabel('Temperature (K)', fontsize=12)
# plt.ylabel('Percent helicity', fontsize=12)
# plt.title('Gen4-CA', fontsize=14)
# plt.errorbar(temps,helicity, yerr=helicity_std, elinewidth=1, ecolor='k', capsize=4, capthick=2, c='k')
# plt.scatter(temps,helicity, c='c', edgecolor='k', s=60)
# width=1.35
# ax.spines['left'].set_linewidth(width)
# ax.spines['bottom'].set_linewidth(width)
# ax.spines['right'].set_linewidth(width)
# ax.spines['top'].set_linewidth(width)
# ax.tick_params(width=width)
# # plt.savefig('{}_helicity_v_temp.svg'.format(save_name), format='svg', dpi=1000, bbox_inches='tight')
# plt.show()

