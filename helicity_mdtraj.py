import mdtraj as md
import os
import numpy as np
import matplotlib.pyplot as plt

save_name = '8_1_combined'
name = 'Gen4 combined'
data_path = ['/media/ee/EE_external/MD/8_1_test32_REMD_1/test32_output/', 
			 '/media/ee/EE_external/MD/8_1_test32_REMD_1/test32_output/sorted_supercloud/',
			 '/media/ee/EE_external/MD/8_1_test37_REMD_2/test37_output/',
			 '/media/ee/EE_external/MD/8_1_test37_REMD_2/test37_output/sorted_supercloud/']
topology = ['/media/ee/EE_external/MD/8_1_test32_REMD_1/8_1_test32_QwikMD.psf',
			'/media/ee/EE_external/MD/8_1_test32_REMD_1/8_1_test32_QwikMD.psf',
			'/media/ee/EE_external/MD/8_1_test37_REMD_2/8_1_test37_QwikMD.psf',
			'/media/ee/EE_external/MD/8_1_test37_REMD_2/8_1_test37_QwikMD.psf']

# save_name = '8_1_cov'
# data_path = ['/media/ee/EE_external/MD/struct81_cov_REMD/81_cov_output/',
# 			 '/media/ee/EE_external/MD/struct81_cov_REMD/81_cov_output/sorted/',
# 			 '/media/ee/EE_external/MD/struct81_cov_REMD/81_cov_output/sorted_2/']
# topology = ['/media/ee/EE_external/MD/struct81_cov_REMD/struct81_cov_REMD_QwikMD_edit.psf',
# 			'/media/ee/EE_external/MD/struct81_cov_REMD/struct81_cov_REMD_QwikMD_edit.psf',
# 			'/media/ee/EE_external/MD/struct81_cov_REMD/struct81_cov_REMD_QwikMD_edit.psf']

# data_path = ['/media/ee/EE_external/MD/struct81_cov_REMD/81_cov_output/']
# topology = ['/media/ee/EE_external/MD/struct81_cov_REMD/struct81_cov_REMD_QwikMD_edit.psf']

# def single_temp_ss(data_path, topologies, i, slice_=None):
# 	start = True
# 	for dp, topology in zip(data_path,topologies):
# 		sorted_dcds = []
# 		dp = os.path.join(dp,str(i))
# 		print(dp)
# 		dir_comps = os.listdir(dp)

# 		sorted_dcds += sorted([f for f in dir_comps if 'sort.dcd' in f])
# 		# print(sorted_dcds)
# 		if start:
# 			t = md.load(os.path.join(dp,sorted_dcds[0]), top=topology)
# 			start = False
# 		if slice_ != None:
# 			t = t[::slice_]
# 		for f in sorted_dcds[1:]:
# 			if slice_ != None:
# 				t = t[::slice_]
# 			t_ = md.load(os.path.join(dp,f), top=topology)
# 			t = t.join(t_)
# 	ss = md.compute_dssp(t)
# 	size = len(ss)
# 	ds = []
# 	for i in range(size):
# 		ds.append(ss[i,1:28][ss[i,1:28]=='H'].shape[0]/ss[i,1:28].shape[0])
# 	ds = np.asarray(ds)
# 	ss_mean, ss_std = ds.mean(), ds.std()
# 	return ss_mean, ss_std

def single_temp_ss(data_path, topologies, i, slice_=None):
	ds = []
	for dp, topology in zip(data_path,topologies):
		sorted_dcds = []
		dp = os.path.join(dp,str(i))
		print(dp)
		dir_comps = os.listdir(dp)

		sorted_dcds += sorted([f for f in dir_comps if 'sort.dcd' in f])
		# print(sorted_dcds)
		
		for f in sorted_dcds:
			t = md.load(os.path.join(dp,f), top=topology)
			if slice_ != None:
				t = t[::slice_]
			ss = md.compute_dssp(t)
			size = len(ss)
			for j in range(size):
				ds.append(ss[j,1:28][ss[j,1:28]=='H'].shape[0]/ss[j,1:28].shape[0])
	ds = np.asarray(ds)
	ss_mean, ss_std = ds.mean(), ds.std()
	return ss_mean, ss_std

helicity, helicity_std = [], []
temps = [300,317,336,356,376,398,422,446,472,500]
for i,t in enumerate(temps):
	ss_m, ss_d = single_temp_ss(data_path, topology, i)
	print(t, ss_m, ss_d)
	helicity.append(ss_m)
	helicity_std.append(ss_d)


fig,ax = plt.subplots(1,1,figsize=(6,5))   
plt.xlabel('Temperature (K)', fontsize=12)
plt.ylabel('Percent helicity', fontsize=12)
plt.title(name, fontsize=14)
plt.errorbar(temps,helicity, yerr=helicity_std, elinewidth=1, ecolor='k', capsize=4, capthick=2, c='k')
plt.scatter(temps,helicity, c='c', edgecolor='k', s=60)
width=1.35
ax.spines['left'].set_linewidth(width)
ax.spines['bottom'].set_linewidth(width)
ax.spines['right'].set_linewidth(width)
ax.spines['top'].set_linewidth(width)
ax.tick_params(width=width)
# plt.savefig('{}_helicity_v_temp.svg'.format(save_name), format='svg', dpi=1000, bbox_inches='tight')
plt.show()

