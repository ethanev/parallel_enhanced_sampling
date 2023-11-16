from pymol import cmd
import os
import math
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import statistics
### run with: pymol -c ./lig_MP_interactions.py

def calc_distances(max_d = 3.5):
	atom_distances = {}
	# mp_atms = cmd.get_model('chain B and id 515 around {} and chain A'.format(max_d))
	# ca_atms = cmd.get_model('chain A around {} and chain B and id 515'.format(max_d))
	mp_atms = cmd.get_model('chain B and id 508-530 around {} and chain A'.format(max_d))
	ca_atms = cmd.get_model('chain A around {} and chain B and id 508-530'.format(max_d)) 

	for c1 in range(len(ca_atms.atom)):
		for c2 in range(len(mp_atms.atom)):
			distance=math.sqrt(sum(map(lambda f: (f[0]-f[1])**2, zip(ca_atms.atom[c1].coord,mp_atms.atom[c2].coord))))
			if distance<float(max_d):
				# atom_distances[(mp_atms.atom[c2].resn+'_'+mp_atms.atom[c2].resi+'_'+mp_atms.atom[c2].name, ca_atms.atom[c1].resn+'_'+ca_atms.atom[c1].resi+'_'+ca_atms.atom[c1].name)] = distance

				# atom_distances[mp_atms.atom[c2].resn+'_'+mp_atms.atom[c2].resi] = distance

				if mp_atms.atom[c2].resn+' '+mp_atms.atom[c2].resi in atom_distances:
					if distance >= atom_distances[mp_atms.atom[c2].resn+' '+mp_atms.atom[c2].resi]:
						continue
					else:
						atom_distances[mp_atms.atom[c2].resn+' '+mp_atms.atom[c2].resi] = distance
				else:
					atom_distances[mp_atms.atom[c2].resn+' '+mp_atms.atom[c2].resi] = distance
	return atom_distances

def min_distance(min_d = 3.0):
	atom_distances = {}
	atms = 0
	while atms == 0:
		min_d += 0.5
		# mp_atms = cmd.get_model('chain B and id 508-530 around {} and chain A'.format(min_d))
		# ca_atms = cmd.get_model('chain A around {} and chain B and id 508-530'.format(min_d))
		mp_atms = cmd.get_model('chain B and id 36447-36469 around {} and chain A'.format(min_d))
		ca_atms = cmd.get_model('chain A around {} and chain B and id 36447-36469'.format(min_d)) 
		atms = len(mp_atms.atom)
	shortest = 100
	for c1 in range(len(ca_atms.atom)):
		for c2 in range(len(mp_atms.atom)):
			distance=math.sqrt(sum(map(lambda f: (f[0]-f[1])**2, zip(ca_atms.atom[c1].coord,mp_atms.atom[c2].coord))))
			if distance<shortest:
				shortest = distance
	return shortest

def process_frame(frame, min_dist):
	cmd.load(os.path.join(path,frame))             
	if not min_dist:
		atom_distances = calc_distances()
	else:
		atom_distances = min_distance()
	for obj in cmd.get_names():
		cmd.delete(obj)
	return atom_distances


#################################################
clusters = pickle.load(open('./Gen4+CA_REMD_clustered_frames.pkl','rb'))
which_clusters = [i for i in range(1,93)]
# labels = ['C'+str(i) for i in which_clusters]
labels = ['All clusters combined']
min_lig_dists_clusters = []
print(labels)
# path = '/media/ee/EE_external/MD/8_1_CA_noncov_REMD/pdbs/0/'
explicit = True
if explicit:
	path = '/media/ee/EE_external/MD/struct81_noncov_EXPLICIT_SOLV/run/pdbs_centered/'
	frames = os.listdir(path)
	for frame in frames:
		dists = process_frame(frame, True)
		min_lig_dists_clusters.append(dists)
	fig = plt.figure(figsize=(4,3))
	plt.title('Explicit solvent Gen4+CA, median = {} '.format(round(statistics.median(min_lig_dists_clusters),2))+r'$\AA$')
	ax = plt.subplot()
	print(len(min_lig_dists_clusters))
	ax.boxplot(min_lig_dists_clusters,labels=['All structures'])
	plt.xticks(fontname='Dejavu Sans',fontsize=10)
	plt.ylabel('Min CA-to-peptide distance '+r'$\AA$',fontname='Dejavu Sans', fontsize=10)
	width=1.35
	ax.spines['left'].set_linewidth(width)
	ax.spines['bottom'].set_linewidth(width)
	ax.spines['right'].set_linewidth(width)
	ax.spines['top'].set_linewidth(width)
	ax.tick_params(width=width)
	plt.savefig('Min_exp_solv_CA_pep_dist_boxplt.svg', format='svg', dpi=1000, bbox_inches='tight')
	exit()
MP01 = ['MET 1', 'ASN 2', 'GLN 3', 'LYS 4', 'TYR 5', 'LYS 6', 'MET 7','ALA 8','LYS 9','ALA 10','CYS 11','PHE 12',
		'PHE 13','ALA 14','PHE 15','LEU 16','GLU 17','HSD 18','LEU 19','LYS 20','LYS 21','ARG 22','LYS 23','LEU 24',
		'TYR 25','PRO 26','MET 27','SER 28','GLY 29']
all_clust = True
min_dist = True
count=0
for c in which_clusters:
	near_residues = {}
	sec_ord_near_residues = {}
	min_lig_dists = []
	print(c, len(clusters[c]))
	for frame in clusters[c]:
		count += 1
		dists = process_frame(frame, min_dist)
		if min_dist:
			min_lig_dists.append(dists)
			continue
		for d in dists:
			if d not in near_residues:
				near_residues[d] = 1
			else:
				near_residues[d] += 1
		combined = []
		if len(dists) > 1:
			seen = []
			for e1 in dists:
				for e2 in dists:
					if (e1 != e2 and [e1,e2] not in seen and [e2,e1] not in seen):
						combined.append((e1+'_'+e2))
						seen.append([e1,e2])
						seen.append([e2,e1])
		if len(combined) == 0:
			continue
		for comb in combined:
			if comb not in sec_ord_near_residues:
				sec_ord_near_residues[comb] = 1
			else:
				sec_ord_near_residues[comb] += 1

	# near_residues = sorted([(k,v) for k,v in near_residues.items()], key=lambda x:x[1], reverse=True)
	# print(near_residues)
	# sec_ord_near_residues = sorted([(k,v) for k,v in sec_ord_near_residues.items()], key=lambda x:x[1], reverse=True)
	# print(sec_ord_near_residues)

	if min_dist:
		if all_clust:
			min_lig_dists_clusters += min_lig_dists
		else:
			min_lig_dists_clusters.append(min_lig_dists)
		continue

	n_to_c_near_res = sorted([(k,v) for k,v in near_residues.items()], key=lambda x:int(x[0].split(' ')[1]))
	n_to_c_near_res = {ele[0]:ele[1]/len(clusters[c]) for ele in n_to_c_near_res}
	x = [i for i in range(1,len(MP01)+1)]
	y = []
	for res in MP01:
		if res in n_to_c_near_res:
			y.append(n_to_c_near_res[res])
		else:
			y.append(0)



	colors = np.linspace(0.0,1.0,29)
	cmap = 'coolwarm'
	c_mapper = cm.get_cmap(cmap)
	colors = [c_mapper(ele) for ele in colors]
	colors = colors[::-1]

	fig = plt.figure(figsize=(8,3)) 
	ax = plt.subplot()
	ax.bar(x,y, tick_label=MP01, edgecolor='k', color=colors)
	plt.xlim([0.4,29.6])
	plt.xticks(rotation=90,fontname='Dejavu Sans',fontsize=10)
	plt.title('Cluster: {}'.format(c),fontname='Dejavu Sans', fontsize=12)
	plt.ylabel('Frames with CA < 4.5 '+r'$\AA$',fontname='Dejavu Sans', fontsize=10)
	width=1.35
	ax.spines['left'].set_linewidth(width)
	ax.spines['bottom'].set_linewidth(width)
	ax.spines['right'].set_linewidth(width)
	ax.spines['top'].set_linewidth(width)
	ax.tick_params(width=width)
	plt.savefig('CA_activesite_Carbon_residence_cluster_{}.svg'.format(c), format='svg', dpi=1000, bbox_inches='tight')

	# plt.show()
	

if min_dist:
	if all_clust:
		fig = plt.figure(figsize=(4,3))
		plt.title('median = {} '.format(round(statistics.median(min_lig_dists_clusters),2))+r'$\AA$')
	else:
		fig = plt.figure(figsize=(8,4)) 
		plt.title('Cluster 1-20',fontname='Dejavu Sans', fontsize=12)
	ax = plt.subplot()
	print(len(min_lig_dists_clusters))
	ax.boxplot(min_lig_dists_clusters,labels=labels)
	plt.xticks(fontname='Dejavu Sans',fontsize=10)
	plt.ylabel('Min CA-to-peptide distance '+r'$\AA$',fontname='Dejavu Sans', fontsize=10)
	width=1.35
	ax.spines['left'].set_linewidth(width)
	ax.spines['bottom'].set_linewidth(width)
	ax.spines['right'].set_linewidth(width)
	ax.spines['top'].set_linewidth(width)
	ax.tick_params(width=width)
	plt.savefig('Min_CA_pep_dist_boxplt_usedallclust_{}.svg'.format(all_clust), format='svg', dpi=1000, bbox_inches='tight')