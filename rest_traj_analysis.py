#!/bin/python
## author: Ethan D. Evans
## date: 10/10/2020
import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
import pyemma

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dcd', help='path to the dcd file of interest')
	parser.add_argument('--pdb', help='path to the pdb file to be used as a topology')
	return parser.parse_args()

def main():
	args = parse_args()
	traj = md.load_dcd(args.dcd, top=args.pdb).remove_solvent()
	psi_indices, phi_indices = [6, 8, 14, 16], [4, 6, 8, 14]
	angles = md.compute_dihedrals(traj, [phi_indices, psi_indices])

	fig = plt.figure(figsize=(5,5))
	ax = plt.gca()
	pyemma.plots.plot_free_energy(angles[:,0],angles[:,1], ax=ax)
	# ax.scatter(angles[:,0],angles[:,1], c='c')
	plt.show()
if __name__ == '__main__':
	main()