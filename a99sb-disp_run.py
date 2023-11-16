#!/bin/bash
# Author: Ethan D. Evans 
# Date: 09/10/2020 (MM/DD/YYYY)

import math, argparse
from sys import stdout
import tempfile, os
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from modeller import *
# import openmmtools
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

def a99sb_disp_backbone_O_H(modeller, system, capped=False, coulscale=0.833333, ljscale=0.5):
	"""
	Require that the H of backbone amide and O of backbone carbonyl are the ONLY atoms labelled
	'O' and 'H'. Otherwise this will alter the LJ calc of those as well.
	"""
	Hs = []
	Os = []
	sigma = 0.150
	epsilon = 1.2552
	for i,res in enumerate(modeller.topology.residues()):
		if res.name != 'HOH':
			for atom in res.atoms():
				if atom.name == 'H':
					Hs.append(atom.index)
				elif atom.name == 'O':
					Os.append(atom.index)
	if capped:
		pass
	else:
		Hs, Os = Hs[1:], Os[:-1]
	nonbondedforces = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == NonbondedForce][0]
	nbps = [nonbondedforces.getExceptionParameters(i) for i in range(nonbondedforces.getNumExceptions())]
	bond_pairs = {(nbp[0],nbp[1]):1 for nbp in nbps}
	print('pre adding O--H LJ, number of exceptions ', nonbondedforces.getNumExceptions())
	for o_ind in Os:
		for h_ind in Hs:
			first = min(o_ind, h_ind)
			second = max(o_ind, h_ind)
			if (first,second) in bond_pairs:
				qq = coulscale*nonbondedforces.getParticleParameters(h_ind)[0]*nonbondedforces.getParticleParameters(o_ind)[0]
				epsilon = ljscale * epsilon
			else:
				qq = nonbondedforces.getParticleParameters(h_ind)[0]*nonbondedforces.getParticleParameters(o_ind)[0]
			nonbondedforces.addException(first, second, qq, sigma, epsilon, replace=True)
	print('post adding O--H LJ, number of exceptions ', nonbondedforces.getNumExceptions())
	return modeller, system

def get_slurm_gpus_on_node(job_id):
	os.system('echo $CUDA_VISIBLE_DEVICES > cuda_dev.{}'.format(job_id))
	f = open('cuda_dev.{}'.format(job_id),'r')
	devs = f.readline().strip().split(',')
	return devs

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', help='path to pdb file')
	parser.add_argument('-f', help='forcefield', default='a99sb-disp')
	parser.add_argument('-i', help='job ID')
	parser.add_argument('-c', help='was the protein capped? applied first and last AAs disp O--H correction', default=False, type=bool)
	parser.add_argument('--box_size', help='size of box in nm', default=6.0, type=float)
	parser.add_argument('--padding', help='padding of box in nm', default=None)
	parser.add_argument('--ion_conc', help='ion concentration in molar', default=0.05, type=float)
	parser.add_argument('--temp', help='simulation temperature in kelvin', default=300.0, type=float)
	parser.add_argument('--sm', help='path to small molecule file if using one', default=None)
	parser.add_argument('--cpu', help='use only CPU platform? Default is False', default=False, type=bool)
	parser.add_argument('--anneal_equil', help='anneal and equilibrate?', default=False, type=bool)
	parser.add_argument('--restart', help='path to check point file to restart the simulation', default=None)
	parser.add_argument('--restart_num', help='what number restart is this?')
	parser.add_argument('--xmls', help='path to xml forcefield files', default='/home/gridsan/eevans/IDPs', type=str)
	parser.add_argument('--solvate', help='solvate or not, default is False', default=False, type=bool)
	return parser.parse_args()

def main():
	args = parse_args()
	model_name = args.p.split('/')[-1][:-4]
	model_name = '_'.join(model_name.split('_')[:2])
	box_size = args.box_size
	ion_conc = args.ion_conc
	temperature = args.temp
	try:
		pad = float(args.padding)
	except:
		pass
	nb_cutoff = 1.0*nanometer
	switch_dist = 0.9*nanometer
	sim_steps = 50000000000

	### NVT min, anneal
	# temperature = 0.0
	if not args.cpu:
		platform = Platform.getPlatformByName('CUDA')
		properties = {'DeviceIndex':'0', 'Precision': 'mixed', 'UseCpuPme':'false'}
	else:
		platform = Platform.getPlatformByName('CPU')
		properties = {'Precision': 'mixed'}

	if args.f == 'a99sb-disp':
		ff = os.path.join(args.xmls,'a99sb-disp_openMM.xml')
		ff_water = os.path.join(args.xmls,'a99SB_disp_water.xml')
		forcefield = ForceField(ff,ff_water)
		# forcefield = ForceField('/scratch/users/eevans/IDPs/a99sb-disp_openMM.xml','/scratch/users/eevans/IDPs/a99SB_disp_water.xml')
		### This uses a special modeller python scripts that must be in that calling folder (in that i also changed where it imports element)
	if args.f == 'amber14':
		forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
	if args.f == 'amber99sbildn':
		forcefield = ForceField('amber99sbildn.xml', 'tip3p.xml')
	if args.sm != None:
		molecule = Molecule(args.sm, file_format='SDF')
		molecule.name = 'LIG'
		# Create the SMIRNOFF template generator with the default installed force field (openff-1.0.0)
		smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule, cache='./l1.json', forcefield='openff-1.2.0')
		forcefield.registerTemplateGenerator(smirnoff.generator)
		pdb = PDBFile(args.p)
	else:
		pdb = PDBFile(args.p)

	modeller = Modeller(pdb.topology, pdb.positions)
	if args.f == 'a99sb-disp' and args.solvate:
		print('adding water for a99sb-disp')
		if args.padding != None:
			modeller.addSolvent(forcefield, model='a99SBdisp_water', padding=pad*nanometer, ionicStrength=ion_conc*molar)
		else:
			modeller.addSolvent(forcefield, model='a99SBdisp_water', boxSize=(box_size,box_size,box_size), ionicStrength=ion_conc*molar)
		modeller.addExtraParticles(forcefield)
	elif args.solvate:
		print('not a99sb disp, adding water')
		if args.padding != None:
			modeller.addSolvent(forcefield, padding=pad*nanometer, ionicStrength=ion_conc*molar)
		else:
			modeller.addSolvent(forcefield, boxSize=(box_size,box_size,box_size), ionicStrength=ion_conc*molar)
		modeller.addExtraParticles(forcefield)

	if args.anneal_equil:
		### create the system:
		system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=nb_cutoff,switchDistance=switch_dist, constraints=HBonds)
		#### add special LJ params for the backbone amide H--carbonyl O
		if args.f == 'a99sb-disp':
			modeller, system = a99sb_disp_backbone_O_H(modeller, system, args.c)
		integrator = LangevinIntegrator(temperature*kelvin, 1/picosecond, 0.002*picoseconds)
		simulation = Simulation(modeller.topology, system, integrator, platform, properties)
		simulation.context.setPositions(modeller.positions)
		positions = simulation.context.getState(getPositions=True).getPositions()
		PDBFile.writeFile(simulation.topology, positions, open('./{}_{}_{}_initial.pdb'.format(model_name, args.f, args.i),'w'))
		# simulation.reporters.append(DCDReporter('./{}_{}_{}_NVT_anneal.dcd'.format(model_name, args.f, args.i), 10000))
		# simulation.reporters.append(StateDataReporter('./{}_{}_{}_NVT_anneal.csv'.format(model_name, args.f, args.i), 10000, step=True, potentialEnergy=True, kineticEnergy=True, temperature=True, volume=True, speed=True))
		simulation.minimizeEnergy()
		temp_ramp = int(temperature/3)
		for i in range(temp_ramp,-1,-1):
			integrator.setTemperature(3*(temp_ramp-i)*kelvin)
			simulation.step(2000)
		positions = simulation.context.getState(getPositions=True).getPositions()
		print('done annealing, running NPT')
		### NPT
		system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=nb_cutoff,switchDistance=switch_dist, constraints=HBonds)
		if args.f == 'a99sb-disp':
			modeller, system = a99sb_disp_backbone_O_H(modeller, system, args.c)
		integrator = LangevinIntegrator(temperature*kelvin, 1/picosecond, 0.002*picoseconds)
		mc_barostat = MonteCarloBarostat(1.01325*bar, temperature*kelvin)
		system.addForce(mc_barostat)
		simulation = Simulation(modeller.topology, system, integrator,platform, properties)
		simulation.context.setPositions(positions)
		# simulation.reporters.append(DCDReporter('./{}_{}_{}_NPT_equil.dcd'.format(model_name, args.f, args.i), 10000, enforcePeriodicBox=False))
		# simulation.reporters.append(StateDataReporter('./{}_{}_{}_NPT_equil.csv'.format(model_name, args.f, args.i), 10000, step=True, potentialEnergy=True, kineticEnergy=True, temperature=True, density=True, volume=True, speed=True))
		simulation.minimizeEnergy()
		simulation.context.setVelocitiesToTemperature(temperature*kelvin)
		simulation.step(500000) ## 1 ns NPT
		positions = simulation.context.getState(getPositions=True).getPositions()
		# PDBFile.writeFile(simulation.topology, positions, open('./{}_{}_{}_NPT_equil.pdb'.format(model_name, args.f, args.i),'w'))
		print('done with NPT equil, starting production run')
	
	### NPT production
	system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=nb_cutoff, switchDistance=switch_dist, constraints=HBonds)
	if args.f == 'a99sb-disp':
		modeller, system = a99sb_disp_backbone_O_H(modeller, system, args.c)
	integrator = LangevinIntegrator(temperature*kelvin, 1/picosecond, 0.002*picoseconds)
	mc_barostat = MonteCarloBarostat(1.01325*bar, temperature*kelvin) #1.01325 bar = 1 atm
	system.addForce(mc_barostat)
	simulation = Simulation(modeller.topology, system, integrator, platform, properties)
	if args.restart == None and args.solvate:
		print('no restart file, but did solvate, treating like initial run')
		if args.anneal_equil:
			simulation.context.setPositions(positions)
		else:
			simulation.context.setPositions(modeller.positions)
		simulation.reporters.append(DCDReporter('./{}_{}_{}_NPT_prod.dcd'.format(model_name, args.f, args.i), 10000))
		simulation.reporters.append(StateDataReporter('./{}_{}_{}_NPT_prod.csv'.format(model_name, args.f, args.i), 10000, step=True, potentialEnergy=True, kineticEnergy=True, temperature=True, density=True, volume=True, speed=True))
		simulation.reporters.append(CheckpointReporter('./{}_{}_{}_NPT_prod.chk'.format(model_name, args.f, args.i), 500000))
		simulation.minimizeEnergy(tolerance=0.001*kilojoule/mole)
		positions = simulation.context.getState(getPositions=True).getPositions()
		PDBFile.writeFile(simulation.topology, positions, open('./{}_{}_{}_pre_NPT_prod.pdb'.format(model_name, args.f, args.i),'w'))
		simulation.context.setVelocitiesToTemperature(temperature*kelvin) 
		simulation.step(sim_steps) 
		positions = simulation.context.getState(getPositions=True).getPositions()
		PDBFile.writeFile(simulation.topology, positions, open('./{}_{}_{}_NPT_prod_end.pdb'.format(model_name, args.f, args.i),'w'))
		print('done')
	else:
		if args.restart_num == None:
			print('please specify a --restart_num')
			exit()
		if args.restart:
			print('setting positions to chk point')
			simulation.loadCheckpoint(args.restart)
		else:
			print('setting positions to pdb...')
			positions = simulation.context.getState(getPositions=True).getPositions()
			simulation.context.setPositions(modeller.positions)
			simulation.context.setVelocitiesToTemperature(temperature*kelvin) 
		# r_name = args.restart.split('_')
		# r_name[3] = args.i
		# r_name = '_'.join(r_name)
		simulation.reporters.append(DCDReporter('./{}_{}_{}_NPT_prod_restart_{}.dcd'.format(model_name, args.f, args.i,args.restart_num), 10000))
		simulation.reporters.append(StateDataReporter('./{}_{}_{}_NPT_prod_restart_{}.csv'.format(model_name, args.f, args.i,args.restart_num), 10000, step=True, potentialEnergy=True, kineticEnergy=True, temperature=True, density=True, volume=True, speed=True))
		simulation.reporters.append(CheckpointReporter('./{}_{}_{}_NPT_prod_restart_{}.chk'.format(model_name, args.f, args.i,args.restart_num), 500000))
		positions = simulation.context.getState(getPositions=True).getPositions()
		PDBFile.writeFile(simulation.topology, positions, open('./{}_{}_{}_NPT_prod_restart_{}.pdb'.format(model_name, args.f, args.i, args.restart_num),'w'))
		simulation.step(sim_steps) # 250 ns NPT ~5 days since its ~42 ns/day on k20s on C3
		positions = simulation.context.getState(getPositions=True).getPositions()
		PDBFile.writeFile(simulation.topology, positions, open('./{}_{}_{}_NPT_prod_restart_{}_end.pdb'.format(model_name, args.f, args.i, args.restart_num),'w'))
		print('done')
	# do this through system.addConstraint(particle1, particle2,distance)
	# likely would have to add a constraint for this...
		# ; angle-derived constraints forOH and SH groups in proteins
		# 	; The constraint A-C isCAlculated from the angle A-B-C and bonds A-B, B-C.
		# 	C  HO      0.195074
		# 	CA HO      0.195074
		# 	CT HO      0.194132
		# 	CT HS      0.235935


if __name__ == '__main__':
	main()