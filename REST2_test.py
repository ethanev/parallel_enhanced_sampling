#!/bin/bash
# Author: Ethan D. Evans 
# Date: 09/10/2020 (MM/DD/YYYY)

import math
from sys import stdout
import tempfile, os
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import openmmtools



def main():
	### CRITICAL: change impropers to the FF of interest, currently hardcoded for a14 / a99disp
	NUM_REPLICAS = 10
	temperature = 300.0
	max_temperature = 500.0
	platform = Platform.getPlatformByName('CUDA')
	properties = {'DeviceIndex':'0', 'Precision': 'mixed', 'UseCpuPme':'false'}
	pdb = PDBFile('./1l2y_trpcage.pdb')
	# pdb = PDBFile('./AAQAA_3_test.pdb')

	# forcefield = ForceField('./a99sb-disp_openMM.xml', './a99SB_disp_water.xml')
	forcefield = ForceField('amber14-all.xml', 'amber14/tip4pew.xml')
	modeller = Modeller(pdb.topology, pdb.positions)
	modeller.addSolvent(forcefield, model='tip4pew', padding=1.0*nanometer)
	# modeller.addExtraParticles(forcefield)
	system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1.0*nanometer, vdwCutoff=1.2*nanometer, constraints=HBonds)
	# ts = openmmtools.states.ThermodynamicState(system=system, temperature=temperature*kelvin)
	# integrator = openmmtools.integrators.LangevinIntegrator(temperature=temperature*kelvin)
	# context = ts.create_context(integrator, platform, properties)
	# context.setPositions(modeller.positions)
	# LocalEnergyMinimizer.minimize(context)
	# context.setVelocitiesToTemperature(300*kelvin)
	# integrator.step(1000)


	### add special LJ params for the backbone amide H -- carbonyl O
	# modeller, system = a99sb_disp_backbone_O_H(modeller, system)
	## need to make sure this isnt adding 2x (one with special LJ and one not) the params
	### scale the solute-solute and solute solvent interactions
	solute_atoms, solvent_atoms = solvent_solute(modeller)
	temps = exp_spacing(temperature, max_temperature, n=NUM_REPLICAS)
	scaling_factors = [temperature/i for i in temps]
	thermo_states = [openmmtools.states.ThermodynamicState(system=system, temperature=temperature*kelvin) for i in range(NUM_REPLICAS)]
	thermo_states = [parameter_scaling(s, solute_atoms, solvent_atoms, solute_scale, nonbonded=True, torsions_p=True) for s,solute_scale in zip(thermo_states,scaling_factors)]
	# state_contexts_ = [t_state.create_context(openmmtools.integrators.LangevinIntegrator(temperature=temperature*kelvin), platform,properties) for t_state in thermo_states]
	# state_contexts__ = []
	# for cont in state_contexts_:
	# 	cont.setPositions(modeller.positions)
	# 	state_contexts__.append(cont)

	# state_contexts__ = [c.setPositions(modeller.positions) for c in state_contexts_]

	# sampler_states = [openmmtools.states.SamplerState.from_context(c) for c in state_contexts__]
	# state = openmmtools.states.ThermodynamicState(system=system, temperature=temperature*kelvin)
	# state.system, _ = parameter_scaling(state.system, solute_atoms, solvent_atoms, solute_scale, nonbonded=True, torsions_p=True)
	move = openmmtools.mcmc.GHMCMove(timestep=0.002*picoseconds,collision_rate=1.0/picoseconds, n_steps=500)
	simulation = openmmtools.multistate.ReplicaExchangeSampler(mcmc_moves=move, number_of_iterations=100)
	# storage_path = tempfile.NamedTemporaryFile(delete=False).name + '.nc'
	storage_path = './test_2.out'
	reporter = openmmtools.multistate.MultiStateReporter(storage_path, checkpoint_interval=1)
	sampler_states = [openmmtools.states.SamplerState(modeller.positions, box_vectors=system.getDefaultPeriodicBoxVectors()) for i in range(NUM_REPLICAS)]
	simulation.create(thermodynamic_states=thermo_states, sampler_states=sampler_states, storage=reporter)
	simulation.minimize()
	print('mined')
	simulation.equilibrate(5)
	print('here')
	simulation.run()
	print('ran')
	exit()

	# testsystem = openmmtools.testsystems.AlanineDipeptideExplicit()
	# n_replicas = 3  # Number of temperature replicas.
	# T_min = 298.0 * kelvin  # Minimum temperature.
	# T_max = 600.0 * kelvin  # Maximum temperature.
	# temperatures = [T_min + (T_max - T_min) * (math.exp(float(i) / float(n_replicas-1)) - 1.0) / (math.e - 1.0) for i in range(n_replicas)]
	# thermodynamic_states = [openmmtools.states.ThermodynamicState(system=testsystem.system, temperature=T) for T in temperatures]
	# move = openmmtools.mcmc.GHMCMove(timestep=2.0*femtoseconds, n_steps=50)
	# simulation = openmmtools.multistate.ReplicaExchangeSampler(mcmc_moves=move, number_of_iterations=50)
	# storage_path = tempfile.NamedTemporaryFile(delete=False).name + '.nc'
	# reporter = openmmtools.multistate.MultiStateReporter(storage_path, checkpoint_interval=1)
	# simulation.create(thermodynamic_states=thermodynamic_states,sampler_states=openmmtools.states.SamplerState(testsystem.positions, box_vectors=testsystem.system.getDefaultPeriodicBoxVectors()),storage=reporter)
	# simulation.run(1)


	# integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
	# simulation = Simulation(modeller.topology, system, integrator, platform, properties)
	# simulation.context.setPositions(modeller.positions)
	# simulation.reporters.append(PDBReporter('./AAQAA3_REST2_01.pdb', 1000))
	# simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True,kineticEnergy=True, temperature=True, volume=True, speed=True))
	# simulation.minimizeEnergy()
	# # for i in range(100,-1,-1):
	#     # integrator.setTemperature(3*(100-i)*kelvin)
	#     # simulation.step(2000)
	# simulation.context.setVelocitiesToTemperature(300*kelvin)
	# simulation.step(200000)
	# position = simulation.context.getState(getPositions=True).getPositions()
	# PDBFile.writeFile(simulation.topology, position, open('./AAQAA_3_ildn_NVT_a-disp.pdb','w'))

	# nonbondedforces.setReactionFieldDielectric(66) # what ever robustelli useds






# ### NPT:
# from simtk.openmm.app import *
# from simtk.openmm import *
# from simtk.unit import *
# from sys import stdout
# platform = Platform.getPlatformByName('CUDA')
# properties = {'DeviceIndex': '0', 'Precision': 'mixed', 'UseCpuPme':'false'}
# pdb = PDBFile('./AAQAA_3_ildn_NVT.pdb')
# forcefield = ForceField('./a99sb-disp_openMM.xml', './a99SB_disp_water.xml')
# modeller = Modeller(pdb.topology, pdb.positions)
# modeller.addSolvent(forcefield, padding=1*nanometer)
# modeller.addExtraParticles(forcefield)
# system = forcefield.createSystem(modeller.topology, 
# 								 nonbondedMethod=PME, 
# 								 nonbondedCutoff=1.0*nanometer, 
# 								 vdwCutoff=1.2*nanometer, 
# 								 constraints=HBonds)
# mc_barostat = MonteCarloBarostat(1*bar, 300*kelvin)
# system.addForce(mc_barostat)
# integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
# simulation = Simulation(modeller.topology, system, integrator,platform, properties)
# simulation.context.setPositions(modeller.positions)
# simulation.reporters.append(PDBReporter('./AAQAA_3_ildn_NVT.pdb', 200000))
# simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True,kineticEnergy=True, temperature=True, density=True, volume=True, speed=True))
# simulation.reporters.append(DCDReporter('AAQAA_3_a99sb-disp.dcd', 5000))
# simulation.minimizeEnergy()
# simulation.context.setVelocitiesToTemperature(300*kelvin)
# simulation.step(200000)
# position = simulation.context.getState(getPositions=True).getPositions()
# PDBFile.writeFile(simulation.topology, position, open('./AAQAA_3_ildn_NpT_a-disp.pdb','w'))


# simulation.reporters.append(PDBReporter('./AAQAA_3_a99sb-disp.pdb', 10000, enforcePeriodicBox=False))


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