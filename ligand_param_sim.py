import math, argparse, parmed
import tempfile, os
from openforcefield.topology import Molecule
from openforcefield.typing.engines.smirnoff import ForceField as sFF
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
####################################
molecule = Molecule('/home/ee/Documents/Evans_lab/projects/MD/IDPs/p27/SJ403_swissparam/SJ403_SM_for_p27.sdf', file_format='SDF')
force_field = sFF('openff-1.2.0.offxml')
# Parametrize the ligand molecule by creating a Topology object from it
sm_system = force_field.create_openmm_system(molecule.to_topology())
sm_pdb = PDBFile('/home/ee/Documents/Evans_lab/projects/MD/IDPs/p27/SJ403_swissparam/homeSJ403_SM_for_p27.pdb')

# Convert OpenMM System object containing ligand parameters into a ParmEd Structure.
ligand_structure = parmed.openmm.load_topology(sm_pdb.topology,sm_system,xyz=sm_pdb.positions)

# protein_system = omm_forcefield.createSystem(protein_pdb.topology)

# Convert the protein System into a ParmEd Structure.
protein_structure = parmed.openmm.load_topology(protein_pdb.topology,protein_system,xyz=protein_pdb.positions)
# sm_protein_structure = receptor_structure + ligand_structure
# Convert the Structure to an OpenMM System in vacuum.

# complex_system = sm_protein_structure.createSystem(nonbondedMethod=PME,nonbondedCutoff=10.0*angstrom,constraints=HBonds,removeCMMotion=False)
####################################
temperature = 300*kelvin

protein_pdb = PDBFile('/home/ee/Documents/Evans_lab/projects/MD/IDPs/p27/p27_relaxed.pdb')
omm_forcefield = ForceField('/home/ee/Documents/Evans_lab/projects/MD/IDPs/a99sb-disp_openMM.xml',
							'/home/ee/Documents/Evans_lab/projects/MD/IDPs/a99SB_disp_water.xml')
modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
print('System has %d atoms' % modeller.topology.getNumAtoms())
modeller.add(ligand_mol.to_topology().to_openmm(), ligand_mol.conformers[0])
print('System has %d atoms' % modeller.topology.getNumAtoms())
modeller.addSolvent(forcefield,model='tip4pew', boxSize=(8.9,8.9,8.9), ionicStrength=0.1*molar)
modeller.addExtraParticles(forcefield)

simulation = Simulation(modeller.topology, system, integrator, platform=platform)
simulation.context.setPositions(modeller.positions)
simulation.minimizeEnergy()
######################################

###############################
molecule.generate_conformers()
molecule.compute_partial_charges_am1bcc()
########################

import math, argparse
from sys import stdout
import tempfile, os
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openforcefield.topology import Molecule
from sys import stdout

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
		if res.name != 'HOH' and res.name != 'LIG':
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

temperature = 0
platform = Platform.getPlatformByName('CUDA')
properties = {'DeviceIndex':'0,1,2', 'Precision': 'mixed', 'UseCpuPme':'false'}

molecule = Molecule('/scratch/users/eevans/IDPs/p27/SJ403_SM_for_p27.sdf', file_format='SDF')
# Create the SMIRNOFF template generator with the default installed force field (openff-1.0.0)
smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule, forcefield='openff-1.2.0')
forcefield = ForceField('/scratch/users/eevans/IDPs/a99sb-disp_openMM.xml','/scratch/users/eevans/IDPs/a99SB_disp_water.xml')
forcefield.registerTemplateGenerator(smirnoff.generator)
combined_pdb = PDBFile('/scratch/users/eevans/IDPs/p27/p27_SM_close.pdb')

# molecule = Molecule('/home/ee/Documents/Evans_lab/projects/MD/IDPs/p27/SJ403_swissparam/SJ403_SM_for_p27.sdf', file_format='SDF')
# # Create the SMIRNOFF template generator with the default installed force field (openff-1.0.0)
# smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule)
# forcefield = ForceField('/home/ee/Documents/Evans_lab/projects/MD/IDPs/a99sb-disp_openMM.xml',
# 							'/home/ee/Documents/Evans_lab/projects/MD/IDPs/a99SB_disp_water.xml')
# forcefield.registerTemplateGenerator(smirnoff.generator)
# combined_pdb = PDBFile('/home/ee/Documents/Evans_lab/projects/MD/IDPs/p27/p27_SM_close.pdb')

modeller = Modeller(combined_pdb.topology, combined_pdb.positions)
# modeller.add(ligand_mol.to_topology().to_openmm(), ligand_mol.conformers[0])
modeller.addSolvent(forcefield,model='tip4pew', boxSize=(8.9,8.9,8.9), ionicStrength=0.1*molar)
modeller.addExtraParticles(forcefield)
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1.2*nanometer,switchDistance=1.0*nanometer, constraints=HBonds)
modeller, system = a99sb_disp_backbone_O_H(modeller, system, True)

integrator = LangevinIntegrator(temperature*kelvin, 1/picosecond, 0.002*picoseconds)
simulation = Simulation(modeller.topology, system, integrator, platform, properties)
simulation.context.setPositions(modeller.positions)
positions = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, positions, open('./p27_SM_premin_parsleyFF.pdb','w'))
simulation.reporters.append(StateDataReporter(stdout, 5000, step=True, potentialEnergy=True, kineticEnergy=True, temperature=True, density=True, volume=True, speed=True))
simulation.minimizeEnergy()
for i in range(104,-1,-1):
	integrator.setTemperature(3*(104-i)*kelvin)
	simulation.step(2000)
simulation.step(250000) ## 0.5 ns NVT
positions = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, positions, open('./p27_SM_NVT_equil.pdb','w'))

print('running NPT equil')
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1.2*nanometer,switchDistance=1.0*nanometer, constraints=HBonds)
modeller, system = a99sb_disp_backbone_O_H(modeller, system, True)
temperature = 310.0
integrator = LangevinIntegrator(temperature*kelvin, 1/picosecond, 0.002*picoseconds)
mc_barostat = MonteCarloBarostat(1*bar, temperature*kelvin)
system.addForce(mc_barostat)
simulation = Simulation(modeller.topology, system, integrator,platform, properties)
simulation.context.setPositions(positions)
simulation.reporters.append(DCDReporter('./p27_SM_NPT_equil.dcd', 1000, enforcePeriodicBox=False))
simulation.reporters.append(StateDataReporter(stdout, 5000, step=True, potentialEnergy=True, kineticEnergy=True, temperature=True, density=True, volume=True, speed=True))
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(temperature*kelvin)
simulation.step(500000) ## 1 ns NPT
positions = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, positions, open('./p27_SM_NPT_equil_final.pdb','w'))


print('starting production')
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1.2*nanometer,switchDistance=1.0*nanometer, constraints=HBonds)
modeller, system = a99sb_disp_backbone_O_H(modeller, system, True)
temperature = 310.0
integrator = LangevinIntegrator(temperature*kelvin, 1/picosecond, 0.002*picoseconds)
mc_barostat = MonteCarloBarostat(1*bar, temperature*kelvin)
system.addForce(mc_barostat)
simulation = Simulation(modeller.topology, system, integrator,platform, properties)
simulation.context.setPositions(positions)
simulation.reporters.append(DCDReporter('./p27_SM_NPT_prod.dcd', 10000, enforcePeriodicBox=False))
simulation.reporters.append(StateDataReporter('./p27_SM_NPT_prod.csv', 10000, step=True, potentialEnergy=True, kineticEnergy=True, temperature=True, density=True, volume=True))
simulation.reporters.append(CheckpointReporter('./p27_SM_NPT_checkpnt.chk', 10000))
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(temperature*kelvin)
simulation.step(125000000)
positions = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, positions, open('./p27_SM_parsleyFF_NPT_prod.pdb','w'))
