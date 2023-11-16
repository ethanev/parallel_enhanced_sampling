
###### first fix the geometry with inexpensive sim:
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
platform = Platform.getPlatformByName('CUDA')
properties = {'DeviceIndex': '0', 'Precision': 'mixed', 'UseCpuPme':'true'}
pdb = PDBFile('./GNSRV.pdb')
pdb.topology.addBond(list(pdb.topology.atoms())[58], list(pdb.topology.atoms())[0], type='Amide')
forcefield = ForceField('amber99sbildn.xml','amber99_obc.xml')
modeller = Modeller(pdb.topology, pdb.positions)
system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff) #, constraints=HBonds)
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds) # 0.002*picoseconds)
simulation = Simulation(modeller.topology, system, integrator, platform, properties)
simulation.context.setPositions(modeller.positions)
simulation.reporters.append(PDBReporter('GNSRV_ildn_implicit.pdb', 100000))
# simulation.reporters.append(DCDReporter('/scratch/users/eevans/cyc_pep/GNSRV_ildn_implicit.pdb', 200000))
simulation.minimizeEnergy()
for i in range(100,-1,-1):
    integrator.setTemperature(3*(100-i)*kelvin)
    simulation.step(2000)

position = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, position, open('GNSRV_ildn_implicit_1.pdb','w'))

### solvate the proper angle structure
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
platform = Platform.getPlatformByName('CUDA')
properties = {'DeviceIndex': '0', 'Precision': 'mixed', 'UseCpuPme':'true'}
pdb = PDBFile('/scratch/users/eevans/cyc_pep/GNSRV_ildn_implicit.pdb')
pdb.topology.addBond(list(pdb.topology.atoms())[58], list(pdb.topology.atoms())[0], type='Amide')
forcefield = ForceField('amber99sbildn.xml', 'tip3p.xml')
modeller = Modeller(pdb.topology, pdb.positions)
modeller.addSolvent(forcefield, model='tip3p', padding=1*nanometer)
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1.2*nanometer) #, constraints=HBonds)
integrator = LangevinIntegrator(0*kelvin, 1/picosecond, 0.001*picoseconds)
simulation = Simulation(modeller.topology, system, integrator,platform, properties)
simulation.context.setPositions(modeller.positions)
simulation.reporters.append(PDBReporter('/scratch/users/eevans/cyc_pep/GNSRV_ildn_NVT_2.pdb', 200000))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True,kineticEnergy=True, temperature=True, volume=True, speed=True))
simulation.minimizeEnergy()
for i in range(100,-1,-1):
    integrator.setTemperature(3*(100-i)*kelvin)
    simulation.step(2000)
simulation.step(20000)

position = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, position, open('/scratch/users/eevans/cyc_pep/GNSRV_ildn_NVT_pre_NPT.pdb','w'))
# position = simulation.context.getState(getPositions=True).getPositions()
# PDBFile.writeFile(simulation.topology, position, open('/scratch/users/eevans/cyc_pep/GNSRV_ildn_NVT_pre_NPT.pdb','w'))


#### NPT
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
platform = Platform.getPlatformByName('CUDA')
properties = {'DeviceIndex': '0', 'Precision': 'double', 'UseCpuPme':'true'}
pdb = PDBFile('/scratch/users/eevans/cyc_pep/GNSRV_ildn_NVT_pre_NPT.pdb')
pdb.topology.addBond(list(pdb.topology.atoms())[58], list(pdb.topology.atoms())[0], type='Amide')
forcefield = ForceField('amber99sbildn.xml', 'tip3p.xml')
modeller = Modeller(pdb.topology, pdb.positions)
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1.2*nanometer) #, constraints=HBonds)
mc_barostat = MonteCarloBarostat(1*bar, 300*kelvin)
system.addForce(mc_barostat)
integrator = LangevinIntegrator(0*kelvin, 1/picosecond, 0.001*picoseconds)
simulation = Simulation(modeller.topology, system, integrator, platform, properties)
simulation.context.setPositions(modeller.positions)
simulation.reporters.append(PDBReporter('/scratch/users/eevans/cyc_pep/GNSRV_ildn_NPT.pdb', 200000))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True,kineticEnergy=True, temperature=True, volume=True, density=True, speed=True))
simulation.minimizeEnergy()
for i in range(100,-1,-1):
    integrator.setTemperature(3*(100-i)*kelvin)
    simulation.step(2000)
simulation.step(200000)

position = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, position, open('/scratch/users/eevans/cyc_pep/GNSRV_ildn_NPT_pre_amoeba.pdb','w'))


### write the file with water for real sim:


#### use the amoeba version of:
forces['NonbondedForce'].setUseDispersionCorrection(use_dispersion_correction)
forces['AmoebaVdwForce'].setUseDispersionCorrection(use_dispersion_correction)


#### anneal with amoeba
### may need to do so without the barostat
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
pdb = PDBFile('./GNSRV_amoeba_150ps_equil_NPT_300k.pdb')
# pdb = PDBFile('./GNSRV_ildn_NPT_pre_amoeba.pdb')
pdb.topology.addBond(list(pdb.topology.atoms())[58], list(pdb.topology.atoms())[0], type='Amide')
forcefield = ForceField('amoeba2013.xml')
modeller = Modeller(pdb.topology, pdb.positions)
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1.0*nanometer, vdwCutoff=1.2*nanometer, polarization='extrapolated')
nonbonded = [f.setForceGroup(1) for f in system.getForces() if isinstance(f,AmoebaVdwForce) or isinstance(f,AmoebaMultipoleForce)]
# integrator = MTSIntegrator(0.5*femtoseconds, [(1,1), (0,8)])
integrator = MTSIntegrator(2*femtoseconds, [(1,1), (0,8)])
thermostat = AndersenThermostat(300*kelvin, 1/picosecond)
mc_barostat = MonteCarloBarostat(1*bar, 300*kelvin)
system.addForce(thermostat)
system.addForce(mc_barostat)
platform = Platform.getPlatformByName('CUDA')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
properties = {'CudaPrecision': 'mixed', 'UseCpuPme':'true'}
simulation = Simulation(modeller.topology, system, integrator, platform, properties)
simulation.context.setPositions(modeller.positions)
simulation.reporters.append(PDBReporter('./GNSRV_amoeba_gpu_volta.pdb', 10000, enforcePeriodicBox=False))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True,kineticEnergy=True, temperature=True, density=True, volume=True, speed=True))
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(300*kelvin)
simulation.step(20000000)
# simulation.step(40000)
# simulation.saveCheckpoint('amoeba_05fs_step_40000steps.chk')
# simulation.saveState('amoeba_05fs_step_40000steps.xml')
# integrator.setStepSize(1*femtosecond)
# simulation.step(40000)
# simulation.saveCheckpoint('amoeba_1fs_step_80000steps.chk')
# simulation.saveState('amoeba_1fs_step_80000steps.xml')
# integrator.setStepSize(2*femtosecond)
simulation.step(2000000)

# force = CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
# force.addGlobalParameter("k", 500*kilocalories_per_mole/angstroms**2)
# force.addPerParticleParameter("x0")
# force.addPerParticleParameter("y0")
# force.addPerParticleParameter("z0")
# atms = list(pdb.topology.atoms())
# pos = list(pdb.positions)
# for i,ele in enumerate(zip(atms,pos)):      
#     if 'chain 0' in str(ele[0]) and ele[0].name in ('CA', 'C', 'N', 'O'):
#         force.addParticle(i, ele[1].value_in_unit(nanometers))
# system.addForce(force)




position = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, position, open('/scratch/users/eevans/cyc_pep/GNSRV_amoeba_150ps_equil_NPT_300k.pdb','w'))

run metadynamics + BEMeta
save DCD / energy / easy and automated analysis 
save the state of a simulation / restart
run on multiple CPUs+GPUs
metadynamics



PDBFile.writeFile(simulation.topology, testsystem.positions, open('./alapep.pdb','w'))

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import math, os, tempfile
import numpy as np
from openmmtools import testsystems, states, mcmc
from openmmtools.multistate import ReplicaExchangeSampler, MultiStateReporter
os.environ['CUDA_VISIBLE_DEVICES'] 
platform = Platform.getPlatformByName('CUDA')
properties = {'Precision':'mixed', 'UseCpuPme':'true', 'DeviceIndex': 'GPU-52c4daf0-7b8a-3121-163e-4c60d2340220'}
testsystem = testsystems.AlanineDipeptideExplicit()
### Bias Potential 1
dih_0 = CustomTorsionForce("theta")
dih_0.addTorsion(4,5,8,14)
dih_1 = CustomTorsionForce("theta")
dih_1.addTorsion(6,8,14,16)
# cv_force = CustomCVForce('dih_0')
# cv_force = CustomCVForce('dih_1')
# cv_force.addCollectiveVariable('dih_1',dih_1)
# cv_force.addCollectiveVariable('dih_0',dih_0)
bv_0 = BiasVariable(dih_0, -np.pi, np.pi, np.pi/10, True)
bv_1 = BiasVariable(dih_1, -np.pi, np.pi, np.pi/10, True)
meta_0 = Metadynamics(testsystem.system, [bv_0,bv_1], 310.15*kelvin, 2.0, 0.1*kilojoules_per_mole, 500, saveFrequency=500, biasDir='./biases_0')
### bias potential 2
# dih_1 = CustomTorsionForce("theta1")
# dih_1.addTorsion(6,8,14,16)
# cv_force_1 = CustomCVForce('dih_1')
# cv_force_1.addCollectiveVariable('dih_1',dih_1)
# bv_1 = BiasVariable(dih_1, -np.pi, np.pi, np.pi/10, True)
# meta_1 = Metadynamics(testsystem.system, [bv_1], 310.15*kelvin, 1.0, 0.05*kilojoules_per_mole, 500, saveFrequency=500, biasDir='./biases_1')
integrator = LangevinIntegrator(310.15*kelvin, 1.0/picosecond, 0.002*picoseconds)
simulation = Simulation(testsystem.topology, testsystem.system, integrator, platform, properties)
simulation.context.setPositions(testsystem.positions)
simulation.context.setVelocitiesToTemperature(310.15*kelvin)
simulation.step(10000)
simulation.reporters.append(DCDReporter('ala.dcd', 5000))
simulation.reporters.append(StateDataReporter(stdout,5000, step=True, 
    potentialEnergy=True, temperature=True, progress=True, remainingTime=True, 
    speed=True, totalSteps=5000000, separator=','))
# Run small-scale simulation (10ns, 5*10^6 steps) and plot the free energy landscape
meta_0.step(simulation, 5000000)

## define the states
# bias_states = [meta_0, meta_1]
# bias_states = [states.ThermodynamicState(system=testsystem.system, temperature=310.15*kelvin),states.ThermodynamicState(system=testsystem.system, temperature=310.15*kelvin)]
# meta_0 = Metadynamics(bias_states[0].system, [bv_0], 310.15*kelvin, 1.0, 0.05*kilojoules_per_mole, 500, saveFrequency=500, biasDir='./biases_0')
# meta_1 = Metadynamics(bias_states[1].system, [bv_1], 310.15*kelvin, 1.0, 0.05*kilojoules_per_mole, 500, saveFrequency=500, biasDir='./biases_1')
# move = mcmc.GHMCMove(timestep=2.0*femtoseconds, collision_rate=2/picosecond, n_steps=1000)
# simulation = ReplicaExchangeSampler(mcmc_moves=move, number_of_iterations=1000)
# storage_path = './PT_MetaD'
# reporter = MultiStateReporter(storage_path, checkpoint_interval=1000)
# simulation.create(thermodynamic_states=[meta_0, meta_1],sampler_states=states.SamplerState(testsystem.positions),storage=reporter)
# simulation.minimize()
# simulation.context.setVelocitiesToTemperature(310.15*kelvin)
# simulation.equilibrate(5)
# simulation.run(n_iterations=20)



integrator = mm.LangevinIntegrator(310.15*kelvin, 1.0/picosecond, 0.002*picoseconds)
simulation = Simulation(molecule.topology, system, integrator)
simulation.context.setPositions(positions)
simulation.context.setVelocitiesToTemperature(310.15*kelvin)
simulation.step(100)
simulation.reporters.append(DCDReporter('ala_test.dcd', 5000))
simulation.reporters.append(StateDataReporter('ala_test.out', 5000, step=True, 
    potentialEnergy=True, temperature=True, progress=True, remainingTime=True, 
    speed=True, totalSteps=5000000, separator=','))
# Run small-scale simulation (10ns, 5*10^6 steps) and plot the free energy landscape
meta.step(simulation, 5000000)





##### understanding checkpoint loading:
from simtk.unit import *
import math, os, tempfile, time
os.environ['OPENMM_CPU_THREADS'] = '2'
import numpy as np
from sys import stdout
from simtk.openmm.app import *
from simtk.openmm import *
from openmmtools import testsystems, states, mcmc
# from openmmtools.multistate import ReplicaExchangeSampler, MultiStateReporter
# platform = Platform.getPlatformByName('CUDA')
# properties = {'Precision':'mixed', 'UseCpuPme':'true', 'DeviceIndex': '0'}
testsystem = testsystems.AlanineDipeptideExplicit()
integrator = LangevinIntegrator(310.15*kelvin, 1.0/picosecond, 0.002*picoseconds)
simulation = Simulation(testsystem.topology, testsystem.system, integrator) #, platform, properties)
simulation.context.setPositions(testsystem.positions)
# simulation.context.setPositions(simulation.context.getState(getPositions=True).getPositions())
simulation.context.setVelocitiesToTemperature(310.15*kelvin)
simulation.reporters.append(DCDReporter('ala.dcd', 5))
simulation.reporters.append(StateDataReporter(stdout,5, step=True, volume=True,
	potentialEnergy=True, kineticEnergy=True, temperature=True, speed=True, density=True))
s1 = time.perf_counter()
simulation.minimizeEnergy()
simulation.step(100)
s2 = time.perf_counter()
print(f"min and 100 steps took {s2 - s1:0.4f} seconds")
positions = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, positions, open('output_1.pdb', 'w'))
simulation.saveCheckpoint('test_1.chk')



from simtk.unit import *
import math, os, tempfile, time
os.environ['OPENMM_CPU_THREADS'] = '2'
import numpy as np
from sys import stdout
from simtk.openmm.app import *
from simtk.openmm import *
from openmmtools import testsystems, states, mcmc
# from openmmtools.multistate import ReplicaExchangeSampler, MultiStateReporter
# platform = Platform.getPlatformByName('CUDA')
# properties = {'Precision':'mixed', 'UseCpuPme':'true', 'DeviceIndex': '0'}
testsystem = testsystems.AlanineDipeptideExplicit()
integrator = LangevinIntegrator(310.15*kelvin, 1.0/picosecond, 0.002*picoseconds)
simulation = Simulation(testsystem.topology, testsystem.system, integrator) #, platform, properties)
simulation.loadCheckpoint('./test_1.chk')
simulation.context.setPositions(simulation.context.getState(getPositions=True).getPositions())
simulation.reporters.append(DCDReporter('ala.dcd', 5))
simulation.reporters.append(StateDataReporter(stdout,5, step=True, volume=True,
	potentialEnergy=True, kineticEnergy=True, temperature=True, speed=True, density=True))
s1 = time.perf_counter()
simulation.step(100)
s2 = time.perf_counter()
print(f"min and 100 steps took {s2 - s1:0.4f} seconds")
positions = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, positions, open('output_2.pdb', 'w'))
simulation.saveCheckpoint('test_2.chk')




meta_0 = Metadynamics(testsystem.system, [bv_0,bv_1], 310.15*kelvin, 2.0, 0.1*kilojoules_per_mole, 500, saveFrequency=500, biasDir='./biases_0')
### bias potential 2
# dih_1 = CustomTorsionForce("theta1")
# dih_1.addTorsion(6,8,14,16)
# cv_force_1 = CustomCVForce('dih_1')
# cv_force_1.addCollectiveVariable('dih_1',dih_1)
# bv_1 = BiasVariable(dih_1, -np.pi, np.pi, np.pi/10, True)
# meta_1 = Metadynamics(testsystem.system, [bv_1], 310.15*kelvin, 1.0, 0.05*kilojoules_per_mole, 500, saveFrequency=500, biasDir='./biases_1')

# Run small-scale simulation (10ns, 5*10^6 steps) and plot the free energy landscape
meta_0.step(simulation, 5000000)