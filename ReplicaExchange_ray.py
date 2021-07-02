import math, os, tempfile,copy
import time, random
import ray
import numpy as np
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4"
ray.init(num_cpus=8, num_gpus=3) 
# ray.init(num_cpus=3, memory=5_000_000_000, object_store_memory=5_000_000_000)
# ray.init(num_cpus=8, num_gpus=8, memory=20_000_000_000, object_store_memory=20_000_000_000) # for cluster, 20Gb memory
## ray start --head --num-gpus=2 --num-cpus=6 --memory 20000000000
# ray.init(address='auto', redis_password='5241590000000000')
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,2"

### currently cant inherit from a ray actor class...turn this on if you just want to use Replica
# @ray.remote(num_cpus=1, num_gpus=0.33)
# class Replica:
# 	def __init__(self, rep_num, replica_params, simulation_params):
# 		self._rep_num = rep_num
# 		self._replica_params = replica_params
# 		self._simulation_params = simulation_params
# 		self._initial = True

# 	def _get_integrator(self):
# 		if 'langevin' in self._simulation_params['integrator_type']:
# 			return LangevinIntegrator(self._replica_params['temp']*kelvin, 
# 									  self._simulation_params['damping']/picosecond, 
# 									  self._simulation_params['timestep']*picoseconds)
# 		else:
# 			print('need to implement integrators other than langevin')

# 	def _build(self):
# 		self.integrator = self._get_integrator()
# 		platform = Platform.getPlatformByName(self._simulation_params['platform'])
# 		if self._simulation_params['platform'] == 'CUDA':
# 			print('CUDA')
# 			properties = {'Precision':self._simulation_params['Precision'], 'UseCpuPme':self._simulation_params['UseCpuPme'], 'CudaDeviceIndex':self._replica_params['device']} 
# 			self.context = Context(self.system, self.integrator, platform, properties)
# 		else:
# 			self.context = Context(self.system, self.integrator)
# 		self.context.setPositions(self.modeller.positions)

# 	def make_simulation(self, pdb, forcefield):
# 		self._pdb = pdb
# 		self._forcefield = forcefield
# 		self.modeller = Modeller(self._pdb.topology, self._pdb.positions)
# 		self.modeller.addSolvent(self._forcefield, padding=self._simulation_params['padding']) #, model=self._simulation_params['water'], padding=self._simulation_params['padding'])
# 		if self._simulation_params['water'] != 'tip3p' and self._simulation_params['water'] != 'implicit':
# 			self.modeller.addExtraParticles(self._forcefield)
# 		self.system = self._forcefield.createSystem(self.modeller.topology, nonbondedMethod=PME, nonbondedCutoff=0.9*nanometer, vdwCutoff=1.0*nanometer, constraints=HBonds)
# 		self.topology = self.modeller.topology
# 		self._write_dir = os.path.join(self._simulation_params['write_path'],'{}'.format(self._rep_num))
# 		if not os.path.exists(self._write_dir):
# 			os.mkdir(self._write_dir)
# 		self._build()
# 		self._subclassbuild()

# 	def update(self):
# 		'''
# 		For subclass-specific parameter updating (e.g. REST2 scaling after a switch)
# 		'''
# 		pass

# 	def step(self, steps, minimize, reset_vel):
# 		print('inner step')
# 		if minimize:
# 			tolerance = 10*kilojoule/mole
# 			maxIterations = 0
# 			LocalEnergyMinimizer.minimize(self.context, tolerance, maxIterations)
# 		if reset_vel:
# 			print('setting / resetting velocities')
# 			temp = self.integrator.getTemperature()
# 			self.context.setVelocitiesToTemperature(temp*kelvin)
# 		print('running for {} steps...'.format(steps))
# 		self.integrator.step(steps)
# 		print('rep {} done!'.format(self._rep_num))
# 		self.state = self.context.getState(getPositions=True, getVelocities=True)
# 		return self.state.getTime()

# 	def get_energies(self):
# 		'''
# 		Gets the energies of a replica under all thermodynamic states 
# 		'''
# 		energies = []
# 		for i in range(self._simulation_params['num_replicas']):
# 			self.update(i)
# 			energies.append(self.context.getState(PotentialEnergy=True).getPotentialEnergy())
# 		return energies

# 	def _subclassbuild(self):
# 		'''
# 		Again, placeholder to be used by subclasses
# 		'''
# 		pass

# 	def sync(self):
# 		'''
# 		does nothing but block ray processes to sync simulations 
# 		'''
# 		pass

# @ray.remote(num_cpus=1, num_gpus=0.2)
@ray.remote(num_cpus=1, num_gpus=1)
class REST2Replica():
	def __init__(self, rep_num, replica_params, simulation_params):
		self._rep_num = rep_num
		self._step = 0
		self._replica_params = replica_params
		self._simulation_params = simulation_params
		self._initial = True
		self._R = 0.00831446261815324*kilojoules/moles/kelvin
		self._state = None
		self._reset_vel = False
		self._min = False
		self._current_chk = None
		self._prev_chk = None
		self._make_files = True

	def update(self, new_state, calc_energy=False):
		if self._initial:
			init = True
			self._initial = False
		else:
			init = False
		if new_state != self._state and not calc_energy:
			self._reset_vel = True
			self._min = True
			self._state = new_state
		scale = self._scale_to_params[new_state]
		self._parameter_scaling(self.solute_atoms, self.solvent_atoms, scale, initial=init)

		# forces = self._scale_to_params[new_state]
		# [f.updateParametersInContext(self.context) for f in forces]
		# for i,force in enumerate(self._scale_to_params[new_state]):
			# force.updateParametersInContext(self.context)
			# nonbondedforces = [f for f in [self.context.getSystem().getForce(i) for i in range(self.system.getNumForces())] if type(f) == NonbondedForce][0]
			# print(nonbondedforces.getNumExceptions())
			# for i in range(3):
				# print(force.getParticleParameters(i))

	def _write_reporters(self):
		## use context.createCheckpoint for checkpoints
		# print('reporters')
		if self._make_files:
			self._make_files = False
			self.out_name = '{}_sim_{}_restart_{}'.format(self._simulation_params['io_name'],self._rep_num,self._simulation_params['restart'])
			if self._simulation_params['dcd_freq'] != 0:
				# print('dcd file')
				dcd_name = self.out_name + '.dcd'
				self._dcd_file = DCDFile(open(os.path.join(self._write_dir,dcd_name), 'wb'), self.topology, self._simulation_params['timestep']*picoseconds)
			if self._simulation_params['state_freq'] != 0:
				# print('state file')
				state_name = self.out_name + '.csv'
				self._state_file = open(os.path.join(self._write_dir,state_name), 'w')
				self._state_file.write('time-(ps), step, potential_energy_(KJ/mol), kinetic_energy_(KJ/mol)\n')
		#### for writing	
		if self._step % self._simulation_params['chk_freq'] == 0:
			# print('chk file')
			self._prev_chk = self._current_chk
			self._current_chk = self.out_name + '_step_{}.chk'.format(self._step)
			chk_file = open(os.path.join(self._write_dir,self._current_chk),'wb')
			chk_file.write(self.context.createCheckpoint())
			chk_file.close()
			if self._prev_chk != None:
				# print('old and removing: ', os.path.join(self._write_dir,self._prev_chk))
				os.system('rm {}'.format(os.path.join(self._write_dir,self._prev_chk)))
		if self._step % self._simulation_params['dcd_freq'] == 0:
			# print('writing dcd')
			state = self.context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True)
			self._dcd_file.writeModel(state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors())
		if self._step % self._simulation_params['state_freq'] == 0:
			# print('writing state')
			if self._simulation_params['state_freq'] != self._simulation_params['dcd_freq']:
				state = self.context.getStart(getEnergy=True)
			data = [state.getTime()._value, self._step, state.getPotentialEnergy()._value, state.getKineticEnergy()._value]
			info = ','.join(str(d) for d in data)
			info += '\n'
			self._state_file.write(info)
			self._state_file.flush()
		# print('done reporters')

	def _get_integrator(self):
		if 'langevin' in self._simulation_params['integrator_type']:
			return LangevinIntegrator(self._replica_params['temp']*kelvin, 
									  self._simulation_params['damping']/picosecond, 
									  self._simulation_params['timestep']*picoseconds)
		else:
			print('need to implement integrators other than langevin')

	def _build(self):
		self.integrator = self._get_integrator()
		platform = Platform.getPlatformByName(self._simulation_params['platform'])
		
		# os.environ["CUDA_VISIBLE_DEVICES"]=self._replica_params['device']
		if self._simulation_params['platform'] == 'CUDA':
			print('making on device: ',self._replica_params['device'])
			properties = {'Precision':self._simulation_params['Precision'], 'UseCpuPme':self._simulation_params['UseCpuPme'], 'DeviceIndex':'0'} #self._replica_params['device']} 
			try:
				self.context = Context(self.system, self.integrator, platform, properties)
			except:
				print('CANNOT MAKE CONTEXT')
				exit()
		else:
			self.context = Context(self.system, self.integrator)
		self.context.setPositions(self.modeller.positions)
		positions = self.context.getState(getPositions=True).getPositions()
		init_rep_pdb = '{}_sim_{}_restart_{}.pdb'.format(self._simulation_params['io_name'],self._rep_num,self._simulation_params['restart'])
		init_rep_pdb_path = os.path.join(self._write_dir,init_rep_pdb)
		PDBFile.writeFile(self.topology, positions, open(init_rep_pdb_path,'w'))
		return

	def _subclassbuild(self):
		self.solute_atoms, self.solvent_atoms = self._solvent_solute()
		self._scale_to_params = {}
		for i, scale in enumerate(self._replica_params['t_scales']):
			# self._scale_to_params[i] = self._parameter_scaling(self.solute_atoms, self.solvent_atoms, scale, initial=init)
			self._scale_to_params[i] = scale
		self.update(self._rep_num)
		return

	def make_simulation(self, pdb, forcefield):
		self._pdb = pdb
		self._forcefield = forcefield
		self.modeller = Modeller(self._pdb.topology, self._pdb.positions)
		self.modeller.addSolvent(self._forcefield, padding=self._simulation_params['padding'], ionicStrength=self._simulation_params['ion_conc']) #, model=self._simulation_params['water'], padding=self._simulation_params['padding'])
		if self._simulation_params['water'] != 'tip3p' and self._simulation_params['water'] != 'implicit':
			self.modeller.addExtraParticles(self._forcefield)
		self.system = self._forcefield.createSystem(self.modeller.topology, nonbondedMethod=PME, nonbondedCutoff=0.9*nanometer, vdwCutoff=1.0*nanometer, constraints=HBonds)
		self.topology = self.modeller.topology
		self._write_dir = os.path.join(self._simulation_params['write_path'],'{}'.format(self._rep_num))
		if not os.path.exists(self._write_dir):
			os.mkdir(self._write_dir)
		self._build()
		self._subclassbuild()

	def step(self, steps, minimize, reset_vel, write):
		# print('inner step')
		if minimize or self._min:
			tolerance = 5*kilojoule/mole
			maxIterations = 0
			# print('minimzing')
			LocalEnergyMinimizer.minimize(self.context, tolerance, maxIterations)
			self._min = False
		if reset_vel or self._reset_vel:
			# print('setting / resetting velocities')
			temp = self._simulation_params['temp']
			self.context.setVelocitiesToTemperature(temp*kelvin)
			self._reset_vel = False
		# nonbondedforces = [f for f in [self.context.getSystem().getForce(i) for i in range(self.context.getSystem().getNumForces())] if type(f) == NonbondedForce][0]
		# for i in range(2):
			# print(nonbondedforces.getParticleParameters(i))
		# print('stepping')
		self.integrator.step(steps)
		self._step += steps
		if write:
			self._write_reporters()
		# print('inner step done')
		# self.state = self.context.getState(getPositions=True, getVelocities=True)
		# return self.state.getTime()

	def get_energies(self):
		'''
		Gets the energies of a replica under all thermodynamic states 
		'''
		energies = []
		for i in range(self._simulation_params['num_replicas']):
			self.update(i, calc_energy=True)
			energies.append(self.context.getState(getEnergy=True).getPotentialEnergy()/(self._R*self._replica_params['temp']*kelvin))
		return energies

	def _check_nans(self):
	    if np.isnan(coordinates).any():
	        raise RuntimeError(f'NaN in coordinates of {self._rep_num}')
	    if np.isnan(velocities).any():
	        raise RuntimeError(f'NaN in velocities of {self._rep_num}')

	def sync(self):
		'''
		does nothing but block ray processes to sync simulations 
		'''
		pass
		
	def _solvent_solute(self):
		solute_atoms = []
		solvent_atoms = []
		for i, res in enumerate(self.topology.residues()):
			if res.name.upper() not in ['HOH','WAT','SOL','H2O','CL','NA','MG','K','RB','LI','I','F','BR','CA']: ### generalize to ligand?
				for atom in res.atoms():
					solute_atoms.append(atom.index)
			else:
				for atom in res.atoms():
					solvent_atoms.append(atom.index)
		return solute_atoms, solvent_atoms

	def _parameter_scaling(self, solute_atoms, solvent_atoms, scale_factor, initial=False, nonbonded=True, torsions_p=True, torsions_np=False, bonds=False, angles=False):
		'''
		take all the solute-solute and solvent-solute interactions and scale them via the scale_factor.
		The scale factor (effective temp) for solu-solu is Bj/B0, while solu-solv its sqrt(Bj/B0)
		'''
		if initial:
			self.params_nonbond = {}
			self.params_nonbondexcept = {}
			self.params_torsion = {}
		if nonbonded:
			# nonbondedforces = [f for f in [self.context.getSystem().getForce(i) for i in range(self.context.getSystem().getNumForces())] if type(f) == NonbondedForce][0]
			nonbondedforces = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == NonbondedForce][0]
			# for i in range(2):
				# print(nonbondedforces.getParticleParameters(i))
			for ind in solute_atoms:
				q, sigma, eps = nonbondedforces.getParticleParameters(ind)
				if initial:
					self.params_nonbond[ind] = [q, eps]
				else:
					q, eps = self.params_nonbond[ind][0], self.params_nonbond[ind][1]
				nonbondedforces.setParticleParameters(ind, math.sqrt(scale_factor)*q, sigma, scale_factor*eps)
			for ind in range(nonbondedforces.getNumExceptions()):
				i, j, q, sigma, eps = nonbondedforces.getExceptionParameters(ind)
				if i in solute_atoms and j in solute_atoms:
					if initial:
						self.params_nonbondexcept[ind] = [q, eps]
					else:
						q, eps = self.params_nonbondexcept[ind][0], self.params_nonbondexcept[ind][1]
					nonbondedforces.setExceptionParameters(ind, i, j, scale_factor*q, sigma, scale_factor*eps)
			nonbondedforces.updateParametersInContext(self.context)
		if torsions_p:
			### set specifically for the a99sb-disp FF I custom made from Robuselli et al / Paul's github FF gromacs/desmond release
			# impropers = [4.60240, 43.93200, 4.18400]
			impropers = [43.932, 4.184,4.6024] ## openMM ff14SB
			# torsionforces = [f for f in [self.context.getSystem().getForce(i) for i in range(self.context.getSystem().getNumForces())] if type(f) == PeriodicTorsionForce][0]
			torsionforces = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == PeriodicTorsionForce][0]			
			for ind in range(torsionforces.getNumTorsions()):
				i,j,k,l,period,angle,const = torsionforces.getTorsionParameters(ind)
				if i in solute_atoms and j in solute_atoms and k in solute_atoms and l in solute_atoms:
					if const._value not in impropers or torsions_np:
						if initial:
							self.params_torsion[ind] = [const]
						else:
							const = self.params_torsion[ind][0]
						torsionforces.setTorsionParameters(ind,i,j,k,l,period,angle,scale_factor*const)
			torsionforces.updateParametersInContext(self.context)
		if bonds:
			bondforces = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == HarmonicBondForce][0]
			print('Not implemented, does nothing')
		if angles:
			angleforces = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == HarmonicAngleForce][0]
			print('Not implemented, does nothing')
		# for i in range(2):
			# print(nonbondedforces.getParticleParameters(i))
		return [nonbondedforces, torsionforces]





####################################################################
###### Base class for running multicopy simulations
###### controls general launching and coordination
###### following the openMMtools version it doesn't implement the exchanges itself, left to subclasses
class MultiReplicaSampler:
	def __init__(self, **kwargs):
		self._pdb = None
		self._write_path = None
		self._io_name = None
		self._num_replicas = None
		## to use a different number of work threads or GPUs than replicas
		self._num_workers = None 
		self._simulation_params = None
		self._minimize_replica = {}
		self.replicas = {}
		self._replica_to_state = []
		self._replica_type = None
		self._initial = True
		self._steps = 0

	@property
	def write_path(self):
		return self._write_path

	@write_path.setter
	def write_path(self, path):
		if not os.path.exists(path):
			os.mkdir(path)
		self._write_path = path

	@property
	def io_name(self):
		return self._io_name

	@io_name.setter
	def io_name(self, name):
		self._io_name = name

	@property
	def replica_type(self):
		return self._replica_type

	@replica_type.setter
	def replica_type(self, rep_type):
		self._replica_type = rep_type

	def pdb(self, pdb):
		self._pdb = PDBFile(pdb)
	
	@property
	def num_replicas(self):
		return self._num_replicas

	@num_replicas.setter
	def num_replicas(self, num):
		self._num_replicas = int(num)
		self._minimize_replica = {i:True for i in range(int(num))}

	def _add_replica_params(self, replica_params):
		self._replica_params = replica_params

	def _add_simulation_params(self, simulation_params):
		self._simulation_params = simulation_params
		self.num_replicas = self._simulation_params['num_replicas']
		self._replica_to_state = np.array([i for i in range(self.num_replicas)])
		assert 'platform' in simulation_params
		self._platform = simulation_params['platform']
		assert 'num_workers' in simulation_params
		self._num_workers = simulation_params['num_workers']
		if 'write_path' in simulation_params:
			self.write_path = simulation_params['write_path']
		if 'io_name' in simulation_params:
			self.io_name = simulation_params['io_name']

	def _write_replica_states(self):
		pass

	def launch_workers(self):
		self.replicas = {i:self.replica_type.remote(i, self._replica_params[i], self._simulation_params) for i in range(self.num_replicas)}
		synced = [ray.get(self.replicas[i].sync.remote()) for i in range(self.num_replicas)]

	def make_simulations(self, forcefield):
		for i in range(self.num_replicas):
			self.replicas[i].make_simulation.remote(self._pdb, forcefield)
		synced = [ray.get(self.replicas[i].sync.remote()) for i in range(self.num_replicas)]

	def _calc_energies(self):
		energies = ray.get([self.replicas[i].get_energies.remote() for i in range(self.num_replicas)])
		energies = np.vstack([energies[i] for i in range(len(energies))])
		return energies
			
	def step(self, steps=1, minimize=False, reset_vel=False, write=True):
		for i in range(self.num_replicas):
			print(f'running {steps} steps')
			self.replicas[i].step.remote(steps, minimize, reset_vel, write)
		self._steps += steps
		# synced = [ray.get(self.replicas[i].sync.remote()) for i in range(self.num_replicas)]

class ReplicaExchangeSampler(MultiReplicaSampler):
	def __init__(self, **kwargs):
		super(ReplicaExchangeSampler, self).__init__(**kwargs)	
		self._exchanges_attempted = {}
		self._exchanges_accepted = {}
		# index in the replica, value is the state its in
		self._neighbors = True
		self._priority = True
		self._num_swaps = 0
		self._num_swaps_prop = 0
		
	def run(self, num_runs, step_size, minimize=False, reset_vel=False):
		while num_runs > 0:
			num_runs -= 1
			print('runs left: ', num_runs)
			# print('outter calc energy')
			state_energy_matrix = self._calc_energies()
			# print('exchange')
			self._attempt_exchange(state_energy_matrix)
			# print('step')
			self.step(step_size, minimize, reset_vel)
			# print('write outer states')
			self._write_replica_states()

	def _attempt_exchange(self, state_energy_matrix):
		if self._neighbors:
			## find which sim is in which specific state (ie scaling factor or temp)
			swaps = {}
			nonpriority_swaps = []
			# print('energy: ', state_energy_matrix)
			# print('pre exchange: ',self._replica_to_state)
			for s1 in range(self.num_replicas-1):
				self._num_swaps_prop += 1
				s2 = s1+1
				i = int(np.where(self._replica_to_state==s1)[0])
				j = int(np.where(self._replica_to_state==s2)[0])
				state_i_pe_i = state_energy_matrix[i,s1]
				state_i_pe_j = state_energy_matrix[i,s2]
				state_j_pe_j = state_energy_matrix[j,s2]
				state_j_pe_i = state_energy_matrix[j,s1]
				delta_pe = (state_i_pe_j+state_j_pe_i-state_j_pe_j-state_i_pe_i)
				# print(delta_pe, math.exp(-delta_pe))
				if delta_pe <= 0 or math.exp(-delta_pe) > np.random.rand():
					if self._priority and s1%2 == 0:
						self._num_swaps += 1
						swaps[i] = s2
						swaps[j] = s1
					elif not self._priority and s1%2 ==1:
						self._num_swaps += 1
						swaps[i] = s2
						swaps[j] = s1
					else:
						nonpriority_swaps.append((i,s1,j,s2))
			for swap in nonpriority_swaps:
				if (swap[0] not in swaps and swap[2] not in swaps):
					self._num_swaps += 1
					swaps[swap[0]] = swap[3]
					swaps[swap[2]] = swap[1]
			for s in swaps:
				self._replica_to_state[s] = swaps[s]
			# print('post exchange: ', self._replica_to_state)
			[self.replicas[i].update.remote(self._replica_to_state[i]) for i in range(self.num_replicas)]
			self._priority = not self._priority


class REST2Sampler(ReplicaExchangeSampler):
	def __init__(self, **kwargs):
		super(REST2Sampler,self).__init__(**kwargs)

	def add_simulation_params(self, simulation_params):
		self._add_simulation_params(simulation_params)
		temp = self._simulation_params['temp']
		min_t = self._simulation_params['min_temp']
		max_t = self._simulation_params['max_temp']
		reps = self.num_replicas
		temps = self._set_temp_spacing(min_t,max_t,reps)
		print('temperatures: ', temps)
		self.scaling_factors = [temp/i for i in temps]
		print('scaling factors: ', self.scaling_factors)
		replica_params = {i:{'temp':temp,'t_scales':self.scaling_factors} for i, _ in enumerate(self.scaling_factors)}
		# self.rep_to_theotemp_and_scale = {i:{'theoretical_temp':temp, 'scale':scale} for temp, scale in zip(temps, scaling_factors)}
		if self._simulation_params['platform'] == 'CUDA':
			for i in range(reps):
				indexer = i%len(self._simulation_params['gpus'])
				replica_params[i]['device'] = self._simulation_params['gpus'][indexer]
		self._add_replica_params(replica_params)

	def _write_replica_states(self):
		if self._initial:
			self.rep_file = open(os.path.join(self.write_path,self.io_name+'_rep_states_restart_{}.csv'.format(self._simulation_params['restart'])),'w') 
			header = 'step,'
			header += ','.join(str(i) for i in range(self.num_replicas))
			header += '\n'
			self.rep_file.write(header)
			self._initial = False
		data_line = str(self._steps)+',' + ','.join(str(self.scaling_factors[self._replica_to_state[i]]) for i in range(self.num_replicas)) + '\n'
		self.rep_file.write(data_line)
		self.rep_file.flush()

	@staticmethod
	def _set_temp_spacing(min_t, max_t, n):
		factor = (max_t/min_t)**(1/(n-1))
		temps = []
		for i in range(1,n+1):
			temps.append(min_t*factor**(i-1))
		return temps

class BiasExchangeReplicaSampler(MultiReplicaSampler):
	def __init__(self, **kwargs):
		super(BiasExchangeReplicaSampler, self).__init__()
		self._replica_biases = {}
		self._metad_sims = {}
		self._bias_params = {}

	def add_biases(self, rep_to_bias):
		for rep in rep_to_bias:
			self._replica_biases[rep] = rep_to_bias[rep]

	def add_bias_params(self, bias_params):
		for rep in bias_params:
			self._bias_params[rep] = bias_params[rep]

	def _calc_energies(self, rep_num_all_positions):
		rep_num, positions = rep_num_all_positions[0], rep_num_all_positions[1]
		device, temp, damping, ts = self._simulation_params[rep_num]
		testsystem = testsystems.AlanineDipeptideExplicit()
		integrator = LangevinIntegrator(temp*kelvin, damping/picosecond, ts*picoseconds)
		platform = Platform.getPlatformByName(self._platform)
		if self._platform == 'CUDA':
			device = self.rep_to_gpu[str(device)]
			#### usecpupme MUST BE OFF TO RUN MULTIPLE GPU
			properties = {'Precision':'single' , 'UseCpuPme':'false', 'CudaDeviceIndex':device} 
			context = Context(testsystem.system, integrator, platform)
			# simulation = Simulation(testsystem.topology, testsystem.system, integrator, platform, properties)
		else:
			context = Context(testsystem.system, integrator, platform)
			# simulation = Simulation(testsystem.topology, testsystem.system, integrator)
		energies = {}
		#### THIS IS THE FULL PE, I ONLY WANT TO BIAS PE ... NEED TO ISOLATE THAT COMPONENT OF THE ENERGY

		for rep, pos in positions:
			context.setPositions(pos)
			state = context.getState(getEnergy=True)
			PE = state.getPotentialEnergy()
			energies[int(rep)] = PE
			# print('for potential of replica: ', rep_num, '  PE of: {} is {}'.format(rep, PE))
		return rep_num, energies

	def calc_energies(self):
		positions = [[rep_num, pos] for rep_num, pos in self._rep_to_pos.items()]
		p = multiprocessing.Pool(self._workers)
		energies = p.map(self._calc_energies, [[str(i), positions] for i in range(self.num_replicas)])
		self.energies = energies
		print('all energies', energies)
		p.close()
		p.join()

	def to_metadynamics_simulations(self):
		for rep,sim in self._simulations.items(): 
			self._metad_sims[rep] = Metadynamics(sim.system, self._replica_biases[rep],
									self._simulation_params[rep][1]*kelvin, self._bias_params[rep][0], 
									self._bias_params[rep][1]*kilojoules_per_mole, self._bias_params[rep][2], 
									saveFrequency=self._bias_params[rep][3], biasDir=self._bias_params[rep][4])

	def _meta_step(self, sim_and_steps):
		sim_and_steps[0].step(sim_and_steps[1], sim_and_steps[2])

	def meta_step(self, steps=1):
		p = multiprocessing.Pool(self._workers)
		p.map(self._meta_step, [[i, steps] for i in range(self.num_replicas)])
		p.close()

def main():
	rest2 = True
	metadynamics = False
	### REST2 testing
	if rest2:
		forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
		simulation_params = {'num_replicas':3, 'num_workers':3, 
							 'platform':'CUDA', 'gpus':['0'],
							 'min_temp':300.0, 'max_temp':450.0, 'temp':300.0,
							 'damping':1, 'timestep':0.002,
							 'Precision':'mixed', 'UseCpuPme':'false',
							 'integrator_type':'langevin', 'water':'tip3p',
							 'padding':1.0*nanometer, 'write_path':'./ala_rest_4',
							 'io_name':'ala_rest_4', 'ion_conc':0.005*molar,
							 'chk_freq':50000, 'dcd_freq':1000, 'state_freq':1000,
							 'restart':0}
		rest_replica_sampler = REST2Sampler()
		rest_replica_sampler.replica_type = REST2Replica
		rest_replica_sampler.pdb('./ala_dipep.pdb') #'./AAQAA_3_test.pdb', './IDPs/trpcage/random_trpcage.pdb', 
		rest_replica_sampler.add_simulation_params(simulation_params)
		rest_replica_sampler.launch_workers()
		rest_replica_sampler.make_simulations(forcefield)
		s1 = time.perf_counter()
		# # rep_reporters = ['dcd', 'state']
		# # rest_replica_sampler.reporters = rep_reporters
		print('equil steps')
		rest_replica_sampler.step(50000, minimize=True, reset_vel=True, write=False)
		print('runs')
		rest_replica_sampler.run(1000000, 1000)
		# rest_replica_sampler.step(1000)
		# rest_replica_sampler.step(1000)
		# rest_replica_sampler.step(1000)
		# rest_replica_sampler.step(1000)
		# rest_replica_sampler.step(1000)
		# rest_replica_sampler.step(1000)
		# rest_replica_sampler.step(1000)
		# rest_replica_sampler.step(1000)
		s2 = time.perf_counter()
		print(f"total processing time took {s2 - s1:0.4f} seconds")
		## rest_replica_sampler.calc_energies()
		exit()

	if metadynamics:
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
		bvs = {'0':[bv_0], '1':[bv_1]}
		### order of params: amplitude, gauss height, # steps to per apply, # steps per save, output dir
		bias_params = {'0': [2.0, 0.1, 500, 500, './biases_0'],
					   '1': [2.0, 0.1, 500, 500, './biases_1']}

		### order of params: device_index, temp, damping, ts
		replica_params = {'0':[0, 310.5, 5, 0.002],
						  '1':[1, 310.5, 5, 0.002]}

		replica_sampler = BiasExchangeReplicaSampler()
		replica_sampler.write_path = './test'
		replica_sampler.num_replicas = 2
		replica_sampler.num_workers = 2
		replica_sampler.platform = 'CPU'
		# replica_sampler.platform = 'CUDA'
		replica_sampler.add_replica_params(replica_params)
		replica_sampler.add_biases(bvs)
		replica_sampler.add_bias_params(bias_params)
		replica_sampler.make_simulations()

		### minimize the simulations, set velocities to temp and run 20ps equilibration
		### min doesnt work with GPU version for diala sim
		replica_sampler.minimize() 

		rep_reporters = ['dcd', 'state']
		replica_sampler.reporters = rep_reporters
		replica_sampler.step(2500)
		replica_sampler.calc_energies()
		exit()
		replica_sampler.to_metadynamics_simulations()
		replica_sampler.meta_step(1000000)

if __name__ == '__main__':
	main()