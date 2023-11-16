import math, argparse
from sys import stdout
import tempfile, os
from openforcefield.topology import Molecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from modeller import *
nb_cutoff = 1.0*nanometer
switch_dist = 0.9*nanometer
sim_steps = 5000
temp = 310
platform = Platform.getPlatformByName('CUDA')
properties = {'DeviceIndex':'0', 'Precision': 'mixed', 'UseCpuPme':'false'}
ff = os.path.join('/home/gridsan/eevans/IDPs/','a99sb-disp_openMM.xml')
ff_water = os.path.join('/home/gridsan/eevans/IDPs/','a99SB_disp_water.xml')
forcefield = ForceField(ff,ff_water)
molecule = Molecule('/home/gridsan/eevans/IDPs/L1.sdf', file_format='SDF')
molecule.name = 'LIG'
smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule, forcefield='openff-1.2.0')
forcefield.registerTemplateGenerator(smirnoff.generator)
# pdb = PDBFile('/home/gridsan/eevans/IDPs/NUPR1_SM_7_a99sb-disp_7_pre_NPT_prod.pdb')
# system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=nb_cutoff, switchDistance=switch_dist, constraints=HBonds)
pdb = PDBFile('/home/gridsan/eevans/IDPs/NUPR1_SM_5.pdb')
modeller = Modeller(pdb.topology, pdb.positions)
modeller.addSolvent(forcefield, model='a99SBdisp_water', boxSize=Vec3(10.0,10.0,10.0)*nanometers, ionicStrength=0.1*molar)
modeller.addExtraParticles(forcefield)
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=nb_cutoff, switchDistance=switch_dist, constraints=HBonds)
modeller, system = a99sb_disp_backbone_O_H(modeller, system, False)
integrator = LangevinIntegrator(temp*kelvin, 1/picosecond, 0.002*picoseconds)
mc_barostat = MonteCarloBarostat(1.01325*bar, temp*kelvin) #1.01325 bar = 1 atm
system.addForce(mc_barostat)
simulation = Simulation(modeller.topology, system, integrator, platform, properties)

def a99sb_disp_backbone_O_H(modeller, system, capped=False, coulscale=0.833333, ljscale=0.5):
    """
    Require that the H of backbone amide and O of backbone carbonyl are the ONLY atoms labelled
    'O' and 'H'. Otherwise this will alter the LJ calc of those as well.
    """
    Hs = []
    Os = []
    sigma = 0.150
    epsilon = 1.2552
    print('getting Hs and Os')
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