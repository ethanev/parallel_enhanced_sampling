# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 10:14:00 2021

@author: Anika
"""
#Loading OpenFF Forcefield and parmed
from openff.toolkit.typing.engines.smirnoff.forcefield import ForceField
from simtk.openmm.app import PDBFile
import parmed
from simtk import unit
import numpy as np
import argparse

#Declare Input papramaters
parser = argparse.ArgumentParser(description = 'Parameterize AB')
parser.add_argument('-f', required=True, type = str, help='File name for input PDB')

#Import Arguments
args = parser.parse_args()
File_name = args.f + '.pdb'

#Select force field of choice
parsley = ForceField('openff_unconstrained-1.3.0.offxml')

#Loading up an OpenFF Topology from amorphadiene's SMILES <can be changed to other ligands>
from openff.toolkit.topology import Molecule, Topology

AD = Molecule.from_smiles('CC1=CCC(CC1)C(=CCC=C(C)C)C', allow_undefined_stereo=True)

#Import PDB file <change name as necessary>
pdbfile = PDBFile(File_name)

#Generate topology
top_ff = Topology.from_openmm(pdbfile.topology, unique_molecules=[AD])
top_mm = top_ff.to_openmm() #OpenMM topology

#Set box vectors for OpenFF topology
top_ff.box_vectors = np.array([4, 4, 4]) * unit.nanometer

#Create an OpenMM System
openmm_sys = parsley.create_openmm_system(top_ff)

# Convert OpenMM System to a ParmEd structure.
parmed_structure = parmed.openmm.load_topology(top_mm, openmm_sys, pdbfile.positions) #Topology from smiles sequence and atom positions from PDB file

# Export GROMACS files.
parmed_structure.save('AB_OpenFF_pdb.top', overwrite=True) #topology <change name as desired>
parmed_structure.save('AB_OpenFF_pdb.gro', overwrite=True) #GRO structure file <change name as desired>

#Validate Conversion
from simtk import openmm
for force in openmm_sys.getForces():
        if isinstance(force, openmm.NonbondedForce):
                    break
print(force.getCutoffDistance())
print(force.getUseSwitchingFunction())
print(force.getNonbondedMethod() == openmm.NonbondedForce.PME)