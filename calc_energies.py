
from copy import deepcopy

from openff.units import unit
from openff.units.openmm import from_openmm, to_openmm
from openmm import LangevinIntegrator, app
from openmm import unit as openmm_unit

import openff.toolkit.typing.engines.smirnoff.parameters as offtk_parameters
from openff.toolkit import ForceField, Molecule



def minimize_and_visualize(molecule, forcefield):
    # Sort out our input data
    mol_topology = molecule.to_topology()
    mol_system = forcefield.create_openmm_system(
        mol_topology,
        charge_from_molecules=[molecule],
    )

    # Set up the minimization and point calculation
    integrator = LangevinIntegrator(
        300 * openmm_unit.kelvin,
        1 / openmm_unit.picosecond,
        0.002 * openmm_unit.picoseconds,
    )
    simulation = app.Simulation(mol_topology.to_openmm(), mol_system, integrator)
    simulation.context.setPositions(to_openmm(molecule.conformers[0]))

    # Get the initial energy
    initial_potential = simulation.context.getState(getEnergy=True).getPotentialEnergy()

    # Energy minimize
    simulation.minimizeEnergy()
    minimized_state = simulation.context.getState(getPositions=True, getEnergy=True)
    minimized_potential = minimized_state.getPotentialEnergy()
    minimized_coords = from_openmm(minimized_state.getPositions(asNumpy=True))

    # Visualize
    vis_mol = deepcopy(molecule)
    vis_mol.conformers[0] = minimized_coords
    view = vis_mol.visualize(backend="nglview")
    print(
        f"Initial energy is {initial_potential.format('%0.1F')};",
        f"Minimized energy is {minimized_potential.format('%0.1F')}",
    )
    return view

