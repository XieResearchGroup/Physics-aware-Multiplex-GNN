import os
import Bio
from preprocess_rna_pdb import REV_ATOM_TYPES
import numpy as np


class SampleToPDB():
    def __init__(self):
        pass

    def to_pdb(self, sample, path, name):
        # Convert sample to pdb
        unique_batches = np.unique(sample.batch.cpu().numpy())

        for batch in unique_batches:
            mask = sample.batch == batch
            self.write_pdb(sample.x[mask], path, name[batch])

    def to_xyz(self, sample, path, name, post_fix:str=''):
        # Convert sample to xyz
        unique_batches = np.unique(sample.batch.cpu().numpy())

        for batch in unique_batches:
            mask = sample.batch == batch
            self.write_xyz(sample.x[mask], path, name[batch], post_fix)

    def write_xyz(self, x, path, name, post_fix:str=''):
        atoms_pos = x[:, :3].cpu().numpy()
        atoms_pos *= 10  # Convert back to angstroms
        atoms_types = x[:, 3:7].cpu().numpy()
        atom_names = [REV_ATOM_TYPES[np.argmax(atom)] for atom in atoms_types]

        name = name.replace(".pdb", "")
        name = name + post_fix
        name = name + '.xyz' if not name.endswith('.xyz') else name
        # Save the structure as a xyz file
        os.makedirs(path, exist_ok=True)
        out_path = os.path.join(path, name)
        with open(out_path, 'w') as f:
            f.write(f"{len(atoms_pos)}\n")
            f.write(f"{name}\n")
            for atom, pos in zip(atom_names, atoms_pos):
                f.write(f"{atom} {pos[0]} {pos[1]} {pos[2]}\n")


    def write_pdb(self, x, path, name):
        atoms_pos = x[:, :3].cpu().numpy()
        atoms_pos *= 10  # Convert back to angstroms
        atoms_types = x[:, 3:7].cpu().numpy()
        atom_names = [REV_ATOM_TYPES[np.argmax(atom)] for atom in atoms_types]
    
        structure = self.create_structure(atoms_pos, atom_names, name)
        
        name = name + '.pdb' if not name.endswith('.pdb') else name
        # Save the structure as a PDB file
        os.makedirs(path, exist_ok=True)
        out_path = os.path.join(path, name)
        io = Bio.PDB.PDBIO()
        io.set_structure(structure)
        io.save(out_path)



    def create_structure(self, coords, atoms, name):
        # Create an empty structure
        structure = Bio.PDB.Structure.Structure(name)
        
        # Create a model within the structure
        model = Bio.PDB.Model.Model(0)
        structure.add(model)
        
        # Create a chain within the model
        chain = Bio.PDB.Chain.Chain('A')
        model.add(chain)
        
        # Create atoms and add them to the chain
        residue_id = 1
        for coord, atom in zip(coords, atoms):
            residue_name = 'A' # atom.get_parent().get_resname()
            
            # Create a residue
            residue = Bio.PDB.Residue.Residue((' ', residue_id, ' '), residue_name, ' ')
            chain.add(residue)
            
            # Create an atom
            new_atom = Bio.PDB.Atom.Atom(atom, coord, 0, 0, ' ', atom, 0, ' ')
            residue.add(new_atom)
            residue_id += 1
        
        return structure
        
