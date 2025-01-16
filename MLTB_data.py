import os
import re
import sys
import itertools
from torch_geometric.data import Data
import torch
import numpy as np
#from pymatgen import Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.periodic_table import Element
from scipy.sparse import csr_matrix
from tqdm import tqdm
from utils_.compute_nb import find_neighbors
from utils_.compute_tp import compute_tp_cc

#A = np.array([1, 2, 3])
#B = np.array([[1.1, 2.2, 3.3], [0.2, 0.4, 0.5], [1, 2, 3]])
#print(f"A: {A}")
#print(f"B: {B}")
#print(f"B.T: {B.T}")
#print(f"np.dot(A,B): {np.dot(A,B)}")
#sys.exit()

# Initialization
data_dir = './'  # Directory containing data folders
max_orbitals = 40  # Maximum number of orbitals for "ORBStandard"
#  awk -v a=0 '/Sorbital/{a=a+$4}/Porbital/{a=a+3*$4}/Dorbital/{a=a+5*$4}/Forbital/{a=a+7*$4}END{print a}' *.orb

######################################################################################################################

dir_list = os.listdir(data_dir)
folder_pattern = re.compile(r'^\d{3}$')
folder_numbers = sorted(int(dir_item) for dir_item in dir_list
                        if os.path.isdir(os.path.join(data_dir, dir_item)) and folder_pattern.match(dir_item))

# Check for missing folder numbers
previous_number = folder_numbers[0] - 1
for current_number in folder_numbers:
    if current_number != previous_number + 1:
        print(f"Warning: Folder number {previous_number + 1:03d} is missing.")
    previous_number = current_number

# Output the number of valid folders
print(f"Found {len(folder_numbers)} valid folders.")

######################################################################################################################
# ORBStandard
orbitals_dict = {
 'Ag': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'Al': {'orbitals': {'d': 1, 'f': 0, 'p': 4, 's': 4, 'tot': 21},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'Ar': {'orbitals': {'d': 1, 'f': 0, 'p': 2, 's': 2, 'tot': 13},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'As': {'orbitals': {'d': 1, 'f': 0, 'p': 2, 's': 2, 'tot': 13},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'Au': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'B': {'orbitals': {'d': 1, 'f': 0, 'p': 2, 's': 2, 'tot': 13},
       'radius_angstroms': 4.233418,
       'radius_au': 8.0},
 'Ba': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 5.291772,
        'radius_au': 10.0},
 'Be': {'orbitals': {'d': 0, 'f': 0, 'p': 1, 's': 4, 'tot': 7},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'Bi': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 2, 'tot': 25},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'Br': {'orbitals': {'d': 1, 'f': 0, 'p': 2, 's': 2, 'tot': 13},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'C': {'orbitals': {'d': 1, 'f': 0, 'p': 2, 's': 2, 'tot': 13},
       'radius_angstroms': 3.704241,
       'radius_au': 7.0},
 'Ca': {'orbitals': {'d': 1, 'f': 0, 'p': 2, 's': 4, 'tot': 15},
        'radius_angstroms': 4.762595,
        'radius_au': 9.0},
 'Cd': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'Cl': {'orbitals': {'d': 1, 'f': 0, 'p': 2, 's': 2, 'tot': 13},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'Co': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 4.233418,
        'radius_au': 8.0},
 'Cr': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 4.233418,
        'radius_au': 8.0},
 'Cs': {'orbitals': {'d': 1, 'f': 0, 'p': 2, 's': 4, 'tot': 15},
        'radius_angstroms': 5.291772,
        'radius_au': 10.0},
 'Cu': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 4.233418,
        'radius_au': 8.0},
 'F': {'orbitals': {'d': 1, 'f': 0, 'p': 2, 's': 2, 'tot': 13},
       'radius_angstroms': 3.704241,
       'radius_au': 7.0},
 'Fe': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 4.233418,
        'radius_au': 8.0},
 'Ga': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 2, 'tot': 25},
        'radius_angstroms': 4.233418,
        'radius_au': 8.0},
 'Ge': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 2, 'tot': 25},
        'radius_angstroms': 4.233418,
        'radius_au': 8.0},
 'H': {'orbitals': {'d': 0, 'f': 0, 'p': 1, 's': 2, 'tot': 5},
       'radius_angstroms': 3.175063,
       'radius_au': 6.0},
 'He': {'orbitals': {'d': 0, 'f': 0, 'p': 1, 's': 2, 'tot': 5},
        'radius_angstroms': 3.175063,
        'radius_au': 6.0},
 'Hf': {'orbitals': {'d': 2, 'f': 2, 'p': 2, 's': 4, 'tot': 34},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'Hg': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 4.762595,
        'radius_au': 9.0},
 'I': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 2, 'tot': 25},
       'radius_angstroms': 3.704241,
       'radius_au': 7.0},
 'In': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 2, 'tot': 25},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'Ir': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'K': {'orbitals': {'d': 1, 'f': 0, 'p': 2, 's': 4, 'tot': 15},
       'radius_angstroms': 4.762595,
       'radius_au': 9.0},
 'Kr': {'orbitals': {'d': 1, 'f': 0, 'p': 2, 's': 2, 'tot': 13},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'Li': {'orbitals': {'d': 0, 'f': 0, 'p': 1, 's': 4, 'tot': 7},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'Mg': {'orbitals': {'d': 1, 'f': 0, 'p': 2, 's': 4, 'tot': 15},
        'radius_angstroms': 4.233418,
        'radius_au': 8.0},
 'Mn': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 4.233418,
        'radius_au': 8.0},
 'Mo': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'N': {'orbitals': {'d': 1, 'f': 0, 'p': 2, 's': 2, 'tot': 13},
       'radius_angstroms': 3.704241,
       'radius_au': 7.0},
 'Na': {'orbitals': {'d': 1, 'f': 0, 'p': 2, 's': 4, 'tot': 15},
        'radius_angstroms': 4.233418,
        'radius_au': 8.0},
 'Nb': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 4.233418,
        'radius_au': 8.0},
 'Ne': {'orbitals': {'d': 1, 'f': 0, 'p': 2, 's': 2, 'tot': 13},
        'radius_angstroms': 3.175063,
        'radius_au': 6.0},
 'Ni': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 4.233418,
        'radius_au': 8.0},
 'O': {'orbitals': {'d': 1, 'f': 0, 'p': 2, 's': 2, 'tot': 13},
       'radius_angstroms': 3.704241,
       'radius_au': 7.0},
 'Os': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'P': {'orbitals': {'d': 1, 'f': 0, 'p': 2, 's': 2, 'tot': 13},
       'radius_angstroms': 3.704241,
       'radius_au': 7.0},
 'Pb': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 2, 'tot': 25},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'Pd': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'Pt': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'Rb': {'orbitals': {'d': 1, 'f': 0, 'p': 2, 's': 4, 'tot': 15},
        'radius_angstroms': 5.291772,
        'radius_au': 10.0},
 'Re': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'Rh': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'Ru': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'S': {'orbitals': {'d': 1, 'f': 0, 'p': 2, 's': 2, 'tot': 13},
       'radius_angstroms': 3.704241,
       'radius_au': 7.0},
 'Sb': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 2, 'tot': 25},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'Sc': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 4.233418,
        'radius_au': 8.0},
 'Se': {'orbitals': {'d': 1, 'f': 0, 'p': 2, 's': 2, 'tot': 13},
        'radius_angstroms': 4.233418,
        'radius_au': 8.0},
 'Si': {'orbitals': {'d': 1, 'f': 0, 'p': 2, 's': 2, 'tot': 13},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'Sn': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 2, 'tot': 25},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'Sr': {'orbitals': {'d': 1, 'f': 0, 'p': 2, 's': 4, 'tot': 15},
        'radius_angstroms': 4.762595,
        'radius_au': 9.0},
 'Ta': {'orbitals': {'d': 2, 'f': 2, 'p': 2, 's': 4, 'tot': 34},
        'radius_angstroms': 4.233418,
        'radius_au': 8.0},
 'Tc': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'Te': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 2, 'tot': 25},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'Ti': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 4.233418,
        'radius_au': 8.0},
 'Tl': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 2, 'tot': 25},
        'radius_angstroms': 3.704241,
        'radius_au': 7.0},
 'V': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
       'radius_angstroms': 4.233418,
       'radius_au': 8.0},
 'W': {'orbitals': {'d': 2, 'f': 2, 'p': 2, 's': 4, 'tot': 34},
       'radius_angstroms': 4.233418,
       'radius_au': 8.0},
 'Xe': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 2, 'tot': 25},
        'radius_angstroms': 4.233418,
        'radius_au': 8.0},
 'Y': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
       'radius_angstroms': 4.233418,
       'radius_au': 8.0},
 'Zn': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 4.233418,
        'radius_au': 8.0},
 'Zr': {'orbitals': {'d': 2, 'f': 1, 'p': 2, 's': 4, 'tot': 27},
        'radius_angstroms': 4.233418,
        'radius_au': 8.0}}
# Find max number for each of s/p/d/f orbitals
#max_s=0
#for element, data in orbitals_dict.items():
#    s_value = data['orbitals']['f']
#    if s_value > max_s:
#        max_s = s_value
#print(f"max_s: {max_s}")
#sys.exit()
# Totnumber = 4*1 + 4*3 + 2*5 + 2*7 = 40

# Cal Ham_site in 40 for each element
def Find_orbsite(ele_str):
    numorb = orbitals_dict[ele_str]['orbitals']
    s_site = list(range(numorb['s']))
    p_site = [x + 4 for x in range(3 * numorb['p'])]
    d_site = [x + 16 for x in range(5 * numorb['d'])]
    f_site = [x + 26 for x in range(7 * numorb['f'])]
    orbsite = s_site + p_site + d_site + f_site
    return orbsite
#orbsite = Find_orbsite('Zr')
#print(f"orbsite: {orbsite}")
#sys.exit() 

#def final_etot(filename):
#    ETot = 0
#    with open(filename, 'r') as file:
#        for line in file:
#            if 'FINAL_ETOT' in line:
#                parts = line.split()
#                ETot = float(parts[-2])
#                break
#    return ETot

def final_etot(filename):
    with open(filename, 'r') as file:
        for line in file:
            if 'E-Fermi' in line:
                parts = line.split()
                ETot = float(parts[-1])
                break
    return ETot

######################################################################################################################

Dataset = {}
for idx_folder, folder_number in enumerate(tqdm(folder_numbers)):

    folder_path = os.path.join(data_dir, f"{folder_number:03d}")
    stru_file_path = os.path.join(folder_path, 'POSCAR')
    HR_file_path = os.path.join(folder_path, 'data-HR-sparse_SPIN0.csr')
    scf_file_path = os.path.join(folder_path, 'E-Fermi')    

    poscar = Poscar.from_file(stru_file_path) # poscar has properities in class Structure (base: IStructure and SiteCollection)
    lattice = poscar.structure.lattice.matrix
    elements = [el.symbol for el in poscar.structure.composition.elements] 
    coords = poscar.structure.cart_coords
    num_atoms = [poscar.structure.num_sites]
    num_elements = [poscar.structure.ntypesp]
    num_atoms_element = poscar.natoms
    num_orb_element = [orbitals_dict[el]['orbitals']['tot'] for el in elements]
    num_orb_atom = [orb for atoms, orb in zip(num_atoms_element, num_orb_element) for _ in range(atoms)]
    element_atom = [ele for atoms, ele in zip(num_atoms_element, elements) for _ in range(atoms)]
    atomic_numbers = poscar.structure.atomic_numbers
    atomblock_rangeend = abre = list(itertools.accumulate(num_orb_atom))
    atomblock_rangestart = abrs = [0] + atomblock_rangeend[:-1]
    total_energy = final_etot(scf_file_path)
    #print(f"total_energy: {total_energy}")

    # GPTFF
    #lattice = np.array(lattice, dtype=np.float64)
    #coords = np.array(coords, dtype=np.float64)
    #i, j, offset, d_ij = find_neighbors(coords, lattice, 5.0, np.array([1,1,1], dtype=np.int32))

    #print(f"num_orb_element: {num_orb_element}")
    #print(f"num_orb_atom: {num_orb_atom}")
    #print(f"element_atom: {element_atom}")
    #print(f"abrs: {abrs}")
    #print(f"abre: {abre}")
    #print(f"atomblock_rangestart: {atomblock_rangestart}")
    #print(f"atomblock_rangeend: {atomblock_rangeend}")
    #sys.exit()

    #print(f"lattice: {lattice}")
    #print(f"elements: {elements}")
    #print(f"coords: {coords}")
    #print(f"num_atoms: {num_atoms}")
    #print(f"num_elements: {num_elements}")
    #print(f"num_atoms_element: {num_atoms_element}")
    #print(f"atomic_numbers: {atomic_numbers}")

    cells = []
    pairs = []
    bonds = []
    Hon = []
    Hoff = []
    maxnum = 0
    with open(HR_file_path, 'r') as hr:
        hr.readline() 
        dim_ham = int(hr.readline().split()[4])
        num_ham = int(hr.readline().split()[4])
        for idx_ham in range(num_ham):
            line = hr.readline().split()
            cell = list(map(int, line[:3]))
            if int(line[3]) != 0:
                val = list(map(float, hr.readline().split()));
                col = list(map(int, hr.readline().split()));
                row = list(map(int, hr.readline().split()));
                ham_cell = csr_matrix((val, col, row), shape=[dim_ham, dim_ham])
                #print(f"ham_cell.shape: {ham_cell.shape}")

                for a_i in range(num_atoms[0]):
                    for a_j in range(num_atoms[0]):

                        #print(f"element_atom[a_i]: {element_atom[a_i]}")
                        #print(f"element_atom[a_j]: {element_atom[a_j]}")
                        #print(f"abrs[a_i]: {abrs[a_i]}")
                        #print(f"abre[a_i]: {abre[a_i]}")
                        #print(f"abrs[a_j]: {abrs[a_j]}")
                        #print(f"abre[a_j]: {abre[a_j]}")
                        ham_block = ham_cell[ abrs[a_i]:abre[a_i], abrs[a_j]:abre[a_j] ]

                        if cell[0]==0 and cell[1]==0 and cell[2]==0 and a_i==a_j:

                            Hon_block = np.zeros((40, 40))
                            site_a_ij = np.array(Find_orbsite(element_atom[a_i]))
                            for s_i in range(ham_block.shape[0]):
                                for s_j in range(ham_block.shape[1]):
                                    Hon_block[site_a_ij[s_i],site_a_ij[s_j]] = ham_block[s_i,s_j]
                            Hon.append(Hon_block.flatten())                        

                        else:

                            Rcut = orbitals_dict[element_atom[a_i]]['radius_angstroms'] + orbitals_dict[element_atom[a_j]]['radius_angstroms'] 
                            Distance = np.linalg.norm(coords[a_j] - coords[a_i] + np.dot(cell, lattice.T))
                            #print(f"orbitals_dict[element_atom[a_i]]['radius_angstroms']: {orbitals_dict[element_atom[a_i]]['radius_angstroms']}")
                            #print(f"orbitals_dict[element_atom[a_j]]['radius_angstroms']: {orbitals_dict[element_atom[a_j]]['radius_angstroms']}")
                            #print(f"Rcut: {Rcut}")
                            #print(f"coords[a_j]: {coords[a_j]}")
                            #print(f"coords[a_i]: {coords[a_i]}")
                            #print(f"cell: {cell}")
                            #print(f"lattice.T: {lattice.T}")
                            #print(f"(cell @ lattice.T): {(cell @ lattice.T)}")
                            #print(f"np.dot(cell, lattice.T): {np.dot(cell, lattice.T)}")
                            #print(f"Distance: {Distance}")
                            #if Distance <= Rcut:
                            #if ham_block.nnz != 0:
                            if ham_block.nnz != 0:
                                if max(abs(ham_block.data)) > 1e-5:
                                    Hoff_block = np.zeros((40, 40))
                                    site_a_i = np.array(Find_orbsite(element_atom[a_i]))
                                    site_a_j = np.array(Find_orbsite(element_atom[a_j]))
                                    #print(f"ham_block.shape: {ham_block.shape}")
                                    #print(f"site_a_i: {site_a_i}")
                                    #print(f"site_a_j: {site_a_j}")
                                    for s_i in range(ham_block.shape[0]):
                                        for s_j in range(ham_block.shape[1]):
                                            #print(f"s_i: {s_i}")
                                            #print(f"s_j: {s_j}")
                                            Hoff_block[site_a_i[s_i],site_a_j[s_j]] = ham_block[s_i,s_j]
                                    Hoff.append(Hoff_block.flatten())
                                    pairs.append([a_i, a_j])
                                    cells.append(cell)
                            #else:
                            #    if ham_block.nnz != 0:
                            #        #print(f"type(ham_block.data): {ham_block.data}")
                            #        #print(f"ham_block.data.shape: {ham_block.data.shape}")
                            #        #print(f"np.abs(ham_block.data): {np.abs(ham_block.data)}")
                            #        #print(f"np.abs(ham_block.data).max(): {np.abs(ham_block.data).max()}")
                            #        maxnum = max(np.abs(ham_block.data).max(), maxnum)

    #print(f"len(cells): {len(cells)}")
    #print(f"len(pairs): {len(pairs)}")
    #print(f"len(Hon): {len(Hon)}")
    #print(f"len(Hoff): {len(Hoff)}") 
    #print(f"maxnum: {maxnum}")
    #print(f"type(Hon): {type(Hon)}")
    #print(f"type(Hon[0]): {type(Hon[0])}")
    #print(f"Hon[0]: {Hon[0]}")
    #Hon = np.stack(Hon, axis=0)
    #print(f"Hon.shape: {Hon.shape}")
    #print(f"type(Hon): {type(Hon)}")
    #print(f"type(Hon[0]): {type(Hon[0])}")
    #print(f"Hon[0]: {Hon[0]}")

    assert all(block.shape == (1600,) for block in Hon), "All elements in Hon must have shape (1600,)"
    assert all(block.shape == (1600,) for block in Hoff), "All elements in Hoff must have shape (1600,)"

    ##########################################################################################################################
    lattice = np.array(lattice, dtype=np.float64)
    coords = np.array(coords, dtype=np.float64)

    i = np.array(pairs, dtype=np.int32)[:,0].T
    j = np.array(pairs, dtype=np.int32)[:,1].T
    offset = np.array(cells, dtype=np.int32)
    d_ij = []
    for idx, (a_i, a_j, offset_val) in enumerate(zip(i, j, offset)):
        #print(f"pos[a_j,:]: {pos[a_j,:]}")
        #print(f"pos[a_i,:]: {pos[a_i,:]}")
        #print(f"offset_val: {offset_val}")
        #print(f"cell: {cell}")
        #print(f"np.dot(offset_val, cell): {np.dot(offset_val, cell)}")
        #print(f"pos[a_j,:]+np.dot(offset_val, cell): {pos[a_j,:]+np.dot(offset_val, cell)}")
        #print(f"pos[a_j,:] - pos[a_i,:] + np.dot(offset_val, cell): {pos[a_j,:] - pos[a_i,:] + np.dot(offset_val, cell)}")
        d_ij_one = np.linalg.norm(coords[a_j,:] - coords[a_i,:] + np.dot(offset_val, cell))
        d_ij.append(d_ij_one)
        #if a_i==gg_i and a_j==gg_j and offset_val[0]==gg_os[0] and offset_val[1]==gg_os[1] and offset_val[2]==gg_os[2]:
        #    print(f"2. a_j: {a_j}")
        #    print(f"2. a_i: {a_i}")
        #    print(f"2. pos[a_j,:]: {pos[a_j,:]}")
        #    print(f"2. pos[a_i,:]: {pos[a_i,:]}")
        #    print(f"2. offset_val: {offset_val}")
        #    print(f"2. d_ij_one: {d_ij_one}")
    d_ij = np.array(d_ij, dtype=np.float64)

    nbr_atoms = np.array([i, j], dtype=np.int32).T
    bonds_r = np.array(d_ij, dtype=np.float32).T
    num_bonds = bonds_r.shape[0]

    #print(f"nbr_atoms: {nbr_atoms}")
    #print(f"nbr_atoms.shape: {nbr_atoms.shape}")
    #print(f"bonds_r: {bonds_r}")
    #print(f"bonds_r.shape: {bonds_r.shape}")

    if len(nbr_atoms) == 0:
        n_triple_atoms = np.array([0] * num_atoms, dtype=np.int32)
        n_triple_bond = np.array([], dtype=np.int32)
        triple_indices = np.array([], dtype=np.int32).reshape(-1, 2)
    else:
        n_triple_atom, n_triple_bond, triple_indices = compute_tp_cc(nbr_atoms, d_ij, 3.5, int(num_atoms[0])) # a_cut=3.5

    n_triple_stru = np.array([np.sum(n_triple_atom)], dtype=np.int32) # num of triple (bond pairs) in total

    # Get vectors of each bond
    offset_dist = np.dot(offset, lattice) # Get vectors of each cell
    vec_diff_ij = coords[nbr_atoms[:, 1], :] - coords[nbr_atoms[:, 0], :] + offset_dist
    #print(f"vec_diff_ij.shape: {vec_diff_ij.shape} / vec_diff_ij: {vec_diff_ij}")
    #print(f"bonds_r: {bonds_r}") # Check OK
    #print(f"d_ij: {np.linalg.norm(vec_diff_ij, axis=1)}")

    triple_vec_ij = vec_diff_ij[triple_indices[:, 0]]
    triple_vec_ik = vec_diff_ij[triple_indices[:, 1]]
    triple_dist_ij = bonds_r[triple_indices[:, 0]]
    triple_dist_ik = bonds_r[triple_indices[:, 1]]

    #print(f"triple_vec_ij.shape: {triple_vec_ij.shape}")
    #print(f"triple_vec_ik.shape: {triple_vec_ik.shape}")
    #print(f"triple_dist_ij.shape: {triple_dist_ij.shape}")
    #print(f"triple_dist_ik.shape: {triple_dist_ik.shape}")

    triple_vec_ij_reshaped = triple_vec_ij[:, None, :]
    triple_vec_ik_reshaped = triple_vec_ik[:, :, None]
    triple_a_jik_numerator = np.matmul(triple_vec_ij_reshaped, triple_vec_ik_reshaped).squeeze(-1)
    triple_a_jik_denominator = triple_dist_ij[:, None] * triple_dist_ik[:, None]
    triple_a_jik = triple_a_jik_numerator / (triple_a_jik_denominator + 1e-6)
    triple_a_jik = np.clip(triple_a_jik, -1.0, 1.0) * (1 - 1e-6)
    triple_a_jik = triple_a_jik.squeeze()
    ######################################################################################################################

    #Dataset[idx_folder] = Data(
    #        num_atoms=torch.tensor(num_atoms, dtype=torch.long),
    #        atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
    #        lattice=torch.tensor(lattice, dtype=torch.float32),
    #        coords=torch.tensor(coords, dtype=torch.float32),
    #        cells=torch.tensor(cells, dtype=torch.long),
    #        pairs=torch.tensor(pairs, dtype=torch.long),
    #        Hon=torch.tensor(np.stack(Hon, axis=0), dtype=torch.float32),
    #        Hoff=torch.tensor(np.stack(Hoff, axis=0), dtype=torch.float32),
    #        Hamiltonian=torch.tensor(np.vstack((Hon, Hoff)), dtype=torch.float32),
    #        nbr_atoms=torch.tensor(nbr_atoms, dtype=torch.long),
    #        bonds_r=torch.tensor(bonds_r, dtype=torch.float32),
    #        vec_diff_ij=torch.tensor(vec_diff_ij, dtype=torch.float32),
    #        triple_indices=torch.tensor(triple_indices, dtype=torch.long),
    #        n_triple_atom=torch.tensor(n_triple_atom, dtype=torch.long),
    #        n_triple_bond=torch.tensor(n_triple_bond, dtype=torch.long),
    #        n_triple_stru=torch.tensor(n_triple_stru, dtype=torch.long),
    #        triple_dist_ij=torch.tensor(triple_dist_ij, dtype=torch.float32),
    #        triple_dist_ik=torch.tensor(triple_dist_ik, dtype=torch.float32),
    #        triple_a_jik=torch.tensor(triple_a_jik, dtype=torch.float32),
    #        energy=torch.tensor([total_energy], dtype=torch.float32)
    #    )

    Dataset[idx_folder] = Data(
                atom_fea=torch.tensor(atomic_numbers, dtype=torch.long),
                cell=torch.tensor(lattice, dtype=torch.float32),
                pos=torch.tensor(coords, dtype=torch.float32),
                offset=torch.tensor(cells, dtype=torch.float32),
                num_atoms=torch.tensor(num_atoms, dtype=torch.long),
                nbr_atoms=torch.tensor(nbr_atoms, dtype=torch.long),
                bonds_r=torch.tensor(bonds_r, dtype=torch.float32),
                vec_diff_ij=torch.tensor(vec_diff_ij, dtype=torch.float32),
                triple_indices=torch.tensor(triple_indices, dtype=torch.long),
                n_triple_atom=torch.tensor(n_triple_atom, dtype=torch.long),
                n_triple_bond=torch.tensor(n_triple_bond, dtype=torch.long),
                n_triple_stru=torch.tensor(n_triple_stru, dtype=torch.long),
                triple_dist_ij=torch.tensor(triple_dist_ij, dtype=torch.float32),
                triple_dist_ik=torch.tensor(triple_dist_ik, dtype=torch.float32),
                triple_a_jik=torch.tensor(triple_a_jik, dtype=torch.float32),
                energy=torch.tensor([total_energy], dtype=torch.float32),
                Hon=torch.tensor(np.stack(Hon, axis=0), dtype=torch.float32),
                Hoff=torch.tensor(np.stack(Hoff, axis=0), dtype=torch.float32),
                Hamiltonian=torch.tensor(np.vstack((Hon, Hoff)), dtype=torch.float32)
            )
print(f"Dataset: {Dataset}")

dataset_path = os.path.join('./', 'MLTB_data.npz')
np.savez(dataset_path, Dataset=Dataset)


