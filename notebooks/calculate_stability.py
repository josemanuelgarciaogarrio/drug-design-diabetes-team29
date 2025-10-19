import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta.core.scoring import *
from datetime import datetime

# Inicializar PyRosetta
pyrosetta.init()
init_time = datetime.now()

# Cargar tu estructura de AlphaFold2
pose = pose_from_pdb("example2.pdb")

# Crear función de scoring (energía)
scorefxn = get_fa_scorefxn()  # Función de energía full-atom

# Calcular energía total
total_score = scorefxn(pose)

print("="*50)
print("ANÁLISIS DE ESTABILIDAD")
print("="*50)
print(f"Energía Total (REU): {total_score:.2f}")
print()

# Desglose detallado de energías
print("Desglose de términos energéticos:")
print("-"*50)

# Obtener términos individuales
energies = pose.energies()
score_types = [
    fa_atr,      # Atracción de van der Waals
    fa_rep,      # Repulsión de van der Waals
    fa_sol,      # Solvatación
    fa_elec,     # Electrostática
    hbond_sc,    # Puentes de hidrógeno (cadena lateral)
    hbond_bb_sc, # Puentes de hidrógeno (backbone-sidechain)
    omega,       # Ángulo omega
    rama_prepro, # Ramachandran
    p_aa_pp,     # Propensión de aminoácidos
]

for score_type in score_types:
    value = energies.total_energies()[score_type]
    print(f"{score_type.name:20s}: {value:10.2f}")

print("="*50)

# Energía por residuo (útil para identificar regiones problemáticas)
print("\nEnergía por residuo (top 10 más inestables):")
print("-"*50)

residue_energies = []
for i in range(1, pose.total_residue() + 1):
    res_energy = pose.energies().residue_total_energy(i)
    residue_energies.append((i, pose.residue(i).name3(), res_energy))

# Ordenar por energía (más alta = menos estable)
residue_energies.sort(key=lambda x: x[2], reverse=True)

print("Residuo#  Tipo   Energía (REU)")
for i, (res_num, res_name, energy) in enumerate(residue_energies[:10]):
    print(f"{res_num:8d}  {res_name:4s}  {energy:10.2f}")

print("="*50)

final_time = datetime.now() - init_time

print(f"Tiempo de ejecución: {final_time}")
print(f"Tiempo en segundos: {final_time.total_seconds()}")
