import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from datetime import datetime


# Inicializar
pyrosetta.init('-mute all')  # Silenciar output verboso

def analizar_estructura(pdb_file):
    """Análisis completo de estabilidad"""
    
    pose = pose_from_pdb(pdb_file)
    scorefxn = get_fa_scorefxn()
    
    # 1. Energía total
    total_energy = scorefxn(pose)
    
    # 2. Energía por residuo
    energy_per_residue = total_energy / pose.total_residue()
    
    # 3. Relajación (FastRelax) para estimar estabilidad
    print("Ejecutando FastRelax para optimizar estructura...")
    from pyrosetta.rosetta.protocols.relax import FastRelax
    
    relax = FastRelax()
    relax.set_scorefxn(scorefxn)
    
    # Clonar pose para no modificar original
    relaxed_pose = pose.clone()
    relax.apply(relaxed_pose)
    
    relaxed_energy = scorefxn(relaxed_pose)
    energy_change = relaxed_energy - total_energy
    
    # 4. RMSD entre original y relajada
    from pyrosetta.rosetta.core.scoring import CA_rmsd
    rmsd = CA_rmsd(pose, relaxed_pose)
    
    # Resultados
    print("\n" + "="*60)
    print("RESUMEN DE ESTABILIDAD")
    print("="*60)
    print(f"Archivo: {pdb_file}")
    print(f"Número de residuos: {pose.total_residue()}")
    print()
    print(f"Energía inicial:        {total_energy:.2f} REU")
    print(f"Energía post-relax:     {relaxed_energy:.2f} REU")
    print(f"Cambio energético:      {energy_change:.2f} REU")
    print(f"Energía/residuo:        {energy_per_residue:.2f} REU/res")
    print(f"RMSD (Cα):              {rmsd:.2f} Å")
    print()
    
    # Interpretación
    print("INTERPRETACIÓN:")
    print("-"*60)
    if energy_change < -5:
        print("⚠️  Estructura MUY INESTABLE (cambio > -5 REU)")
    elif energy_change < -2:
        print("⚠️  Estructura algo inestable (cambio -2 a -5 REU)")
    elif energy_change < 0:
        print("✓  Estructura aceptable (cambio < -2 REU)")
    else:
        print("✓✓ Estructura MUY ESTABLE (energía aumentó o igual)")
    
    if rmsd > 2.0:
        print(f"⚠️  RMSD alto ({rmsd:.2f} Å) - estructura se deformó mucho")
    elif rmsd > 1.0:
        print(f"⚠️  RMSD moderado ({rmsd:.2f} Å)")
    else:
        print(f"✓  RMSD bajo ({rmsd:.2f} Å) - estructura estable")
    
    print("="*60)
    
    # Guardar estructura relajada
    output_file = pdb_file.replace('.pdb', '    .pdb')
    relaxed_pose.dump_pdb(output_file)
    print(f"\nEstructura relajada guardada en: {output_file}")
    
    return {
        'total_energy': total_energy,
        'relaxed_energy': relaxed_energy,
        'energy_change': energy_change,
        'rmsd': rmsd,
        'energy_per_residue': energy_per_residue
    }

# Uso
if __name__ == "__main__":
    init_time = datetime.now()
    analizar_estructura("example2.pdb")
    final_time = datetime.now() - init_time
    print(f"Tiempo de ejecución: {final_time}")
    print(f"Tiempo en segundos: {final_time.total_seconds()}")
