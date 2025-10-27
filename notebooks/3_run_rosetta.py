# run_rosetta_complete.py
import os
import time
import glob
from datetime import datetime

# INICIALIZAR PYROSETTA
import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta.core.scoring import *
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.scoring import CA_rmsd

pyrosetta.init('-mute all')

def analizar_estructura_completa(pdb_file):
    """Análisis COMPLETO con Rosetta REF15"""
    start_time = time.time()
    
    print("=" * 70)
    print(f"🧬 ANÁLISIS ROSETTA COMPLETO: {os.path.basename(pdb_file)}")
    print("=" * 70)
    
    # CARGAR ESTRUCTURA
    pose = pose_from_pdb(pdb_file)
    scorefxn = get_fa_scorefxn()  # REF15
    
    # 1. ENERGÍA TOTAL Y DESGLOSE
    total_energy = scorefxn(pose)
    energy_per_residue = total_energy / pose.total_residue()
    
    print(f"📊 INFORMACIÓN BÁSICA:")
    print(f"   Residuos: {pose.total_residue()}")
    print(f"   Energía REF15: {total_energy:.2f} REU")
    print(f"   Energía/residuo: {energy_per_residue:.2f} REU/res")
    
    # DESGLOSE ENERGÉTICO
    print(f"\n🔍 DESGLOSE ENERGÉTICO:")
    print("-" * 40)
    energies = pose.energies()
    score_terms = [
        fa_atr,      # Atracción van der Waals
        fa_rep,      # Repulsión van der Waals  
        fa_sol,      # Solvatación
        fa_elec,     # Electrostática
        hbond_sc,    # Puentes H cadena lateral
        hbond_bb_sc, # Puentes H backbone-sidechain
        omega,       # Ángulo omega
        rama_prepro, # Ramachandran
        p_aa_pp,     # Propensión aminoácidos
    ]
    
    for score_type in score_terms:
        value = energies.total_energies()[score_type]
        print(f"   {score_type.name:20}: {value:8.2f} REU")
    
    # ENERGÍA POR RESIDUO (TOP 10 MÁS INESTABLES)
    print(f"\n⚠️  RESIDUOS MÁS INESTABLES (Top 10):")
    print("-" * 40)
    residue_energies = []
    for i in range(1, pose.total_residue() + 1):
        res_energy = pose.energies().residue_total_energy(i)
        residue_energies.append((i, pose.residue(i).name3(), res_energy))
    
    residue_energies.sort(key=lambda x: x[2], reverse=True)
    
    print("   Residuo#  Tipo   Energía (REU)")
    for i, (res_num, res_name, energy) in enumerate(residue_energies[:10]):
        print(f"   {res_num:8d}  {res_name:4s}  {energy:10.2f}")
    
    # 2. FASTRELAX
    print(f"\n🔄 EJECUTANDO FASTRELAX...")
    relax = FastRelax()
    relax.set_scorefxn(scorefxn)
    relaxed_pose = pose.clone()
    relax.apply(relaxed_pose)
    
    relaxed_energy = scorefxn(relaxed_pose)
    energy_change = relaxed_energy - total_energy
    rmsd = CA_rmsd(pose, relaxed_pose)
    
    print(f"\n📈 RESULTADOS POST-RELAX:")
    print("-" * 40)
    print(f"   Energía post-relax: {relaxed_energy:.2f} REU")
    print(f"   Cambio energético:  {energy_change:.2f} REU")
    print(f"   RMSD (Cα):         {rmsd:.2f} Å")
    
    # 3. INTERPRETACIÓN
    print(f"\n🎯 INTERPRETACIÓN:")
    print("-" * 40)
    
    # Por energía/residuo
    if energy_per_residue < -2.0:
        print("   ✅ EXCELENTE: Energía/residuo < -2.0 REU")
    elif energy_per_residue < -1.5:
        print("   ✅ BUENA: Energía/residuo < -1.5 REU")
    elif energy_per_residue < -1.0:
        print("   ⚠️  ACEPTABLE: Energía/residuo < -1.0 REU")
    else:
        print("   ❌ PROBLEMÁTICA: Energía/residuo > -1.0 REU")
    
    # Por cambio energético post-relax
    if energy_change < -5:
        print("   ⚠️  MUY INESTABLE: cambio > -5 REU")
    elif energy_change < -2:
        print("   ⚠️  ALGO INESTABLE: cambio -2 a -5 REU")
    elif energy_change < 0:
        print("   ✅ ACEPTABLE: cambio < -2 REU")
    else:
        print("   ✅✅ MUY ESTABLE: energía aumentó/mantuvo")
    
    # Por RMSD
    if rmsd < 1.0:
        print(f"   ✅ RMSD bajo: {rmsd:.2f} Å")
    elif rmsd < 2.0:
        print(f"   ⚠️  RMSD moderado: {rmsd:.2f} Å")
    else:
        print(f"   ❌ RMSD alto: {rmsd:.2f} Å")
    
    # GUARDAR ESTRUCTURA RELAJADA
    output_file = pdb_file.replace('.pdb', '_relaxed.pdb')
    relaxed_pose.dump_pdb(output_file)
    print(f"\n💾 Estructura relajada: {os.path.basename(output_file)}")
    
    end_time = time.time()
    print(f"\n⏱️  Tiempo análisis: {end_time - start_time:.2f} segundos")
    print("=" * 70)
    
    return {
        'archivo': pdb_file,
        'residuos': pose.total_residue(),
        'energia_inicial': total_energy,
        'energia_relajada': relaxed_energy,
        'energia_por_residuo': energy_per_residue,
        'rmsd': rmsd,
        'energy_change': energy_change
    }

def analizar_carpeta_completa(carpeta):
    """Analiza TODOS los PDBs en una carpeta"""
    start_total = time.time()
    
    print("🔥 INICIANDO ANÁLISIS ROSETTA COMPLETO")
    print("=" * 70)
    
    pdb_files = glob.glob(f"{carpeta}/*.pdb")
    
    if not pdb_files:
        print("❌ No se encontraron archivos PDB")
        return
    
    print(f"📁 Encontrados {len(pdb_files)} archivos PDB")
    
    resultados = []
    for pdb_file in pdb_files:
        try:
            resultado = analizar_estructura_completa(pdb_file)
            resultados.append(resultado)
        except Exception as e:
            print(f"❌ Error con {os.path.basename(pdb_file)}: {e}")
    
    # REPORTE COMPARATIVO
    if resultados:
        print("\n" + "=" * 70)
        print("🏆 REPORTE COMPARATIVO - MEJORES DISEÑOS")
        print("=" * 70)
        
        # Ordenar por energía/residuo (mejor a peor)
        resultados.sort(key=lambda x: x['energia_por_residuo'])
        
        print(f"{'Archivo':<25} {'Residues':<8} {'Energy/Res':<10} {'RMSD':<6} {'Estado':<12}")
        print("-" * 70)
        
        for i, res in enumerate(resultados):
            archivo = os.path.basename(res['archivo'])[:24]
            energy_res = res['energia_por_residuo']
            
            if energy_res < -2.0:
                estado = "EXCELENTE"
            elif energy_res < -1.5:
                estado = "BUENA" 
            elif energy_res < -1.0:
                estado = "ACEPTABLE"
            else:
                estado = "PROBLEMA"
            
            print(f"{i+1:2d}. {archivo:<22} {res['residuos']:<8} {energy_res:<10.2f} {res['rmsd']:<6.2f} {estado:<12}")
    
    end_total = time.time()
    print(f"\n⏱️  TIEMPO TOTAL: {end_total - start_total:.2f} segundos")

# EJECUCIÓN
if __name__ == "__main__":
    # ANALIZAR CARPETA COMPLETA
    modelo          = 'test_6B3J'
    carpeta_diseños = f"/workspace/outputs/{modelo}/all_pdb"  
    analizar_carpeta_completa(carpeta_diseños)