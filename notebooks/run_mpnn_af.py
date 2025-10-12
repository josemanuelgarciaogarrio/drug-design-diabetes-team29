import os
import sys
import subprocess
import time

# Configurar paths
sys.path.append('/workspace/RFdiffusion')
sys.path.append('/workspace/colabdesign')

def run_proteinmpnn_alphafold(path, contigs, copies=1, num_seqs=8, 
                             initial_guess=False, num_recycles=1,
                             use_multimer=False, rm_aa="C", 
                             mpnn_sampling_temp=0.1, num_designs=1):
    """
    Ejecuta ProteinMPNN para generar secuencias y AlphaFold para validar
    Versión adaptada para contenedor Docker
    """
    
    print("=" * 60)
    print("🧬 PROTEINMPNN + ALPHAFOLD VALIDATION")
    print("=" * 60)
    
    # Verificar que el archivo PDB existe
    pdb_file = f"/workspace/RFdiffusion/outputs/{path}_0.pdb"
    if not os.path.exists(pdb_file):
        print(f"❌ No se encuentra el archivo PDB: {pdb_file}")
        # Buscar en otras ubicaciones
        alt_locations = [
            f"/workspace/outputs/{path}_0.pdb",
            f"/dev/shm/{path}_0.pdb"
        ]
        for alt_loc in alt_locations:
            if os.path.exists(alt_loc):
                pdb_file = alt_loc
                print(f"✅ Encontrado en: {pdb_file}")
                break
        else:
            print("💡 Ejecuta primero RFdiffusion para generar el backbone")
            return False
    
    # Convertir contigs a string
    if isinstance(contigs, list):
        contigs_str = ":".join(contigs)
    else:
        contigs_str = str(contigs)
    
    # Construir opciones
    opts = [
        f"--pdb={pdb_file}",
        f"--loc=/workspace/RFdiffusion/outputs/{path}_mpnn",  # Carpeta separada
        f"--contig={contigs_str}",
        f"--copies={copies}",
        f"--num_seqs={num_seqs}",
        f"--num_recycles={num_recycles}",
        f"--rm_aa={rm_aa}",
        f"--mpnn_sampling_temp={mpnn_sampling_temp}",
        f"--num_designs={num_designs}"
    ]
    
    if initial_guess:
        opts.append("--initial_guess")
    if use_multimer:
        opts.append("--use_multimer")
    
    # Construir comando
    cmd = ['python3', '/workspace/colabdesign/rf/designability_test.py'] + opts
    
    print("📋 PARÁMETROS:")
    print(f"   PDB: {pdb_file}")
    print(f"   Contigs: {contigs_str}")
    print(f"   Secuencias: {num_seqs}")
    print(f"   Recycles: {num_recycles}")
    print(f"   Temp: {mpnn_sampling_temp}")
    print(f"   Initial guess: {initial_guess}")
    print(f"   Comando: {' '.join(cmd)}")
    print("=" * 60)
    
    # Ejecutar
    print("🚀 Ejecutando ProteinMPNN + AlphaFold...")
    start_time = time.time()
    
    try:
        # Cambiar al directorio de trabajo
        original_cwd = os.getcwd()
        os.chdir('/workspace')
        
        result = subprocess.run(cmd, check=True)
        end_time = time.time()
        
        print(f"✅ Validación completada en {end_time - start_time:.1f}s")
        
        # Verificar resultados
        results_dir = f"/workspace/RFdiffusion/outputs/{path}_mpnn"
        if os.path.exists(results_dir):
            print(f"📁 Resultados en: {results_dir}")
            
            # Listar archivos generados
            files = os.listdir(results_dir)
            seq_files = [f for f in files if f.endswith('.seq')]
            pdb_files = [f for f in files if 'af' in f and f.endswith('.pdb')]
            scores_files = [f for f in files if 'scores' in f]
            
            print(f"   📄 Secuencias generadas: {len(seq_files)}")
            print(f"   🧪 Modelos AlphaFold: {len(pdb_files)}")
            print(f"   📊 Archivos de scores: {len(scores_files)}")
            
            # Mostrar algunos resultados si existen
            if scores_files:
                scores_file = os.path.join(results_dir, scores_files[0])
                try:
                    with open(scores_file, 'r') as f:
                        print(f"   📈 Primeros scores: {f.readline().strip()}")
                except:
                    pass
        
        # Regresar al directorio original
        os.chdir(original_cwd)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en ProteinMPNN/AlphaFold: {e}")
        return False
    except FileNotFoundError as e:
        print(f"❌ Archivo no encontrado: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

# FUNCIÓN PARA EJECUCIÓN AUTOMÁTICA DESPUÉS DE RFDIFFUSION
def run_validation_after_rfdiffusion(rfdiffusion_name, rfdiffusion_contigs, **kwargs):
    """
    Ejecuta validación automáticamente después de RFdiffusion
    """
    print("\n" + "="*60)
    print("🔄 EJECUTANDO VALIDACIÓN AUTOMÁTICA")
    print("="*60)
    
    return run_proteinmpnn_alphafold(
        path=rfdiffusion_name,
        contigs=rfdiffusion_contigs,
        **kwargs
    )

# EJEMPLO DE USO DIRECTO
if __name__ == "__main__":
    # Parámetros por defecto
    run_proteinmpnn_alphafold(
        path="test_profesor2",  # Mismo nombre que tu diseño RFdiffusion
        contigs="50-100",       # Mismos contigs
        copies=1,
        num_seqs=8,
        initial_guess=False,
        num_recycles=1,
        use_multimer=False,
        rm_aa="C",
        mpnn_sampling_temp=0.1,
        num_designs=1
    )