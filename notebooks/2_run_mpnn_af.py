import os
import sys
import subprocess
import time

start_time_total = time.time()

# Configurar paths
sys.path.append('/workspace/RFdiffusion')
sys.path.append('/workspace/colabdesign')

#############
print("=" * 60)
print("🔍 INFORMACIÓN DEL ENTORNO Y ARCHIVOS")
print("=" * 60)
print("📁 Directorio actual:", os.getcwd())

#############
print("\n📂 ARCHIVOS EN /workspace/outputs/ (VOLUMEN DOCKER):")
try:
    output_files = os.listdir('/workspace/outputs/')
    for file in output_files:
        file_path = f"/workspace/outputs/{file}"
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path) / 1024
            print(f"   - {file} ({size:.1f} KB)")
        else:
            print(f"   - {file}/ [carpeta]")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("=" * 60)
#############

def run_proteinmpnn_alphafold(path, contigs, copies=1, num_seqs=8, 
                             initial_guess=False, num_recycles=1,
                             use_multimer=False, rm_aa="C", 
                             mpnn_sampling_temp=0.1, num_designs=1):
    """
    Ejecuta ProteinMPNN para generar secuencias y AlphaFold para validar
    """
    
    print("=" * 60)
    print("🧬 PROTEINMPNN + ALPHAFOLD VALIDATION")
    print("=" * 60)
    
    pdb_file = f"/workspace/outputs/{path}_0.pdb"
    if not os.path.exists(pdb_file):
        print(f"❌ No se encuentra el archivo PDB: {pdb_file}")
        return False
    
    # Convertir contigs a string
    if isinstance(contigs, list):
        contigs_str = ":".join(contigs)
    else:
        contigs_str = str(contigs)
    
    output_dir = f"/workspace/outputs/{path}"
    
    # Construir opciones
    opts = [
        f"--pdb={pdb_file}",
        f"--loc={output_dir}",
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
    print(f"   Output: {output_dir}")
    print(f"   Contigs: {contigs_str}")
    print(f"   Secuencias: {num_seqs}")
    print(f"   Recycles: {num_recycles}")
    print("=" * 60)
    
    # Ejecutar
    print("🚀 Ejecutando ProteinMPNN + AlphaFold...")
    start_time = time.time()
    
    try:
        original_cwd = os.getcwd()
        os.chdir('/workspace')
        
        result = subprocess.run(cmd, check=True)
        end_time = time.time()
        
        print(f"✅ Validación completada en {end_time - start_time:.1f}s")
        
        # ⬇️⬇️⬇️ NOMBRES REALES (basados en lo que ves) ⬇️⬇️⬇️
        expected_files = [
            f"{output_dir}/mpnn_results.csv",      # Este sí existe
            f"{output_dir}/best.pdb",              # Mejor diseño
            f"{output_dir}/best_design0.pdb",      # Mejor diseño alternativo
            f"{output_dir}/design.fasta",          # Secuencias
            f"{output_dir}/scores.json",           # Scores
            f"{output_dir}/af_pred.pdb",           # Predicción AlphaFold
            f"{output_dir}/seq.fasta"              # Secuencias alternativo
        ]
        
        print("📁 ARCHIVOS GENERADOS:")
        files_found = []
        for file_path in expected_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024
                print(f"   ✅ {os.path.basename(file_path)} ({file_size:.1f} KB)")
                files_found.append(file_path)
        
        # Si no se encontraron archivos, mostrar todos los que hay
        if not files_found:
            print("   🔍 Buscando todos los archivos en la carpeta...")
            if os.path.exists(output_dir):
                all_files = os.listdir(output_dir)
                for file in sorted(all_files):
                    full_path = os.path.join(output_dir, file)
                    if os.path.isfile(full_path):
                        file_size = os.path.getsize(full_path) / 1024
                        print(f"   📄 {file} ({file_size:.1f} KB)")
        
        os.chdir(original_cwd)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

# EJECUCIÓN PRINCIPAL
if __name__ == "__main__":    
    # Ejecutar validación
    run_proteinmpnn_alphafold(
        path="test_6B3J",
        contigs="12-15 R311-337",
        copies=1,
        num_seqs=32,
        initial_guess=False,
        num_recycles=1,
        use_multimer=False,
        rm_aa="C",
        mpnn_sampling_temp=0.1,
        num_designs=1
    )

end_time_total = time.time()
print("=" * 60)
print(f"TIEMPO TOTAL DE EJECUCIÓN: {end_time_total - start_time_total:.2f} segundos")
print("=" * 60)