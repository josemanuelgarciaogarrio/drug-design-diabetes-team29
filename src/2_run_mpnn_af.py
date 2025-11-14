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
print("INFORMACIÓN DEL ENTORNO Y ARCHIVOS")
print("=" * 60)
print("Directorio actual:", os.getcwd())

#############
print("\n ARCHIVOS EN /workspace/outputs/ (VOLUMEN DOCKER):")
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
    print(f"   - Error: {e}")

print("=" * 60)
#############

#############
print("VERIFICANDO HARDWARE DISPONIBLE:")

try:
    import jax
    print(f"   JAX devices: {jax.devices()}")
    print(f"   JAX backend: {jax.default_backend()}")
except Exception as e:
    print(f"   Error JAX: {e}")

try:
    import torch
    print(f"   PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   PyTorch GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"   Error PyTorch: {e}")
##############

def run_proteinmpnn_alphafold(path, 
                              contigs, 
                              copies=1, 
                              num_seqs=8, 
                              initial_guess=False, 
                              num_recycles=1,
                              use_multimer=False, 
                              rm_aa="C", 
                              mpnn_sampling_temp=0.1, 
                              num_designs=1,
                              design_num=0):
    """
    Ejecuta ProteinMPNN para generar secuencias y AlphaFold para validar
    """
    
    print("=" * 60)
    print(" - PROTEINMPNN + ALPHAFOLD VALIDATION")
    print("=" * 60)
    
    pdb_file = f"/workspace/outputs/{path}_{design_num}.pdb"
    if not os.path.exists(pdb_file):
        print(f" - No se encuentra el archivo PDB: {pdb_file}")
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
    
    print("PARÁMETROS:")
    print(f"   PDB: {pdb_file}")
    print(f"   Output: {output_dir}")
    print(f"   Contigs: {contigs_str}")
    print(f"   Secuencias: {num_seqs}")
    print(f"   Recycles: {num_recycles}")
    print("=" * 60)
    
    # Ejecutar
    print("Ejecutando ProteinMPNN + AlphaFold...\n\n")
    start_time = time.time()
    
    try:
        original_cwd = os.getcwd()
        os.chdir('/workspace')
        
        result = subprocess.run(cmd, check=True)
        end_time = time.time()
        
        print(f"Validación completada en {end_time - start_time:.1f}s")
        
        expected_files = [
            f"{output_dir}/mpnn_results.csv",      # Este sí existe
            f"{output_dir}/best.pdb",              # Mejor diseño
            f"{output_dir}/best_design0.pdb",      # Mejor diseño alternativo
            f"{output_dir}/design.fasta",          # Secuencias
            f"{output_dir}/scores.json",           # Scores
            f"{output_dir}/af_pred.pdb",           # Predicción AlphaFold
            f"{output_dir}/seq.fasta"              # Secuencias alternativo
        ]
        
        print("ARCHIVOS GENERADOS:")
        files_found = []
        for file_path in expected_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024
                print(f" {os.path.basename(file_path)} ({file_size:.1f} KB)")
                files_found.append(file_path)
        
        # Si no se encontraron archivos, mostrar todos los que hay
        if not files_found:
            print("Buscando todos los archivos en la carpeta...")
            if os.path.exists(output_dir):
                all_files = os.listdir(output_dir)
                for file in sorted(all_files):
                    full_path = os.path.join(output_dir, file)
                    if os.path.isfile(full_path):
                        file_size = os.path.getsize(full_path) / 1024
                        print(f"  {file} ({file_size:.1f} KB)")
        
        os.chdir(original_cwd)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"Error inesperado: {e}")
        return False


def run_proteinmpnn_only(path, 
                        contigs, 
                        copies=1, 
                        num_seqs=8, 
                        initial_guess=False,
                        num_recycles=1,
                        use_multimer=False, 
                        rm_aa="C", 
                        mpnn_sampling_temp=0.1, 
                        num_designs=1,
                        design_num=0):
    """
    Ejecuta SOLAMENTE ProteinMPNN para generar secuencias sin validación con AlphaFold
    """
    
    print("=" * 60)
    print(" PROTEINMPNN ONLY (SIN ALPHAFOLD)")
    print("=" * 60)
    
    pdb_file = f"/workspace/outputs/{path}_{design_num}.pdb"
    if not os.path.exists(pdb_file):
        print(f"No se encuentra el archivo PDB: {pdb_file}")
        return False
    
    # Convertir contigs a string
    if isinstance(contigs, list):
        contigs_str = ":".join(contigs)
    else:
        contigs_str = str(contigs)
    
    output_dir = f"/workspace/outputs/{path}_mpnn_only"
    
    # Crear script temporal que solo ejecuta MPNN
    mpnn_only_script = """
import os
import sys
sys.path.append('/workspace/colabdesign')

from colabdesign.mpnn import mk_mpnn_model
from colabdesign.af import mk_af_model
import pandas as pd
import numpy as np

from string import ascii_uppercase, ascii_lowercase
alphabet_list = list(ascii_uppercase+ascii_lowercase)

def get_info(contig):
    F = []
    free_chain = False
    fixed_chain = False
    sub_contigs = [x.split("-") for x in contig.split("/")]
    for n,(a,b) in enumerate(sub_contigs):
        if a[0].isalpha():
            L = int(b)-int(a[1:]) + 1
            F += [1] * L
            fixed_chain = True
        else:
            L = int(b)
            F += [0] * L
            free_chain = True
    return F,[fixed_chain,free_chain]

# Parámetros
pdb_filename = "{pdb_file}"
output_dir = "{output_dir}"
contigs_str = "{contigs_str}"
copies = {copies}
num_seqs = {num_seqs}
initial_guess = {initial_guess}
use_multimer = {use_multimer}
rm_aa = "{rm_aa}"
sampling_temp = {sampling_temp}
num_designs = {num_designs}

# VALIDACIÓN DE ARGUMENTOS REQUERIDOS 
if None in [pdb_filename, output_dir, contigs_str]:
    print(" Missing Required Arguments: pdb, loc, contigs")
    sys.exit(1)
if rm_aa == "":
    rm_aa = None

# Parse contigs
contigs = []
for contig_str in contigs_str.replace(" ",":").replace(",",":").split(":"):
    if len(contig_str) > 0:
        contig = []
        for x in contig_str.split("/"):
            if x != "0": contig.append(x)
        contigs.append("/".join(contig))

chains = alphabet_list[:len(contigs)]
info = [get_info(x) for x in contigs]
fixed_pos = []
fixed_chains = []
free_chains = []
both_chains = []

for pos,(fixed_chain,free_chain) in info:
    fixed_pos += pos
    fixed_chains += [fixed_chain and not free_chain]
    free_chains += [free_chain and not fixed_chain]
    both_chains += [fixed_chain and free_chain]

# Preparar modelo AF solo para estructura 
flags = {{"initial_guess":initial_guess,
        "best_metric":"rmsd",
        "use_multimer":use_multimer,
        "model_names":["model_1_multimer_v3" if use_multimer else "model_1_ptm"]}}

if sum(both_chains) == 0 and sum(fixed_chains) > 0 and sum(free_chains) > 0:
    protocol = "binder"
    print("protocol=binder")
    target_chains = []
    binder_chains = []
    for n,x in enumerate(fixed_chains):
        if x: target_chains.append(chains[n])
        else: binder_chains.append(chains[n])
    af_model = mk_af_model(protocol="binder",**flags)
    prep_flags = {{"target_chain":",".join(target_chains),
                "binder_chain":",".join(binder_chains),
                "rm_aa":rm_aa}}
                
elif sum(fixed_pos) > 0:
    protocol = "partial"
    print("protocol=partial")
    af_model = mk_af_model(protocol="fixbb", 
                           use_templates=True, **flags)
    rm_template = np.array(fixed_pos) == 0
    prep_flags = {{"chain":",".join(chains),
                 "rm_template":rm_template,
                 "rm_template_seq":rm_template,
                 "copies":copies,
                 "homooligomer":copies>1,
                 "rm_aa":rm_aa}}
else:
    protocol = "fixbb"
    print("protocol=fixbb")
    af_model = mk_af_model(protocol="fixbb",**flags)
    prep_flags = {{"chain":",".join(chains),
                 "copies":copies,
                 "homooligomer":copies>1,
                 "rm_aa":rm_aa}}

# Ejecutar solo MPNN

batch_size = 8
if num_seqs < batch_size:    
    batch_size = num_seqs

print("Running ProteinMPNN only...")

mpnn_model = mk_mpnn_model()
os.makedirs(output_dir, exist_ok=True)
data = []

# CREAR CARPETA PARA PDBs
pdb_dir = os.path.join(output_dir, "all_pdbs")
os.makedirs(pdb_dir, exist_ok=True)

with open(f"{output_dir}/design.fasta", "w") as fasta:
    for m in range(num_designs):
        if num_designs == 0:
            current_pdb = pdb_filename
        else:
            current_pdb = pdb_filename.replace("_0.pdb", f"_{{m}}.pdb")
        
        af_model.prep_inputs(current_pdb, **prep_flags)
        if protocol == "partial":
            p = np.where(fixed_pos)[0]
            af_model.opt["fix_pos"] = p[p < af_model._len]
        mpnn_model.get_af_inputs(af_model)
        out = mpnn_model.sample(num=num_seqs//batch_size, batch=batch_size, temperature=sampling_temp)
        
        for n in range(num_seqs):
            score_line = [f'design:{{m}} n:{{n}}', f'mpnn_score:{{out["score"][n]:.3f}}']
            line = f'>{{"|".join(score_line)}}\\n{{out["seq"][n]}}'
            fasta.write(line + "\\n")
            print(f"design:{{m}} n:{{n}} mpnn_score:{{out['score'][n]:.3f}} {{out['seq'][n]}}")
            data.append([m, n, out["score"][n], out["seq"][n]])

# Guardar resultados MPNN
df = pd.DataFrame(data, columns=["design", "n", "score", "seq"])
df.to_csv(f'{output_dir}/mpnn_results.csv')
print(f"MPNN only completed. Results saved to {output_dir}")

""".format(
        pdb_file=pdb_file,
        output_dir=output_dir,
        contigs_str=contigs_str,
        copies=copies,
        num_seqs=num_seqs,
        initial_guess=str(initial_guess),
        use_multimer=str(use_multimer),
        rm_aa=rm_aa,
        sampling_temp=mpnn_sampling_temp,
        num_designs=num_designs
    )
    
    # Guardar script temporal
    temp_script_path = "/tmp/mpnn_only.py"
    with open(temp_script_path, "w") as f:
        f.write(mpnn_only_script)
    
    print("PARÁMETROS MPNN ONLY:")
    print(f"   PDB: {pdb_file}")
    print(f"   Output: {output_dir}")
    print(f"   Contigs: {contigs_str}")
    print(f"   Secuencias: {num_seqs}")
    print(f"   Initial guess: {initial_guess}")
    print(f"   Use multimer: {use_multimer}")
    print(f"   Sampling temp: {mpnn_sampling_temp}")
    print("=" * 60)
    
    # Ejecutar script temporal
    print("Ejecutando ProteinMPNN (sin AlphaFold)...")
    start_time = time.time()
    
    try:
        result = subprocess.run(['python3', temp_script_path], check=True, capture_output=True, text=True)
        end_time = time.time()
        
        print(f"ProteinMPNN completado en {end_time - start_time:.1f}s")
        
        # Verificar archivos generados
        expected_files = [
            f"{output_dir}/design.fasta",
            f"{output_dir}/mpnn_results.csv",
            f"{output_dir}/all_pdbs/"
        ]
        
        print("ARCHIVOS GENERADOS POR MPNN:")
        for file_path in expected_files:
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    pdb_files = os.listdir(file_path)
                    print(f"   SÍ: {os.path.basename(file_path)}/ ({len(pdb_files)} archivos PDB)")
                else:
                    file_size = os.path.getsize(file_path) / 1024
                    print(f"   SÍ: {os.path.basename(file_path)} ({file_size:.1f} KB)")
            else:
                print(f"   NO: {os.path.basename(file_path)} (no encontrado)")
        
        # Limpiar script temporal
        os.remove(temp_script_path)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error ejecutando MPNN: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error inesperado: {e}")
        return False

# EJECUCIÓN PRINCIPAL
if __name__ == "__main__":    
    # Ejecutar validación
    run_proteinmpnn_alphafold(
        path="Nov13_6B3J_nohotspots_final",
        contigs="20-20/0 R30-127/R138-336/R345-400",
        copies=1,
        num_seqs=16,
        initial_guess=True,
        num_recycles=1,
        use_multimer=True,
        rm_aa="C",
        mpnn_sampling_temp=0.1,
        num_designs=1,
        design_num=0
    )


### EJECUCION MPNN Solamente
#if __name__ == "__main__":    
#    # OPCIÓN 1: Ejecutar solo MPNN
#    run_proteinmpnn_only(
#        path="T16B3J_R_KB_HS_KB_vFinal",
#        contigs="20-20/0 R30-127/R138-336/R345-400",
#        copies=1,
#        num_seqs=8,
#        initial_guess=True,
#        num_recycles=1,
#        use_multimer=True,
#        rm_aa="C",
#        mpnn_sampling_temp=0.1,
#        num_designs=1,
#        design_num=0
#    )

end_time_total = time.time()
print("=" * 60)
print(f"TIEMPO TOTAL DE EJECUCIÓN: {end_time_total - start_time_total:.2f} segundos")
print("=" * 60)