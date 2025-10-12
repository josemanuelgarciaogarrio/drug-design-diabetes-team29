###################
###################

import os
import sys
import subprocess
import torch  # <- AÑADIDO para verificación GPU

sys.path.append('/workspace/RFdiffusion')


# VERIFICACIÓN GPU AL INICIO
print("=" * 50)
print("VERIFICACIÓN GPU:")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("ADVERTENCIA: No se detectó GPU - Usando CPU (muy lento)")
print("=" * 50)

#########################################################
################### parametros
name = "test_profesor2"
contigs = "50-100" 
pdb = ""
iterations = 50
hotspot = ""
num_designs = 1
visual = "image"  # "none", "image", o "interactive"

# NUEVOS PARÁMETROS AÑADIDOS
symmetry = ""  # "C2", "C3", "D2", etc.
symmetry_order = ""  # 2, 3, 4, etc.
chains = ""  # "A,B", "A,B,C", etc.

##########################################
#pdb_file = download_pdb(pdb_id)
##########################################

#########################################################
#Comand
cmd = [
    'python3', 
    'RFdiffusion/run_inference.py',
    f'inference.output_prefix=outputs/{name}',
    f'contigmap.contigs=[{contigs}]',
    f'inference.num_designs={num_designs}',
    f'diffuser.T={iterations}',
    f'inference.dump_pdb=True',
    f'inference.dump_pdb_path=/dev/shm'
]

# AÑADIR OPCIONALES
if pdb:
    cmd.append(f'inference.input_pdb={pdb}')
if hotspot:
    cmd.append(f'ppi.hotspot_res=[{hotspot}]')

# NUEVOS PARÁMETROS OPCIONALES AÑADIDOS
if symmetry:
    cmd.append(f'inference.symmetry={symmetry}')
if symmetry_order:
    cmd.append(f'inference.symmetry_order={symmetry_order}')
if chains:
    cmd.append(f'inference.chains={chains}')

# MOSTRAR PARÁMETROS MEJORADO
print("Parameters RFDIFFUSION:")
print(f"Nombre: {name}")
print(f"Contigs: {contigs}")
print(f"PDB: {pdb if pdb else 'None'}")
print(f"Iteraciones: {iterations}")
print(f"Hotspot: {hotspot if hotspot else 'None'}")
print(f"Diseños: {num_designs}")
print(f"Simetría: {symmetry if symmetry else 'None'}")
print(f"Orden simetría: {symmetry_order if symmetry_order else 'None'}")
print(f"Cadenas: {chains if chains else 'None'}")
print(f"Visual: {visual}")
print(f"Comando: {' '.join(cmd)}")
print("=" * 50)

# EJECUTAR CON MONITOREO DE TIEMPO
print("Iniciando diseño de proteína... \n\n\n\n")

import time
start_time = time.time()

result = subprocess.run(cmd)

end_time = time.time()
execution_time = end_time - start_time

# VERIFICAR ARCHIVO EN LA RUTA CORRECTA
print("=" * 50)
if result.returncode == 0:
    # Buscar en /dev/shm (donde realmente se guarda)
    pdb_path = f"/dev/shm/{name}_0.pdb"
    
    if os.path.exists(pdb_path):
        file_size = os.path.getsize(pdb_path) / 1024
        print(f"Diseño completado en {time.time() - start_time:.1f}s")
        print(f"Archivo: {pdb_path}")
        print(f"Tamaño: {file_size:.1f} KB")
        
        # OPCIONAL: Copiar a outputs para persistencia
        outputs_dir = "/workspace/RFdiffusion/outputs/"
        os.makedirs(outputs_dir, exist_ok=True)
        import shutil
        shutil.copy(pdb_path, f"{outputs_dir}{name}_0.pdb")
        print(f"Copiado a: {outputs_dir}{name}_0.pdb")
    else:
        print(f"Archivo no encontrado en: {pdb_path}")
        print("Buscando en otras ubicaciones...")
        
        # Buscar en cualquier lugar
        find_cmd = ['find', '/workspace', '-name', f'{name}_0.pdb', '-type', 'f']
        find_result = subprocess.run(find_cmd, capture_output=True, text=True)
        if find_result.stdout:
            print(f"🔍 Encontrado en: {find_result.stdout}")
else:
    print(f"Error: Código {result.returncode}")