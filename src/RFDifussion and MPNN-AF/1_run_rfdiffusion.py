###################
###################

import os
import sys
import subprocess
import torch
import requests  # <- AÑADIDO para descargar PDBs
import time
import shutil

start_time_total = time.time()

sys.path.append('/workspace/RFdiffusion')

# FUNCIÓN PARA DESCARGAR PDBs y QUITAR CADENA ESPECÍFICA
def quitar_cadena(pdb_file, cadena_a_quitar):
    """Quita una cadena específica del PDB"""
    # Leer todo el contenido
    with open(pdb_file, 'r') as f:
        lineas = f.readlines()
    
    # Filtrar y escribir
    with open(pdb_file, 'w') as f:
        for linea in lineas:
            # Si es línea ATOM/HETATM Y es la cadena a quitar → SKIP
            if (linea.startswith('ATOM') or linea.startswith('HETATM')):
                if len(linea) > 21 and linea[21:22] == cadena_a_quitar:
                    continue  # Saltar esta línea
            f.write(linea)  # Escribir todas las demás líneas
    
    print(f"- Cadena {cadena_a_quitar} removida de {pdb_file}")
    return pdb_file

# FUNCIÓN DESCARGAR MODIFICADA
def download_pdb(pdb_id, remove_chain=None):
    """
    Descarga un PDB desde RCSB
    remove_chain: Letra de cadena a remover (ej: 'P', 'A', etc.)
    """
    if not pdb_id or len(pdb_id) != 4:
        return pdb_id
    
    pdb_file = f"/workspace/RFdiffusion/{pdb_id}.pdb"
    
    if os.path.exists(pdb_file):
        print(f"- PDB encontrado: {pdb_file}")
        return pdb_file
    print("=" * 50)
    print(f"*** Descargando {pdb_id} desde RCSB...")
    try:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            with open(pdb_file, 'w') as f:
                f.write(response.text)
            print(f"- PDB descargado: {pdb_file}")
            
            # Remover cadena SOLO si se especifica
            if remove_chain is not None:
                original_file = f"/workspace/RFdiffusion/{pdb_id}_ORIGINAL.pdb"
                shutil.copy2(pdb_file, original_file)
                print(f"- Copia original guardada: {original_file}")
                
                pdb_file = quitar_cadena(pdb_file, remove_chain)
            
            return pdb_file
        else:
            print(f"- Error descargando {pdb_id}")
            return None
    except Exception as e:
        print(f"- Error de conexión: {e}")
        return None
    print("=" * 50)

###################
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

##########################################
##########################################
################### parametros
name = "Nov13_6B3J_nohotspots_final"
contigs = "20-20/0 R30-127/R138-336/R345-400"  # ← Cadenas explícitas
pdb = "6B3J" 
iterations = 100
hotspot = "" 
num_designs = 1
visual = "image"
symmetry = ""
symmetry_order = ""
chains = ""
chain_to_remove = None #"P"  #None si 
##########################################
##########################################

if pdb and len(pdb) == 4:  # Si es un ID de PDB (4 caracteres)
    pdb_file_actual = download_pdb(pdb, 
                               remove_chain=chain_to_remove)
    if pdb_file_actual:
        print(f"- Usando PDB: {pdb_file_actual}")
    else:
        print(f"-  Continuando sin PDB base")
        pdb_file_actual = ""
else:
    pdb_file_actual = pdb  # Usar el valor original
##########################################

#########################################################
# Comando
cmd = [
    'python3', 
    'RFdiffusion/run_inference.py',
    f'inference.output_prefix=outputs/{name}',
    f'contigmap.contigs=[{contigs}]',
    f'inference.num_designs={num_designs}',
    f'diffuser.T={iterations}',
    f'inference.dump_pdb=True',
    f'inference.dump_pdb_path=/dev/shm',
]

# AÑADIR OPCIONALES
if pdb_file_actual:  # Usar el archivo descargado (si existe)
    cmd.append(f'inference.input_pdb={pdb_file_actual}')
if hotspot:
    cmd.append(f'ppi.hotspot_res=[{hotspot}]')

# NUEVOS PARÁMETROS OPCIONALES
if symmetry:
    cmd.append(f'inference.symmetry={symmetry}')
if symmetry_order:
    cmd.append(f'inference.symmetry_order={symmetry_order}')
if chains:
    cmd.append(f'inference.chains={chains}')

# MOSTRAR PARÁMETROS MEJORADO
print("=" * 50)
print("Parameters RFDIFFUSION:")
print(f"Nombre: {name}")
print(f"Contigs: {contigs}")
print(f"PDB: {pdb_file_actual if pdb_file_actual else 'None (de novo)'}")
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
    pdb_path = f"/dev/shm/{name}.pdb"
    
    if os.path.exists(pdb_path):
        file_size = os.path.getsize(pdb_path) / 1024
        print(f"Diseño completado en {execution_time:.1f}s")
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
        find_cmd = ['find', '/workspace', '-name', f'{name}_*.pdb', '-type', 'f']
        find_result = subprocess.run(find_cmd, capture_output=True, text=True)
        if find_result.stdout:
            print(f"Encontrado en: {find_result.stdout}")
        else:
            print("No se encontró el archivo PDB generado")
else:
    print(f"Error: Código {result.returncode}")


###########################
# COPIAR ARCHIVOS PDB A TU PC (SOLO ESTO AGREGADO)
import shutil
os.makedirs("/workspace/outputs/", exist_ok=True)

# Copiar PDB original (si existe)
if os.path.exists(f"/workspace/RFdiffusion/{pdb}_ORIGINAL.pdb"):
    shutil.copy2(f"/workspace/RFdiffusion/{pdb}_ORIGINAL.pdb", f"/workspace/outputs/{pdb}_ORIGINAL.pdb")
    print(f"- {pdb}_ORIGINAL.pdb")

# Copiar PDB sin cadena P (si existe)  
if os.path.exists(f"/workspace/RFdiffusion/{pdb}.pdb"):
    shutil.copy2(f"/workspace/RFdiffusion/{pdb}.pdb", f"/workspace/outputs/{pdb}_SIN_P.pdb")
    print(f"- {pdb}_SIN_P.pdb")

print("Archivos PDB en tu carpeta 'outputs'")
###########################

end_time_total = time.time()
print("=" * 60)
print(f"TIEMPO TOTAL DE EJECUCIÓN: {end_time_total - start_time_total:.2f} segundos")
print("=" * 60)