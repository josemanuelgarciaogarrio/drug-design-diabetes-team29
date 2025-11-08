import pandas as pd
from pathlib import Path
import os

columns = ["plddt", "i_ptm", "seq"]
output_file = "train.csv"
separator = ","

def process(main_path_raw):
    main_path = Path(main_path_raw)
    list_data = []
    print(f"Buscando archivos en: {main_path_raw}\n")

    archivos_csv_encontrados = list(main_path.glob("**/*.csv"))

    if not archivos_csv_encontrados:
        print("¡Error! No se encontraron archivos CSV (.csv) en la ruta especificada.")
    else:
        #2
        print(f"Se encontraron {len(archivos_csv_encontrados)} archivos. Procesando...")
        for archivo in archivos_csv_encontrados:
            try:
                df_temp = pd.read_csv(archivo, sep=separator, encoding="utf-8")
                df_filter = df_temp[columns]
                list_data.append(df_filter) 
                print(f"  [OK] Procesado: {archivo.name} con {df_temp.shape[0]} registros")
            except KeyError:
                # Error común: si una de las 'columnas_deseadas' no existe
                print(f"  [AVISO] Saltando archivo: {archivo.name}. No tiene todas las columnas: {columns}")
            except Exception as e:
                # Captura cualquier otro error de lectura
                print(f"  [ERROR] No se pudo leer el archivo: {archivo.name}. Detalle: {e}")
        #3
        if list_data:
            df_final = pd.concat(list_data, ignore_index=True)
            print(f"\nTotal de registros: {df_final.shape[0]}")
            final_path = os.path.join(main_path_raw, output_file)
            df_final.to_csv(final_path, index=False, encoding='utf-8-sig')
            print(f"\n¡ÉXITO! Se ha creado el archivo consolidado en:")
            print(final_path)
            print(f"Total de filas consolidadas: {len(df_final)}")
        else:
            print("\nNo se procesó ningún archivo con éxito. No se generó el archivo de salida.")
