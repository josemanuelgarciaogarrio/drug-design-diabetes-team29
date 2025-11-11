import csv
import logging

logger = logging.getLogger(__name__)

def create_output_file(name="output/outputs.csv", dataset=None, preds_plddt=None, preds_iptm=None):
    with open(name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["sequence","pred_plddt" ,"pred_iptm"])
        for data, pred_plddt, pred_iptm in zip(dataset, preds_plddt, preds_iptm):
            writer.writerow([data[0], pred_plddt, pred_iptm])
    logger.info(f"Archivo generado en: {name}")

