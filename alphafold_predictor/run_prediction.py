import sys
import argparse
import logging
from src.batch_predictor import BatchPredictor
from src.config import LOG_LEVEL

def setup_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        description='Predictor de proteínas usando ESM + MLP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  Predicción individual:
    python run_prediction.py --mode single --sequence "MKTLILAFLFASA"
    
  Desde Docker:
    docker run batch-predictor:latest --mode single --sequence "MKTLILAFLFASA"
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['single'],
        help='Modo de predicción: single (una secuencia)'
    )

    parser.add_argument(
        '--sequence',
        type=str,
        required=True,
        help='Secuencia de aminoácidos para predecir'
    )

    args = parser.parse_args()
    try:
        logger.info("=" * 60)
        logger.info("Iniciando Batch Predictor")
        logger.info("=" * 60)
        logger.info("Cargando modelos...")
        predictor = BatchPredictor()

        if args.mode == 'single':
            logger.info(f"Modo: Predicción individual")
            logger.info(f"Secuencia: {args.sequence}")
            logger.info("-" * 60)
            result = predictor.predict_single(args.sequence)
            logger.info("=" * 60)
            if result['status'] == 'success':
                logger.info("PREDICCIÓN EXITOSA")
                logger.info("-" * 60)
                print(f"\nSecuencia: {result['sequence']}")
                print(f"Predicción: {result['prediction']:.6f}")
                print(f"Timestamp: {result['timestamp']}\n")
                # Exit code 0 (éxito)
                sys.exit(0)
            else:
                logger.error("PREDICCIÓN FALLIDA")
                logger.error("-" * 60)
                print(f"\nError: {result['error']}\n")
                # Exit code 1 (error)
                sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("\nProceso interrumpido por el usuario")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Error inesperado {str(e)}", exc_info=True)
        print(f"\nError inesperado: {str(e)}\n")
        sys.exit(1)

if __name__ == '__main__':
    main()