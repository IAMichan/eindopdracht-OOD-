"""
Main entry point voor Pasfoto Validatie Applicatie
"""
import sys
import logging
from PyQt6.QtWidgets import QApplication

from src.gui.main_window import PhotoBoothWindow


def setup_logging():
    """Setup logging configuratie"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pasfoto_validatie.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main functie"""
    # Setup logging
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Starting Pasfoto Validatie Applicatie")

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Pasfoto Validatie")
    app.setOrganizationName("Gemeente")

    # Create and show main window
    window = PhotoBoothWindow()
    window.show()

    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
