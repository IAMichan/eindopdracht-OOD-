# Pasfoto Validatie Applicatie

## Projectbeschrijving
Een object-oriented desktop applicatie voor gemeenten waarmee burgers in een fotohokje een pasfoto kunnen maken die automatisch wordt gevalideerd door een AI/ML-model.

## Functionaliteiten
- Real-time camera preview in fotohokje
- Automatische validatie van pasfoto's volgens officiële eisen:
  - Correcte belichting en scherpte
  - Neutrale gezichtsuitdrukking en gesloten mond
  - Juiste positie en afstand van het gezicht
  - Geen reflecties of schaduwen
  - Ogen goed zichtbaar en niet bedekt
- Real-time feedback en suggesties
- Lokale opslag van goedgekeurde foto's
- Offline werking (geen internet vereist)

## Technische Stack
- **Python 3.11+**
- **OpenCV** - Computer vision en camera toegang
- **MediaPipe** - Gezichtsdetectie
- **TensorFlow Lite** - AI/ML model
- **PyQt6** - Desktop GUI
- **SQLite** - Metadata opslag

## Project Structuur
```
pasfoto-validatie/
├── src/
│   ├── models/          # AI/ML modellen
│   ├── validators/      # Validatie business logic
│   ├── services/        # Core services
│   ├── gui/            # Desktop interface
│   ├── repositories/    # Data toegang laag
│   └── utils/          # Utilities
├── tests/              # Unit en integration tests
├── data/               # Dataset en trained models
├── docs/               # Documentatie
└── requirements.txt    # Python dependencies
```

## Dataset
BioID Face Database: https://www.bioid.com/face-database/

## Installatie
Zie [INSTALL.md](docs/INSTALL.md)

## Testing
Zie [TESTPLAN.md](docs/TESTPLAN.md)

## Licentie
Ontwikkeld voor gemeentelijke IT-omgevingen
