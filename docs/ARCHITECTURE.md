# Architectuur Documentatie

## Overzicht
De applicatie volgt een gelaagde architectuur met strikte scheiding van concerns volgens object-oriented principes (SOLID).

## Architectuur Lagen

### 1. Presentation Layer (GUI)
- **PhotoBoothWindow** - Hoofd applicatie window
- **CameraWidget** - Real-time camera preview
- **ValidationFeedbackWidget** - Visuele feedback
- **ResultDialog** - Resultaten weergave

### 2. Service Layer
- **PhotoCaptureService** - Camera beheer
- **ValidationService** - Orchestratie van validaties
- **StorageService** - Foto opslag
- **ReportingService** - Logging en rapportage

### 3. Domain Layer (Validators)
- **IValidator** (Interface) - Base validator contract
- **BrightnessValidator** - Belichting controle
- **SharpnessValidator** - Scherpte controle
- **FacePositionValidator** - Gezichtspositie controle
- **FacialExpressionValidator** - Gezichtsuitdrukking controle
- **EyeVisibilityValidator** - Ogen zichtbaarheid
- **ReflectionValidator** - Reflectie detectie
- **ShadowValidator** - Schaduw detectie

### 4. Model Layer
- **FaceDetectionModel** - MediaPipe face detection
- **FacialLandmarkModel** - Gezichtskenmerken detectie
- **ExpressionClassifier** - ML model voor expressie classificatie

### 5. Data Access Layer (Repositories)
- **IPhotoRepository** (Interface) - Data toegang contract
- **SQLitePhotoRepository** - SQLite implementatie
- **FileSystemStorage** - Bestandssysteem opslag

## Design Patterns

### Strategy Pattern
Verschillende validatie strategieën kunnen worden toegepast:
```
IValidator
  ├── BrightnessValidator
  ├── SharpnessValidator
  └── FacePositionValidator
```

### Factory Pattern
```
ValidatorFactory -> creates validators based on configuration
ModelFactory -> creates AI models based on type
```

### Repository Pattern
```
IPhotoRepository
  └── SQLitePhotoRepository
```

### Observer Pattern
```
ValidationService (Subject)
  └── ValidationFeedbackWidget (Observer)
```

### Singleton Pattern
```
ConfigurationManager - Applicatie configuratie
```

## Class Diagram

```
┌─────────────────────────┐
│   PhotoBoothWindow      │
├─────────────────────────┤
│ - cameraWidget          │
│ - feedbackWidget        │
│ - validationService     │
├─────────────────────────┤
│ + startCapture()        │
│ + validatePhoto()       │
└──────────┬──────────────┘
           │ uses
           ▼
┌─────────────────────────┐
│   ValidationService     │
├─────────────────────────┤
│ - validators: List      │
│ - models: Dict          │
├─────────────────────────┤
│ + validate(photo)       │
│ + addValidator()        │
└──────────┬──────────────┘
           │ uses
           ▼
┌─────────────────────────┐
│   <<Interface>>         │
│   IValidator            │
├─────────────────────────┤
│ + validate(photo)       │
│ + getName()             │
└──────────┬──────────────┘
           │ implements
     ┌─────┴─────┬─────────────┐
     ▼           ▼             ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│Brightness│ │Sharpness │ │FacePos.  │
│Validator │ │Validator │ │Validator │
└──────────┘ └──────────┘ └──────────┘
```

## Component Diagram

```
┌────────────────────────────────────────────┐
│           Desktop Application              │
│  ┌──────────────────────────────────────┐  │
│  │         GUI Layer (PyQt6)            │  │
│  └─────────────────┬────────────────────┘  │
│                    │                        │
│  ┌─────────────────▼────────────────────┐  │
│  │       Service Layer                  │  │
│  │  - ValidationService                 │  │
│  │  - PhotoCaptureService               │  │
│  │  - StorageService                    │  │
│  └─────────────────┬────────────────────┘  │
│                    │                        │
│  ┌─────────────────▼────────────────────┐  │
│  │       Validator Layer                │  │
│  │  - Multiple Validator Strategies     │  │
│  └─────────────────┬────────────────────┘  │
│                    │                        │
│  ┌─────────────────▼────────────────────┐  │
│  │       Model Layer                    │  │
│  │  - MediaPipe Face Detection          │  │
│  │  - TensorFlow Expression Classifier  │  │
│  └─────────────────┬────────────────────┘  │
│                    │                        │
│  ┌─────────────────▼────────────────────┐  │
│  │       Repository Layer               │  │
│  │  - SQLitePhotoRepository             │  │
│  │  - FileSystemStorage                 │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
           │                    │
           ▼                    ▼
    ┌──────────┐         ┌──────────┐
    │ SQLite   │         │ Local    │
    │ Database │         │ Files    │
    └──────────┘         └──────────┘
```

## Data Flow

1. **Foto Capture**
   - CameraWidget -> PhotoCaptureService -> OpenCV Camera

2. **Validatie Process**
   - Photo -> ValidationService -> Validators -> Models -> ValidationResult

3. **Opslag**
   - ValidationResult -> StorageService -> Repository -> Database/FileSystem

## SOLID Principles Toepassing

### Single Responsibility
Elke class heeft één specifieke verantwoordelijkheid:
- Validators: alleen validatie logica
- Services: orchestratie
- Repositories: data toegang

### Open/Closed
Nieuwe validators kunnen worden toegevoegd zonder bestaande code te wijzigen via IValidator interface.

### Liskov Substitution
Alle validator implementaties kunnen IValidator vervangen.

### Interface Segregation
Kleine, specifieke interfaces (IValidator, IPhotoRepository).

### Dependency Inversion
High-level modules (Services) zijn niet afhankelijk van low-level modules (Repositories), maar van abstracties (Interfaces).
