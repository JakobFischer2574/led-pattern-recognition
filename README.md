# LED Method Development

Dieses Repository ist **bewusst eine Entwicklungsumgebung** für zwei visuelle Erkennungsansätze (Classic CV und YOLO-Gerüst) und **noch keine finale Evaluationspipeline**.

## Ziel dieses Schritts
- Entwicklung, Debugging und Iteration beider Verfahren auf Frames/Videos.
- Gemeinsame Detektor-Schnittstelle (`BaseDetector`) für spätere vergleichbare Evaluation.
- Vorbereitung von Annotierungsdaten für YOLO.

## Projektstruktur
Siehe Ordnerstruktur im Repository (`data/`, `configs/`, `scripts/`, `src/`, `results/`).

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Nutzung
### 1) Frames aus Video extrahieren
```bash
python scripts/extract_frames.py --video data/raw/videos/video_001.mp4 --output data/sampled_frames/video_001 --step 30
```

### 2) LED-ROIs visuell prüfen
```bash
python scripts/inspect_frames.py --frame data/sampled_frames/video_001/video_001_frame_000030.jpg --output data/debug/classic_cv/inspect_video_001_000030.jpg
```

### 3) Classic-CV-Ansatz testen
```bash
python scripts/test_classic_cv.py --input data/sampled_frames/video_001 --output-csv results/development_runs/classic_cv_video_001.csv
```
- Gibt pro Frame LED-Array aus.
- Speichert Debug-Bilder mit ROI, Zustand und Helligkeitsmerkmalen.

### 4) Frames für YOLO-Annotation vorbereiten
```bash
python scripts/prepare_yolo_frames.py --input data/sampled_frames/video_001 --output data/annotation_candidates/video_001 --step 2
```

### 5) YOLO-Detektor-Gerüst testen
```bash
python scripts/test_yolo_detector.py --frame data/sampled_frames/video_001/video_001_frame_000030.jpg
```
Ohne Modellpfad wird absichtlich eine klare Fehlermeldung ausgegeben.

## Architekturhinweise
- `src/detectors/base_detector.py`: Gemeinsame Schnittstelle (`detect(frame) -> DetectionResult`).
- `ClassicCVDetector`: fester ROI-basierter Startpunkt mit konfigurierbaren Schwellenwerten.
- `YOLODetector`: vorbereitetes Gerüst (Modell-Laden + gemeinsame Schnittstelle), noch kein fertiger Trainings-/Inferenzen-Flow.

## TODOs für spätere Schritte
- Finale Evaluationspipeline (metrikengetrieben, reproduzierbar).
- Statistische Tests / Signifikanzprüfung für Bachelorarbeit.
- Temporale Blinkmustererkennung (Sequenzen statt Einzelframes).
- Vollständiger YOLO-Trainingsprozess inkl. Datensatzversionierung.
- Ressourcenmessung (CPU/RAM/Energie) für fairen Methodenvergleich.
