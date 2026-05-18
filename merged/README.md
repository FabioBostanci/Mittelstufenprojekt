# Mittelstufenprojekt: Schweinehaltungs-Klassifikation (PLF)

Kaggle-Wettbewerb „Multi-View Pig Posture Recognition" der Michigan State University.  
Aufgabe: Klassifikation von fünf Schweinehaltungen (Lateral_lying_left, Lateral_lying_right, Sitting, Standing, Sternal_lying) aus Stallkamerabildern anhand vorgegebener Bounding Boxes. Bewertungsmetrik: Macro-averaged F1-Score.  
Bearbeitet von Fabio, Lara und Yusuf.

---

## Verzeichnisstruktur

```
Mittelstufenprojekt/
├── Fabio/          individuelle Experimente (Fabio)
├── Lara/           individuelle Experimente (Lara)
├── Yusuf/          individuelle Experimente (Yusuf)
└── merged/         konsolidiertes Best-of, Golden Baseline
```

---

## merged/

### `01_golden_baseline_resnet50.ipynb`
ResNet50 als einfachst mögliches Referenzmodell, das alle nachfolgenden Experimente schlagen müssen. Ausschließlich Train Set 2, GroupShuffleSplit nach `frame_id` zur Vermeidung von Data Leakage zwischen Train und Validierung, minimale Augmentation (nur horizontales Spiegeln), 5 Epochen, Adam mit lr=1e-4, inverse Klassenhäufigkeiten als Verlustgewichtung, kein Scheduler, kein Mixed Precision.

### `merged.ipynb`
Konsolidiertes Abschluss-Notebook nach CRISP-DM-Struktur. Vereint die besten Erkenntnisse aller drei Teammitglieder mit Quellenangaben pro Codezelle. Architektur: EfficientNetV2-M mit zweistufigem Training (Phase 1: eingefrorenes Backbone, nur Kopf; Phase 2: vollständiges Fine-Tuning mit lr=5e-5). Focal Loss (γ=2) gegen Klassenungleichgewicht, Mixed Precision (AMP), GroupShuffleSplit, RAM-Preloading mit ThreadPoolExecutor, umfangreiche Augmentationspipeline (ColorJitter, RandomPerspective, RandomErasing).

---

## Fabio/

### `00_read_data.ipynb`
Erstes Einlesen der Rohdaten. Lädt Train1-, Train2- und Test-CSV, zeigt Spaltenstruktur, stellt je fünf zufällige Bilder mit eingezeichneten Bounding Boxes dar.

### `01_eda.ipynb`
Explorative Datenanalyse. Ergänzt das Einlesen um Klassenverteilungsdiagramme für alle fünf Haltungsklassen. Stellt das starke Ungleichgewicht fest (Standing ~40 %, Sitting ~3 %).

### `02_separate_cameras.ipynb`
EDA mit Fokus auf Kameraverteilung. Extrahiert Kamerakennung aus den Dateinamen per Regex, analysiert, wie stark einzelne Kameras in Train1 und Train2 dominieren, und zieht Rückschlüsse auf den Distribution Shift zwischen den Trainingsmengen.

### `03_first_simple_model.ipynb`
Erstes Modellexperiment mit PyTorch und Albumentations-Augmentation. Enthält noch EDA-Zellen aus den Vorgänger-Notebooks (einige als Raw-Zellen deaktiviert) und testet einen ersten Trainingsansatz.

### `04_loadingtime_optimization.ipynb`
Untersuchung der Ladezeiten. Vergleicht verschiedene Strategien zur Beschleunigung des Datenladens (DataLoader-Parameter, Bildvorverarbeitung), um Trainingsschleifen weniger durch I/O zu bremsen.

### `05_dataoptimize.ipynb`
Datenoptimierung. Identifiziert fehlende Werte in den CSVs und bewertet Gegenmaßnahmen zum Klassenungleichgewicht (Sitting stark unterrepräsentiert). Erste Überlegungen zu Klassen­gewichtung und Oversampling.

### `06_newmodel.ipynb`
Neue Modelliteration auf Basis der EDA-Erkenntnisse. Überarbeitet die Dataset- und DataLoader-Implementierung und testet eine überarbeitete Trainingsstruktur.

### `07_allgpus.ipynb`
Multi-GPU-Training mit `DataParallel`. Einführung von Mixed Precision (AMP) und RAM-Preloading über `ThreadPoolExecutor`. Reduziert Trainingszeit deutlich durch parallele Bildverarbeitung auf mehreren V100-GPUs.

### `08_convnext_large.ipynb`
Experiment mit ConvNeXt Large als Backbone. Vergleicht Lernverhalten und Speicherbedarf gegenüber ResNet-basierten Modellen.

### `09_efficientnet_weighted_loss.ipynb`
EfficientNet mit gewichteter Verlustfunktion. Testet inverse Klassenhäufigkeiten als Gewichte in der Cross-Entropy-Loss, um die Sitting-Klasse stärker zu berücksichtigen.

### `10_optimize_efficientnet.ipynb`
Hyperparameter-Optimierung für EfficientNet. Variiert Lernrate, Batch-Größe und Augmentationsparameter.

### `11_further_optimize.ipynb`
Weitergehende Optimierung. Einführung von Focal Loss (γ=2), EfficientNetV2-M als Backbone, vollständige Augmentationspipeline mit `RandomPerspective` und `RandomErasing`, OOM-Präventionsroutine vor Phase 2 (VRAM-Bereinigung vor Backbone-Unfreeze).

### `12_convnext_2ndtry.ipynb`
Zweiter ConvNeXt-Versuch. Wesentliche Neuerung: `GroupShuffleSplit` nach `frame_id` statt zufälligem Split, um Data Leakage zu verhindern (mehrere Schweine pro Kamerabild in beiden Splits verursacht überhöhte Validierungsscores).

### `13_try_swin.ipynb`
Experiment mit dem Swin Transformer als Alternative zu CNN-Backbones. Vergleicht Trainingsverhalten und Inference-Geschwindigkeit auf dem verfügbaren Hardware-Setup.

---

## Lara/

### Versuch 1

#### `00_einlesen.ipynb`
Einlesen aller drei Datensätze (Train1, Train2, Test) mit PIL. Zeigt je fünf Pig-Instances mit Bounding Box. Beobachtung: Trainingsbilder sind deutlich dunkler als Testbilder — wird als Augmentationshinweis vermerkt.

#### `01_eda.ipynb`
Detaillierte EDA. Klassenverteilung, Bildgrößenverteilung, Anzahl Schweine pro Bild, Dekonstruktion des Dateinamens zur Kameraidentifikation (pen, Kameratyp, Kameranummer, Zeitstempel). Klärt, dass „8–10 Schweine pro Batch" sich auf den Stall (pen), nicht auf ein einzelnes Bild bezieht.

#### `02_restnet50.ipynb`
Erstes ResNet50-Baseline-Modell mit eigenem `Learner`-Objekt (Freeze/Unfreeze-Strategie) und `TrainingMonitor`. Verwendet Train1. Dokumentiert Probleme: Darstellungsfehler im Monitor, Lernrate zu hoch für Phase 2.

#### `03_fixes.ipynb`
Korrekturen am Versuch-1-Modell. Behebt Fehler in der Trainingsschleife und Monitordarstellung aus `02_restnet50.ipynb`.

### Versuch 2 (ResNet50)

#### `01_two_header(ups).ipynb`
Fehlgeschlagener Versuch eines Zwei-Kopf-Modells (Klassifikation + Regression). Notebook dokumentiert, warum der Ansatz verworfen wurde.

#### `02_nur_klassifikation.ipynb`
Reines Klassifikationsmodell mit ResNet50, Freeze/Unfreeze-Learner und strukturierter Trainingsschleife. Basis für alle nachfolgenden Versuche.

#### `03_mehr_albumentation_und_klassengewichtung.ipynb`
Erweiterung der Augmentationspipeline (ColorJitter, HorizontalFlip) und Einführung von inversen Klassengewichten in der Verlustfunktion.

#### `04_train2.ipynb`
Umstieg auf Train Set 2 für das Training, da es mehr Kameraperspektiven abdeckt und der Testverteilung näher kommt.

#### `05_mehrere_gpus.ipynb`
Multi-GPU-Setup für ResNet50.

#### `Modelle speichen,laden,etc.ipynb`
Hilfss-Notebook mit Routinen zum Speichern, Laden und Exportieren von Modellen.

### Versuch 3 (ResNet50)

#### `3.1.ipynb`
Bereinigter Neustart auf Basis der besten Versuch-2-Konfiguration. Behebt einen Datenfehler (Train1-DataFrame statt Train2-DataFrame wurde verwendet). Albumentations-Augmentation, 80/20-Split nach `image_id`.

#### `3.2_höhere startauflösung.ipynb`
Erhöht die Eingabeauflösung von 224×224 auf 320×320 Pixel, um feinere Körpermerkmale der Schweine besser zu erfassen.

#### `3.3_mehr_gpus_möglich.ipynb`
Explizite GPU-Auswahl per `CUDA_VISIBLE_DEVICES`, um einzelne GPUs gezielt zuzuweisen.

#### `3.4_mehr_padding.ipynb`
Vergrößert den Padding-Bereich um die Bounding Boxes. Ziel: Weniger abgeschnittene Körperteile an den Rändern, mehr Kontext für das Modell.

### Versuch 4 (ResNeXt50_32x4d)

#### `4.1_resnext50_32x4d.ipynb`
Wechsel von ResNet50 zu ResNeXt50_32x4d. Erstmalig: vollständiges Preloading aller Bilder in den RAM über `ThreadPoolExecutor`, um I/O-Engpässe beim Training zu eliminieren.

#### `4.2_resnext50_32x4d_ohne_padding.ipynb`
Identisch mit 4.1, jedoch ohne zusätzliches Padding um die Bounding Boxes. Vergleich, ob Padding die Genauigkeit verbessert oder verschlechtert.

### Versuch 5 (EfficientNet-B3)

#### `5.0.ipynb`
Wechsel zu EfficientNet-B3 mit allen vier verfügbaren GPUs. Eingabeauflösung 300×300 (Standardgröße für EfficientNet-B3), RAM-Preloading beibehalten.

#### `5.1.ipynb`
Weiterführende EfficientNet-B3-Experimente auf Basis von 5.0, Anpassungen an Augmentation und Trainingsparametern.

#### `Untitled.ipynb`
Scratch-Notebook aus Versuch 5, enthält unvollständige Experimente.

---

## Yusuf/

### `00_daten_einlesen.ipynb`
Einlesen beider Trainings-CSVs und des Test-CSVs. Übersicht über Spaltennamen und erste Zeilen der Daten.

### `01_eda.ipynb`
EDA: Klassenverteilung, Visualisierung von Multi-View-Beispielen (beide Kameraperspektiven nebeneinander), Analyse der Bounding-Box-Größen relativ zur Bildfläche. Erkenntnis: Schweine belegen meist nur 2,5–10 % der Bildfläche, Cropping auf Bounding Boxes sinnvoll.

### `02_datenvorbereitung.ipynb`
Datenvorbereitung: Cropping aller Pig-Instances auf ihre Bounding Boxes, Skalierung auf 224×224, 80/20-Trainings-/Validierungssplit, Erstellung eines repräsentativen Subsamples von 10 000 Bildern.

### `03_Modellvorbereitung.ipynb`
Vorbereitung der Modellarchitektur: Dataset-Klasse, Transformationspipeline (Resize, ImageNet-Normalisierung), DataLoader-Konfiguration.

### `04_Modell-Definition und Training.ipynb`
Modell­definition und Training. Enthält außerdem einen Multi-View-Ansatz (`PigMultiViewDataset`/`MultiViewNet`): zwei Kameraperspektiven desselben Schweins werden gemeinsam in ein Netz gegeben, Feature-Konkatenation vor dem Klassifikationskopf.

### `05_Evaluation.ipynb`
Detaillierte Fehleranalyse nach dem Training: Confusion Matrix, Classification Report, visueller Stichprobencheck. Befund: Trotz 83 % Accuracy enthält der Datensatz fehlerhafte Labels, die das Modell mitlernt.

### `06_Validierung.ipynb`
Modell­validierung und weitergehende Fehleranalyse. Untersucht systematische Fehlklassifikationen und deren Ursachen (Okklusionen, Labelrauschen, schwierige Klassen).

### `Mittelstufenprojekt.ipynb`
Kombiniertes Übersichts-Notebook: Klassenverteilung, Kameramatching für den Multi-View-Ansatz, Analyse von Bounding Boxes ohne erkannte Schweine (leere BBoxes), Heatmap der Klassenverteilung über beide Trainingsmengen.
