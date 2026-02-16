# TODOlista
## Środowisko (RunPod + MMSeg)
- [ ] Przygotować instancję RunPod (CUDA + Python 3.10)
- [ ] Zainstalować:
  - [ ] PyTorch (zgodny z CUDA)
  - [ ] mmcv
  - [ ] mmengine
  - [ ] mmsegmentation
- [ ] Sklonować repo mmsegmentation
- [ ] Wykonać testowy tools/test.py (sprawdzenie czy środowisko działa)
- [ ] Skonfigurować ścieżkę do datasetu Cityscapes (data_root)
## Dataset – Cityscapes
- [ ] Zweryfikować strukturę katalogów:
  - [ ] leftImg8bit/train
  - [ ] leftImg8bit/val
  - [ ] gtFine/train
  - [ ] gtFine/val
- [ ] Sprawdzić czy mmseg poprawnie ładuje dataset
- [ ] Zwizualizować 1 batch (czy maski są poprawne)
- [ ] Ustalić stały pipeline augmentacji (ten sam dla wszystkich eksperymentów)
## Baseline – oryginalny SegFormer
- [ ] Modele do testu
  - [ ] MiT-B0
  - [ ] MiT-B1
  - [ ] MiT-B2
  - [ ] MiT-B3
  - [ ] (opcjonalnie) MiT-B5
- [ ] Rozdzielczości
  - [ ] 512×1024
  - [ ] 640×1280
  - [ ] 768×768
  - [ ] 1024×1024 (sliding window)
- [ ] Metryki do zapisania
  - [ ] mIoU
  - [ ] mAcc
  - [ ] aAcc
  - [ ] F1 / Dice
  - [ ] IoU per klasa
  - [ ] Czas inferencji (ms / obraz)
  - [ ] FPS
  - [ ] Peak VRAM
- [ ] Utworzyć tabelę wyników baseline (model × rozdzielczość)
## Implementacja własnych headów
### Head A – Conv Head
- [ ] Zamienić końcowy MLP-fuse na:
  - [ ] 1x1 Conv
  - [ ] Depthwise 3x3 Conv
  - [ ] 1x1 Conv
  - [ ] Classifier
- [ ] Dodać nową klasę head w mmseg
- [ ] Przygotować config eksperymentalny
### Head B – LSTM / GRU Head
- [ ] Po concatenation feature map:
  - [ ] Flatten H×W → sekwencja
  - [ ] 1 warstwa GRU/LSTM (mały hidden)
  - [ ] Reshape do mapy cech
  - [ ] Classifier
- [ ] Dodać head do mmseg
- [ ] Przygotować config
### Head C – Gated Fusion
- [ ] Dodać mechanizm wagowania skal:
  - [ ] Global pooling
  - [ ] MLP
  - [ ] Sigmoid gate
- [ ] Wagi zastosować przed concatenation
- [ ] Przygotować config
## Trening (minimalny koszt)
### Tryb szybki (rekomendowany)
- [ ] Start z checkpointu city.160k
- [ ] Zamrozić encoder
- [ ] Trenować tylko head (10k–40k iteracji)
- [ ] Zapisać checkpoint co N iteracji
### Eksperymenty minimalne
- [ ]  B0 + (baseline vs A vs B vs C)
- [ ]  B2 + (baseline vs najlepszy head)
- [ ]  B3 + (opcjonalnie najlepszy head)
- [ ]  Powtórzyć kluczowe eksperymenty dla 2 seedów
## Analiza wyników
- [ ] Tabela zbiorcza (wszystkie modele i heady)
- [ ] Wykres mIoU vs czas inferencji
- [ ] Wykres mIoU vs liczba parametrów
- [ ] IoU per klasa (top poprawy / top spadki)
- [ ] Analiza granic obiektów (czy conv poprawia boundary?)
- [ ] 10–20 przykładów wizualnych (baseline vs zmodyfikowany head)
## Elementy do rozdziału eksperymentalnego
- [ ] Opis architektury SegFormer
- [ ] Opis oryginalnego head (MLP decoder)
- [ ] Opis modyfikacji (A, B, C)
- [ ] Uzasadnienie architektoniczne zmian
- [ ] Tabela wyników
- [ ] Analiza trade-off accuracy vs speed
- [ ] Wnioski końcowe (czy zmiana head ma sens?)
