# Virtual Coach — Guida passo passo per la demo dal vivo

Questa guida ti porta da zero a una demo funzionante dei 3 task del progetto Virtual Coach, utilizzando uno squat come esercizio dimostrativo. Al termine, un utente esterno (il professore) potrà vedere il sistema funzionare in tempo reale.

---

## Panoramica di cosa succede durante la demo

L'utente tiene il telefono in tasca, sta fermo in piedi davanti alla webcam. Il sistema dice "STANDING — utente pronto" (Task 1). L'utente inizia a muoversi, il sistema rileva "MOVING" e attiva la sessione. L'utente esegue degli squat: sullo schermo appare lo scheletro sovrapposto al corpo (Task 3) e un contatore che avanza ad ogni ripetizione completata (Task 2), con feedback sulla qualità di ciascuna ripetizione.

---

## Cosa ti serve

**Hardware**

- Un computer con webcam (portatile o fisso + webcam USB)
- Uno smartphone (Android o iPhone) con phyphox installato
- Entrambi i dispositivi connessi alla **stessa rete WiFi**

**Software sul computer**

- Python 3.9+ (consigliato: Anaconda)
- Jupyter Notebook o JupyterLab
- Le librerie Python elencate sotto

**File del progetto** (tutti nella stessa cartella)

| File | Da dove viene |
|---|---|
| `virtual_coach_live_demo_squat.ipynb` | il notebook della demo |
| `task3_module.py` | esportato dal Task 3 |
| `best_2d.pt` | classificatore 2D vincitore (JTMNet) |
| `best_3d.pt` | classificatore 3D vincitore (PartLSTM) |
| `meta.json` | metadati dei modelli |
| `virtual_coach_imu.phyphox` | esperimento personalizzato per phyphox *(opzionale ma consigliato)* |

---

## STEP 1 — Preparare l'ambiente Python

### Opzione A: con Anaconda (consigliata)

```bash
# Crea un ambiente dedicato
conda create -n virtualcoach python=3.11 -y
conda activate virtualcoach

# Installa le dipendenze
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install mediapipe opencv-python numpy pandas matplotlib scipy scikit-learn tqdm jupyter
```

### Opzione B: con pip (senza Anaconda)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install mediapipe opencv-python numpy pandas matplotlib scipy scikit-learn tqdm jupyter
```

> **Nota per Mac Apple Silicon (M1/M2/M3):** PyTorch e MediaPipe funzionano nativamente. Usa `pip install torch torchvision` senza `--index-url`.

> **Opzionale (per il modello Task 1 quando sarà pronto):** `pip install tensorflow joblib`

### Verifica che tutto sia installato

```bash
python -c "import torch; import mediapipe; import cv2; print('OK:', torch.__version__, mediapipe.__version__, cv2.__version__)"
```

Deve stampare le versioni senza errori.

---

## STEP 2 — Organizzare i file

Crea una cartella dedicata e mettici dentro tutti i file:

```
📂 demo_virtual_coach/
   ├── virtual_coach_live_demo_squat.ipynb
   ├── task3_module.py
   ├── best_2d.pt
   ├── best_3d.pt
   ├── meta.json
   └── virtual_coach_imu.phyphox        ← opzionale, per il telefono
```

> **Importante:** tutti i file devono stare nella **stessa** cartella. Il notebook li cerca lì.

---

## STEP 3 — Primo test in modalità simulazione

Prima di collegare hardware reale, verifica che tutto funzioni in simulazione.

1. Apri un terminale, vai nella cartella dei file, e lancia Jupyter:

```bash
cd demo_virtual_coach
jupyter notebook
```

2. Apri `virtual_coach_live_demo_squat.ipynb`.

3. **Nella prima cella di codice** (Pannello di controllo), verifica che sia impostato:

```python
SIMULATE_TASK1  = True    # simulazione telefono
SIMULATE_WEBCAM = True    # simulazione webcam
```

4. **Esegui tutte le celle** in ordine (Kernel → Restart & Run All, oppure Shift+Enter cella per cella).

5. **Cosa devi vedere:**
   - Cella "TASK 1": stampa `STANDING (motion=0.014)` poi `MOVING (motion=0.3xx)` → "Via libera: SÌ"
   - Cella "Classificatore": stampa `2D: JTMNet (acc=0.936) | 3D: PartLSTM (acc=0.936)` e `SELEZIONATO: 3D PartLSTM`
   - Cella "Demo integrata": il contatore avanza da 1/5 a 5/5 e stampa "★ Allenamento completato: 1 x 5 BodyWeightSquats."
   - Due grafici: uno con il segnale IMU (9 canali), uno con il segnale cinematico + contatore + confidenza

Se vedi tutto questo, il software funziona. Passiamo all'hardware.

---

## STEP 4 — Preparare il telefono (Task 1)

### 4.1 Installare phyphox

- **Android:** Google Play Store → cerca "phyphox" → installa (app gratuita della RWTH Aachen)
- **iPhone:** App Store → cerca "phyphox" → installa

### 4.2 Caricare l'esperimento personalizzato (consigliato)

L'esperimento custom `virtual_coach_imu.phyphox` registra accelerometro e giroscopio in contemporanea (gli esperimenti predefiniti di phyphox ne leggono solo uno alla volta).

**Per caricarlo sul telefono, scegli UNO di questi metodi:**

- **Via email/messaggio:** invia il file `.phyphox` a te stesso, aprilo sul telefono → phyphox si apre e lo importa automaticamente.
- **Via cavo/cloud:** copia il file nella memoria del telefono, poi aprilo con phyphox dall'app File.
- **Via browser:** se il computer e il telefono sono sulla stessa WiFi, puoi anche usare AirDrop (iPhone) o Nearby Share (Android).

Dopo l'importazione, l'esperimento appare nella sezione "Virtual Coach" dell'app.

> **Alternativa senza file custom:** se non riesci a caricare il file, apri l'esperimento predefinito "Acceleration (with g)" nella sezione "Raw Sensors". Il notebook funziona anche con il solo accelerometro (il giroscopio sarà riempito con zeri, il gate STANDING/MOVING funziona lo stesso).

### 4.3 Attivare lo streaming WiFi

1. Apri l'esperimento "Virtual Coach IMU" (o "Acceleration with g").
2. Tocca il menu **⋮** (tre puntini, in alto a destra).
3. Seleziona **"Enable remote access"** (o "Abilita accesso remoto").
4. Sullo schermo compare un indirizzo tipo: **`http://192.168.1.50:8080`**
5. **Annotalo** — ti serve nello step successivo.
6. Premi il **pulsante ▶** (play) per avviare la registrazione.

> **Importante:** il telefono e il computer devono essere sulla **stessa rete WiFi**. Se sei in un'aula universitaria, verifica che la rete non blocchi il traffico tra dispositivi (le reti "guest" spesso lo fanno). In caso, usa l'hotspot del telefono e connettici il computer.

### 4.4 Verificare la connessione

Apri un browser sul computer e vai all'indirizzo che hai annotato (es. `http://192.168.1.50:8080`). Devi vedere l'interfaccia web di phyphox con i grafici in tempo reale. Se non la vedi, il traffico è bloccato dalla rete.

---

## STEP 5 — Configurare il notebook per la demo dal vivo

Apri il notebook e modifica **solo la prima cella di codice** (Pannello di controllo):

```python
# --- Task 1 (telefono) ---
SIMULATE_TASK1  = False                          # ← cambia da True a False
PHYPHOX_URL     = "http://192.168.1.50:8080"     # ← metti l'IP del TUO telefono

# --- Task 2/3 (webcam) ---
SIMULATE_WEBCAM = False                          # ← cambia da True a False
WEBCAM_INDEX    = 0                              # ← 0 = webcam predefinita
LIVE_MAX_SECONDS = 60                            # ← secondi di cattura (aumenta se servono più squat)
```

> **Consiglio per la prima volta:** cambia un flag alla volta. Prima testa solo il telefono (`SIMULATE_TASK1=False`, `SIMULATE_WEBCAM=True`), poi solo la webcam (`SIMULATE_TASK1=True`, `SIMULATE_WEBCAM=False`), infine entrambi.

---

## STEP 6 — Eseguire la demo dal vivo

### Preparazione (30 secondi prima)

1. Il telefono è in tasca o in vita, phyphox è in play con remote access attivo.
2. Il computer è aperto con il notebook, la webcam è attiva.
3. Posizionati davanti alla webcam: il corpo intero deve essere visibile (dalla testa ai piedi), con buona illuminazione.
4. Distanza dalla webcam: circa 2–3 metri.

### Esecuzione

1. **Esegui tutte le celle** fino alla cella "🎬 Demo integrata" (cella 24).
2. In alternativa, se hai già eseguito le celle precedenti, puoi rilanciare solo la cella della demo.

### Cosa succede durante la demo

**Fase 1 — Task 1 (telefono, ~3 secondi)**
- Il sistema legge il segnale dal telefono.
- Se stai **fermo in piedi**: stampa `gate = STANDING` → *"utente pronto"*.
- Appena **inizi a muoverti**: stampa `gate = MOVING` → *"avvio la sessione di esercizio"*.

**Fase 2+3 — Task 3 + Task 2 (webcam, finché non finisci gli squat)**
- Sullo schermo si apre una finestra con il feed della webcam.
- Sul tuo corpo è sovrapposto lo **scheletro** verde (Task 3 — 33 keypoint).
- In alto a sinistra appare l'**HUD** con:
  - `[Task1] contesto: MOVING`
  - `[Task2] riconosciuto: BodyWeightSquats (87%)`
  - `[Task2] ripetizioni: 3`
  - Feedback testuale (set/rep/qualità)
- Ad ogni squat completato, nella console appare `✓ ripetizione X/5`.
- Dopo 5 ripetizioni: `★ Allenamento completato`.

**Premi `q`** nella finestra del video per interrompere prima del tempo, se necessario.

**Al termine** il notebook stampa:
- Un riepilogo con `exercise_recognized`, `counter_state`, `counter_feedback`
- Un grafico a 3 pannelli: segnale cinematico con soglie, contatore, confidenza del riconoscimento
- La qualità (ROM) di ciascuna ripetizione

---

## STEP 7 — Cosa fare se qualcosa non va

### Il telefono non è raggiungibile

```
phyphox non raggiungibile a http://192.168.1.50:8080 (...)
```

**Cause e soluzioni:**
- L'esperimento non è in play → premi ▶ su phyphox.
- Remote access non è attivo → menu ⋮ → "Enable remote access".
- Rete diversa → connetti entrambi alla stessa WiFi (o usa l'hotspot del telefono).
- IP sbagliato → controlla l'IP mostrato da phyphox e aggiorna `PHYPHOX_URL`.
- **Piano B:** rimetti `SIMULATE_TASK1 = True` e la demo parte lo stesso con dati sintetici.

### La webcam non si apre

```
Webcam non apribile -> fallback sintetico.
```

**Cause e soluzioni:**
- Un'altra applicazione sta usando la webcam (Zoom, Teams, ecc.) → chiudila.
- Indice webcam sbagliato → prova `WEBCAM_INDEX = 1` o `2`.
- Permessi → su macOS, vai in Preferenze di Sistema → Privacy → Fotocamera → abilita Terminal/Python.
- **Piano B:** rimetti `SIMULATE_WEBCAM = True`.

### Il contatore non avanza

**Cause possibili:**
- Sei troppo vicino alla webcam e MediaPipe non vede le ginocchia/caviglie → allontanati (2–3 m).
- Illuminazione scarsa → accendi la luce.
- Squat troppo leggeri (range of motion < 18°) → scendi di più.
- Il classificatore non riconosce lo squat → non importa, il contatore avanza comunque grazie al segnale cinematico (angolo del ginocchio). Se neanche quello funziona, il segnale non ha abbastanza oscillazione.

### MediaPipe non si scarica

```
Detector non inizializzato (serve internet la prima volta)
```

Il modello MediaPipe (`pose_landmarker_full.task`, ~30 MB) si scarica automaticamente al primo avvio. Serve internet. Una volta scaricato, resta nella cartella `models/` e non serve più internet.

**Soluzione offline:** scarica manualmente il file da https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task e mettilo in `models/pose_landmarker_full.task`.

---

## STEP 8 — Configurazioni avanzate (opzionali)

### Cambiare il numero di ripetizioni e set

Nella prima cella:

```python
REPS_PER_SET = 10     # 10 ripetizioni per set
N_SETS       = 3      # 3 set
```

### Aggiungere il modello Task 1 (quando sarà pronto)

Quando il collega ti manda il modello Keras del Task 1:

1. Crea una sottocartella `task1_models/` nella cartella della demo.
2. Mettici:
   - il file `.h5` del modello (es. `inception_model.h5`)
   - il file `scaler_raw.pkl` (lo StandardScaler fittato sui 9 canali grezzi del training UCI-HAR)
3. Il notebook li trova automaticamente e li usa al posto del gate model-free.

### Usare solo l'accelerometro (senza esperimento custom)

Se non riesci a caricare l'esperimento custom sul telefono, apri l'esperimento predefinito **"Acceleration (with g)"** in phyphox. Il notebook gestisce automaticamente l'assenza del giroscopio (lo riempie con zeri). Il gate STANDING/MOVING funziona ugualmente bene con il solo accelerometro.

---

## Checklist rapida (da stampare per il giorno della demo)

```
□  File nella cartella: notebook + task3_module.py + best_2d.pt + best_3d.pt + meta.json
□  Ambiente Python attivato (conda activate virtualcoach)
□  Test in simulazione superato (5/5 reps, nessun errore)
□  Telefono: phyphox installato, esperimento aperto, ▶ premuto, remote access attivo
□  IP phyphox annotato e inserito in PHYPHOX_URL
□  Computer e telefono sulla stessa WiFi
□  Browser apre http://IP:8080 → vedo i grafici phyphox
□  SIMULATE_TASK1 = False, SIMULATE_WEBCAM = False nel notebook
□  Webcam funzionante (nessun'altra app la sta usando)
□  Posizione: 2-3 m dalla webcam, corpo intero visibile, buona luce
□  Pronto: eseguo le celle → sto fermo → mi muovo → faccio gli squat
```
