# Van-der-Pol-Equation-via-PINNs
This repository implements a Physics-Informed Neural Network (PINN) to solve and simulate the non-linear Van der Pol oscillator equation. By embedding the governing differential equations directly into the neural network's loss function, the model accurately captures the system's dynamic behavior and stable limit cycles.

Questo repository contiene il codice e la relazione per il progetto del corso di **Numerical Methods for Scientific Computing** (Università degli Studi di Catania, LM-18), anno accademico 2025-2026.

Autore: **Gabriela Riscica**
-------------
## Descrizione del Progetto

Il progetto affronta la risoluzione dell'equazione differenziale non lineare di **Van der Pol** in regime di forte non linearità (comportamento *stiff*) utilizzando le **Physics-Informed Neural Networks (PINN)**. I modelli PINN standard faticano a convergere su problemi *stiff* a causa del mal-condizionamento dei residui fisici, dello *spectral bias* e della violazione del principio di causalità su lunghi orizzonti temporali. 

Per superare queste limitazioni e garantire un'elevata accuratezza, l'impianto algoritmico si basa su quattro pilastri metodologici fondamentali:
1. **Formulazione nel piano di Liénard**: conversione dell'equazione canonica in un sistema del primo ordine per abbassare l'ordine delle derivate, ed evitare l'esplosione del gradiente bilanciando la scala dei residui.
2. **Hard Constraints**: imposizione esatta delle condizioni iniziali tramite un *Ansatz* lineare, per evitare la calibrazione di pesi aggiuntivi nella funzione di *loss*.
3. **Time-Matching causale**: scomposizione del dominio in sotto-finestre temporali risolte sequenzialmente (*warm-start*), in modo da preservare la causalità e limitare la retropropagazione dell'errore.
4. **Allocazione adattativa (*Adaptive Time-Marching*)**: controllo dinamico del passo temporale basato sull'euristica del gradiente, con meccanismi di *step rejection* e *rollback* per catturare accuratamente le transizioni rapide della dinamica.

## Struttura del Repository

- `vdp_pinn.py`: Script Python contenente l'implementazione del modello PINN (creazione della rete, addestramento ibrido Adam + L-BFGS, time-marching adattativo e generazione dei grafici).
- `Relazione.pdf`: Relazione completa del progetto che documenta dettagliatamente l'approccio teorico, le architetture, la metodologia proposta e l'analisi dei risultati ottenuti.
- 'Relation.pdf': Versione inglese della relazione

## Requisiti

Per eseguire lo script sono necessarie le seguenti librerie:
- `python` (>= 3.8 consigliato)
- `torch`
- `numpy`
- `scipy` (usato esclusivamente per calcolare la soluzione di riferimento *ground truth* con il solutore implicito Radau per il calcolo dell'errore)
- `matplotlib`

## Utilizzo

Per avviare la simulazione e generare i grafici di output, è sufficiente l'esecuzione dello script `vdp_pinn.py` da riga di comando.

**Esempio di esecuzione di base** (con approccio adattativo su un problema molto stiff):
```bash
python vdp_pinn.py --mu 10.0 --mode march
```

**Esempio di fallimento dell'approccio classico** (addestramento su un'unica griglia fissa sull'intero dominio):
```bash
python vdp_pinn.py --mu 10.0 --mode single
```

### Parametri e Opzioni

Di seguito i principali argomenti da riga di comando per configurare la simulazione:

- `--mu`: Regola il grado di non-linearità del sistema (default: `1.0`). L'impostazione di un valore elevato (es. `10.0`) rende l'equazione fortemente *stiff*.
- `--periods`: Numero di periodi (stimati) per l'estensione dell'orizzonte temporale (default: `2.0`). In alternativa è possibile forzare un tempo finale esatto con `--T`.
- `--mode`: Modalità di soluzione. Le scelte possibili sono `march` (default, time-marching a finestre) e `single` (dominio intero).
- `--uniform`: Flag per disabilitare il passo adattativo nel time-marching e forzare un passo costante `--h`.
- `--dx-cap`: Variazione massima consentita per lo stato in una singola finestra (usata come soglia di tolleranza nel time-marching adattativo, default: `0.4`).
- `--width`, `--depth`: Iperparametri strutturali del Multi-Layer Perceptron (default: rete 4x64 con attivazione `Tanh`).
- `--colloc`: Numero di punti di collocazione fisica per ogni singola finestra (default: `256`).
- `--adam`, `--lbfgs`: Iterazioni massime concesse rispettivamente all'ottimizzatore Adam per l'esplorazione, e L-BFGS per il fine-tuning (default: 2500 per Adam, 400 per L-BFGS).

## Output e Metriche

Al termine dell'elaborazione, lo script restituisce in output l'errore relativo globale calcolato in norma L2. Contestualmente, viene prodotto e salvato automaticamente nella directory corrente un grafico png (es. `vdp_mu10_march.png`) che rappresenta 4 *subplot* diagnostici:
1. Evoluzione temporale dello stato $x(t)$
2. Evoluzione temporale della velocità $v(t)$
3. Ritratto di fase
4. Andamento asintotico dell'errore assoluto in scala semilogaritmica.
