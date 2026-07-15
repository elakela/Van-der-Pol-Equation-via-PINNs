# Physics-Informed Neural Networks per l'Equazione di Van der Pol

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Progetto per il corso di **Numerical Methods for Scientific Computing** (Università degli Studi di Catania, LM-18), Anno Accademico 2025-2026.

**Autore:** Gabriela Riscica

---

## 📖 Indice
- [Descrizione del Progetto](#-descrizione-del-progetto)
- [Metodologia](#-metodologia)
- [Struttura del Repository](#-struttura-del-repository)
- [Requisiti e Installazione](#-requisiti-e-installazione)
- [Utilizzo](#-utilizzo)
  - [Esempi di Esecuzione](#esempi-di-esecuzione)
  - [Parametri della CLI](#parametri-della-cli)
- [Output e Metriche](#-output-e-metriche)

---

## 🎯 Descrizione del Progetto

Questo repository contiene il codice e la relazione per la risoluzione dell'equazione differenziale non lineare di **Van der Pol** in regime di forte non linearità (comportamento *stiff*) utilizzando le **Physics-Informed Neural Networks (PINN)**. 

L'equazione di Van der Pol è definita come:

$$ \frac{d^2x}{dt^2} - \mu (1 - x^2) \frac{dx}{dt} + x = 0 $$

dove il parametro $\mu$ controlla la non-linearità e la stiffness del sistema.
I modelli PINN standard faticano a convergere su problemi *stiff* a causa del mal-condizionamento dei residui fisici, dello *spectral bias* e della violazione del principio di causalità su lunghi orizzonti temporali. Questo progetto propone un'architettura avanzata per superare tali ostacoli.

---

## 🔬 Metodologia

Per garantire un'elevata accuratezza, l'impianto algoritmico si basa su quattro pilastri metodologici fondamentali:

1. **Formulazione nel piano di Liénard**: 
   Conversione dell'equazione canonica in un sistema del primo ordine. Questo abbassa l'ordine delle derivate ed evita l'esplosione del gradiente, bilanciando la scala dei residui.
2. **Hard Constraints**: 
   Imposizione esatta delle condizioni iniziali tramite un *Ansatz* lineare, per evitare la calibrazione di pesi aggiuntivi nella funzione di *loss*.
3. **Time-Matching causale**: 
   Scomposizione del dominio in sotto-finestre temporali risolte sequenzialmente (*warm-start*), in modo da preservare la causalità e limitare la retropropagazione dell'errore.
4. **Allocazione adattativa (*Adaptive Time-Marching*)**: 
   Controllo dinamico del passo temporale basato sull'euristica del gradiente, con meccanismi di *step rejection* e *rollback* per catturare accuratamente le transizioni rapide della dinamica.

---

## 📂 Struttura del Repository

| File/Directory | Descrizione |
|---|---|
| 📄 [`vdp_pinn.py`](vdp_pinn.py) | Script Python principale. Contiene l'implementazione del modello PINN, l'addestramento ibrido (Adam + L-BFGS), il time-marching adattativo e la generazione dei grafici. |
| 📄 `Relazione.pdf` / `Relation.pdf` | Relazione completa del progetto che documenta dettagliatamente l'approccio teorico, le architetture, la metodologia proposta e l'analisi dei risultati ottenuti. |

---

## ⚙️ Requisiti e Installazione

Per eseguire lo script sono necessarie le seguenti librerie:

```bash
pip install torch numpy scipy matplotlib
```

*Nota: `scipy` è utilizzato esclusivamente per calcolare la soluzione di riferimento (ground truth) tramite il solutore implicito Radau, al fine di valutare l'errore del modello PINN.*

---

## 🚀 Utilizzo

Per avviare la simulazione e generare i grafici di output, è sufficiente eseguire lo script `vdp_pinn.py` da riga di comando.

### Esempi di Esecuzione

**Esecuzione di base (Consigliata)**
Approccio adattativo su un problema molto stiff ($\mu = 10.0$):
```bash
python vdp_pinn.py --mu 10.0 --mode march
```

**Esecuzione con approccio classico (Per confronto)**
Addestramento su un'unica griglia fissa sull'intero dominio (soggetto a fallimento per alti valori di $\mu$):
```bash
python vdp_pinn.py --mu 10.0 --mode single
```

### Parametri della CLI

Di seguito i principali argomenti da riga di comando per configurare la simulazione:

| Argomento | Default | Descrizione |
| :--- | :---: | :--- |
| `--mu` | `1.0` | Regola il grado di non-linearità del sistema. Valori elevati (es. `10.0`) rendono l'equazione fortemente *stiff*. |
| `--periods` | `2.0` | Numero di periodi stimati per l'estensione dell'orizzonte temporale. In alternativa usare `--T` per forzare il tempo finale. |
| `--mode` | `march` | Modalità di soluzione: `march` (time-marching a finestre) o `single` (dominio intero). |
| `--uniform` | *False* | Flag per disabilitare il passo adattativo nel time-marching e forzare un passo costante (`--h`). |
| `--dx-cap` | `0.4` | Variazione massima consentita per lo stato in una singola finestra (usata nel time-marching adattativo). |
| `--width` | `64` | Larghezza (numero di neuroni per livello) del Multi-Layer Perceptron. |
| `--depth` | `4` | Profondità (numero di livelli nascosti) del Multi-Layer Perceptron. Attivazione `Tanh`. |
| `--colloc` | `256` | Numero di punti di collocazione fisica per ogni singola finestra. |
| `--adam` | `2500` | Iterazioni massime per l'ottimizzatore Adam (fase di esplorazione). |
| `--lbfgs` | `400` | Iterazioni massime per l'ottimizzatore L-BFGS (fase di fine-tuning). |

---

## 📊 Output e Metriche

Al termine dell'elaborazione, lo script calcola e stampa a schermo l'**errore relativo globale in norma L2**.

Contestualmente, viene generato e salvato un grafico in formato PNG (es. `vdp_mu10_march.png`) contenente 4 *subplot* diagnostici:
1. Evoluzione temporale dello stato $x(t)$
2. Evoluzione temporale della velocità $v(t)$
3. Ritratto di fase (Phase portrait)
4. Andamento asintotico dell'errore assoluto in scala semilogaritmica

---
