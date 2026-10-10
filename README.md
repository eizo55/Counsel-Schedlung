# Optimizing the Small Council: A Genetic Algorithm Approach

Master course project (HTW Berlin, *Programmierkonzepte und Algorithmen*).

Schedules nine tasks of a Game of Thrones "small council" (time slot, location and attendees out of 25 characters) with a genetic algorithm that maximizes an overall value score under hard and soft constraints. An island model runs several populations in parallel.

## Contents

| File | Description |
|---|---|
| `Genetic Algorithim Paper.docx` | Written paper describing the problem, method and results |
| `zeitplanung_got_dataset.ipynb` | Main notebook (in German): data loading, GA implementation, experiments, plots |
| `island_worker.py` | Worker code for the parallel island model; must be in the same folder as the notebook |

## Running it

1. Install dependencies: `pip install numpy pandas matplotlib jupyter`
2. Put the dataset file `task_scheduling.json` next to the notebook (it is not included in this repository).
3. Start `jupyter notebook` and run `zeitplanung_got_dataset.ipynb` from top to bottom.
