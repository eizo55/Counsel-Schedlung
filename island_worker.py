# island_worker.py  ── must be in the same folder as your notebook
import json, random, copy
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import logging
import time


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger("island_worker")


def _emit_log(message, log_queue=None):
    logger.info(message)
    if log_queue is not None:
        try:
            log_queue.put(("INFO", message))
        except Exception:
            pass
    else:
        print(message, flush=True)


TIMESLOTS = [
    "Day1_AM","Day1_PM","Day2_AM","Day2_PM","Day3_AM",
    "Day3_PM","Day4_AM","Day4_PM","Day5_AM","Day5_PM"
]

@dataclass
class Character:
    name: str
    home_location: str
    available: List[str]
    preferred_tasks: List[str]
    disliked_tasks: List[str]
    likes_people: List[str]
    dislikes_people: List[str]
    def available_at(self, ts): return ts in self.available

@dataclass
class Task:
    name: str
    required_people: List[str]
    min_people: int
    max_people: int
    preferred_people: List[str]
    forbidden_people: List[str]
    preferred_locations: List[str]
    allowed_locations: List[str]
    preferred_timeslots: List[str]
    value: int


def _build_objects(tasks_raw, chars_raw):
    tasks = {t["name"]: Task(
        name=t["name"],
        required_people=t.get("required_people", []),
        min_people=t["min_people"], max_people=t["max_people"],
        preferred_people=t.get("preferred_people", []),
        forbidden_people=t.get("forbidden_people", []),
        preferred_locations=t.get("preferred_locations", []),
        allowed_locations=t.get("allowed_locations", []),
        preferred_timeslots=t.get("preferred_timeslots", []),
        value=t["value"],
    ) for t in tasks_raw}

    chars = {c["name"]: Character(
        name=c["name"],
        home_location=c.get("home_location", "Unknown"),
        available=c["available"],
        preferred_tasks=c.get("preferred_tasks", []),
        disliked_tasks=c.get("disliked_tasks", []),
        likes_people=c.get("likes_people", []),
        dislikes_people=c.get("dislikes_people", []),
    ) for c in chars_raw}
    return tasks, chars


def _fitness(schedule, tasks, chars, scoring):
    score = 0.0
    S = scoring

    # -----------------------------
    # Calculate real double bookings
    # -----------------------------
    person_slots = {}

    for sl in schedule.values():
        for p in sl["attendees"]:
            person_slots.setdefault(p, []).append(sl["timeslot"])

    for p, slots in person_slots.items():
        if p not in chars:
            continue

        unique_slots = set(slots)

        for ts in unique_slots:
            count = slots.count(ts)

            # Penalize only the extra assignments causing conflict
            if count > 1:
                score += S["person_double_booked_penalty"] * (count - 1)

    # -----------------------------
    # Score each task
    # -----------------------------
    for tname, slot in schedule.items():

        task = tasks[tname]

        ts = slot["timeslot"]
        loc = slot["location"]
        att = slot["attendees"]

        aset = set(att)

        # Base task value
        score += task.value

        # -------------------------
        # Location scoring
        # -------------------------
        if loc not in task.allowed_locations:
            score += S["wrong_location_penalty"]

        elif loc in task.preferred_locations:
            score += S["preferred_location_bonus"]

        # -------------------------
        # Timeslot scoring
        # -------------------------
        if ts in task.preferred_timeslots:
            score += S["preferred_timeslot_bonus"]

        # -------------------------
        # Required people
        # -------------------------
        for r in task.required_people:
            if r not in aset:
                score += S["missing_required_person_penalty"]

        # -------------------------
        # Forbidden people
        # -------------------------
        for f in task.forbidden_people:
            if f in aset:
                score += S["forbidden_person_in_task_penalty"]

        # -------------------------
        # Preferred people
        # -------------------------
        for pref in task.preferred_people:
            if pref in aset:
                score += S["preferred_person_bonus"]

        # -------------------------
        # Availability
        # -------------------------
        for p in att:
            if p not in chars:
                continue

            if not chars[p].available_at(ts):
                score += S["person_scheduled_when_unavailable_penalty"]

        # -------------------------
        # Sympathie / Antipathie
        # bidirectional
        # -------------------------
        for i, p1 in enumerate(att):

            if p1 not in chars:
                continue

            ch1 = chars[p1]

            for p2 in att[i+1:]:

                if p2 not in chars:
                    continue

                ch2 = chars[p2]

                # Either direction likes
                if p2 in ch1.likes_people:
                    score += S["positive_pair_bonus_same_task"]

                if p1 in ch2.likes_people:
                    score += S["positive_pair_bonus_same_task"]

                # Either direction dislikes
                if p2 in ch1.dislikes_people:
                    score += S["negative_pair_penalty_same_task"]

                if p1 in ch2.dislikes_people:
                    score += S["negative_pair_penalty_same_task"]


    return score


def _rand_schedule(tasks, chars, locations):
    s = {}
    for tname, task in tasks.items():
        loc = random.choice(task.allowed_locations) if task.allowed_locations else random.choice(locations)
        ts  = random.choice(TIMESLOTS)
        att = [p for p in task.required_people if p in chars]
        pool = [n for n in task.preferred_people
                if n in chars and n not in att and n not in task.forbidden_people]
        random.shuffle(pool)
        for p in pool:
            if len(att) >= task.max_people: break
            att.append(p)
        others = [n for n in chars
                  if n not in att and n not in task.forbidden_people
                  and chars[n].available_at(ts)]
        random.shuffle(others)
        while len(att) < task.max_people and others:
            att.append(others.pop())
        s[tname] = {"timeslot": ts, "location": loc, "attendees": att}
    return s


def _tournament(pop, fits, k=4):
    idx = random.sample(range(len(pop)), min(k, len(pop)))
    return copy.deepcopy(pop[max(idx, key=lambda i: fits[i])])


# the same as in zeitplanung_got_dataset; 
# made two so i can check the difference between the two implementations
def _crossover(p1, p2):
    c1, c2 = {}, {}
    for k in p1:
        if random.random() < 0.5:
            c1[k] = copy.deepcopy(p1[k]); c2[k] = copy.deepcopy(p2[k])
        else:
            c1[k] = copy.deepcopy(p2[k]); c2[k] = copy.deepcopy(p1[k])
    return c1, c2

# the same as in zeitplanung_got_dataset; 
# made two so i can check the difference between the two implementations
def _mutate_half_biased(sched, tasks, chars, mr=0.3, mutation_ratio = 50):
    m = copy.deepcopy(sched)
    weighted_mutation_list = (["timeslot", "location"] + ["attendees"] * mutation_ratio)
    for tname, slot in m.items():
        if random.random() > mr: continue
        task = tasks[tname]
        # Weightetd choice of mutation type: timeslot, location, or attendees modification
        #tp = random.choice(["timeslot", "location", "attendees1", "attendees2", "attendees3", "attendees4", "attendees5", "attendees6", "attendees7", "attendees8", "attendees9", "attendees10"])
        
        tp = random.choice(weighted_mutation_list)
        
        if tp == "timeslot":
            slot["timeslot"] = random.choice(TIMESLOTS)
        elif tp == "location":
            if task.allowed_locations:
                slot["location"] = random.choice(task.allowed_locations)
        else:
            op  = random.choice(["replace", "add", "remove", "replace_with_preferred","replace_with_preferred1","replace_with_preferred2", "replace_with_friend", "replace_with_friend1", "replace_with_friend2"])
            att = slot["attendees"]; ts = slot["timeslot"]

            if op == "replace" and att:
                i = random.randrange(len(att))
                c = [n for n in chars if n not in att
                     and n not in task.forbidden_people
                     and chars[n].available_at(ts)]
                if c: att[i] = random.choice(c)
            
            if op == ("replace_with_friend" or op == "replace_with_friend1" or op == "replace_with_friend2") and att:
                i = random.randrange(len(att))
                remaining = [p for j, p in enumerate(att) if j != i]

                candidates = []
                weights = []

                for n in chars:
                    if (n not in att
                            and n not in task.forbidden_people
                            and chars[n].available_at(ts)):
                        score = sum(n in chars[p].likes_people for p in remaining)
                        if score > 0:
                            candidates.append(n)
                            weights.append(score)

                if candidates:
                    att[i] = random.choices(candidates, weights=weights, k=1)[0]

            elif  op in ("replace_with_preferred","replace_with_preferred1","replace_with_preferred2") and att:
                i = random.randrange(len(att))
                c = [n for n in task.preferred_people if n not in att
                        and n not in task.forbidden_people
                        and n in chars
                        and chars[n].available_at(ts)]
                if c: att[i] = random.choice(c)

            elif op == "add" and len(att) < task.max_people:
                c = [n for n in chars if n not in att
                     and n not in task.forbidden_people
                     and chars[n].available_at(ts)]
                if c: att.append(random.choice(c))

            elif op == "remove" and len(att) > task.min_people:
                rem = [p for p in att if p not in task.required_people]
                if rem: att.remove(random.choice(rem))
    return m

def _mutate(sched, tasks, chars, mr=0.3, mutation_ratio=50):
    m = copy.deepcopy(sched)
    weighted_mutation_list = (["timeslot", "location"] + ["attendees"] * mutation_ratio)

    for tname, slot in m.items():
        if random.random() > mr:
            continue

        task = tasks[tname]

        tp = random.choice(weighted_mutation_list)

        if tp == "timeslot":
            slot["timeslot"] = random.choice(TIMESLOTS)

        elif tp == "location":
            if task.allowed_locations:
                slot["location"] = random.choice(task.allowed_locations)

        else:
            op = random.choice([
                "replace",
                "add",
                "remove",
                "replace_with_preferred",
                "replace_with_preferred1",
                "replace_with_preferred2",
                "replace_with_friend",
                "replace_with_friend1",
                "replace_with_friend2"
            ])

            att = slot["attendees"]
            ts = slot["timeslot"]

            if op == "replace" and att:
                i = random.randrange(len(att))
                c = [
                    n for n in chars
                    if n not in att
                    and n not in task.forbidden_people
                    and chars[n].available_at(ts)
                ]
                if c:
                    att[i] = random.choice(c)

            elif op in (
                "replace_with_friend",
                "replace_with_friend1",
                "replace_with_friend2"
            ) and att:

                i = random.randrange(len(att))
                remaining = [p for j, p in enumerate(att) if j != i]

                candidates = []
                weights = []

                for n in chars:
                    if (
                        n not in att
                        and n not in task.forbidden_people
                        and chars[n].available_at(ts)
                    ):
                        score = sum(
                            n in chars[p].likes_people
                            for p in remaining
                        )

                        if score > 0:
                            candidates.append(n)
                            weights.append(score)

                if candidates:
                    att[i] = random.choices(
                        candidates,
                        weights=weights,
                        k=1
                    )[0]

            elif op in (
                "replace_with_preferred",
                "replace_with_preferred1",
                "replace_with_preferred2"
            ) and att:

                i = random.randrange(len(att))

                c = [
                    n for n in task.preferred_people
                    if n not in att
                    and n not in task.forbidden_people
                    and n in chars
                    and chars[n].available_at(ts)
                ]

                if c:
                    att[i] = random.choice(c)

            elif op == "add" and len(att) < task.max_people:
                c = [
                    n for n in chars
                    if n not in att
                    and n not in task.forbidden_people
                    and chars[n].available_at(ts)
                ]

                if c:
                    att.append(random.choice(c))

            elif op == "remove" and len(att) > task.min_people:
                rem = [
                    p for p in att
                    if p not in task.required_people
                ]

                if rem:
                    att.remove(random.choice(rem))

    return m

def _run_island_worker(args, record_snapshots=False, target_fit = 353):
    """Run one island of a standard genetic algorithm with migration."""

    # Define Args depending on cases; just to make the subproccess hand over the logs to the main process 
    # without breaking the loop. This is primarily for debugging, you can ignore it. 
    if len(args) >= 16:
        (tasks_raw, chars_raw, locations, scoring,
         pop_size, n_gen, crossover_rate, mutation_rate,
         mutation_ratio, elitism, k, seed,
         immigrants, pop, log_queue, target_generation_wanted) = args
     # 1. # logging turned on and pop provided
    elif len(args) == 15: 
        (tasks_raw, chars_raw, locations, scoring,
         pop_size, n_gen, crossover_rate, mutation_rate,
         mutation_ratio, elitism, k, seed,
         immigrants, pop, log_queue) = args
        target_generation_wanted = False
    # 2. # logging turned on, but no pop provided
    elif len(args) == 14:
        (tasks_raw, chars_raw, locations, scoring,
         pop_size, n_gen, crossover_rate, mutation_rate,
         mutation_ratio, elitism, k, seed,
         immigrants, log_queue) = args
        pop = None
        target_generation_wanted = False
    # 3. # logging turned off, but pop provided
    elif len(args) == 13:
        (tasks_raw, chars_raw, locations, scoring,
         pop_size, n_gen, crossover_rate, mutation_rate,
         elitism, k, seed, immigrants, pop) = args
        log_queue = None
        mutation_ratio = 50
        target_generation_wanted = False
    # 4. # logging turned off, no pop provided
    else:
        (tasks_raw, chars_raw, locations, scoring,
         pop_size, n_gen, crossover_rate, mutation_rate,
         elitism, k, seed, immigrants) = args
        pop = None
        log_queue = None
        mutation_ratio = 50
        target_generation_wanted = False

    random.seed(seed)
    np.random.seed(seed)

    tasks, chars = _build_objects(tasks_raw, chars_raw)

    # if its the first round the initial population is created here, 
    # otherwise the one from the prev iteration is used
    if pop == None:
        #_emit_log("[island_worker] Generating initial population", log_queue)
        pop = [_rand_schedule(tasks, chars, locations) for _ in range(pop_size)]
    fits = [_fitness(ind, tasks, chars, scoring) for ind in pop]

    best_s = copy.deepcopy(pop[int(np.argmax(fits))])
    best_f = max(fits)
    hist = [best_f]
    snapshots = []
    target_generation = -1
    target_time = -1

    for gen in range(n_gen):
        if immigrants and gen == 0:
            #_emit_log(f"[island_worker] Applying {len(immigrants)} immigrants at generation 0", log_queue)
            weak_idx = np.argsort(fits)[: len(immigrants)]
            for idx, imm in zip(weak_idx, immigrants):
                if imm:
                    pop[int(idx)] = copy.deepcopy(imm)
            fits = [_fitness(ind, tasks, chars, scoring) for ind in pop]

        elite_idx = np.argsort(fits)[-max(1, elitism):]
        new_pop = [copy.deepcopy(pop[i]) for i in elite_idx]

        while len(new_pop) < pop_size:
            p1 = _tournament(pop, fits, k)
            p2 = _tournament(pop, fits, k)
            if random.random() < crossover_rate:
                c1, c2 = _crossover(p1, p2)
            else:
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

            new_pop.append(_mutate(c1, tasks, chars, mutation_rate, mutation_ratio))
            if len(new_pop) < pop_size:
                new_pop.append(_mutate(c2, tasks, chars, mutation_rate, mutation_ratio))

        pop = new_pop[:pop_size]
        fits = [_fitness(ind, tasks, chars, scoring) for ind in pop]
        gf = max(fits)
        if gf > best_f:
            best_f = gf
            best_s = copy.deepcopy(pop[int(np.argmax(fits))])
        hist.append(best_f)

        if best_f >= target_fit and target_generation_wanted and target_generation == -1:
            target_generation = gen
            target_time =  time.perf_counter() 
        if record_snapshots and gen % 50 == 0:
            snapshots.append((gen, copy.deepcopy(best_s), best_f))

    #_emit_log(f"[island_worker] Finished with best fitness {best_f:.2f}", log_queue)

    top_idx = np.argsort(fits)[-5:]
    emigrants = [copy.deepcopy(pop[i]) for i in top_idx]

    if record_snapshots:
        return best_s, best_f, hist, emigrants, pop, snapshots
    if target_generation_wanted:
        return best_s, best_f, hist, emigrants, pop, target_generation, target_time
    return best_s, best_f, hist, emigrants, pop

def run_island(args):
    """Top-level pickleable worker for a normal island-based genetic algorithm."""
    return _run_island_worker(args, record_snapshots=False)

def run_island_vis(args):
    """Top-level pickleable worker with snapshot support for visualization."""
    return _run_island_worker(args, record_snapshots=True)

def run_island_worker_random_init(args, record_snapshots=False,target_fit=353):
    """Run one island of a standard genetic algorithm with migration."""

    # Define Args depending on cases; just to make the subproccess hand over the logs to the main process 
    # without breaking the loop. This is primarily for debugging, you can ignore it. 
    # 1. # logging turned on and pop provided
    if len(args) >= 15: 
        (tasks_raw, chars_raw, locations, scoring,
         pop_size, n_gen, crossover_rate, mutation_rate,
         mutation_ratio, elitism, k, seed,
         immigrants, pop, log_queue, target_generation_wanted) = args
    # 2. # logging turned on, but no pop provided
    elif len(args) == 14: 
        (tasks_raw, chars_raw, locations, scoring,
         pop_size, n_gen, crossover_rate, mutation_rate,
         mutation_ratio, elitism, k, seed,
         immigrants, log_queue) = args
        pop = None
    # 3. # logging turned off, but pop provided
    elif len(args) == 13:
        (tasks_raw, chars_raw, locations, scoring,
         pop_size, n_gen, crossover_rate, mutation_rate,
         elitism, k, seed, immigrants, pop) = args
        log_queue = None
        mutation_ratio = 50
    # 4. # logging turned off, no pop provided
    else:
        (tasks_raw, chars_raw, locations, scoring,
         pop_size, n_gen, crossover_rate, mutation_rate,
         elitism, k, seed, immigrants) = args
        pop = None
        log_queue = None
        mutation_ratio = 50

    random.seed(seed)
    np.random.seed(seed)

    tasks, chars = _build_objects(tasks_raw, chars_raw)

    # if its the first round the initial population is created here, 
    # otherwise the one from the prev iteration is used
    if pop == None:
        #_emit_log("[island_worker] Generating initial population", log_queue)
        pop = [_rand_schedule_random_init(tasks, chars, locations) for _ in range(pop_size)]
    fits = [_fitness(ind, tasks, chars, scoring) for ind in pop]

    best_s = copy.deepcopy(pop[int(np.argmax(fits))])
    best_f = max(fits)
    hist = [best_f]
    snapshots = []

    target_generation = -1
    target_time = -1


    for gen in range(n_gen):
        if immigrants and gen == 0:
            #_emit_log(f"[island_worker] Applying {len(immigrants)} immigrants at generation 0", log_queue)
            weak_idx = np.argsort(fits)[: len(immigrants)]
            for idx, imm in zip(weak_idx, immigrants):
                if imm:
                    pop[int(idx)] = copy.deepcopy(imm)
            fits = [_fitness(ind, tasks, chars, scoring) for ind in pop]

        elite_idx = np.argsort(fits)[-max(1, elitism):]
        new_pop = [copy.deepcopy(pop[i]) for i in elite_idx]

        while len(new_pop) < pop_size:
            p1 = _tournament(pop, fits, k)
            p2 = _tournament(pop, fits, k)
            if random.random() < crossover_rate:
                c1, c2 = _crossover(p1, p2)
            else:
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

            new_pop.append(_mutate(c1, tasks, chars, mutation_rate, mutation_ratio))
            if len(new_pop) < pop_size:
                new_pop.append(_mutate(c2, tasks, chars, mutation_rate, mutation_ratio))

        pop = new_pop[:pop_size]
        fits = [_fitness(ind, tasks, chars, scoring) for ind in pop]
        gf = max(fits)
        if gf > best_f:
            best_f = gf
            best_s = copy.deepcopy(pop[int(np.argmax(fits))])
        hist.append(best_f)

        if record_snapshots and gen % 50 == 0:
            snapshots.append((gen, copy.deepcopy(best_s), best_f))
        if best_f >= target_fit and target_generation_wanted and target_generation == -1:
                    target_generation = gen
                    target_time =  time.perf_counter()
    #_emit_log(f"[island_worker] Finished with best fitness {best_f:.2f}", log_queue)

    top_idx = np.argsort(fits)[-5:]
    emigrants = [copy.deepcopy(pop[i]) for i in top_idx]

    if record_snapshots:
        return best_s, best_f, hist, emigrants, pop, snapshots
    if target_generation_wanted:
        return best_s, best_f, hist, emigrants, pop, target_generation, target_time
    return best_s, best_f, hist, emigrants, pop

def _rand_schedule_random_init(tasks, chars, locations):
    s = {}
    for tname, task in tasks.items():
        loc = random.choice(locations)
        ts  = random.choice(TIMESLOTS)
        att = []
        pool = [n for n in chars]
        random.shuffle(pool)
        for p in pool:
            if len(att) >= task.max_people: break
            att.append(p)
        s[tname] = {"timeslot": ts, "location": loc, "attendees": att}
    return s

def _mutate_non_biased(sched, tasks, chars, mr=0.3, mutation_ratio = 50):
    m = copy.deepcopy(sched)
    weighted_mutation_list = (["timeslot", "location"] + ["attendees"] * mutation_ratio)
    for tname, slot in m.items():
        if random.random() > mr: continue
        task = tasks[tname]
        # Weightetd choice of mutation type: timeslot, location, or attendees modification
        #tp = random.choice(["timeslot", "location", "attendees1", "attendees2", "attendees3", "attendees4", "attendees5", "attendees6", "attendees7", "attendees8", "attendees9", "attendees10"])
        
        tp = random.choice(weighted_mutation_list)
        
        if tp == "timeslot":
            slot["timeslot"] = random.choice(TIMESLOTS)
        elif tp == "location":
            if task.allowed_locations:
                slot["location"] = random.choice(task.allowed_locations)
        else:
            op  = random.choice(["replace", "add", "remove"])
            att = slot["attendees"]; ts = slot["timeslot"]

            if op == "replace" and att:
                i = random.randrange(len(att))
                c = [n for n in chars if n not in att
                     and n not in task.forbidden_people
                     and chars[n].available_at(ts)]
                if c: att[i] = random.choice(c)

            elif op == "add" and len(att) < task.max_people:
                c = [n for n in chars if n not in att
                     and n not in task.forbidden_people
                     and chars[n].available_at(ts)]
                if c: att.append(random.choice(c))

            elif op == "remove" and len(att) > task.min_people:
                rem = [p for p in att if p not in task.required_people]
                if rem: att.remove(random.choice(rem))
    return m

def _run_island_worker_non_biased_mutation(args, record_snapshots=False, target_fit = 353):
    """Run one island of a standard genetic algorithm with migration. and _mutate_non_biased"""

    # Define Args depending on cases; just to make the subproccess hand over the logs to the main process 
    # without breaking the loop. This is primarily for debugging, you can ignore it. 
    # 1. # logging turned on and pop provided
    if len(args) >= 15: 
        (tasks_raw, chars_raw, locations, scoring,
         pop_size, n_gen, crossover_rate, mutation_rate,
         mutation_ratio, elitism, k, seed,
         immigrants, pop, log_queue, target_generation_wanted) = args
    # 2. # logging turned on, but no pop provided
    elif len(args) == 14:
        (tasks_raw, chars_raw, locations, scoring,
         pop_size, n_gen, crossover_rate, mutation_rate,
         mutation_ratio, elitism, k, seed,
         immigrants, log_queue) = args
        pop = None
    # 3. # logging turned off, but pop provided
    elif len(args) == 13:
        (tasks_raw, chars_raw, locations, scoring,
         pop_size, n_gen, crossover_rate, mutation_rate,
         elitism, k, seed, immigrants, pop) = args
        log_queue = None
        mutation_ratio = 50
    # 4. # logging turned off, no pop provided
    else:
        (tasks_raw, chars_raw, locations, scoring,
         pop_size, n_gen, crossover_rate, mutation_rate,
         elitism, k, seed, immigrants) = args
        pop = None
        log_queue = None
        mutation_ratio = 50

    target_generation = -1
    target_time = -1

    random.seed(seed)
    np.random.seed(seed)

    tasks, chars = _build_objects(tasks_raw, chars_raw)

    # if its the first round the initial population is created here, 
    # otherwise the one from the prev iteration is used
    if pop == None:
        #_emit_log("[island_worker] Generating initial population", log_queue)
        pop = [_rand_schedule(tasks, chars, locations) for _ in range(pop_size)]
    fits = [_fitness(ind, tasks, chars, scoring) for ind in pop]

    best_s = copy.deepcopy(pop[int(np.argmax(fits))])
    best_f = max(fits)
    hist = [best_f]
    snapshots = []

    for gen in range(n_gen):
        if immigrants and gen == 0:
            #_emit_log(f"[island_worker] Applying {len(immigrants)} immigrants at generation 0", log_queue)
            weak_idx = np.argsort(fits)[: len(immigrants)]
            for idx, imm in zip(weak_idx, immigrants):
                if imm:
                    pop[int(idx)] = copy.deepcopy(imm)
            fits = [_fitness(ind, tasks, chars, scoring) for ind in pop]

        elite_idx = np.argsort(fits)[-max(1, elitism):]
        new_pop = [copy.deepcopy(pop[i]) for i in elite_idx]

        while len(new_pop) < pop_size:
            p1 = _tournament(pop, fits, k)
            p2 = _tournament(pop, fits, k)
            if random.random() < crossover_rate:
                c1, c2 = _crossover(p1, p2)
            else:
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

            new_pop.append(_mutate_non_biased(c1, tasks, chars, mutation_rate, mutation_ratio))
            if len(new_pop) < pop_size:
                new_pop.append(_mutate_non_biased(c2, tasks, chars, mutation_rate, mutation_ratio))

        pop = new_pop[:pop_size]
        fits = [_fitness(ind, tasks, chars, scoring) for ind in pop]
        gf = max(fits)
        if gf > best_f:
            best_f = gf
            best_s = copy.deepcopy(pop[int(np.argmax(fits))])
            
        hist.append(best_f)
       
        if record_snapshots and gen % 50 == 0:
            snapshots.append((gen, copy.deepcopy(best_s), best_f))

        #_emit_log(f"[island_worker] Finished with best fitness {best_f:.2f}", log_queue)
        if best_f >= target_fit and target_generation_wanted and target_generation == -1:
                    target_generation = gen
                    target_time =  time.perf_counter() 
    top_idx = np.argsort(fits)[-5:]
    emigrants = [copy.deepcopy(pop[i]) for i in top_idx]

    if record_snapshots:
        return best_s, best_f, hist, emigrants, pop, snapshots
    if target_generation_wanted:
        return best_s, best_f, hist, emigrants, pop, target_generation, target_time
    return best_s, best_f, hist, emigrants, pop
