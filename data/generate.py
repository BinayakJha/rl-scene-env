"""Generate templated spoken lines with gold scene-plan labels.
Train and test use DISJOINT nouns and DISJOINT templates (for spatial lines),
and disjoint abstract templates, so test measures generalisation.
Usage: python data/generate.py  -> data/train.jsonl, data/test.jsonl
"""
import json, random, os

NUM_WORDS = {1: ["a", "one"], 2: ["two", "2"], 3: ["three", "3"], 4: ["four", "4"], 5: ["five", "5"]}
RELATIONS = {  # surface phrase -> canonical relation type
    "on": "on", "on top of": "on", "under": "under", "beneath": "under",
    "behind": "behind", "in front of": "in_front_of", "next to": "next_to",
    "beside": "next_to", "inside": "inside", "in": "inside",
    "to the left of": "left_of", "left of": "left_of",
    "to the right of": "right_of", "right of": "right_of",
}
# (singular, plural)
TRAIN_SUBJ = [("cat","cats"),("dog","dogs"),("kid","kids"),("person","people"),("bird","birds"),
  ("apple","apples"),("book","books"),("chair","chairs"),("student","students"),("ball","balls"),
  ("cup","cups"),("bicycle","bicycles"),("flower","flowers"),("robot","robots"),("child","children")]
TRAIN_TGT = ["table","tree","house","car","box","bench","door","teacher","desk","bus"]
TEST_SUBJ = [("rabbit","rabbits"),("lamp","lamps"),("woman","women"),("penguin","penguins"),
  ("candle","candles"),("horse","horses"),("bottle","bottles"),("musician","musicians")]
TEST_TGT = ["fountain","piano","tent","lighthouse","van","sofa"]

TRAIN_TEMPL = ["{n} {s} {r} the {t}", "put {n} {s} {r} the {t}", "show me {n} {s} {r} the {t}",
  "there are {n} {s} {r} the {t}", "I want {n} {s} standing {r} the {t}", "draw {n} {s} {r} the {t}"]
TEST_TEMPL = ["imagine {n} {s} {r} the {t}", "can you place {n} {s} {r} the {t}",
  "so we have {n} {s} sitting {r} the {t}", "let's see {n} {s} {r} the {t}"]

# abstract lines: (text, gold treatment). 'hold' = show nothing new.
TRAIN_ABS = [
  ("our revenue is growing but we keep less of it", "chart"),
  ("sales went up every quarter this year", "chart"),
  ("user signups doubled since march", "chart"),
  ("costs rose faster than income last year", "chart"),
  ("compare the old plan with the new plan", "comparison"),
  ("cats versus dogs as pets", "comparison"),
  ("renting versus buying a home", "comparison"),
  ("the difference between weather and climate", "comparison"),
  ("how a bill becomes a law", "process"),
  ("first we collect data then we clean it then we train the model", "process"),
  ("the steps of photosynthesis", "process"),
  ("how coffee gets from farm to cup", "process"),
  ("a red bicycle", "object"), ("a big old oak tree", "object"), ("the eiffel tower", "object"),
  ("wait let me think", "hold"), ("hmm give me a second", "hold"),
  ("as I said before", "hold"), ("let me repeat that", "hold"),
  ("sorry I meant the same thing", "hold"), ("okay so yeah", "hold"),
]
TEST_ABS = [
  ("profits are flat even though customers grew", "chart"),
  ("our churn dropped by half this quarter", "chart"),
  ("iphone compared with android", "comparison"),
  ("public school versus private school", "comparison"),
  ("how vaccines train the immune system", "process"),
  ("the water cycle from rain to river to sea", "process"),
  ("a golden retriever", "object"), ("a tall glass skyscraper", "object"),
  ("um just a moment", "hold"), ("like I mentioned earlier", "hold"),
  ("no wait that is what I already said", "hold"), ("right right okay", "hold"),
]

def spatial(rng, subjs, tgts, templs):
    k = rng.randint(1, 5); s = rng.choice(subjs); t = rng.choice(tgts)
    phrase = rng.choice(list(RELATIONS)); word = rng.choice(NUM_WORDS[k])
    text = rng.choice(templs).format(n=word, s=s[0] if k == 1 else s[1], r=phrase, t=t)
    gold = {"objects": [{"name": s[0], "count": k, "relation": {"type": RELATIONS[phrase], "target": t}},
                        {"name": t, "count": 1, "relation": None}], "treatment": "object"}
    return {"text": text, "gold": gold, "kind": "spatial"}

def abstract(rng, pool):
    text, tr = rng.choice(pool)
    return {"text": text, "gold": {"objects": [], "treatment": tr}, "kind": "abstract"}

def build(n, seed, subjs, tgts, templs, abs_pool, abs_frac=0.25):
    rng = random.Random(seed); out, seen = [], set()
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        ex = abstract(rng, abs_pool) if rng.random() < abs_frac else spatial(rng, subjs, tgts, templs)
        if ex["kind"] == "spatial" and ex["text"] in seen: continue
        seen.add(ex["text"]); out.append(ex)
    return out

if __name__ == "__main__":
    d = os.path.dirname(os.path.abspath(__file__))
    for name, rows in [("train", build(2000, 0, TRAIN_SUBJ, TRAIN_TGT, TRAIN_TEMPL, TRAIN_ABS)),
                       ("test", build(300, 1, TEST_SUBJ, TEST_TGT, TEST_TEMPL, TEST_ABS))]:
        with open(os.path.join(d, f"{name}.jsonl"), "w") as f:
            for i, r in enumerate(rows): f.write(json.dumps({"id": f"{name}-{i}", **r}) + "\n")
        print(name, len(rows))
