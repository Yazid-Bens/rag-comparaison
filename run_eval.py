import time, argparse, os
import pandas as pd
from tqdm import tqdm
import config
from llm import STATS
from data import LOADERS, build_corpus, build_chunks
from indexing import Indexes
from strategies import STRATEGIES, answer
from metrics import (f1, exact_match, recall_at_k, precision_at_k, mrr,
                     context_precision, faithfulness, answer_relevancy)

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=config.DATASETS)
ap.add_argument("--methods", nargs="+", default=list(STRATEGIES))
ap.add_argument("--scale", type=float, default=1.0, help="multiply question counts")
ap.add_argument("--no-judge", action="store_true")
ap.add_argument("--judge-every", type=int, default=1, help="LLM-judge metrics every k-th question")
ap.add_argument("--resume", action="store_true", help="skip methods already in results_<ds>.csv")
args = ap.parse_args()

all_rows = []
for ds in args.datasets:
    print(f"\n=== {ds} ===")
    n = max(5, int(config.N_PER_DATASET[ds] * args.scale))
    questions = LOADERS[ds](n)
    chunks = build_chunks(build_corpus(questions))
    print(f"{len(questions)} questions, {len(chunks)} chunks")
    idx = Indexes(chunks, config.EMBED_MODELS.get(ds, config.EMBED_MODEL))
    path = f"results_{ds}.csv"
    ds_rows = pd.read_csv(path).to_dict("records") if args.resume and os.path.exists(path) else []
    done = {r["method"] for r in ds_rows}

    for name in args.methods:
        if name in done:
            continue
        s = STRATEGIES[name]()
        for qid, q in enumerate(tqdm(questions, desc=f"{ds}/{name}")):
            c0 = STATS["calls"]
            t0 = time.time()
            retrieved = s.run(q["question"], idx)
            t_ret = time.time() - t0
            t1 = time.time()
            ans = answer(q["question"], retrieved)
            t_gen = time.time() - t1
            row = {"dataset": ds, "method": name, "qid": qid, "question": q["question"],
                   "gold": q["answers"][0], "pred": ans,
                   "f1": f1(ans, q["answers"]), "em": exact_match(ans, q["answers"]),
                   "recall@5": recall_at_k(retrieved, q["gold_titles"], config.TOP_K),
                   "precision@5": precision_at_k(retrieved, q["gold_titles"], config.TOP_K),
                   "mrr": mrr(retrieved, q["gold_titles"]),
                   "retrieval_time": t_ret, "generation_time": t_gen,
                   "total_time": t_ret + t_gen, "llm_calls": STATS["calls"] - c0}
            if not args.no_judge and qid % args.judge_every == 0:
                row["context_precision"] = context_precision(q["question"], retrieved)
                row["faithfulness"] = faithfulness(q["question"], ans, retrieved)
                row["answer_relevancy"] = answer_relevancy(q["question"], ans)
            ds_rows.append(row)
        pd.DataFrame(ds_rows).to_csv(path, index=False)
    all_rows += ds_rows

df = pd.DataFrame(all_rows)
df.to_csv(config.RESULTS_CSV, index=False)
cols = [c for c in df.columns if c not in ("dataset", "method", "qid", "question", "gold", "pred")]
summary = df.groupby(["dataset", "method"])[cols].mean().round(4)
summary.to_csv("summary.csv")
print(summary)