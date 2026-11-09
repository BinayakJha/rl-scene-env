"""Build docs/code_guide.html from docs/guide_template.html, then print it to PDF.

Why a builder: every number in the guide is read from results/*.json and every code snippet
is copied from the real source files, so the guide can never drift from the code or invent a
number. Run:  python docs/build_guide.py
"""
import html
import importlib
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

KEYWORDS = r"\b(def|class|return|if|elif|else|for|while|in|not|and|or|import|from|with|as|None|True|False|break|continue|yield|lambda)\b"


def highlight(code):
    out = []
    for line in code.rstrip().split("\n"):
        cut = line.find("#")
        body, comment = (line[:cut], line[cut:]) if cut >= 0 else (line, "")
        pieces = re.split(r"(\"[^\"]*\"|'[^']*')", body)  # odd pieces are string literals
        body = ""
        for n, piece in enumerate(pieces):
            if n % 2:
                body += f'<span class="s">{html.escape(piece)}</span>'
            else:
                body += re.sub(KEYWORDS, r'<span class="k">\1</span>', html.escape(piece))
        if comment:
            body += f'<span class="c">{html.escape(comment)}</span>'
        out.append(body)
    return "\n".join(out)


def code_of(spec):
    """'scene_repair.env:SceneRepairEnv.step' -> the real source of that function."""
    module_name, qualname = spec.split(":")
    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    source = inspect.getsource(obj)
    lines = source.split("\n")
    indent = len(lines[0]) - len(lines[0].lstrip())
    source = "\n".join(l[indent:] for l in lines)
    # drop the docstring: the guide explains it in words already
    source = re.sub(r'(\):\n|:\n)(\s+)""".*?"""\n', r"\1", source, count=1, flags=re.S)
    return f'<pre class="code"><div class="where">{module_name.replace(".", "/")}.py</div>{highlight(source)}</pre>'


def pct(x):
    return f"{100 * x:.1f}%"


def ci(d, fmt="{:.3f}"):
    return f"{fmt.format(d['mean'])} <span class='ci'>[{fmt.format(d['ci_low'])}, {fmt.format(d['ci_high'])}]</span>"


def ci_pct(d):
    return f"{100*d['mean']:.1f}% <span class='ci'>[{100*d['ci_low']:.1f}, {100*d['ci_high']:.1f}]</span>"


LABELS = {"random": "Random", "bc": "Behavior cloning", "ppo": "PPO", "planner": "Adaptive replanner"}


def results_table(rows):
    head = ("<table class='res'><tr><th>Policy</th><th>Full success</th><th>Mean reward R</th>"
            "<th>Goals met (s)</th><th>Actions used</th><th>UNDOs / ep.</th></tr>")
    body = ""
    for name in ("random", "bc", "ppo", "planner"):
        if name not in rows:
            continue
        p = rows[name]["pooled"]
        body += (f"<tr><td><b>{LABELS[name]}</b></td><td>{ci_pct(p['success'])}</td><td>{ci(p['reward'])}</td>"
                 f"<td>{ci(p['s'])}</td><td>{ci(p['actions_used'], '{:.2f}')}</td><td>{p['undos']['mean']:.2f}</td></tr>")
    return head + body + "</table>"


def seed_table(rows):
    out = "<table class='res small'><tr><th>Policy</th><th>Seed</th><th>Test success</th><th>Test mean R</th><th>Train-check success</th><th>Val success</th></tr>"
    for name in ("bc", "ppo"):
        r = rows[name]
        for n, s in enumerate(r["per_seed"]):
            out += (f"<tr><td>{LABELS[name]}</td><td>{s['seed']}</td><td>{pct(s['success']['mean'])}</td>"
                    f"<td>{s['reward']['mean']:.3f}</td><td>{pct(r['train_vs_val_success']['train_check'][n])}</td>"
                    f"<td>{pct(r['train_vs_val_success']['val'][n])}</td></tr>")
    return out + "</table>"


def ablation_table(variants):
    names = {"main": "Main (noise on, UNDO on)", "deterministic": "No noise", "no_undo": "No UNDO"}
    out = "<table class='res small'><tr><th>Setting</th>" + "".join(
        f"<th>{LABELS[p]}</th>" for p in ("random", "bc", "ppo", "planner")) + "</tr>"
    for v, rows in variants.items():
        out += f"<tr><td><b>{names[v]}</b></td>" + "".join(
            f"<td>{ci_pct(rows[p]['pooled']['success'])}</td>" for p in ("random", "bc", "ppo", "planner")) + "</tr>"
    return out + "</table>"


def file_table():
    rows = ""
    for path in sorted(list((ROOT / "scene_repair").rglob("*.py")) + list((ROOT / "scripts").glob("*.py"))
                       + list((ROOT / "tests").glob("*.py")) + [ROOT / "run_all.py"]):
        if path.name in ("__init__.py", "_setup.py", "conftest.py") and path.parent.name != "scene_repair":
            continue
        text = path.read_text()
        doc = re.search(r'^"""(.*?)(\n|""")', text, re.S)
        first = doc.group(1).strip() if doc else ""
        rows += (f"<tr><td><code>{path.relative_to(ROOT)}</code></td><td>{len(text.splitlines())}</td>"
                 f"<td>{html.escape(first)}</td></tr>")
    return "<table class='files'><tr><th>File</th><th>Lines</th><th>Its one job (first line of its docstring)</th></tr>" + rows + "</table>"


def main():
    res = json.load(open(ROOT / "results" / "test_results.json"))
    stats = json.load(open(ROOT / "results" / "test_scene_stats.json"))
    run_log = json.load(open(ROOT / "results" / "run_log.json"))
    splits = json.load(open(ROOT / "results" / "data" / "splits_summary.json"))
    main_rows = res["variants"]["main"]
    ppo_curves = [json.load(open(ROOT / "results" / "curves" / f"ppo_main_seed{s}.json")) for s in (0, 1, 2)]
    bc_curves = [json.load(open(ROOT / "results" / "curves" / f"bc_main_seed{s}.json")) for s in (0, 1, 2)]
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip().splitlines()[-1]
    m = lambda p, k: main_rows[p]["pooled"][k]["mean"]
    facts = {
        "RESULTS_TABLE": results_table(main_rows),
        "SEED_TABLE": seed_table(main_rows),
        "ABLATION_TABLE": ablation_table(res["variants"]),
        "FILE_TABLE": file_table(),
        "TESTS": html.escape(tests),
        "TEST_SCENES": str(res["test_scenes"]),
        "ROLLOUTS": str(res["rollouts_per_scene"]),
        "TEST_EPISODES": f"{res['test_scenes'] * res['rollouts_per_scene']:,}",
        "VAL_SCENES": str(splits["val"]["scenes"]),
        "TRAINCHECK_SCENES": str(splits["train_check"]["scenes"]),
        "TEST_TEMPLATES": str(splits["test"]["templates"]),
        "VAL_TEMPLATES": str(splits["val"]["templates"]),
        "CPU": res["hardware"]["cpu"], "CORES": str(res["hardware"]["cores"]),
        "TORCH": res["hardware"]["torch"], "PYVER": res["hardware"]["python"],
        "PPO_STEPS": f"{ppo_curves[0]['env_steps']:,}",
        "PPO_STEPS_TOTAL": f"{sum(c['env_steps'] for c in ppo_curves):,}",
        "PPO_MINUTES": f"{sum(c['train_seconds'] for c in ppo_curves) / 3 / 60:.1f}",
        "BC_DEMO_EPISODES": f"{bc_curves[0]['demo_episodes']:,}",
        "BC_DEMO_PAIRS": ", ".join(f"{c['demo_pairs']:,}" for c in bc_curves),
        "BC_MINUTES": f"{sum(c['train_seconds'] for c in bc_curves) / 3 / 60:.1f}",
        "PLANNER_MS": f"{main_rows['planner']['ms_per_decision']:.2f}",
        "PLANNER_EXP": f"{main_rows['planner']['expansions_total']:,}",
        "PLANNER_DECISIONS": f"{main_rows['planner']['decisions']:,}",
        "PLANNER_LIMIT_HITS": str(main_rows['planner']['decisions_hitting_limit']),
        "EXP_LIMIT": f"{main_rows['planner']['expansion_limit']:,}",
        "TOTAL_MIN": f"{run_log['seconds']['total'] / 60:.0f}",
        "TRAIN_MIN": f"{run_log['seconds']['training'] / 60:.0f}",
        "MEAN_FIX": f"{stats['mean_shortest_fix']:.2f}",
        "FIX_DIST": ", ".join(f"{k} move{'s' if k != '1' else ''}: {v}" for k, v in stats["shortest_noise_free_fix"].items()),
        "OBJ_DIST": ", ".join(f"{k} objects: {v}" for k, v in stats["objects"].items()),
        "GOAL_DIST": ", ".join(f"{k} goals: {v}" for k, v in stats["goals"].items()),
    }
    fails = json.load(open(ROOT / "results" / "failure_analysis.json"))
    facts.update({"PPO_FAILS": f"{fails['ppo']['failures']:,}", "PPO_STOPPED": f"{fails['ppo']['stopped_early']:,}",
                  "PPO_OOB": f"{fails['ppo']['out_of_budget']:,}"})
    for p in ("bc", "ppo", "planner"):
        facts[f"{p.upper()}_WASTE"] = f"{fails[p]['wasted_actions_per_episode']:.2f}"
    for p in ("random", "bc", "ppo", "planner"):
        facts[f"{p.upper()}_SUCC"] = pct(m(p, "success"))
        facts[f"{p.upper()}_R"] = f"{m(p, 'reward'):.3f}"
        facts[f"{p.upper()}_ACT"] = f"{m(p, 'actions_used'):.2f}"
    for v in ("deterministic", "no_undo"):
        for p in ("random", "bc", "ppo", "planner"):
            facts[f"{v.upper()}_{p.upper()}_SUCC"] = pct(res["variants"][v][p]["pooled"]["success"]["mean"])

    text = (ROOT / "docs" / "guide_template.html").read_text()
    text = re.sub(r"\{\{CODE:([^}]+)\}\}", lambda mt: code_of(mt.group(1)), text)
    missing = set(re.findall(r"\{\{([A-Z_0-9]+)\}\}", text)) - set(facts)
    assert not missing, f"unknown tokens: {missing}"
    text = re.sub(r"\{\{([A-Z_0-9]+)\}\}", lambda mt: facts[mt.group(1)], text)
    (ROOT / "docs" / "code_guide.html").write_text(text)
    subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu",
                    "--no-pdf-header-footer", f"--print-to-pdf={ROOT / 'docs' / 'code_guide.pdf'}",
                    (ROOT / "docs" / "code_guide.html").as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote docs/code_guide.html and docs/code_guide.pdf")


if __name__ == "__main__":
    main()
