"""
Ollama Model Battle Arena — Streamlit App
Run: streamlit run ollama_battle_app.py
"""

import time
import json
import re
import subprocess
import threading
import streamlit as st
from openai import OpenAI

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚔️ Ollama Battle Arena",
    page_icon="⚔️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] { font-family: 'Space Mono', monospace; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }

.stApp { background: #0a0a0f; color: #e2e8f0; }

.arena-header {
    text-align: center;
    padding: 2rem 0 1rem;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    border: 1px solid #e94560;
    margin-bottom: 2rem;
}
.arena-header h1 { font-size: 2.8rem; color: #e94560; margin: 0; letter-spacing: -1px; }
.arena-header p  { color: #94a3b8; margin: 0.5rem 0 0; font-size: 0.85rem; }

.model-card {
    background: #12121f;
    border: 1px solid #2d2d4e;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.model-card:hover { border-color: #e94560; }
.model-card h4 { color: #e94560; margin: 0 0 0.5rem; font-size: 1rem; }

.response-box {
    background: #0d0d1a;
    border: 1px solid #2d2d4e;
    border-left: 3px solid #e94560;
    border-radius: 8px;
    padding: 1rem;
    font-size: 0.82rem;
    color: #cbd5e1;
    white-space: pre-wrap;
    max-height: 220px;
    overflow-y: auto;
    margin: 0.5rem 0;
}

.score-bar-wrap { margin: 0.5rem 0; }
.score-label { font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px; }
.score-bar-bg { background: #1e1e30; border-radius: 4px; height: 14px; overflow: hidden; }
.score-bar-fill { height: 100%; border-radius: 4px; transition: width 0.6s ease; }

.verdict-box {
    background: #1a1a2e;
    border-radius: 6px;
    padding: 0.5rem 0.8rem;
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 0.5rem;
    font-style: italic;
}

.winner-badge {
    display: inline-block;
    background: #e94560;
    color: white;
    font-size: 0.7rem;
    font-weight: bold;
    padding: 2px 8px;
    border-radius: 20px;
    margin-left: 8px;
    vertical-align: middle;
}

.champion-card {
    text-align: center;
    background: linear-gradient(135deg, #1a1a2e, #0f3460);
    border: 2px solid #e94560;
    border-radius: 16px;
    padding: 2rem;
    margin-top: 2rem;
}
.champion-card h2 { color: #e94560; font-size: 2rem; margin: 0; }
.champion-card h3 { color: #ffd700; font-size: 1.4rem; margin: 0.5rem 0 0; }

.pull-status {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
    color: #7ee787;
    font-family: 'Space Mono', monospace;
    margin: 0.3rem 0;
}

.stButton > button {
    background: #e94560 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.5rem !important;
}
.stButton > button:hover { background: #c73652 !important; }

div[data-testid="stTextInput"] input {
    background: #12121f !important;
    border: 1px solid #2d2d4e !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
}

.challenge-header {
    background: #12121f;
    border: 1px solid #2d2d4e;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin: 1.5rem 0 1rem;
}
.challenge-header h3 { color: #e94560; margin: 0; font-size: 1.1rem; }
.challenge-header p  { color: #64748b; margin: 0.3rem 0 0; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Ollama helpers ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434/v1"

def get_ollama_client():
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

def list_local_models() -> list[str]:
    """Return list of model names already pulled in Ollama."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        lines = result.stdout.strip().splitlines()[1:]  # skip header
        return [line.split()[0] for line in lines if line.strip()]
    except Exception:
        return []

def pull_model(model: str, status_placeholder) -> bool:
    """Pull a model via `ollama pull`, streaming progress to the placeholder."""
    try:
        status_placeholder.markdown(f'<div class="pull-status">⬇️ Pulling <b>{model}</b> — this may take a while…</div>', unsafe_allow_html=True)
        process = subprocess.Popen(
            ["ollama", "pull", model],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        last_line = ""
        for line in process.stdout:
            line = line.strip()
            if line:
                last_line = line
                status_placeholder.markdown(
                    f'<div class="pull-status">⬇️ {model}: {last_line[-120:]}</div>',
                    unsafe_allow_html=True,
                )
        process.wait()
        if process.returncode == 0:
            status_placeholder.markdown(f'<div class="pull-status">✅ {model} ready!</div>', unsafe_allow_html=True)
            return True
        else:
            status_placeholder.markdown(f'<div class="pull-status">❌ Failed to pull {model}</div>', unsafe_allow_html=True)
            return False
    except FileNotFoundError:
        status_placeholder.markdown('<div class="pull-status">❌ `ollama` command not found. Is Ollama installed?</div>', unsafe_allow_html=True)
        return False

def ensure_models(models: list[str], pull_placeholders: dict) -> dict[str, bool]:
    """Check each model; pull if missing. Returns {model: ok}."""
    local = list_local_models()
    results = {}
    for model in models:
        if any(model in m for m in local):
            pull_placeholders[model].markdown(f'<div class="pull-status">✅ {model} already present</div>', unsafe_allow_html=True)
            results[model] = True
        else:
            results[model] = pull_model(model, pull_placeholders[model])
    return results

# ── Battle engine ──────────────────────────────────────────────────────────────
CHALLENGES = [
    {
        "name": "🧠 Logic Puzzle",
        "prompt": "A farmer has 17 sheep. All but 9 die. How many sheep are left? Explain your reasoning step by step.",
        "scoring_hint": "Correct answer is 9. Score higher for clear, concise reasoning.",
    },
    {
        "name": "🎨 Creative Writing",
        "prompt": "Write a compelling 3-sentence story that starts with 'The last robot on Earth finally learned to cry.'",
        "scoring_hint": "Score on originality, emotional depth, and narrative quality.",
    },
    {
        "name": "💻 Code Challenge",
        "prompt": "Write a Python one-liner that returns a list of all prime numbers up to 50. Explain how it works.",
        "scoring_hint": "Score on correctness, elegance, and clarity of explanation.",
    },
    {
        "name": "🌍 General Knowledge",
        "prompt": "Name three underrated countries to visit and give one compelling reason for each. Be specific — avoid clichés.",
        "scoring_hint": "Score on specificity, insight, and avoiding generic tourist tropes.",
    },
]

JUDGE_SYSTEM = """You are a strict but fair AI judge evaluating language model responses.
Score on a scale of 1–10. Respond ONLY with valid JSON, nothing else, no markdown:
{"score": <integer 1-10>, "verdict": "<one concise sentence>"}"""

def query_model(client, model: str, prompt: str, system: str = None) -> tuple[str, float]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(model=model, messages=messages)
        text = resp.choices[0].message.content.strip()
    except Exception as exc:
        text = f"[ERROR: {exc}]"
    return text, time.perf_counter() - start

def parse_judgment(raw: str) -> dict:
    match = re.search(r'\{[^{}]*"score"\s*:\s*\d+[^{}]*\}', raw, re.DOTALL)
    candidate = match.group(0) if match else re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        score_m  = re.search(r'"score"\s*:\s*(\d+)', raw)
        verdict_m = re.search(r'"verdict"\s*:\s*"([^"]+)"', raw)
        if score_m:
            return {"score": int(score_m.group(1)), "verdict": verdict_m.group(1) if verdict_m else raw[:100]}
        return {"score": 5, "verdict": f"Parse failed. Raw: {raw[:100]}"}

def score_color(score: int) -> str:
    if score >= 8: return "#4ade80"
    if score >= 5: return "#facc15"
    return "#f87171"

# ── Session state defaults ─────────────────────────────────────────────────────
for key, default in {
    "battle_done": False,
    "battle_results": [],
    "scoreboard": {},
    "models_ready": False,
    "chosen_models": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="arena-header">
  <h1>⚔️ OLLAMA BATTLE ARENA</h1>
  <p>Pit three local models against each other across four challenges</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — Model selection
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.models_ready:

    st.markdown("""
    <div style="text-align:center;margin-bottom:1.5rem;">
      <p style="color:#94a3b8;font-size:0.9rem;">
        Type any Ollama model tag below — missing models are
        <b style="color:#e94560">auto-pulled</b> automatically.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### 🥊 Fighter 1")
        m1 = st.text_input("f1", value="llama3.1:latest", label_visibility="collapsed",
                           key="inp_m1", placeholder="e.g. llama3.1:latest")
    with col2:
        st.markdown("##### 🥊 Fighter 2")
        m2 = st.text_input("f2", value="mistral:latest", label_visibility="collapsed",
                           key="inp_m2", placeholder="e.g. mistral:latest")
    with col3:
        st.markdown("##### 🥊 Fighter 3")
        m3 = st.text_input("f3", value="gemma2:latest", label_visibility="collapsed",
                           key="inp_m3", placeholder="e.g. gemma2:latest")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🎯 Challenges they'll face")
    ch_cols = st.columns(4)
    for col, c in zip(ch_cols, CHALLENGES):
        col.markdown(f"""
        <div style="background:#12121f;border:1px solid #2d2d4e;border-radius:10px;
                    padding:0.8rem;text-align:center;font-size:0.82rem;color:#94a3b8;">
          <div style="font-size:1.5rem">{c['name'].split()[0]}</div>
          <div style="color:#e2e8f0;font-weight:bold;margin:4px 0">{' '.join(c['name'].split()[1:])}</div>
          <div style="font-size:0.72rem">{c['prompt'][:60]}…</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    models_input = [x.strip() for x in [m1, m2, m3] if x.strip()]

    if len(set(models_input)) < 3:
        st.warning("⚠️ Please enter 3 distinct model names above.")
    else:
        if st.button("⚔️  PREPARE FIGHTERS & START BATTLE", use_container_width=True):
            st.session_state.chosen_models = models_input
            st.markdown("### 🔧 Checking & Pulling Models…")
            pull_cols = st.columns(3)
            pull_placeholders = {
                models_input[0]: pull_cols[0].empty(),
                models_input[1]: pull_cols[1].empty(),
                models_input[2]: pull_cols[2].empty(),
            }
            ready = ensure_models(models_input, pull_placeholders)
            if all(ready.values()):
                st.session_state.models_ready = True
                st.session_state.battle_done = False
                st.session_state.battle_results = []
                st.session_state.scoreboard = {m: 0 for m in models_input}
                st.rerun()
            else:
                failed = [m for m, ok in ready.items() if not ok]
                st.error(f"❌ Failed to pull: {', '.join(failed)}")
    st.stop()

# From here models are confirmed ready
models = st.session_state.chosen_models

st.markdown("### 🔧 Model Status")
pull_cols = st.columns(3)
pull_placeholders = {
    models[0]: pull_cols[0].empty(),
    models[1]: pull_cols[1].empty(),
    models[2]: pull_cols[2].empty(),
}
for m, ph in pull_placeholders.items():
    ph.markdown(f'<div class="pull-status">✅ {m} ready</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — Battle
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.battle_done:
    client = get_ollama_client()
    scoreboard = {m: 0 for m in models}
    results = []

    st.markdown("---")
    st.markdown("## 🥊 Battle in Progress…")

    for challenge in CHALLENGES:
        st.markdown(f"""
        <div class="challenge-header">
          <h3>{challenge['name']}</h3>
          <p>{challenge['prompt']}</p>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(3)
        round_scores = {}

        for col, model in zip(cols, models):
            with col:
                st.markdown(f"**🤖 {model}**")
                with st.spinner("Querying…"):
                    response, elapsed = query_model(client, model, challenge["prompt"])

                judge_prompt = (
                    f"Challenge: {challenge['prompt']}\n"
                    f"Scoring guidance: {challenge['scoring_hint']}\n"
                    f"Response to judge:\n{response}\n\n"
                    f'Return ONLY: {{"score": <1-10>, "verdict": "<one sentence>"}}'
                )
                raw_judgment, _ = query_model(client, models[0], judge_prompt, system=JUDGE_SYSTEM)
                judgment = parse_judgment(raw_judgment)
                score = max(1, min(10, int(judgment.get("score", 5))))
                verdict = judgment.get("verdict", "")

                round_scores[model] = score
                scoreboard[model] += score

                color = score_color(score)
                pct = score * 10

                st.markdown(f'<div class="response-box">{response}</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="score-bar-wrap">
                  <div class="score-label">Score: <b style="color:{color}">{score}/10</b> &nbsp;⏱ {elapsed:.1f}s</div>
                  <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width:{pct}%; background:{color};"></div>
                  </div>
                </div>
                <div class="verdict-box">💬 {verdict}</div>
                """, unsafe_allow_html=True)

        winner = max(round_scores, key=round_scores.get)
        st.success(f"🏆 Round winner: **{winner}** ({round_scores[winner]}/10)")
        results.append({"challenge": challenge["name"], "scores": round_scores, "winner": winner})

    st.session_state.battle_done = True
    st.session_state.battle_results = results
    st.session_state.scoreboard = scoreboard
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 3 — Final scoreboard
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.battle_done:
    st.markdown("---")
    st.markdown("## 🏆 Final Scoreboard")

    ranked = sorted(st.session_state.scoreboard.items(), key=lambda x: x[1], reverse=True)
    max_score = len(CHALLENGES) * 10
    medals = ["🥇", "🥈", "🥉"]

    cols = st.columns(3)
    for i, (col, (model, total)) in enumerate(zip(cols, ranked)):
        pct = (total / max_score) * 100
        color = score_color(int(pct / 10))
        col.markdown(f"""
        <div class="model-card">
          <h4>{medals[i]} {model}</h4>
          <div class="score-label">Total: <b style="color:{color}">{total}/{max_score}</b> ({pct:.0f}%)</div>
          <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{pct}%; background:{color};"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    champion = ranked[0][0]
    st.markdown(f"""
    <div class="champion-card">
      <h2>🎉 CHAMPION</h2>
      <h3>{champion}</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Round-by-Round Results")
    for r in st.session_state.battle_results:
        with st.expander(f"{r['challenge']} — Winner: {r['winner']}"):
            for model, score in sorted(r["scores"].items(), key=lambda x: x[1], reverse=True):
                color = score_color(score)
                is_winner = "🏆" if model == r["winner"] else "  "
                st.markdown(f"""
                <div style="margin:4px 0">
                  {is_winner} <b>{model}</b>
                  <span style="color:{color}; float:right">{score}/10</span>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Run New Battle with Different Models", use_container_width=True):
        for key in ["models_ready", "battle_done", "battle_results", "scoreboard", "chosen_models"]:
            st.session_state[key] = [] if key in ("battle_results", "chosen_models") else (False if key != "scoreboard" else {})
        st.rerun()
