"""LoanLens - single-file Flask app (backend + UI) for a scikit-learn decision tree."""
import math
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template_string, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "decisiontree.pkl"))

# Which predicted class means "approved". Flip with APPROVED_CLASS=0 if your
# training labels are the other way round.
APPROVED_CLASS = int(os.getenv("APPROVED_CLASS", "1"))

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Only load pickle files you created yourself - unpickling can run code.
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
FEATURES = list(model.feature_names_in_)

# Label encodings (alphabetical, as sklearn's LabelEncoder produces them).
GENDER = {"female": 0, "male": 1}
EDUCATION = {"Associate": 0, "Bachelor": 1, "Doctorate": 2, "High School": 3, "Master": 4}
HOME = {"MORTGAGE": 0, "OTHER": 1, "OWN": 2, "RENT": 3}
INTENT = {"DEBTCONSOLIDATION": 0, "EDUCATION": 1, "HOMEIMPROVEMENT": 2,
          "MEDICAL": 3, "PERSONAL": 4, "VENTURE": 5}
PREV_DEFAULT = {"No": 0, "Yes": 1}

# Options shown in the UI: (value, label), in a natural display order.
OPTIONS = {
    "gender": [("female", "Female"), ("male", "Male")],
    "education": [("High School", "High school"), ("Associate", "Associate"),
                  ("Bachelor", "Bachelor's"), ("Master", "Master's"), ("Doctorate", "Doctorate")],
    "home": [("RENT", "Rent"), ("MORTGAGE", "Mortgage"), ("OWN", "Own"), ("OTHER", "Other")],
    "intent": [("PERSONAL", "Personal"), ("DEBTCONSOLIDATION", "Debt consolidation"),
               ("EDUCATION", "Education"), ("HOMEIMPROVEMENT", "Home improvement"),
               ("MEDICAL", "Medical"), ("VENTURE", "Business venture")],
    "default": [("No", "No"), ("Yes", "Yes")],
}

# field -> (label, min, max, is_integer)
NUMERIC = {
    "person_age": ("Age", 18, 100, True),
    "person_income": ("Annual income", 1, 10_000_000, False),
    "person_emp_exp": ("Years of employment", 0, 80, True),
    "loan_amnt": ("Loan amount", 1, 10_000_000, False),
    "loan_int_rate": ("Interest rate", 0, 40, False),
    "cb_person_cred_hist_length": ("Credit history", 0, 80, True),
    "credit_score": ("Credit score", 300, 850, True),
}
CATEGORICAL = {
    "person_gender": GENDER,
    "person_education": EDUCATION,
    "person_home_ownership": HOME,
    "loan_intent": INTENT,
    "previous_loan_defaults_on_file": PREV_DEFAULT,
}
LABELS = {
    "person_age": "Age", "person_gender": "Gender", "person_education": "Education",
    "person_income": "Annual income", "person_emp_exp": "Employment experience",
    "person_home_ownership": "Home ownership", "loan_amnt": "Loan amount",
    "loan_intent": "Loan purpose", "loan_int_rate": "Interest rate",
    "loan_percent_income": "Loan as % of income", "cb_person_cred_hist_length": "Credit history",
    "credit_score": "Credit score", "previous_loan_defaults_on_file": "Previous default",
}

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Front end (HTML + CSS + JS in one string, rendered with Jinja)
# ---------------------------------------------------------------------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LoanLens – Loan decision check</title>
  <meta name="description" content="Check a loan application with a decision tree model.">
  <meta name="theme-color" content="#0F2A33">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Sora:wght@500;600;700&display=swap" rel="stylesheet">
  <style>
:root{
  --ink:#0F2A33; --ink-2:#17404C; --teal:#0E7C7B; --teal-d:#0A5F5E; --mist:#EEF3F3;
  --card:#fff; --line:#D6E1E2; --text:#16303A; --muted:#5C7680; --amber:#F2A93B;
  --ok:#5BD9A4; --no:#FF7F88; --err:#C0373F;
  --ease:cubic-bezier(.2,.8,.2,1);
}
*{box-sizing:border-box;margin:0}
html{scroll-behavior:smooth}
body{font:400 16px/1.5 "DM Sans",system-ui,sans-serif;color:var(--text);background:var(--mist);-webkit-font-smoothing:antialiased}
h1,h2,h3,legend,.brand{font-family:Sora,system-ui,sans-serif}
.shell{display:grid;grid-template-columns:minmax(360px,5fr) 7fr;min-height:100vh}

/* Panel */
.panel{background:var(--ink);color:#EAF4F4;background-image:radial-gradient(120% 70% at 0% 0%,var(--ink-2) 0%,transparent 60%)}
.panel-inner{position:sticky;top:0;min-height:100vh;padding:44px clamp(28px,4vw,56px);display:flex;flex-direction:column;gap:36px}
.brand{display:flex;align-items:center;gap:10px;font-weight:600;font-size:20px;letter-spacing:-.01em}
.intro h1{font-size:clamp(30px,3.2vw,42px);line-height:1.1;font-weight:700;letter-spacing:-.025em;margin-bottom:14px}
.intro p{color:#A9C3C6;max-width:34ch}
.fine{margin-top:auto;font-size:13px;color:#7FA0A5}

/* Result */
.result{border:1px solid rgba(255,255,255,.14);border-radius:18px;padding:28px;background:rgba(255,255,255,.04);display:grid;justify-items:start;gap:6px}
.seal{width:104px;height:104px;margin-bottom:10px;overflow:visible}
.seal path,.seal circle{fill:none;stroke-linecap:round;stroke-linejoin:round;stroke-width:6}
.ring{stroke:#7FA0A5;stroke-dasharray:6 12;transform-origin:60px 60px;animation:spin 22s linear infinite}
.tick,.cross{stroke:transparent;stroke-dasharray:100;stroke-dashoffset:100}
.result h2{font-size:24px;font-weight:600;letter-spacing:-.015em}
.result p{color:#A9C3C6}
.result[data-state=loading] .ring{animation-duration:1.2s}
.result[data-state=approved] .ring,.result[data-state=declined] .ring{stroke-dasharray:327;stroke-dashoffset:327;animation:draw .8s var(--ease) forwards}
.result[data-state=approved] .ring,.result[data-state=approved] h2{stroke:var(--ok);color:var(--ok)}
.result[data-state=declined] .ring{stroke:var(--no)} .result[data-state=declined] h2{color:var(--no)}
.result[data-state=error] h2{color:var(--no)}
.result[data-state=approved] .tick{stroke:var(--ok);animation:draw-mark .5s .6s var(--ease) forwards}
.result[data-state=declined] .cross{stroke:var(--no);animation:draw-mark .5s .6s var(--ease) forwards}
.result[data-state=approved] .seal,.result[data-state=declined] .seal{animation:pop .6s var(--ease)}
.result[data-state=approved] #r-title,.result[data-state=declined] #r-title,
.result[data-state=approved] #r-text,.result[data-state=declined] #r-text{animation:rise .5s .35s both var(--ease)}
.factors{width:100%;margin-top:18px;padding-top:18px;border-top:1px solid rgba(255,255,255,.12)}
.factors h3{font-size:14px;font-weight:600;color:#CFE2E3;margin-bottom:12px}
.factors ul{list-style:none;padding:0;display:grid;gap:12px}
.factors li{display:grid;grid-template-columns:1fr auto;gap:2px 12px;font-size:14px;animation:rise .5s both var(--ease);animation-delay:calc(.7s + var(--k)*.08s)}
.factors li b{font-weight:500;color:#fff}.factors li span{color:#A9C3C6}
.factors .bar{grid-column:1/-1;height:4px;border-radius:4px;background:rgba(255,255,255,.1);overflow:hidden}
.factors .bar i{display:block;height:100%;width:0;background:var(--amber);border-radius:4px;transition:width .9s var(--ease) calc(.9s + var(--k)*.08s)}

/* Form */
.main{padding:44px clamp(24px,5vw,72px)}
.form{max-width:760px;margin:0 auto;display:grid;gap:34px}
.form-head{display:flex;justify-content:space-between;align-items:baseline;gap:16px}
.form-head h2{font-size:26px;font-weight:600;letter-spacing:-.02em}
.link{background:none;border:0;color:var(--teal);font:500 15px "DM Sans";cursor:pointer;text-decoration:underline;text-underline-offset:3px}
.group{border:0;padding:0 0 0 22px;border-left:3px solid var(--line);transition:border-color .3s}
.group:focus-within{border-color:var(--teal)}
legend{float:left;width:100%;margin:0 0 16px -22px;padding-left:22px;font-size:17px;font-weight:600;letter-spacing:-.01em}
.grid{clear:both;display:grid;grid-template-columns:1fr 1fr;gap:20px 22px}
.field{display:grid;gap:7px;align-content:start}.wide{grid-column:1/-1}
label,.label{font-size:14px;font-weight:500;color:var(--text)}
input[type=number],select{width:100%;height:46px;padding:0 14px;font:inherit;color:var(--text);background:var(--card);border:1px solid var(--line);border-radius:10px;transition:border-color .2s,box-shadow .2s}
select{appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M1 1l5 5 5-5' fill='none' stroke='%235C7680' stroke-width='2' stroke-linecap='round'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 14px center;padding-right:36px}
input::placeholder{color:#9DB1B6}
input:focus-visible,select:focus-visible{outline:none;border-color:var(--teal);box-shadow:0 0 0 4px rgba(14,124,123,.18)}
input[aria-invalid=true]{border-color:var(--err);animation:shake .35s}
.affix{position:relative}.affix i{position:absolute;left:14px;top:50%;transform:translateY(-50%);font-style:normal;color:var(--muted)}.affix input{padding-left:30px}
.err{font-size:13px;color:var(--err);min-height:0}
.err:empty{display:none}
.seg{display:grid;grid-auto-flow:column;grid-auto-columns:1fr;height:46px;padding:4px;background:#E3EBEC;border-radius:12px;gap:4px}
.seg label{position:relative;display:grid}
.seg input{position:absolute;opacity:0;inset:0;cursor:pointer}
.seg span{display:grid;place-items:center;border-radius:9px;font-size:15px;color:var(--muted);transition:background .25s var(--ease),color .25s,box-shadow .25s}
.seg input:checked+span{background:var(--card);color:var(--ink);font-weight:600;box-shadow:0 1px 3px rgba(15,42,51,.18)}
.seg input:focus-visible+span{outline:3px solid rgba(14,124,123,.35)}
.row{display:flex;justify-content:space-between;align-items:baseline}
output{font:600 15px Sora,sans-serif;color:var(--teal-d);font-variant-numeric:tabular-nums}
output em{font:500 13px "DM Sans";font-style:normal;color:var(--muted);margin-left:6px}
input[type=range]{--fill:50%;-webkit-appearance:none;appearance:none;width:100%;height:6px;border-radius:6px;cursor:pointer;background:linear-gradient(90deg,var(--teal) var(--fill),#D3DFE0 var(--fill))}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:22px;height:22px;border-radius:50%;background:#fff;border:3px solid var(--teal);box-shadow:0 2px 6px rgba(15,42,51,.25);transition:transform .2s var(--ease)}
input[type=range]::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:#fff;border:3px solid var(--teal)}
input[type=range]:active::-webkit-slider-thumb{transform:scale(1.2)}
input[type=range]:focus-visible{outline:3px solid rgba(14,124,123,.3);outline-offset:6px}
.meter{height:8px;border-radius:8px;background:#D3DFE0;overflow:hidden}
.meter div{height:100%;width:0;border-radius:8px;background:var(--teal);transition:width .5s var(--ease),background .3s}
.actions{display:flex;gap:14px;align-items:center}
.btn{position:relative;display:inline-flex;align-items:center;justify-content:center;gap:10px;min-width:200px;height:52px;padding:0 28px;border:0;border-radius:12px;background:var(--teal);color:#fff;font:600 16px "DM Sans";cursor:pointer;transition:background .2s,transform .15s var(--ease),box-shadow .2s;box-shadow:0 6px 18px rgba(14,124,123,.3)}
.btn:hover{background:var(--teal-d)}.btn:active{transform:scale(.98)}
.btn:focus-visible,.ghost:focus-visible,.link:focus-visible{outline:3px solid rgba(242,169,59,.7);outline-offset:3px}
.btn[disabled]{cursor:progress;opacity:.9}
.spinner{display:none;width:18px;height:18px;border:3px solid rgba(255,255,255,.4);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite}
.btn[disabled] .spinner{display:block}
.ghost{height:52px;padding:0 20px;background:none;border:0;color:var(--muted);font:500 16px "DM Sans";cursor:pointer}.ghost:hover{color:var(--text)}

/* Motion */
.reveal{animation:rise .7s both var(--ease);animation-delay:calc(var(--i)*.08s)}
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes draw{to{stroke-dashoffset:0}}
@keyframes draw-mark{to{stroke-dashoffset:0}}
@keyframes pop{0%{transform:scale(.85)}60%{transform:scale(1.06)}100%{transform:scale(1)}}
@keyframes shake{25%{transform:translateX(-4px)}75%{transform:translateX(4px)}}

@media (max-width:960px){
  .shell{grid-template-columns:1fr}
  .panel-inner{position:static;min-height:0;gap:28px;padding:32px 24px}
  .fine{margin-top:0}
}
@media (max-width:560px){.grid{grid-template-columns:1fr}.actions{flex-direction:column-reverse;align-items:stretch}.btn{width:100%}}
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation-duration:.01ms!important;animation-delay:0s!important;animation-iteration-count:1!important;transition-duration:.01ms!important;scroll-behavior:auto!important}
}
</style>
</head>
<body>
<div class="shell">

  <!-- Decision panel -->
  <aside class="panel">
    <div class="panel-inner">
      <div class="brand reveal" style="--i:0">
        <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="16" r="14" fill="none" stroke="#F2A93B" stroke-width="3"/><circle cx="16" cy="16" r="5" fill="#F2A93B"/></svg>
        <span>LoanLens</span>
      </div>

      <div class="intro reveal" style="--i:1">
        <h1>Loan decision check</h1>
        <p>Fill in the applicant's details. A decision tree trained on past applications gives a decision in a moment.</p>
      </div>

      <section id="result" class="result reveal" style="--i:2" data-state="idle" aria-live="polite">
        <svg class="seal" viewBox="0 0 120 120" aria-hidden="true">
          <circle class="ring" cx="60" cy="60" r="52"/>
          <path class="tick" d="M38 62 L54 78 L84 44"/>
          <path class="cross" d="M43 43 L77 77 M77 43 L43 77"/>
        </svg>
        <h2 id="r-title">Waiting for an application</h2>
        <p id="r-text">Your result will appear here.</p>
        <div class="factors" id="factors" hidden>
          <h3>What this model weighs most</h3>
          <ul id="factor-list"></ul>
        </div>
      </section>

      <p class="fine reveal" style="--i:3">For demonstration only. Not financial advice or a real credit decision.</p>
    </div>
  </aside>

  <!-- Form -->
  <main class="main">
    <form id="form" class="form" novalidate>
      <header class="form-head reveal" style="--i:2">
        <h2>Application details</h2>
        <button type="button" class="link" id="example">Fill with an example</button>
      </header>

      <fieldset class="group reveal" style="--i:3">
        <legend>Applicant</legend>
        <div class="grid">
          <div class="field">
            <label for="person_age">Age</label>
            <input id="person_age" name="person_age" type="number" min="18" max="100" step="1" inputmode="numeric" placeholder="e.g. 29" required>
            <span class="err" data-err="person_age"></span>
          </div>
          <div class="field">
            <span class="label" id="l-gender">Gender</span>
            <div class="seg" role="radiogroup" aria-labelledby="l-gender">
              {% for v, t in options.gender %}
              <label><input type="radio" name="person_gender" value="{{ v }}" {{ 'checked' if loop.first }}><span>{{ t }}</span></label>
              {% endfor %}
            </div>
          </div>
          <div class="field">
            <label for="person_education">Education</label>
            <select id="person_education" name="person_education">
              {% for v, t in options.education %}<option value="{{ v }}" {{ 'selected' if v == 'Bachelor' }}>{{ t }}</option>{% endfor %}
            </select>
          </div>
          <div class="field">
            <label for="person_emp_exp">Years of employment</label>
            <input id="person_emp_exp" name="person_emp_exp" type="number" min="0" max="80" step="1" inputmode="numeric" placeholder="e.g. 4" required>
            <span class="err" data-err="person_emp_exp"></span>
          </div>
        </div>
      </fieldset>

      <fieldset class="group reveal" style="--i:4">
        <legend>Income and home</legend>
        <div class="grid">
          <div class="field">
            <label for="person_income">Annual income</label>
            <div class="affix"><i>$</i><input id="person_income" name="person_income" type="number" min="1" step="any" inputmode="decimal" placeholder="e.g. 60000" required></div>
            <span class="err" data-err="person_income"></span>
          </div>
          <div class="field">
            <label for="person_home_ownership">Home ownership</label>
            <select id="person_home_ownership" name="person_home_ownership">
              {% for v, t in options.home %}<option value="{{ v }}">{{ t }}</option>{% endfor %}
            </select>
          </div>
        </div>
      </fieldset>

      <fieldset class="group reveal" style="--i:5">
        <legend>Loan</legend>
        <div class="grid">
          <div class="field">
            <label for="loan_amnt">Loan amount</label>
            <div class="affix"><i>$</i><input id="loan_amnt" name="loan_amnt" type="number" min="1" step="any" inputmode="decimal" placeholder="e.g. 12000" required></div>
            <span class="err" data-err="loan_amnt"></span>
          </div>
          <div class="field">
            <label for="loan_intent">Purpose</label>
            <select id="loan_intent" name="loan_intent">
              {% for v, t in options.intent %}<option value="{{ v }}">{{ t }}</option>{% endfor %}
            </select>
          </div>
          <div class="field wide">
            <div class="row"><label for="loan_int_rate">Interest rate</label><output data-for="loan_int_rate"></output></div>
            <input id="loan_int_rate" name="loan_int_rate" type="range" min="5" max="20" step="0.05" value="11">
          </div>
          <div class="field wide lpi">
            <div class="row"><span class="label">Loan as share of income</span><output id="lpi-text">–</output></div>
            <div class="meter"><div id="lpi-bar"></div></div>
          </div>
        </div>
      </fieldset>

      <fieldset class="group reveal" style="--i:6">
        <legend>Credit history</legend>
        <div class="grid">
          <div class="field">
            <label for="cb_person_cred_hist_length">Credit history length (years)</label>
            <input id="cb_person_cred_hist_length" name="cb_person_cred_hist_length" type="number" min="0" max="80" step="1" inputmode="numeric" placeholder="e.g. 5" required>
            <span class="err" data-err="cb_person_cred_hist_length"></span>
          </div>
          <div class="field">
            <span class="label" id="l-def">Previous loan default on file</span>
            <div class="seg" role="radiogroup" aria-labelledby="l-def">
              {% for v, t in options.default %}
              <label><input type="radio" name="previous_loan_defaults_on_file" value="{{ v }}" {{ 'checked' if loop.first }}><span>{{ t }}</span></label>
              {% endfor %}
            </div>
          </div>
          <div class="field wide">
            <div class="row"><label for="credit_score">Credit score</label><output><span data-for="credit_score"></span> <em id="band"></em></output></div>
            <input id="credit_score" name="credit_score" type="range" min="300" max="850" step="1" value="650">
          </div>
        </div>
      </fieldset>

      <div class="actions reveal" style="--i:7">
        <button type="submit" class="btn" id="submit"><span class="btn-text">Check decision</span><span class="spinner" aria-hidden="true"></span></button>
        <button type="reset" class="ghost" id="reset">Clear</button>
      </div>
    </form>
  </main>
</div>
<script>
(() => {
  const $ = (s) => document.querySelector(s);
  const form = $("#form"), result = $("#result"), btn = $("#submit");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---- Sliders: filled track + live value ----
  const band = (s) => s < 580 ? "Poor" : s < 670 ? "Fair" : s < 740 ? "Good" : s < 800 ? "Very good" : "Exceptional";
  function syncSlider(r) {
    r.style.setProperty("--fill", ((r.value - r.min) / (r.max - r.min)) * 100 + "%");
    const out = document.querySelector(`[data-for="${r.id}"]`);
    out.textContent = r.id === "loan_int_rate" ? (+r.value).toFixed(2) + "%" : r.value;
    if (r.id === "credit_score") $("#band").textContent = band(+r.value);
  }
  document.querySelectorAll("input[type=range]").forEach((r) => {
    r.addEventListener("input", () => syncSlider(r));
    syncSlider(r);
  });

  // ---- Loan share of income (calculated for the user) ----
  function syncShare() {
    const inc = parseFloat($("#person_income").value), amt = parseFloat($("#loan_amnt").value);
    const bar = $("#lpi-bar"), txt = $("#lpi-text");
    if (!(inc > 0) || !(amt > 0)) { txt.textContent = "–"; bar.style.width = "0"; return; }
    const pct = amt / inc;
    txt.textContent = (pct * 100).toFixed(0) + "%";
    bar.style.width = Math.min(pct / 0.6, 1) * 100 + "%";
    bar.style.background = pct > 0.35 ? "#C0373F" : pct > 0.2 ? "#F2A93B" : "#0E7C7B";
  }
  ["person_income", "loan_amnt"].forEach((id) => $("#" + id).addEventListener("input", syncShare));

  // ---- Result panel ----
  function setResult(state, title, text) {
    result.dataset.state = state;
    $("#r-title").textContent = title;
    $("#r-text").textContent = text;
    $("#factors").hidden = true;
  }

  function showFactors(list) {
    const ul = $("#factor-list");
    ul.replaceChildren();
    const max = Math.max(...list.map((f) => f.weight));
    list.forEach((f, k) => {
      const li = document.createElement("li");
      li.style.setProperty("--k", k);
      const b = document.createElement("b"); b.textContent = f.label;
      const v = document.createElement("span"); v.textContent = f.value;
      const bar = document.createElement("div"); bar.className = "bar";
      const fill = document.createElement("i"); fill.dataset.w = (f.weight / max) * 100 + "%";
      fill.style.setProperty("--k", k);
      bar.append(fill); li.append(b, v, bar); ul.append(li);
    });
    $("#factors").hidden = false;
    requestAnimationFrame(() => requestAnimationFrame(() =>
      ul.querySelectorAll("i").forEach((i) => (i.style.width = i.dataset.w))));
  }

  function clearErrors() {
    document.querySelectorAll(".err").forEach((e) => (e.textContent = ""));
    form.querySelectorAll("[aria-invalid]").forEach((i) => i.removeAttribute("aria-invalid"));
  }
  function showErrors(errors) {
    let first = null;
    Object.entries(errors).forEach(([k, msg]) => {
      const el = document.querySelector(`[data-err="${k}"]`), input = form.elements[k];
      if (el) el.textContent = msg;
      if (input && input.setAttribute) { input.setAttribute("aria-invalid", "true"); first ||= input; }
    });
    if (first) first.focus();
  }

  // ---- Submit ----
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearErrors();
    const data = Object.fromEntries(new FormData(form));
    ["person_age","person_income","person_emp_exp","loan_amnt","loan_int_rate","cb_person_cred_hist_length","credit_score"]
      .forEach((k) => (data[k] = data[k] === "" ? null : Number(data[k])));

    btn.disabled = true;
    setResult("loading", "Checking application…", "Running the decision tree.");
    const started = performance.now();
    try {
      const res = await fetch("/predict", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
      });
      const body = await res.json();
      // keep the loading state visible long enough to read
      await new Promise((r) => setTimeout(r, Math.max(0, 600 - (performance.now() - started))));
      if (res.status === 422) {
        setResult("error", "Check the highlighted fields", "Some details are missing or out of range.");
        showErrors(body.errors);
      } else if (!res.ok) {
        throw new Error();
      } else {
        const ok = body.approved;
        setResult(ok ? "approved" : "declined", ok ? "Approved" : "Not approved",
          ok ? "The model predicts this application would be approved."
             : "The model predicts this application would not be approved.");
        showFactors(body.factors);
      }
    } catch {
      setResult("error", "Couldn't get a decision", "The server didn't respond. Try again in a moment.");
    } finally {
      btn.disabled = false;
      if (innerWidth <= 960) result.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "center" });
    }
  });

  // ---- Example + reset ----
  const example = { person_age: 29, person_emp_exp: 5, person_income: 62000, loan_amnt: 12000,
    cb_person_cred_hist_length: 6, loan_int_rate: 12.4, credit_score: 640 };
  $("#example").addEventListener("click", () => {
    Object.entries(example).forEach(([k, v]) => (form.elements[k].value = v));
    form.elements.person_education.value = "Bachelor";
    form.elements.person_home_ownership.value = "RENT";
    form.elements.loan_intent.value = "PERSONAL";
    document.querySelectorAll("input[type=range]").forEach(syncSlider);
    syncShare(); clearErrors();
  });
  form.addEventListener("reset", () => setTimeout(() => {
    document.querySelectorAll("input[type=range]").forEach(syncSlider);
    syncShare(); clearErrors();
    setResult("idle", "Waiting for an application", "Your result will appear here.");
  }));
})();
</script>
</body>
</html>
"""


def validate(payload):
    """Return (clean_values, errors)."""
    clean, errors = {}, {}
    for key, (label, lo, hi, is_int) in NUMERIC.items():
        try:
            val = float(payload.get(key))
            if not math.isfinite(val):
                raise ValueError
        except (TypeError, ValueError):
            errors[key] = f"Enter a number for {label.lower()}."
            continue
        if not lo <= val <= hi:
            errors[key] = f"{label} must be between {lo:,} and {hi:,}."
        elif is_int and val != int(val):
            errors[key] = f"{label} must be a whole number."
        else:
            clean[key] = int(val) if is_int else val
    for key, mapping in CATEGORICAL.items():
        if payload.get(key) not in mapping:
            errors[key] = f"Choose a valid option for {LABELS[key].lower()}."
        else:
            clean[key] = mapping[payload[key]]
    if "person_emp_exp" in clean and "person_age" in clean and clean["person_emp_exp"] > clean["person_age"]:
        errors["person_emp_exp"] = "Employment can't be longer than the applicant's age."
    return clean, errors


def top_factors(values, raw, n=5):
    """The model's most influential inputs, with this applicant's values."""
    imp = model.feature_importances_
    shown = {
        "person_income": f"${raw['person_income']:,.0f}",
        "loan_amnt": f"${raw['loan_amnt']:,.0f}",
        "loan_int_rate": f"{raw['loan_int_rate']:.2f}%",
        "loan_percent_income": f"{values['loan_percent_income'] * 100:.0f}%",
        "credit_score": str(raw["credit_score"]),
        "previous_loan_defaults_on_file": raw["previous_loan_defaults_on_file"],
        "person_age": str(raw["person_age"]),
        "person_emp_exp": f"{raw['person_emp_exp']} yrs",
        "cb_person_cred_hist_length": f"{raw['cb_person_cred_hist_length']} yrs",
    }
    out = []
    for i in np.argsort(imp)[::-1]:
        name = FEATURES[i]
        value = shown.get(name) or str(raw.get(name, ""))
        out.append({"label": LABELS[name], "value": value, "weight": round(float(imp[i]) * 100, 1)})
        if len(out) == n:
            break
    return out


@app.route("/")
def index():
    return render_template_string(INDEX_HTML, options=OPTIONS)


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    clean, errors = validate(payload)
    if errors:
        return jsonify(errors=errors), 422

    income, amount = clean["person_income"], clean["loan_amnt"]
    row = dict(clean)
    row["loan_percent_income"] = round(amount / income, 2)
    x = np.array([[row[name] for name in FEATURES]], dtype=float)
    pred = int(model.predict(x)[0])
    approved = pred == APPROVED_CLASS

    return jsonify(
        approved=approved,
        loan_percent_income=row["loan_percent_income"],
        factors=top_factors(row, payload),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
