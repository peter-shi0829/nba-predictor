let games = [];
let teams = {};
let idx = 0;

const pct = v => Math.round(v * 100) + "%";
const fmt = v => (v === null || v === undefined) ? "–" : v;

async function load() {
  try {
    const [pred, teamData] = await Promise.all([
      fetch("predictions.json").then(r => {
        if (!r.ok) throw new Error("predictions.json " + r.status);
        return r.json();
      }),
      fetch("teams.json").then(r => {
        if (!r.ok) throw new Error("teams.json " + r.status);
        return r.json();
      }),
    ]);
    teams = teamData;
    games = pred.games;
    document.getElementById("updated").textContent =
      "Last updated " + new Date(pred.generated_at).toLocaleString();
    if (pred.mode === "retro") {
      const note = document.getElementById("mode-note");
      note.textContent =
        "Offseason. Looking back at the last playoffs: model pick vs what happened.";
      note.classList.remove("hidden");
    }
    if (!games.length) {
      document.getElementById("card").innerHTML =
        "<p class='empty'>No games in the next week. Check back soon.</p>";
      document.getElementById("prev").disabled = true;
      document.getElementById("next").disabled = true;
      return;
    }
    const dots = document.getElementById("dots");
    games.forEach(() => dots.appendChild(document.createElement("span")));
    render();
  } catch (err) {
    document.getElementById("card").innerHTML =
      "<p class='empty'>Predictions are updating. Check back in a few minutes.</p>";
    document.getElementById("prev").disabled = true;
    document.getElementById("next").disabled = true;
  }
}

function teamName(abbr) {
  return (teams[abbr] && teams[abbr].name) || abbr;
}

function teamColor(abbr, fallback) {
  return (teams[abbr] && teams[abbr].color) || fallback;
}

function statRow(label, h, a) {
  return `<div class="stat-row"><span>${fmt(h)}</span>` +
         `<span class="stat-label">${label}</span><span>${fmt(a)}</span></div>`;
}

function pickWasRight(g) {
  const pickedHome = g.home.win_prob >= 0.5;
  return pickedHome === (g.actual.winner === g.home.abbr);
}

function render() {
  const g = games[idx];
  const card = document.getElementById("card");
  card.style.setProperty("--home-color", teamColor(g.home.abbr, "#1d428a"));
  card.style.setProperty("--away-color", teamColor(g.away.abbr, "#c8102e"));
  card.innerHTML = `
    ${g.is_playoff ? '<span class="badge">PLAYOFFS</span>' : ""}
    <div class="teams">
      <span class="team">${teamName(g.home.abbr)}</span>
      <span class="tip">${g.time_et || g.date}</span>
      <span class="team">${teamName(g.away.abbr)}</span>
    </div>
    <div class="probs">
      <b>${pct(g.home.win_prob)}</b>
      <span>win probability</span>
      <b>${pct(g.away.win_prob)}</b>
    </div>
    <div class="bar"><div class="bar-fill" style="width:${g.home.win_prob * 100}%"></div></div>
    <div class="stats">
      ${statRow("Offensive rating", g.home.stats.ortg, g.away.stats.ortg)}
      ${statRow("Defensive rating", g.home.stats.drtg, g.away.stats.drtg)}
      ${statRow("Pace", g.home.stats.pace, g.away.stats.pace)}
      ${statRow("Net rating, last 10", g.home.stats.net_last10, g.away.stats.net_last10)}
      ${statRow("Days of rest", g.home.stats.rest_days, g.away.stats.rest_days)}
    </div>
    <p class="why">${g.explanation}</p>
    ${g.actual ? `<p class="actual">Final: ${g.actual.winner} won ` +
      `${g.actual.home_pts}-${g.actual.away_pts}. ` +
      `${pickWasRight(g) ? "Model got it right." : "Model missed this one."}</p>` : ""}
  `;
  document.querySelectorAll("#dots span").forEach(
    (d, i) => d.classList.toggle("on", i === idx));
  document.getElementById("prev").disabled = idx === 0;
  document.getElementById("next").disabled = idx === games.length - 1;
}

function move(delta) {
  const next = idx + delta;
  if (next < 0 || next >= games.length) return;
  idx = next;
  render();
}

document.getElementById("prev").addEventListener("click", () => move(-1));
document.getElementById("next").addEventListener("click", () => move(1));
document.addEventListener("keydown", e => {
  if (e.key === "ArrowLeft") move(-1);
  if (e.key === "ArrowRight") move(1);
});

load();
