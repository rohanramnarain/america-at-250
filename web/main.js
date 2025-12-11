import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm";
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

const DATA_URL = "./race_sent_timeseries_combined.csv";
const SENTENCES_URL = "./race_sentences_with_sentiment_both_models.csv";
const COLOR_SCALE = {
  "Black/African American": "#f59e0b",
  "Indigenous/Native": "#ef4444",
  "Asian/Asian American": "#10b981",
  "Latino/Hispanic": "#6366f1",
  "White/European American": "#f472b6",
};
const raceSelect = document.querySelector("#race-select");
const metricSelect = document.querySelector("#metric-select");
const decadeSelect = document.querySelector("#decade-select");
const chartEl = document.querySelector("#chart");
const statsEl = document.querySelector("#stats");
const detailEl = document.querySelector("#detail-panel");
const state = {
  race: null,
  metric: "sentiment_vader",
  decade: null,
};
let data = [];
let raceOptions = [];
let sentenceIndex = new Map();

init().catch((err) => {
  console.error(err);
  chartEl.innerHTML = `<p style="color:#ff708a">Failed to load data: ${err.message}</p>`;
});

async function init() {
  const [tsResponse, sentencesResponse] = await Promise.all([
    fetch(DATA_URL),
    fetch(SENTENCES_URL),
  ]);

  if (!tsResponse.ok) {
    throw new Error(`Unable to fetch ${DATA_URL} (${tsResponse.status})`);
  }
  if (!sentencesResponse.ok) {
    throw new Error(
      `Unable to fetch ${SENTENCES_URL} (${sentencesResponse.status})`
    );
  }

  const [tsText, sentencesText] = await Promise.all([
    tsResponse.text(),
    sentencesResponse.text(),
  ]);

  data = d3.csvParse(tsText, (d) => ({
    race: d.race,
    decade: +d.decade,
    sentiment_vader: +d.sentiment_vader,
    sentiment_transformer: +d.sentiment_transformer,
    n_sentences: +d.n_sentences,
  }));

  const sentenceRows = d3
    .csvParse(sentencesText, (d) => ({
      race: d.race,
      decade: toDecade(+d.hist_year),
      sentence: (d.sentence || "").trim(),
    }))
    .filter((row) => Number.isFinite(row.decade) && row.sentence);

  sentenceIndex = buildSentenceIndex(sentenceRows);

  const races = Array.from(new Set(data.map((d) => d.race))).sort();
  raceOptions = ["ALL", ...races];
  raceSelect.innerHTML = raceOptions
    .map((race) => `<option value="${race}">${race}</option>`)
    .join("");

  state.race = raceOptions[0];
  raceSelect.value = state.race;

  raceSelect.addEventListener("change", (event) => {
    state.race = event.target.value;
    render();
  });

  metricSelect.addEventListener("change", (event) => {
    state.metric = event.target.value;
    render();
  });

  decadeSelect.addEventListener("change", (event) => {
    const value = event.target.value;
    state.decade = value ? Number(value) : null;
    render(false);
  });

  render();
}

function render(resetDecade = true) {
  const showAll = state.race === "ALL";
  const filtered = (showAll ? data.slice() : data.filter((d) => d.race === state.race))
    .sort((a, b) => a.decade - b.decade);
  const metricLabel =
    state.metric === "sentiment_vader"
      ? "Average sentiment (VADER)"
      : "Average sentiment (DistilBERT)";

  updateDecadeOptions(filtered, showAll, resetDecade);

  const domain = showAll ? raceOptions.filter((r) => r !== "ALL") : undefined;
  const range = domain?.map((race) => COLOR_SCALE[race] || "#ffffff");

  const marks = [
    Plot.ruleY([0], { stroke: "#4b4d60", strokeOpacity: 0.6 }),
    Plot.line(filtered, {
      x: "decade",
      y: (d) => d[state.metric],
      strokeWidth: 2,
      stroke: showAll ? "race" : "#89f0ff",
    }),
    Plot.dot(filtered, {
      x: "decade",
      y: (d) => d[state.metric],
      r: (d) => 3 + Math.min(8, Math.log1p(d.n_sentences)),
      fill: showAll ? "race" : "#89f0ff",
      stroke: "#090a12",
    }),
  ];

  if (!showAll && state.decade !== null) {
    const selectedPoints = filtered.filter((d) => d.decade === state.decade);
    if (selectedPoints.length) {
      marks.push(
        Plot.ruleX([state.decade], {
          stroke: "#fbbf24",
          strokeOpacity: 0.6,
        }),
        Plot.dot(selectedPoints, {
          x: "decade",
          y: (d) => d[state.metric],
          r: 9,
          fill: "#fde68a",
          stroke: "#facc15",
          strokeWidth: 2,
        })
      );
    }
  }

  const plot = Plot.plot({
    height: 420,
    marginLeft: 60,
    marginBottom: 45,
    x: {
      label: "Decade",
      tickFormat: (d) => d3.format("d")(d),
    },
    y: {
      label: metricLabel,
      domain: [-1, 1],
      grid: true,
    },
    color: showAll
      ? {
          legend: true,
          domain,
          range,
        }
      : {
          legend: false,
          range: ["#89f0ff"],
        },
    marks,
  });

  chartEl.innerHTML = "";
  chartEl.append(plot);
  updateStats(filtered, showAll);
  updateDetailPanel(filtered, showAll, metricLabel);
}

function updateStats(filtered, showAll) {
  if (!filtered.length) {
    statsEl.innerHTML = `<p>No data for ${state.race}.</p>`;
    return;
  }

  const average = d3.mean(filtered, (d) => d[state.metric]);
  const totalSentences = d3.sum(filtered, (d) => d.n_sentences);

  if (showAll) {
    statsEl.innerHTML = `
      <article class="stat-card">
        <span>Races plotted</span>
        <strong>${raceOptions.length - 1}</strong>
      </article>
      <article class="stat-card">
        <span>Series average (${metricLabelShort(state.metric)})</span>
        <strong>${average.toFixed(3)}</strong>
      </article>
      <article class="stat-card">
        <span>Total sentences</span>
        <strong>${totalSentences}</strong>
      </article>
    `;
    return;
  }

  const latest = filtered[filtered.length - 1];
  statsEl.innerHTML = `
    <article class="stat-card">
      <span>Latest decade</span>
      <strong>${latest.decade}</strong>
    </article>
    <article class="stat-card">
      <span>Latest sentiment</span>
      <strong>${latest[state.metric].toFixed(3)}</strong>
    </article>
    <article class="stat-card">
      <span>Series average (${metricLabelShort(state.metric)})</span>
      <strong>${average.toFixed(3)}</strong>
    </article>
    <article class="stat-card">
      <span>Sentence count</span>
      <strong>${totalSentences}</strong>
    </article>
  `;
}

function metricLabelShort(metricKey) {
  return metricKey === "sentiment_vader" ? "VADER" : "DistilBERT";
}

function updateDecadeOptions(filtered, showAll, resetDecade) {
  if (showAll || !filtered.length) {
    decadeSelect.innerHTML = `<option value="">Select a race</option>`;
    decadeSelect.disabled = true;
    if (resetDecade) {
      state.decade = null;
    }
    return;
  }

  const decades = Array.from(new Set(filtered.map((d) => d.decade))).sort(
    (a, b) => a - b
  );

  if (
    resetDecade ||
    state.decade === null ||
    !decades.includes(state.decade)
  ) {
    state.decade = decades[decades.length - 1] ?? null;
  }

  decadeSelect.disabled = false;
  decadeSelect.innerHTML = decades
    .map((decade) => `
      <option value="${decade}" ${decade === state.decade ? "selected" : ""}>
        ${decade}
      </option>
    `)
    .join("");
  decadeSelect.value = state.decade ?? "";
}

function updateDetailPanel(filtered, showAll, metricLabel) {
  if (showAll) {
    detailEl.innerHTML = `
      <strong>Race + decade details</strong>
      <span>Select a specific race to inspect its decades and sample sentences.</span>
    `;
    return;
  }

  if (!filtered.length || state.decade === null) {
    detailEl.innerHTML = `
      <strong>Race + decade details</strong>
      <span>No data for the current selection.</span>
    `;
    return;
  }

  const current = filtered.find((d) => d.decade === state.decade);
  if (!current) {
    detailEl.innerHTML = `
      <strong>Race + decade details</strong>
      <span>No data for the selected decade.</span>
    `;
    return;
  }

  const sentences = getSentences(state.race, state.decade);
  const sentenceList = sentences.length
    ? `<ul>${sentences.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ul>`
    : `<span class="sentences-empty">No stored sentences for this decade.</span>`;

  detailEl.innerHTML = `
    <strong>${state.race}</strong>
    <span>Decade: ${state.decade}</span>
    <span>${metricLabel}: ${current[state.metric].toFixed(3)}</span>
    <span>Sentences tagged: ${current.n_sentences}</span>
    <div class="sentence-samples">
      <em>Sample sentences</em>
      ${sentenceList}
    </div>
  `;
}

function buildSentenceIndex(rows, maxSentences = 5) {
  const index = new Map();
  rows.forEach((row) => {
    if (!index.has(row.race)) {
      index.set(row.race, new Map());
    }
    const decadeMap = index.get(row.race);
    if (!decadeMap.has(row.decade)) {
      decadeMap.set(row.decade, []);
    }
    const bucket = decadeMap.get(row.decade);
    if (bucket.length < maxSentences) {
      bucket.push(row.sentence);
    }
  });
  return index;
}

function getSentences(race, decade) {
  const raceMap = sentenceIndex.get(race);
  if (!raceMap) return [];
  return raceMap.get(decade) || [];
}

function toDecade(year) {
  if (!Number.isFinite(year)) return null;
  return Math.floor(year / 10) * 10;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.innerText = str;
  return div.innerHTML;
}
