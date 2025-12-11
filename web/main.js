import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm";
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

const DATA_URL = "./race_sent_timeseries_combined.csv";
const raceSelect = document.querySelector("#race-select");
const metricSelect = document.querySelector("#metric-select");
const chartEl = document.querySelector("#chart");
const statsEl = document.querySelector("#stats");
const state = {
  race: null,
  metric: "sentiment_vader",
};
let data = [];

init().catch((err) => {
  console.error(err);
  chartEl.innerHTML = `<p style="color:#ff708a">Failed to load data: ${err.message}</p>`;
});

async function init() {
  const response = await fetch(DATA_URL);
  if (!response.ok) {
    throw new Error(`Unable to fetch ${DATA_URL} (${response.status})`);
  }
  const text = await response.text();
  data = d3.csvParse(text, (d) => ({
    race: d.race,
    decade: +d.decade,
    sentiment_vader: +d.sentiment_vader,
    sentiment_transformer: +d.sentiment_transformer,
    n_sentences: +d.n_sentences,
  }));

  const races = Array.from(new Set(data.map((d) => d.race))).sort();
  raceSelect.innerHTML = races
    .map((race) => `<option value="${race}">${race}</option>`)
    .join("");

  state.race = races[0];
  raceSelect.value = state.race;

  raceSelect.addEventListener("change", (event) => {
    state.race = event.target.value;
    render();
  });

  metricSelect.addEventListener("change", (event) => {
    state.metric = event.target.value;
    render();
  });

  render();
}

function render() {
  const filtered = data
    .filter((d) => d.race === state.race)
    .sort((a, b) => a.decade - b.decade);
  const metricLabel =
    state.metric === "sentiment_vader"
      ? "Average sentiment (VADER)"
      : "Average sentiment (DistilBERT)";

  const plot = Plot.plot({
    height: 420,
    marginLeft: 60,
    marginBottom: 45,
    x: {
      label: "Decade",
      tickFormat: (d) => d,
    },
    y: {
      label: metricLabel,
      domain: [-1, 1],
      grid: true,
    },
    color: {
      legend: false,
      range: ["#89f0ff"],
    },
    marks: [
      Plot.ruleY([0], { stroke: "#4b4d60", strokeOpacity: 0.6 }),
      Plot.line(filtered, {
        x: "decade",
        y: (d) => d[state.metric],
        strokeWidth: 2,
      }),
      Plot.dot(filtered, {
        x: "decade",
        y: (d) => d[state.metric],
        r: (d) => 3 + Math.min(8, Math.log1p(d.n_sentences)),
      }),
      Plot.tip(filtered, Plot.pointerX({
        x: "decade",
        y: (d) => d[state.metric],
        title: (d) =>
          `Decade: ${d.decade}\n${metricLabel}: ${d[state.metric].toFixed(3)}\nSentences: ${d.n_sentences}`,
      })),
    ],
  });

  chartEl.innerHTML = "";
  chartEl.append(plot);
  updateStats(filtered);
}

function updateStats(filtered) {
  if (!filtered.length) {
    statsEl.innerHTML = `<p>No data for ${state.race}.</p>`;
    return;
  }

  const latest = filtered[filtered.length - 1];
  const average = d3.mean(filtered, (d) => d[state.metric]);
  const totalSentences = d3.sum(filtered, (d) => d.n_sentences);

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
      <span>Series average</span>
      <strong>${average.toFixed(3)}</strong>
    </article>
    <article class="stat-card">
      <span>Sentence count</span>
      <strong>${totalSentences}</strong>
    </article>
  `;
}
