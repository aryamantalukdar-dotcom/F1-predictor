/* eslint-env browser */

const $ = (sel) => document.querySelector(sel);
const fmt = (n, d = 2) => (Number.isFinite(n) ? Number(n).toFixed(d) : "—");

async function fetchPrediction(fresh = false) {
  const url = fresh ? "/api/predict?fresh=1" : "/api/predict";
  const res = await fetch(url);
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

function renderRaceBanner(race) {
  const el = $("#race-banner");
  const date = new Date(race.date + "T00:00:00");
  const dateStr = date.toLocaleDateString(undefined, {
    weekday: "long",
    month: "long",
    day: "numeric",
    year: "numeric",
  });
  el.innerHTML = `
    <div>
      <h1 class="race-name">${race.race_name}</h1>
      <p class="race-meta">${race.circuit_name} · ${race.locality}, ${race.country} · Round ${race.round}</p>
    </div>
    <div class="race-date">${dateStr}</div>
  `;
}

function renderWeather(weather) {
  const el = $("#weather-card");
  const rainPct = Math.round((weather.rain_probability || 0) * 100);
  const rainClass = rainPct >= 50 ? "rain-warn" : "";
  const note = weather.is_forecast
    ? "Open-Meteo forecast"
    : "Climatological estimate (forecast not yet available)";
  el.innerHTML = `
    <h2>Race-Day Weather</h2>
    <div class="weather-grid">
      <div class="weather-stat">
        <div class="label">Rain probability</div>
        <div class="value ${rainClass}">${rainPct}%</div>
      </div>
      <div class="weather-stat">
        <div class="label">Temperature</div>
        <div class="value">${fmt(weather.temperature_c, 1)}°C</div>
      </div>
      <div class="weather-stat">
        <div class="label">Precipitation</div>
        <div class="value">${fmt(weather.precipitation_mm, 1)} mm</div>
      </div>
      <div class="weather-stat">
        <div class="label">Wind</div>
        <div class="value">${fmt(weather.wind_kph, 0)} kph</div>
      </div>
    </div>
    <p class="muted" style="margin-top:14px">${note}</p>
  `;
}

function renderPole(pole, polePredictions) {
  const el = $("#pole-card");
  if (!pole) {
    el.innerHTML = "<h2>Pole Position Prediction</h2><div class='card-loading'>No prediction available</div>";
    return;
  }
  const others = (polePredictions || []).slice(1, 5);
  el.innerHTML = `
    <h2>Pole Position Prediction</h2>
    <div class="pole-driver">
      <div class="pole-code">${pole.code}</div>
      <div class="pole-info">
        <div class="pole-name">${pole.driver_name}</div>
        <div class="pole-team">${pole.constructor}</div>
      </div>
    </div>
    ${
      others.length
        ? `
      <div class="pole-secondary">
        <div class="muted">Most likely front runners</div>
        <ul class="pole-secondary-list">
          ${others
            .map(
              (d) => `
            <li>
              <span class="pos">${d.rank}.</span>
              <span class="code">${d.code}</span>
              <span>${d.driver_name}</span>
            </li>`,
            )
            .join("")}
        </ul>
      </div>`
        : ""
    }
  `;
}

function renderRaceTable(predictions) {
  const wrap = $("#race-table-wrap");
  if (!predictions || !predictions.length) {
    wrap.innerHTML = "<div class='card-loading'>No prediction available</div>";
    return;
  }
  const maxProb = Math.max(...predictions.map((d) => d.win_probability));
  const rows = predictions
    .map((d, i) => {
      const podiumClass =
        d.rank === 1
          ? "podium-1"
          : d.rank === 2
            ? "podium-2"
            : d.rank === 3
              ? "podium-3"
              : "";
      const winPct = Math.round(d.win_probability * 1000) / 10;
      const barWidth = (d.win_probability / maxProb) * 80;
      return `
        <tr class="${podiumClass}">
          <td class="rank">P${d.rank}</td>
          <td class="code">${d.code}</td>
          <td class="driver-name">${d.driver_name}</td>
          <td class="team-name">${d.constructor}</td>
          <td class="win-prob">
            ${winPct}%<span class="win-prob-bar" style="width:${barWidth}px"></span>
          </td>
        </tr>
      `;
    })
    .join("");
  wrap.innerHTML = `
    <div class="table-scroll">
    <table>
      <thead>
        <tr>
          <th>Pos</th>
          <th>Code</th>
          <th>Driver</th>
          <th>Team</th>
          <th>Win prob</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    </div>
  `;
}

function renderNews(news, meta) {
  const items = news.items || [];
  $("#news-list").innerHTML = items.length
    ? items
        .map((n) => {
          const when = n.published ? new Date(n.published).toLocaleString() : "";
          return `<li>
            <a href="${n.link}" target="_blank" rel="noopener">${escapeHtml(n.title)}</a>
            <div class="news-meta">${escapeHtml(n.source || "")} ${when ? "· " + when : ""}</div>
          </li>`;
        })
        .join("")
    : "<li class='muted'>No recent news fetched.</li>";

  $("#model-meta").textContent = meta
    ? `Race model: ${meta.race_model} · Pole model: ${meta.pole_model} · ${meta.n_drivers} drivers`
    : "";
}

function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

async function load(fresh = false) {
  const btn = $("#refresh");
  btn.disabled = true;
  btn.querySelector(".refresh-label").textContent = fresh ? "Refreshing…" : "Loading…";
  try {
    const data = await fetchPrediction(fresh);
    renderRaceBanner(data.race);
    renderWeather(data.weather);
    renderPole(data.pole_prediction, data.pole_predictions);
    renderRaceTable(data.race_predictions);
    renderNews(data.news, data.meta);
  } catch (err) {
    console.error(err);
    $("#race-banner").innerHTML = `<div class='race-banner-loading'>Failed to load: ${escapeHtml(err.message)}</div>`;
  } finally {
    btn.disabled = false;
    btn.querySelector(".refresh-label").textContent = "Refresh";
  }
}

$("#refresh").addEventListener("click", () => load(true));
load();
