/**
 * VisioNull Dashboard — Client-side polling & rendering
 */

const POLL_INTERVAL_MS = 3000;
let pollTimer = null;

const eventsContainer = document.getElementById("events-container");
const filterCheckbox = document.getElementById("filter-unacknowledged");
const btnRefresh = document.getElementById("btn-refresh");
const statTotal = document.getElementById("stat-total");
const statUnack = document.getElementById("stat-unacknowledged");
const statLatest = document.getElementById("stat-latest");
const statusDot = document.querySelector(".status-dot");
const statusText = document.querySelector(".status-text");

// ── Fetch helpers ──

async function fetchEvents(acknowledgedOnly) {
    const params = new URLSearchParams();
    if (acknowledgedOnly !== null && acknowledgedOnly !== undefined) {
        params.set("acknowledged", acknowledgedOnly ? "true" : "false");
    }
    params.set("limit", "200");
    const resp = await fetch(`/api/events?${params}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return resp.json();
}

async function acknowledgeEvent(id) {
    const resp = await fetch(`/api/events/${id}/acknowledge`, { method: "POST" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return resp.json();
}

// ── Rendering ──

function formatTimestamp(iso) {
    if (!iso) return "—";
    try {
        const d = new Date(iso);
        return d.toLocaleString();
    } catch {
        return iso;
    }
}

function renderEvent(ev) {
    const card = document.createElement("div");
    card.className = "event-card" + (ev.acknowledged ? " acknowledged" : "");
    card.dataset.id = ev.id;

    const confPct = ((ev.confidence || 0) * 100).toFixed(0);

    card.innerHTML = `
        <div class="event-top">
            <div class="event-info">
                <div>
                    <span class="event-device">${esc(ev.device_name)}</span>
                    <span class="event-location">${esc(ev.device_location)}</span>
                </div>
                <div class="event-message">${esc(ev.message)}</div>
                <div class="event-meta">
                    <span>${formatTimestamp(ev.timestamp)}</span>
                    <span class="confidence-badge">${confPct}% confidence</span>
                    <span>${esc(ev.event_id)}</span>
                </div>
            </div>
            <button class="btn btn-acknowledge"
                    ${ev.acknowledged ? "disabled" : ""}
                    onclick="onAcknowledge(${ev.id})">
                ${ev.acknowledged ? "Acknowledged" : "Acknowledge"}
            </button>
        </div>
    `;
    return card;
}

function renderEvents(events) {
    eventsContainer.innerHTML = "";
    if (events.length === 0) {
        eventsContainer.innerHTML =
            '<p class="events-empty">No fall events recorded yet.</p>';
        return;
    }
    for (const ev of events) {
        eventsContainer.appendChild(renderEvent(ev));
    }
}

function updateStats(allEvents) {
    const total = allEvents.length;
    const unack = allEvents.filter((e) => !e.acknowledged).length;
    statTotal.textContent = total;
    statUnack.textContent = unack;
    if (total > 0) {
        statLatest.textContent = formatTimestamp(allEvents[0].timestamp);
    } else {
        statLatest.textContent = "—";
    }
}

function setConnectionStatus(ok) {
    statusDot.className = "status-dot " + (ok ? "connected" : "error");
    statusText.textContent = ok ? "Connected" : "Connection lost";
}

function esc(s) {
    if (s == null) return "";
    const el = document.createElement("span");
    el.textContent = String(s);
    return el.innerHTML;
}

// ── Actions ──

async function onAcknowledge(id) {
    try {
        await acknowledgeEvent(id);
        await refresh();
    } catch (err) {
        console.error("Acknowledge failed:", err);
    }
}

// Expose globally for inline onclick
window.onAcknowledge = onAcknowledge;

async function refresh() {
    try {
        // Always fetch all for stats
        const allEvents = await fetchEvents(null);
        updateStats(allEvents);

        // Render filtered list
        const showUnackOnly = filterCheckbox.checked;
        const display = showUnackOnly
            ? allEvents.filter((e) => !e.acknowledged)
            : allEvents;
        renderEvents(display);

        setConnectionStatus(true);
    } catch (err) {
        console.error("Refresh failed:", err);
        setConnectionStatus(false);
    }
}

// ── Init ──

function startPolling() {
    refresh();
    pollTimer = setInterval(refresh, POLL_INTERVAL_MS);
}

btnRefresh.addEventListener("click", refresh);
filterCheckbox.addEventListener("change", refresh);

startPolling();
