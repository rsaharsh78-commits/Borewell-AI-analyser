// ================================================================
// JalDrishti v3 — Frontend (wired to real FastAPI backend)
// ================================================================

const API = {
    geocode:       '/api/geocode',
    predict:       '/api/predict',
    heatmap:       '/api/heatmap',
    rainfall:      '/api/rainfall-history',
    gridAnalysis:  '/api/grid-analysis',
    monthlyRisk:   '/api/monthly-risk'
};

const RISK_COLORS = {
    low:  '#00d4aa',
    mod:  '#f59e0b',
    high: '#f97316',
    crit: '#ef4444'
};

function _haversineKm(lat1, lng1, lat2, lng2) {
    const R = 6371;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLng = (lng2 - lng1) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLng/2) * Math.sin(dLng/2);
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}

// ---- State ----
let map;
let marker;
let heatmapCircles = [];
let gridMarkers = [];
let heatmapOn = false;
let heatmapLoaded = false;
let currentLang = 'en';
let currentAdvisory = { en: '', hi: '' };
let currentMonthData = null; // Store fetched monthly data
let precisionMode = false;

// ---- DOM refs ----
const els = {
    map:           document.getElementById('map'),
    search:        document.getElementById('address-search'),
    dropdown:      document.getElementById('search-dropdown'),
    depth:         document.getElementById('planned-depth'),
    month:         document.getElementById('planned-month'),
    radius:        document.getElementById('grid-radius'),
    heatmapToggle: document.getElementById('heatmap-toggle'),
    findBestBtn:   document.getElementById('find-best-spot-btn'),
    printBtn:      document.getElementById('print-btn'),
    overlay:       document.getElementById('coord-overlay'),
    stateInit:     document.getElementById('initial-state'),
    stateLoad:     document.getElementById('loading-state'),
    stateRes:      document.getElementById('results-state'),
    precisionToggle: document.getElementById('precision-toggle'),
    crosshair:     document.getElementById('crosshair-overlay'),
    precisionPanel: document.getElementById('precision-panel'),
    precLat:       document.getElementById('prec-lat'),
    precLng:       document.getElementById('prec-lng'),
    useLocationBtn: document.getElementById('use-location-btn'),
};

// ================================================================
// Map
// ================================================================

const tileLayers = {
    satellite: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {attribution: 'Esri', maxZoom: 18}),
    sentinel: L.tileLayer('https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2021_3857/default/g/{z}/{y}/{x}.jpg', {attribution: 'Sentinel-2 Cloudless by EOX', maxZoom: 15}),
    osm: L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {attribution: 'OpenStreetMap', maxZoom: 19}),
    topo: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}', {attribution: 'Esri Topo', maxZoom: 17})
};

let currentBaseLayer = tileLayers.satellite;
let labelsOverlay;

function initMap() {
    map = L.map('map').setView([17.385, 78.486], 11);
    const bounds = [[17.0, 77.4], [17.8, 78.8]];
    map.setMaxBounds(bounds);
    map.on('drag', () => map.panInsideBounds(bounds, { animate: false }));

    currentBaseLayer.addTo(map);

    labelsOverlay = L.tileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
        { maxZoom: 17, pane: 'overlayPane' }
    );
    map.createPane('overlayPane');
    map.getPane('overlayPane').style.zIndex = 650;
    map.getPane('overlayPane').style.pointerEvents = 'none';
    labelsOverlay.addTo(map);

    map.on('mousemove', e => {
        els.overlay.textContent = `Lat: ${e.latlng.lat.toFixed(6)} | Lng: ${e.latlng.lng.toFixed(6)}`;
    });

    map.on('click', e => {
        placeMarker(e.latlng);
        triggerPrediction(e.latlng.lat, e.latlng.lng);
    });

    // Layer control binding
    document.querySelectorAll('input[name="tileLayer"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            map.removeLayer(currentBaseLayer);
            currentBaseLayer = tileLayers[e.target.value];
            currentBaseLayer.addTo(map);
        });
    });
}

function placeMarker(latlng) {
    if (marker) {
        marker.setLatLng(latlng);
    } else {
        const icon = L.divIcon({
            className: 'water-drop-marker',
            iconSize: [24, 24],
            iconAnchor: [12, 24]
        });
        marker = L.marker(latlng, { icon, zIndexOffset: 1000 }).addTo(map);
    }
    map.panTo(latlng);
}

// ================================================================
// Address Search → GET /api/geocode
// ================================================================
function debounce(fn, ms) {
    let t;
    return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

els.search.addEventListener('input', debounce(async e => {
    const q = e.target.value.trim();
    if (q.length < 3) { els.dropdown.classList.add('hidden'); return; }

    try {
        const res = await fetch(`${API.geocode}?q=${encodeURIComponent(q)}`);
        if (!res.ok) throw new Error('Geocode failed');
        const results = await res.json();

        els.dropdown.innerHTML = '';
        results.forEach(r => {
            const div = document.createElement('div');
            div.className = 'dropdown-item';
            div.textContent = r.display_name;
            div.onclick = () => {
                els.search.value = r.display_name;
                els.dropdown.classList.add('hidden');
                placeMarker(L.latLng(r.lat, r.lng));
                triggerPrediction(r.lat, r.lng);
            };
            els.dropdown.appendChild(div);
        });
        els.dropdown.classList.toggle('hidden', results.length === 0);
    } catch (err) {
        console.error('Geocode error:', err);
    }
}, 500));

document.addEventListener('click', e => {
    if (!els.search.contains(e.target) && !els.dropdown.contains(e.target))
        els.dropdown.classList.add('hidden');
});

// ================================================================
// Current Location Detection → HTML5 Geolocation API
// ================================================================
document.getElementById('my-location-btn').addEventListener('click', detectCurrentLocation);

function detectCurrentLocation() {
    const btn = document.getElementById('my-location-btn');
    
    // Check browser support
    if (!navigator.geolocation) {
        showLocationToast('❌ Geolocation is not supported by your browser. Please use a modern browser like Chrome or Edge.', 'error');
        return;
    }
    
    // Set detecting state
    btn.classList.add('detecting');
    btn.textContent = '⏳ Detecting...';
    
    navigator.geolocation.getCurrentPosition(
        // SUCCESS handler
        (position) => {
            const lat = position.coords.latitude;
            const lng = position.coords.longitude;
            const accuracy = Math.round(position.coords.accuracy);
            
            btn.classList.remove('detecting');
            btn.textContent = '📍 My Location';
            
            // Fly to detected location
            map.flyTo([lat, lng], 15, { duration: 1.5 });
            placeMarker(L.latLng(lat, lng));
            
            showLocationToast(
                `📍 Location detected: ${lat.toFixed(6)}, ${lng.toFixed(6)} (±${accuracy}m)`,
                'success'
            );
            
            // Trigger analysis
            triggerPrediction(lat, lng);
        },
        // ERROR handler
        (error) => {
            btn.classList.remove('detecting');
            btn.textContent = '📍 My Location';
            
            let message;
            switch (error.code) {
                case error.PERMISSION_DENIED:
                    message = '🔒 Location access denied. Please allow location permission in your browser settings (click the lock icon in the address bar).';
                    break;
                case error.POSITION_UNAVAILABLE:
                    message = '📡 Location unavailable. Your device could not determine its position. Ensure GPS/Wi-Fi is enabled.';
                    break;
                case error.TIMEOUT:
                    message = '⏱️ Location request timed out. Please try again or move to an area with better GPS signal.';
                    break;
                default:
                    message = '⚠️ An unknown error occurred while detecting location. Please try again.';
            }
            showLocationToast(message, 'error');
        },
        // OPTIONS
        {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 60000
        }
    );
}

function showLocationToast(message, type) {
    // Remove existing toast
    const existing = document.getElementById('location-toast');
    if (existing) existing.remove();
    
    const toast = document.createElement('div');
    toast.id = 'location-toast';
    toast.style.cssText = `
        position: fixed;
        bottom: 60px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        padding: 12px 20px;
        border-radius: 12px;
        font-size: 0.85rem;
        line-height: 1.4;
        max-width: 420px;
        text-align: center;
        backdrop-filter: blur(12px);
        animation: toastIn 0.3s ease-out;
        ${type === 'success' 
            ? 'background: rgba(0,212,170,0.15); border: 1px solid rgba(0,212,170,0.4); color: #00d4aa;'
            : 'background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.4); color: #fca5a5;'
        }
    `;
    toast.textContent = message;
    document.body.appendChild(toast);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        toast.style.animation = 'toastOut 0.3s ease-in forwards';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

// Toast animations (injected once)
if (!document.getElementById('toast-keyframes')) {
    const style = document.createElement('style');
    style.id = 'toast-keyframes';
    style.textContent = `
        @keyframes toastIn { from { opacity:0; transform:translateX(-50%) translateY(20px); } to { opacity:1; transform:translateX(-50%) translateY(0); } }
        @keyframes toastOut { from { opacity:1; transform:translateX(-50%) translateY(0); } to { opacity:0; transform:translateX(-50%) translateY(20px); } }
    `;
    document.head.appendChild(style);
}

// ================================================================
// Prediction → POST /api/predict
// ================================================================
async function triggerPrediction(lat, lng) {
    els.stateInit.classList.remove('active');
    els.stateRes.classList.add('hidden');
    els.stateLoad.classList.remove('hidden');
    els.stateLoad.classList.add('active');

    const pDepth = parseInt(els.depth.value) || 500;
    const pMonth = parseInt(els.month.value) || 1;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);

    try {
        const res = await fetch(API.predict, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: controller.signal,
            body: JSON.stringify({
                latitude: lat,
                longitude: lng,
                planned_depth_ft: pDepth,
                planned_month: pMonth
            })
        });
        clearTimeout(timeoutId);
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        renderDashboard(data);

        els.stateLoad.classList.remove('active');
        els.stateLoad.classList.add('hidden');
        els.stateRes.classList.remove('hidden');

        fetchRainfallHistory(lat, lng);
        fetchMonthlyRisk(lat, lng);

    } catch (err) {
        clearTimeout(timeoutId);
        console.error('Prediction error:', err);
        els.stateLoad.classList.remove('active');
        els.stateLoad.classList.add('hidden');
        els.stateInit.classList.add('active');
        const msg = err.name === 'AbortError' 
            ? 'Request timed out — external data services may be slow. Please try again.'
            : `Prediction failed: ${err.message}`;
        alert(msg);
    }
}

async function fetchRainfallHistory(lat, lng) {
    try {
        const res = await fetch(`${API.rainfall}?lat=${lat}&lng=${lng}`);
        if (!res.ok) return;
        const data = await res.json();
        renderRainfallChart(data);
    } catch (err) {
        console.error('Rainfall history error:', err);
    }
}

async function fetchMonthlyRisk(lat, lng) {
    try {
        const res = await fetch(`${API.monthlyRisk}?lat=${lat}&lng=${lng}`);
        if (!res.ok) return;
        currentMonthData = await res.json();
        renderMonthlyRisk(currentMonthData);
    } catch (err) {
        console.error('Monthly risk error:', err);
    }
}

// ================================================================
// Rendering
// ================================================================
function renderDashboard(data) {
    updateLocationBar(data);
    updateRiskGauge(data.risk_score, data.risk_category, data.confidence);
    
    // Geology
    if (data.detailed_explanations) {
        renderDetailedExplanations('geology', data.detailed_explanations);
    } else {
        document.getElementById('geology-fallback').classList.remove('hidden');
        document.getElementById('geology-details').classList.add('hidden');
        updateGeologyCard(data.geology_info, data.location_info);
    }

    updateDepthCard(data.recommended_depth_ft, parseInt(els.depth.value) || 500);

    // Hydrology
    if (data.detailed_explanations) {
        renderDetailedExplanations('hydrology', data.detailed_explanations);
    } else {
        document.getElementById('hydrology-fallback').classList.remove('hidden');
        document.getElementById('hydrology-details').classList.add('hidden');
        updateHydrologyCard(data.hydrology_summary);
    }

    // Terrain & Soil
    if (data.detailed_explanations) {
        renderDetailedExplanations('terrain', data.detailed_explanations);
        renderDetailedExplanations('soil', data.detailed_explanations);
    } else {
        document.getElementById('terrain-fallback').classList.remove('hidden');
        document.getElementById('terrain-details').classList.add('hidden');
        updateTerrainCard(data.terrain_info);
    }

    updateFactors(data.factors);
    currentAdvisory.en = data.advisory;
    currentAdvisory.hi = data.advisory_hi;
    renderAdvisory();
    updateNearbyStats(data.nearby_stats);

    if (data.data_sources) {
        renderDataSources(data.data_sources);
    }
    
    if (data.monthly_risk) {
        // Optionally use data.monthly_risk if provided in POST response
    }
}

// ---- Details Helper ----
function renderDetailedExplanations(categoryPrefix, explanations) {
    const listEl = document.getElementById(`${categoryPrefix}-details`);
    if (!listEl) return;
    
    const fallbackEl = document.getElementById(`${categoryPrefix}-fallback`);
    if (fallbackEl) fallbackEl.classList.add('hidden');
    listEl.classList.remove('hidden');
    listEl.innerHTML = '';
    
    // Filter explanations somewhat naively based on factor names if needed, 
    // or assume the backend provides category mapped explanations. 
    // For simplicity, we just filter by rough text matching or assume backend provides `category`.
    // Let's assume explanation objects might have a `category` or we guess by factor name.
    
    let filtered = explanations;
    if (categoryPrefix === 'geology') {
        filtered = explanations.filter(e => ['Rock Type', 'Weathered Zone Thickness', 'Lineament Density', 'Extraction Stage'].some(v => e.factor.includes(v)));
    } else if (categoryPrefix === 'hydrology') {
        filtered = explanations.filter(e => ['Rainfall', 'Recharge', 'Pre-Monsoon GWL', 'Post-Monsoon GWL'].some(v => e.factor.includes(v)));
    } else if (categoryPrefix === 'terrain') {
        filtered = explanations.filter(e => ['Slope', 'Distance to Water', 'Impervious'].some(v => e.factor.includes(v)));
    } else if (categoryPrefix === 'soil') {
        filtered = explanations.filter(e => ['Clay', 'Sand', 'Silt', 'pH', 'Soil'].some(v => e.factor.includes(v)));
    }

    filtered.forEach(exp => {
        let badgeClass = 'neutral';
        if (exp.assessment === 'risk' || exp.assessment === 'Risk') badgeClass = 'risk';
        if (exp.assessment === 'advantage' || exp.assessment === 'Advantage') badgeClass = 'advantage';

        const item = document.createElement('div');
        item.className = 'detail-item';
        item.innerHTML = `
            <div class="detail-header" onclick="toggleDetail(this)">
                <div style="flex:1;min-width:0">
                    <span class="detail-label">${exp.factor}</span>
                </div>
                <span class="detail-value">${exp.value}<small style="opacity:0.6;margin-left:2px">${exp.unit || ''}</small></span>
                <span class="detail-badge ${badgeClass}">${badgeClass === 'risk' ? '⚠️' : (badgeClass === 'advantage' ? '✅' : 'ℹ️')} ${exp.assessment}</span>
                <span class="detail-chevron">▾</span>
            </div>
            <div class="detail-body hidden">
                <p class="detail-narrative">${exp.narrative || 'No description available.'}</p>
                <div class="detail-meta">
                    <span class="source-tag">Source: ${exp.source || 'Model'}</span>
                    <span class="detail-rec">💡 ${exp.recommendation || ''}</span>
                </div>
            </div>
        `;
        listEl.appendChild(item);
    });
}

function toggleDetail(header) {
    const body = header.nextElementSibling;
    body.classList.toggle('hidden');
    const chevron = header.querySelector('.detail-chevron');
    chevron.style.transform = body.classList.contains('hidden') ? '' : 'rotate(180deg)';
}

// ---- Data Sources ----
function renderDataSources(sources) {
    const card = document.getElementById('data-sources-card');
    const list = document.getElementById('data-sources-list');
    card.classList.remove('hidden');
    list.innerHTML = '';
    
    sources.forEach(src => {
        let statusClass = 'status-static';
        let statusIcon = '📁';
        if (src.status.toLowerCase() === 'live') {
            statusClass = 'status-live';
            statusIcon = '✅';
        } else if (src.status.toLowerCase() === 'fallback') {
            statusClass = 'status-fallback';
            statusIcon = '⚠️';
        }
        
        list.innerHTML += `
            <div class="source-item">
                <span class="source-name">${src.field} <small style="color:var(--text-secondary)">(${src.source})</small></span>
                <span class="source-status ${statusClass}">${statusIcon} ${src.status}</span>
            </div>
        `;
    });
}

// ---- Monthly Risk Calendar ----
function renderMonthlyRisk(data) {
    const card = document.getElementById('monthly-risk-card');
    const cal = document.getElementById('risk-calendar');
    card.classList.remove('hidden');
    cal.innerHTML = '';
    
    const plannedMonthNum = parseInt(els.month.value) || 1;
    
    data.months.forEach(m => {
        let riskClass = 'low';
        if (m.risk_score > 30) riskClass = 'moderate';
        if (m.risk_score > 50) riskClass = 'high';
        if (m.risk_score > 70) riskClass = 'critical';
        
        const isSelected = m.month === plannedMonthNum ? 'selected' : '';
        
        const cell = document.createElement('div');
        cell.className = `risk-month ${riskClass} ${isSelected}`;
        cell.innerHTML = `
            <div class="month-name">${m.month_name}</div>
            <div class="month-score">${m.risk_score}</div>
        `;
        
        cell.onclick = () => {
            document.querySelectorAll('.risk-month').forEach(c => c.classList.remove('selected'));
            cell.classList.add('selected');
            showMonthDetails(m);
        };
        
        cal.appendChild(cell);
        
        if (isSelected) {
            showMonthDetails(m);
        }
    });
}

function showMonthDetails(m) {
    const panel = document.getElementById('risk-detail-panel');
    panel.classList.remove('hidden');
    
    document.getElementById('risk-detail-month').textContent = m.month_name + ' Analysis';
    document.getElementById('risk-detail-explanation').textContent = m.explanation;
    
    const risksEl = document.getElementById('risk-detail-risks');
    risksEl.innerHTML = `<strong>Risks</strong><ul>${(m.risks||[]).map(r => `<li>${r}</li>`).join('')}</ul>`;
    
    const advEl = document.getElementById('risk-detail-advantages');
    advEl.innerHTML = `<strong>Advantages</strong><ul>${(m.advantages||[]).map(a => `<li>${a}</li>`).join('')}</ul>`;
    
    document.getElementById('risk-detail-recommendation').textContent = '💡 ' + m.recommendation;
}

// ---- Grid Analysis ----
els.findBestBtn.addEventListener('click', async () => {
    if (!marker) {
        alert("Please select a location on the map first.");
        return;
    }
    
    const lat = marker.getLatLng().lat;
    const lng = marker.getLatLng().lng;
    const rad = parseFloat(els.radius.value) || 2.0;
    
    els.findBestBtn.textContent = 'Searching...';
    els.findBestBtn.disabled = true;
    
    try {
        const res = await fetch(API.gridAnalysis, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ latitude: lat, longitude: lng, radius_km: rad })
        });
        
        if (!res.ok) throw new Error('Grid analysis failed');
        const data = await res.json();
        
        renderGridResults(data);
        
    } catch (err) {
        console.error('Grid analysis error:', err);
        alert('Grid analysis failed.');
    } finally {
        els.findBestBtn.textContent = 'Find Best Spot';
        els.findBestBtn.disabled = false;
    }
});

function renderGridResults(data) {
    // Clear old markers
    gridMarkers.forEach(m => map.removeLayer(m));
    gridMarkers = [];
    
    const card = document.getElementById('grid-analysis-card');
    const list = document.getElementById('grid-results-list');
    card.classList.remove('hidden');
    list.innerHTML = '';
    
    // Add recommendation summary
    if (data.best_location) {
        const bestDist = _haversineKm(data.center_lat, data.center_lng, data.best_location.lat, data.best_location.lng);
        const summary = document.createElement('div');
        summary.className = 'grid-recommendation';
        summary.innerHTML = `
            <div class="rec-header">🏆 Best Location Found</div>
            <div class="rec-body">
                Score: <strong>${data.best_location.total_score}/100</strong> · 
                ${bestDist.toFixed(2)} km ${bestDist < 0.1 ? '(right here!)' : 'from your position'} · 
                ${data.best_location.mandal || ''} · ${data.best_location.extraction_category || ''}
            </div>
        `;
        list.appendChild(summary);
    }
    
    // Draw top 3
    data.top_3.forEach((pt, i) => {
        const rank = pt.rank; // 1, 2, 3
        
        // Add marker
        const iconHtml = `<div class="rank-badge rank-${rank}">${rank}</div>`;
        const icon = L.divIcon({
            className: 'grid-rank-icon',
            html: iconHtml,
            iconSize: [32, 32],
            iconAnchor: [16, 16]
        });
        
        const m = L.marker([pt.lat, pt.lng], { icon, zIndexOffset: 2000 - rank }).addTo(map);
        m.on('click', () => {
            placeMarker(L.latLng(pt.lat, pt.lng));
            triggerPrediction(pt.lat, pt.lng);
        });
        gridMarkers.push(m);
        
        // Add to list
        const item = document.createElement('div');
        item.className = 'grid-result-item';
        const distKm = _haversineKm(data.center_lat, data.center_lng, pt.lat, pt.lng);
        item.innerHTML = `
            <div class="rank-badge rank-${rank}">${rank}</div>
            <div class="grid-info">
                <div style="display:flex;justify-content:space-between;">
                    <span>${pt.mandal || 'Unknown'} - ${pt.rock_type || 'Unknown'}</span>
                    <span class="grid-score">${pt.total_score}</span>
                </div>
                <div class="grid-score-bar-bg">
                    <div class="grid-score-bar" style="width: ${pt.total_score}%"></div>
                </div>
                <div style="font-size:0.75rem;color:var(--text-secondary);margin-top:4px;">
                    📍 ${distKm.toFixed(2)} km away · ${pt.reasoning}
                </div>
            </div>
        `;
        item.onclick = () => {
            placeMarker(L.latLng(pt.lat, pt.lng));
            triggerPrediction(pt.lat, pt.lng);
        };
        list.appendChild(item);
    });
    
    // Draw all others as small circles
    data.all_points.forEach(pt => {
        if (pt.rank <= 3) return; // already drawn
        let c = RISK_COLORS.crit;
        if (pt.total_score > 70) c = RISK_COLORS.low;
        else if (pt.total_score > 50) c = RISK_COLORS.mod;
        else if (pt.total_score > 30) c = RISK_COLORS.high;
        
        const m = L.circleMarker([pt.lat, pt.lng], {
            radius: 4,
            fillColor: c,
            color: '#fff',
            weight: 1,
            fillOpacity: 0.8
        }).addTo(map);
        m.bindTooltip(`Score: ${pt.total_score}`);
        m.on('click', () => {
            placeMarker(L.latLng(pt.lat, pt.lng));
            triggerPrediction(pt.lat, pt.lng);
        });
        gridMarkers.push(m);
    });
    
    // Fit bounds
    if (gridMarkers.length > 0) {
        const group = new L.featureGroup(gridMarkers);
        map.fitBounds(group.getBounds().pad(0.1));
    }
}

// ---- Legacy Updaters (fallback) ----
function updateLocationBar(data) {
    document.getElementById('loc-name').textContent =
        `${data.location_info.mandal}, ${data.location_info.district}, TS`;
    document.getElementById('badge-rock').textContent = data.location_info.rock_type;
    document.getElementById('badge-aquifer').textContent = data.location_info.aquifer_type;
}

function updateRiskGauge(score, category, confidence) {
    const elScore = document.getElementById('risk-score');
    const elCat   = document.getElementById('risk-category');
    const elFill  = document.getElementById('gauge-fill');

    elScore.textContent = score;
    elCat.textContent   = category;
    document.getElementById('risk-confidence').textContent = `${Math.round(confidence)}%`;

    let color = RISK_COLORS.low;
    if (score > 30) color = RISK_COLORS.mod;
    if (score > 60) color = RISK_COLORS.high;
    if (score > 85) color = RISK_COLORS.crit;

    elScore.style.color = color;
    elCat.style.color   = color;
    elFill.style.stroke = color;

    const offset = 125.6 - (125.6 * score / 100);
    setTimeout(() => { elFill.style.strokeDashoffset = offset; }, 100);

    // Contextual explanation — placed OUTSIDE gauge, after confidence
    let contextEl = document.getElementById('risk-context');
    if (!contextEl) {
        contextEl = document.createElement('div');
        contextEl.id = 'risk-context';
        contextEl.style.cssText = 'font-size:0.75rem;color:var(--text-secondary);text-align:center;margin-top:10px;line-height:1.5;padding:6px 12px;background:rgba(255,255,255,0.03);border-radius:8px;';
        // Append to .risk-card, NOT to .gauge-content
        document.querySelector('.risk-card').appendChild(contextEl);
    }
    
    if (score > 75) {
        contextEl.innerHTML = `<span style="color:${color}">⚠️ Urban aquifer is over-exploited.</span><br>Try <strong>🎯 Precision Mode</strong> or <strong>Find Best Spot</strong> with a larger radius to find safer zones nearby.`;
    } else if (score > 50) {
        contextEl.innerHTML = `<span style="color:${color}">Moderate risk zone.</span> Consider drilling during Oct-Nov for best recharge conditions.`;
    } else if (score > 30) {
        contextEl.innerHTML = `<span style="color:${color}">Moderate prospects.</span> Geophysical survey recommended before drilling.`;
    } else {
        contextEl.innerHTML = `<span style="color:${color}">✅ Good location for borewell.</span> Sustainable extraction with favorable recharge.`;
    }
}

function updateGeologyCard(geo, loc) {
    document.getElementById('geo-rock').textContent = loc.rock_type;
    document.getElementById('geo-lineament').innerHTML =
        `${geo.lineament_density.toFixed(1)} <small>fractures/km²</small>`;
    document.getElementById('geo-weathered').innerHTML =
        `${geo.weathered_zone_thickness_m.toFixed(1)} <small>m</small>`;
    document.getElementById('geo-bedrock').innerHTML =
        `${geo.depth_to_bedrock_m.toFixed(1)} <small>m</small>`;
    document.getElementById('geo-trend').textContent = geo.dominant_trend;
}

function updateDepthCard(recommended, planned) {
    document.getElementById('rec-depth').textContent =
        `${recommended.min} – ${recommended.max} ft`;
    document.getElementById('display-planned-depth').textContent = `${planned} ft`;

    const scale = 1200;
    const optRange = document.getElementById('optimal-range');
    const markerEl = document.getElementById('planned-marker');

    const startPct  = Math.min((recommended.min / scale) * 100, 100);
    const widthPct  = Math.min(((recommended.max - recommended.min) / scale) * 100, 100 - startPct);
    const markerPct = Math.min((planned / scale) * 100, 100);

    setTimeout(() => {
        optRange.style.left  = `${startPct}%`;
        optRange.style.width = `${widthPct}%`;
        markerEl.style.left  = `${markerPct}%`;
    }, 200);
}

function updateHydrologyCard(hydro) {
    document.getElementById('hydro-rain').innerHTML =
        `${hydro.rainfall_15yr_avg_mm.toFixed(0)} <small>mm</small>`;
    document.getElementById('hydro-monsoon').innerHTML =
        `${hydro.monsoon_avg_mm.toFixed(0)} <small>mm</small>`;
    document.getElementById('hydro-et0').innerHTML =
        `${hydro.et0_avg_mm.toFixed(0)} <small>mm</small>`;
    document.getElementById('hydro-recharge').innerHTML =
        `${hydro.effective_recharge_mm.toFixed(0)} <small>mm</small>`;
}

function renderRainfallChart(histData) {
    const container = document.getElementById('rainfall-chart');
    container.innerHTML = '';
    const years = histData.years || [];
    const rain  = histData.annual_rainfall || [];
    const et0   = histData.annual_et0 || [];
    if (!years.length) return;

    const maxVal = Math.max(...rain, ...et0) * 1.1;

    years.forEach((yr, i) => {
        const wrap = document.createElement('div');
        wrap.className = 'chart-bar-container';
        wrap.title = `${yr}: Rain ${(rain[i] || 0).toFixed(0)}mm, ET₀ ${(et0[i] || 0).toFixed(0)}mm`;

        const bar = document.createElement('div');
        bar.className = 'chart-bar';
        bar.style.height = '0%';

        const dot = document.createElement('div');
        dot.className = 'chart-line-point';
        dot.style.bottom = `${((et0[i] || 0) / maxVal) * 100}%`;

        wrap.appendChild(bar);
        wrap.appendChild(dot);
        container.appendChild(wrap);

        setTimeout(() => {
            bar.style.height = `${((rain[i] || 0) / maxVal) * 100}%`;
        }, 100);
    });
}

function updateTerrainCard(t) {
    document.getElementById('ter-elev').textContent  = `${t.elevation_m.toFixed(0)} m`;
    document.getElementById('ter-slope').textContent  = `${t.slope_degrees.toFixed(1)}°`;
    document.getElementById('ter-twi').textContent    = t.twi.toFixed(1);
    document.getElementById('ter-water').textContent  = `${t.distance_to_waterbody_m.toFixed(0)} m`;
    document.getElementById('ter-land').textContent   = t.land_cover_class;
    document.getElementById('ter-imperv').textContent = `${t.impervious_pct.toFixed(0)}%`;
}

function updateFactors(factors) {
    const list = document.getElementById('factors-list');
    list.innerHTML = '';

    factors.forEach(f => {
        const score = f.score || 0;
        const isNegative = f.contribution === 'Negative';
        
        // Green for positive (reduces risk), Red/Orange for negative (increases risk)
        let barColor, arrow, arrowColor, impactLabel;
        if (isNegative) {
            barColor = score > 0.7 ? '#ef4444' : (score > 0.4 ? '#f97316' : '#f59e0b');
            arrow = '▼';
            arrowColor = '#ef4444';
            impactLabel = f.impact === 'High' ? 'Increases Risk ⚠️' : 'Mild Risk ↗';
        } else {
            barColor = score > 0.7 ? '#00d4aa' : (score > 0.4 ? '#34d399' : '#6ee7b7');
            arrow = '▲';
            arrowColor = '#00d4aa';
            impactLabel = f.impact === 'High' ? 'Reduces Risk ✅' : 'Favorable ↘';
        }

        list.insertAdjacentHTML('beforeend', `
            <div class="factor">
                <div class="factor-header">
                    <span>${f.name}
                        <span style="color:${arrowColor};font-size:.75em">${arrow}</span>
                        <span style="color:var(--text-secondary);font-size:.7em;margin-left:4px">${f.value || ''}</span>
                    </span>
                    <span style="color:${barColor};font-size:.8em">${impactLabel}</span>
                </div>
                <div class="factor-bar-bg">
                    <div class="factor-bar-fill"
                         style="width:0%;background:${barColor}"
                         data-target="${score * 100}%"></div>
                </div>
            </div>
        `);
    });

    setTimeout(() => {
        list.querySelectorAll('.factor-bar-fill').forEach(bar => {
            bar.style.width = bar.dataset.target;
        });
    }, 100);
}

function renderAdvisory() {
    document.getElementById('advisory-text').textContent = currentAdvisory[currentLang];
}
document.getElementById('lang-toggle').addEventListener('click', e => {
    currentLang = currentLang === 'en' ? 'hi' : 'en';
    e.target.textContent = currentLang === 'hi' ? 'Switch to English' : 'Switch to Hindi';
    renderAdvisory();
});

function updateNearbyStats(s) {
    document.getElementById('near-depth').textContent = `${s.avg_depth_ft} ft`;
    document.getElementById('near-success').textContent = `${Math.round(s.estimated_success_rate * 100)}%`;
    
    const extPct = s.extraction_stage_pct;
    const extEl = document.getElementById('near-extract');
    extEl.textContent = `${extPct.toFixed(0)}%`;
    
    // Color-code based on actual value
    if (extPct > 100) {
        extEl.style.color = 'var(--color-crit)';
    } else if (extPct > 70) {
        extEl.style.color = 'var(--color-mod)';
    } else {
        extEl.style.color = 'var(--color-low)';
    }

    const catEl = document.getElementById('near-extract-cat');
    if (extPct > 100) {
        catEl.textContent = 'Over-Exploited';
        catEl.style.color = 'var(--color-crit)';
    } else if (extPct > 70) {
        catEl.textContent = 'Semi-Critical';
        catEl.style.color = 'var(--color-mod)';
    } else {
        catEl.textContent = 'Safe Zone';
        catEl.style.color = 'var(--color-low)';
    }
}

// ================================================================
// Print
// ================================================================
els.printBtn.addEventListener('click', () => window.print());

// ================================================================
// Heatmap Toggle → GET /api/heatmap
// ================================================================
els.heatmapToggle.addEventListener('click', async () => {
    heatmapOn = !heatmapOn;
    els.heatmapToggle.textContent = heatmapOn ? 'Hide Heatmap' : 'Toggle Heatmap';

    if (heatmapOn) {
        if (!heatmapLoaded) {
            try {
                els.heatmapToggle.textContent = 'Loading…';
                const res = await fetch(API.heatmap);
                if (!res.ok) throw new Error('Heatmap request failed');
                const pts = await res.json();

                pts.forEach(pt => {
                    const s = pt.risk_score;
                    let c;
                    if (s > 75)      c = RISK_COLORS.crit;
                    else if (s > 50) c = RISK_COLORS.high;
                    else if (s > 30) c = RISK_COLORS.mod;
                    else             c = RISK_COLORS.low;

                    const circle = L.circleMarker([pt.lat, pt.lng], {
                        radius: 12,
                        fillColor: c,
                        color: 'transparent',
                        fillOpacity: 0.35,
                    }).bindTooltip(
                        `${pt.mandal}: Risk ${s.toFixed(0)}%`,
                        { sticky: true }
                    );
                    heatmapCircles.push(circle);
                });

                heatmapLoaded = true;
                els.heatmapToggle.textContent = 'Hide Heatmap';
            } catch (err) {
                console.error('Heatmap error:', err);
                els.heatmapToggle.textContent = 'Toggle Heatmap';
                heatmapOn = false;
                return;
            }
        }
        heatmapCircles.forEach(c => c.addTo(map));
    } else {
        heatmapCircles.forEach(c => map.removeLayer(c));
    }
});

// ================================================================
// Precision Mode
// ================================================================
function updatePrecisionCoords() {
    const center = map.getCenter();
    els.precLat.textContent = center.lat.toFixed(6);
    els.precLng.textContent = center.lng.toFixed(6);
}

els.precisionToggle.addEventListener('click', () => {
    precisionMode = !precisionMode;
    els.precisionToggle.classList.toggle('active', precisionMode);
    els.crosshair.classList.toggle('hidden', !precisionMode);
    els.precisionPanel.classList.toggle('hidden', !precisionMode);
    
    if (precisionMode) {
        map.setZoom(Math.max(map.getZoom(), 16));
        updatePrecisionCoords();
        map.on('move', updatePrecisionCoords);
    } else {
        map.off('move', updatePrecisionCoords);
    }
});

els.useLocationBtn.addEventListener('click', () => {
    const center = map.getCenter();
    placeMarker(center);
    triggerPrediction(center.lat, center.lng);
});

// ================================================================
// Boot
// ================================================================
document.addEventListener('DOMContentLoaded', initMap);
