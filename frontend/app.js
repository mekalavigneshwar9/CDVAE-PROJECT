console.log("CDVAE App v1.1 - Layout Fixed");
const generateBtn = document.getElementById('generate-btn');
const downloadBtn = document.getElementById('download-btn');
const btnText = document.querySelector('.btn-text');
const btnLoader = document.getElementById('btn-loader');
const formulaVal = document.getElementById('formula-val');
const elementsVal = document.getElementById('elements-val');
const plotContainer = document.getElementById('plot-container');

const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

let currentCif = null;

// -------- COLORS --------
const elementColors = {
    'Ag': '#c0c0c0', 'O': '#ff4d4d', 'C': '#555555', 'N': '#0055ff',
    'S': '#ffff00', 'Pd': '#006994', 'As': '#bd80e3', 'Cu': '#c87a55',
    'Al': '#bfa6a6', 'Mo': '#54b2a9', 'Br': '#a62929', 'Se': '#ffa100'
};

function getElementColor(symbol) {
    return elementColors[symbol] || '#00cec9';
}

// -------- BUTTON CLICK --------
generateBtn.addEventListener('click', async () => {

    // loading state
    btnText.classList.add('hidden');
    btnLoader.classList.remove('hidden');
    generateBtn.disabled = true;

    formulaVal.textContent = "Generating...";
    elementsVal.textContent = "...";
    plotContainer.innerHTML = "<div class='plot-loader'>Generating novel crystal structure...</div>";



    try {
        console.log("Sending generation request...");
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({})
        });

        console.log("Response status:", response.status);

        // ---- HANDLE SERVER ERRORS ----
        if (!response.ok) {
            const text = await response.text();
            console.error("Server error:", text);
            alert("API error: " + text);
            return;
        }

        const data = await response.json();
        console.log("API data:", data);

        // ---- HANDLE API LOGIC ERRORS ----
        if (data.error) {
            alert(data.error);
            return;
        }

        // ---- VALIDATE RESPONSE ----
        if (!data.coordinates || !data.symbols) {
            alert("Invalid data received from backend");
            return;
        }

        // -------- UPDATE UI --------
        formulaVal.textContent = data.formula || "N/A";
        elementsVal.textContent = (data.elements || []).join(', ');
        currentCif = data.cif || null;



        if (currentCif) {
            downloadBtn.classList.remove('hidden');
        }

        renderPlot(data);

    } catch (err) {
        console.error("Fetch error:", err);
        alert("Backend not responding or network error");
    } finally {
        resetBtn();   // ALWAYS reset UI
    }
});

// -------- RESET BUTTON --------
function resetBtn() {
    btnText.classList.remove('hidden');
    btnLoader.classList.add('hidden');
    generateBtn.disabled = false;
}

// -------- PLOT --------
function renderPlot(data) {
    const coords = data.coordinates;
    const symbols = data.symbols;
    const bonds = data.bonds || [];

    plotContainer.innerHTML = "";

    if (!coords.length || !symbols.length) {
        plotContainer.innerHTML = "<p style='color:white'>No atoms generated</p>";
        return;
    }

    const traces = [];
    const elementMap = {};

    // group atoms
    symbols.forEach((sym, i) => {
        if (!elementMap[sym]) {
            elementMap[sym] = { x: [], y: [], z: [], text: [], color: getElementColor(sym) };
        }

        elementMap[sym].x.push(coords[i][0]);
        elementMap[sym].y.push(coords[i][1]);
        elementMap[sym].z.push(coords[i][2]);
        elementMap[sym].text.push(sym);
    });

    // atoms
    for (const [sym, el] of Object.entries(elementMap)) {
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            name: sym,
            x: el.x,
            y: el.y,
            z: el.z,
            text: el.text,
            hovertemplate: '%{text}<extra></extra>',
            marker: {
                size: Math.max(6, 18 - symbols.length / 3),
                color: el.color,
                opacity: 0.9
            }
        });
    }

    // bonds
    bonds.forEach(bond => {
        if (!coords[bond[0]] || !coords[bond[1]]) return;

        const a = coords[bond[0]];
        const b = coords[bond[1]];

        traces.push({
            type: 'scatter3d',
            mode: 'lines',
            x: [a[0], b[0]],
            y: [a[1], b[1]],
            z: [a[2], b[2]],
            line: { width: 2, color: 'white' },
            hoverinfo: 'none',
            showlegend: false
        });
    });

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 0, r: 0, t: 0, b: 0 },
        autosize: true,
        showlegend: true,
        legend: {
            orientation: 'h',
            yanchor: 'bottom',
            y: 0.02,
            xanchor: 'center',
            x: 0.5,
            font: { color: '#a0a5b8', size: 11 },
            bgcolor: 'rgba(0,0,0,0.3)'
        },
        scene: {
            xaxis: { visible: false },
            yaxis: { visible: false },
            zaxis: { visible: false },
            aspectmode: 'cube',
            camera: {
                eye: { x: 1.2, y: 1.2, z: 1.2 }
            }
        }
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };

    Plotly.newPlot('plot-container', traces, layout, config);
}

// Ensure the plot resizes when the window does
window.addEventListener('resize', () => {
    if (plotContainer.innerHTML !== "") {
        Plotly.Plots.resize(plotContainer);
    }
});

// -------- DOWNLOAD CIF --------
downloadBtn.addEventListener('click', () => {
    if (!currentCif) {
        alert("No CIF available");
        return;
    }

    const blob = new Blob([currentCif], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = "crystal.cif";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    URL.revokeObjectURL(url);
});

// -------- HEALTH CHECK --------
async function checkBackend() {
    try {
        const response = await fetch('/api/health');
        if (response.ok) {
            const data = await response.json();
            statusDot.classList.add('online');
            statusText.textContent = `Online (${data.device})`;
        } else {
            throw new Error("Offline");
        }
    } catch (err) {
        statusDot.classList.remove('online');
        statusText.textContent = "Offline - Run app.py";
    }
}

// Initial check
checkBackend();
// Re-check every 30 seconds
setInterval(checkBackend, 30000);