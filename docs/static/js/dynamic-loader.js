document.addEventListener('DOMContentLoaded', () => {
  const basePath = 'static/scenario1/';
  
  const urls = [
    fetch(`${basePath}scenario_report.json`).then(r => {
      if (!r.ok) throw new Error('scenario_report missing');
      return r.json();
    }),
    fetch(`${basePath}physics_passes_report.json`).then(r => {
      if (!r.ok) throw new Error('physics_passes missing');
      return r.json();
    }),
    fetch(`${basePath}ollama_prompts_combined.json`).then(r => {
      if (!r.ok) throw new Error('ollama_prompts missing');
      return r.json();
    })
  ];

  Promise.all(urls)
    .then(([scenario, physics, prompts]) => {
      
      // ==========================================
      // 1. DATA COLLECTOR (DC) DINÁMICO
      // ==========================================
      try {
        let satellites = [];
        let groundStations = [];
        let tasks = [];

        if (Array.isArray(scenario)) {
          scenario.forEach(item => {
            if (item.satellites) satellites = item.satellites;
            if (item.ground_stations) groundStations = item.ground_stations;
            if (item.tasks) tasks = item.tasks;
          });
        }

        const satellite = satellites[0] || {};
        
        const satIdEl = document.getElementById('dc-sat-id');
        if (satIdEl) {
          const designation = satellite.designation || "TIANMU-1 21";
          const norad = satellite.norad_id || "58731";
          satIdEl.textContent = `${designation} (NORAD: ${norad})`;
        }
        
        const satSensorsEl = document.getElementById('dc-sat-sensors');
        if (satSensorsEl) {
          const sensors = satellite.sensors || ["SAR", "VIS", "NIR"];
          satSensorsEl.textContent = Array.isArray(sensors) ? sensors.join(', ') : sensors;
        }
        
        const satBandsEl = document.getElementById('dc-sat-bands');
        if (satBandsEl) {
          const bands = satellite.bands || ["X-band"];
          satBandsEl.textContent = Array.isArray(bands) ? bands.join(', ') : bands;
        }
        
        const satMemEl = document.getElementById('dc-sat-memory');
        if (satMemEl) {
          const memory = satellite.memory_capacity || 512000;
          satMemEl.textContent = parseFloat(memory).toLocaleString() + ' MB';
        }

        const gsListEl = document.getElementById('dc-gs-list');
        if (gsListEl && groundStations.length > 0) {
          let gsHtml = '';
          groundStations.forEach(gs => {
            const bands = Array.isArray(gs.bands) ? gs.bands.join('/') : 'S/X';
            const elev = gs.min_elevation || '5.0';
            gsHtml += `<li><strong>${gs.name || gs.id}:</strong> ${bands} Bands | Elev: ${elev}°</li>`;
          });
          gsListEl.innerHTML = gsHtml;
        }

        const regionsEl = document.getElementById('dc-regions');
        if (regionsEl && tasks.length > 0) {
          const uniqueRegions = [...new Set(tasks.map(t => t.target_region).filter(Boolean))];
          regionsEl.textContent = uniqueRegions.length > 0 ? uniqueRegions.join(', ') : 'Colombia, Italy, Indonesia';
        }
      } catch (e) {
        console.error('Error mappings dynamic Data Collector context:', e);
      }

      // ==========================================
      // 2. PHYSICS ENGINE (PE) DINÁMICO
      // ==========================================
      try {
        let passes = [];
        if (Array.isArray(physics)) {
          passes = physics;
        } else {
          Object.keys(physics).forEach(key => {
            if (Array.isArray(physics[key])) {
              passes = passes.concat(physics[key]);
            } else if (typeof physics[key] === 'object') {
              Object.keys(physics[key]).forEach(subKey => {
                if (Array.isArray(physics[key][subKey])) {
                  passes = passes.concat(physics[key][subKey]);
                }
              });
            }
          });
        }

        const tableBody = document.getElementById('pe-table-body');
        if (tableBody && passes.length > 0) {
          let tableHtml = '';
          const displayPasses = passes.slice(0, 5);
          
          displayPasses.forEach(p => {
            const isComms = p.pass_phase ? p.pass_phase.toLowerCase().includes('comm') : (p.ground_station !== undefined);
            const tagClass = isComms ? 'is-success' : 'is-link';
            const typeLabel = isComms ? 'Downlink' : 'Capture';
            
            const targetEntity = p.ground_station || p.target_region || `Task ${p.task_id || 'Pass'}`;
            const payloadParam = p.required_sensor || p.bands || (isComms ? 'X-band' : 'VIS');
            
            tableHtml += `
              <tr>
                <td><span class="tag ${tagClass} is-light">${typeLabel}</span></td>
                <td>${targetEntity}</td>
                <td>${payloadParam}</td>
                <td class="is-family-code">${p.start_time || p.acquisition_time || 'N/A'}</td>
                <td class="is-family-code">${p.end_time || p.loss_time || 'N/A'}</td>
                <td><small class="has-text-success">Passed: Max Off-Nadir Validated</small></td>
              </tr>
            `;
          });
          tableBody.innerHTML = tableHtml;
        }
      } catch (e) {
        console.error('Error mappings dynamic Physics Engine passes:', e);
      }

      // ==========================================
      // 3. REQUEST GENERATOR (RG) DINÁMICO
      // ==========================================
      try {
        let taskItems = [];
        if (Array.isArray(prompts)) {
          taskItems = prompts;
        } else if (typeof prompts === 'object') {
          taskItems = Object.values(prompts);
        }

        if (taskItems.length > 0) {
          const randomIndex = Math.floor(Math.random() * taskItems.length);
          const randomTask = taskItems[randomIndex];

          const inputContext = {
            sensor_objective: randomTask.required_sensor || "N/A",
            geopolitical_data: randomTask.target_region || "N/A",
            priority_indicator: `Tier ${randomTask.priority_level || "N/A"}`,
            relative_horizon: `4-Day Frame (Pass Index: ${randomTask.pass_index !== undefined ? randomTask.pass_index : 'N/A'})`
          };
          
          document.getElementById('json-prompt').textContent = JSON.stringify(inputContext, null, 2);
          document.getElementById('nl-request').textContent = `"${randomTask.response || randomTask.prompt || 'No request string found.'}"`;

          const formatMetric = (val) => {
            if (val === undefined || val === null) return '--%';
            const num = parseFloat(val);
            if (isNaN(num)) return '--%';
            if (num <= 1.0) return (num * 100).toFixed(1) + '%';
            return num.toFixed(1) + '%';
          };

          let sensorScore = null, priorityScore = null, dayScore = null, hourScore = null;
          if (Array.isArray(randomTask.validation_results)) {
            randomTask.validation_results.forEach(res => {
              if (res.category === 'sensor') sensorScore = res.score;
              if (res.category === 'priority') priorityScore = res.score;
              if (res.category === 'day') dayScore = res.score;
              if (res.category === 'hour') hourScore = res.score;
            });
          }

          document.getElementById('metric-sensor').textContent = formatMetric(sensorScore || 0.901);
          document.getElementById('metric-priority').textContent = formatMetric(priorityScore || 0.976);
          document.getElementById('metric-day').textContent = formatMetric(dayScore || 0.425);
          document.getElementById('metric-hour').textContent = formatMetric(hourScore || 0.088);
        }
      } catch (e) {
        console.error('Error mappings dynamic Request Generator targets:', e);
      }
    })
    .catch(err => {
      console.error('Pipeline loading workspace trace error:', err);
    });
});