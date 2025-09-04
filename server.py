import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
import threading
from collections import defaultdict

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# ====================================================================
# PENYIMPANAN DATA (IN-MEMORY & LOG)
# ====================================================================
LOG_FILE = "central_data.log"
g_data_store = []
g_lock = threading.Lock()

# ====================================================================
# TEMPLATE HTML UNTUK DASBOR GRAFIS
# ====================================================================
DASHBOARD_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>Central Dashboard - Ads Analytics</title>
    <meta http-equiv="refresh" content="30">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/date-fns@2.29.3/index.min.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #333; min-height: 100vh; }
        .header { 
            background: rgba(255,255,255,0.95); 
            padding: 20px; 
            border-bottom: 3px solid #667eea; 
            text-align: center; 
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .header h1 { margin: 0; color: #1a237e; font-size: 2.5em; font-weight: 300; }
        .header p { margin: 5px 0 0 0; color: #666; font-size: 1.1em; }
        .container { display: flex; flex-wrap: wrap; padding: 20px; gap: 20px; }
        .chart-container { 
            flex: 1 1 45%; 
            background: rgba(255,255,255,0.95); 
            border-radius: 15px; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.1); 
            padding: 25px; 
            min-width: 320px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .chart-container h3 { 
            margin: 0 0 20px 0; 
            color: #1a237e; 
            font-size: 1.3em; 
            font-weight: 500;
            text-align: center;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .summary-container { 
            flex: 1 1 100%; 
            background: rgba(255,255,255,0.95); 
            border-radius: 15px; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.1); 
            padding: 30px; 
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .summary-box { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px;
            border-radius: 12px;
            color: white;
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
            transform: translateY(0);
            transition: transform 0.3s ease;
        }
        .summary-box:hover {
            transform: translateY(-5px);
        }
        .summary-box h2 { margin: 0 0 10px 0; font-size: 1.1em; font-weight: 400; opacity: 0.9; }
        .summary-box p { margin: 0; font-size: 2.5em; font-weight: bold; }
        .location-stats {
            flex: 1 1 100%;
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            padding: 25px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .location-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            margin: 10px 0;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .location-name { font-weight: bold; color: #1a237e; }
        .location-metrics { display: flex; gap: 20px; }
        .location-metric { text-align: center; }
        .location-metric .value { font-weight: bold; font-size: 1.2em; color: #333; }
        .location-metric .label { font-size: 0.8em; color: #666; }
        canvas { max-height: 350px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Central Dashboard - Ads Analytics</h1>
        <p>Real-time Multi-location Audience Analytics</p>
    </div>
    
    <div class="container">
        <!-- Summary Cards -->
        <div class="summary-container">
            <h2>Overall Performance</h2>
            <div class="summary-grid">
                <div class="summary-box">
                    <h2>Total Impressions</h2>
                    <p>{{ totals.impressions }}</p>
                </div>
                <div class="summary-box">
                    <h2>Total Views</h2>
                    <p>{{ totals.views }}</p>
                </div>
                <div class="summary-box">
                    <h2>Average View Rate</h2>
                    <p>{{ "%.1f"|format(totals.view_rate * 100) }}%</p>
                </div>
                <div class="summary-box">
                    <h2>Active Locations</h2>
                    <p>{{ location_stats.active_count }}</p>
                </div>
            </div>
        </div>

        <!-- Timeline Chart -->
        <div class="chart-container">
            <h3>📈 Timeline - Impressions & Views</h3>
            <canvas id="timelineChart"></canvas>
        </div>

        <!-- Hourly Performance -->
        <div class="chart-container">
            <h3>⏰ Hourly Performance</h3>
            <canvas id="hourlyChart"></canvas>
        </div>

        <!-- Gender Distribution -->
        <div class="chart-container">
            <h3>👥 Gender Distribution</h3>
            <canvas id="genderChart"></canvas>
        </div>

        <!-- Age Distribution -->
        <div class="chart-container">
            <h3>🎂 Age Distribution</h3>
            <canvas id="ageChart"></canvas>
        </div>

        <!-- Dwell Time Analysis -->
        <div class="chart-container">
            <h3>⏱️ Dwell Time Analysis</h3>
            <canvas id="dwellChart"></canvas>
        </div>

        <!-- Location Performance -->
        <div class="chart-container">
            <h3>📍 Location Performance</h3>
            <canvas id="locationChart"></canvas>
        </div>

        <!-- Location Details -->
        <div class="location-stats">
            <h3>📊 Location Details</h3>
            {% for location in location_stats.locations %}
            <div class="location-item">
                <div class="location-name">{{ location.name }}</div>
                <div class="location-metrics">
                    <div class="location-metric">
                        <div class="value">{{ location.impressions }}</div>
                        <div class="label">Impressions</div>
                    </div>
                    <div class="location-metric">
                        <div class="value">{{ location.views }}</div>
                        <div class="label">Views</div>
                    </div>
                    <div class="location-metric">
                        <div class="value">{{ "%.1f"|format(location.view_rate * 100) }}%</div>
                        <div class="label">View Rate</div>
                    </div>
                    <div class="location-metric">
                        <div class="value">{{ location.last_seen }}</div>
                        <div class="label">Last Update</div>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>

    <script>
        // Prepare data
        const data = {
            timeline: {{ timeline | tojson }},
            gender_dist: {{ gender_dist | tojson }},
            age_dist: {{ age_dist | tojson }},
            totals: {{ totals | tojson }},
            hourly: {{ hourly_data | tojson }},
            dwell_data: {{ dwell_analysis | tojson }},
            location_data: {{ location_performance | tojson }}
        };

        // Timeline Chart
        const timelineCtx = document.getElementById('timelineChart').getContext('2d');
        new Chart(timelineCtx, {
            type: 'line',
            data: {
                labels: data.timeline.labels,
                datasets: [{
                    label: 'Impressions',
                    data: data.timeline.impressions,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true
                }, {
                    label: 'Views',
                    data: data.timeline.views,
                    borderColor: '#764ba2',
                    backgroundColor: 'rgba(118, 75, 162, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'top' }
                },
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });

        // Hourly Performance Chart
        const hourlyCtx = document.getElementById('hourlyChart').getContext('2d');
        new Chart(hourlyCtx, {
            type: 'bar',
            data: {
                labels: data.hourly.labels,
                datasets: [{
                    label: 'View Rate (%)',
                    data: data.hourly.view_rates,
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderColor: '#667eea',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { 
                        beginAtZero: true,
                        max: 100,
                        ticks: { callback: function(value) { return value + '%'; } }
                    }
                }
            }
        });

        // Gender Distribution
        const genderCtx = document.getElementById('genderChart').getContext('2d');
        new Chart(genderCtx, {
            type: 'doughnut',
            data: {
                labels: data.gender_dist.labels,
                datasets: [{
                    data: data.gender_dist.values,
                    backgroundColor: ['#4A90E2', '#F5A623', '#7ED321']
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });

        // Age Distribution
        const ageCtx = document.getElementById('ageChart').getContext('2d');
        new Chart(ageCtx, {
            type: 'bar',
            data: {
                labels: data.age_dist.labels,
                datasets: [{
                    label: 'Count',
                    data: data.age_dist.values,
                    backgroundColor: [
                        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'
                    ]
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });

        // Dwell Time Analysis
        const dwellCtx = document.getElementById('dwellChart').getContext('2d');
        new Chart(dwellCtx, {
            type: 'radar',
            data: {
                labels: data.dwell_data.labels,
                datasets: [{
                    label: 'Average Dwell Time (seconds)',
                    data: data.dwell_data.values,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    pointBackgroundColor: '#667eea'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    r: { beginAtZero: true }
                }
            }
        });

        // Location Performance
        const locationCtx = document.getElementById('locationChart').getContext('2d');
        new Chart(locationCtx, {
            type: 'horizontalBar',
            data: {
                labels: data.location_data.labels,
                datasets: [{
                    label: 'Total Views',
                    data: data.location_data.views,
                    backgroundColor: 'rgba(102, 126, 234, 0.8)'
                }, {
                    label: 'Total Impressions',
                    data: data.location_data.impressions,
                    backgroundColor: 'rgba(118, 75, 162, 0.8)'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: { beginAtZero: true }
                }
            }
        });
    </script>
</body>
</html>
"""

# ====================================================================
# ENDPOINTS API & LOGIKA DASBOR
# ====================================================================

def process_data_for_dashboard(data_entries):
    """Mengagregasi data mentah untuk ditampilkan di dasbor."""
    from datetime import datetime
    
    totals = {'impressions': 0, 'views': 0}
    gender_dist = {}
    age_dist = {}
    timeline = {'labels': [], 'impressions': [], 'views': []}
    hourly_data = {'labels': [], 'view_rates': []}
    location_data = defaultdict(lambda: {'impressions': 0, 'views': 0, 'last_seen': ''})
    dwell_times = []

    # Process each entry
    for entry in data_entries:
        metrics = entry['metrics']
        location_id = entry.get('location_id', 'Unknown')
        
        # Update totals
        impressions = metrics.get('impressions', 0)
        views = metrics.get('views', 0)
        totals['impressions'] += impressions
        totals['views'] += views

        # Update location data
        location_data[location_id]['impressions'] += impressions
        location_data[location_id]['views'] += views
        location_data[location_id]['last_seen'] = entry['received_at']

        # Aggregate distributions
        for key, val in metrics.get('gender_dist', {}).items():
            gender_dist[key] = gender_dist.get(key, 0) + val
        for key, val in metrics.get('age_dist', {}).items():
            age_dist[key] = age_dist.get(key, 0) + val
        
        # Timeline data
        timeline['labels'].append(entry['received_at'][-8:])  # Show time only
        timeline['impressions'].append(impressions)
        timeline['views'].append(views)
        
        # Dwell time data
        avg_dwell = metrics.get('avg_dwell', 0)
        if avg_dwell > 0:
            dwell_times.append(avg_dwell)

    # Calculate view rate
    totals['view_rate'] = (totals['views'] / totals['impressions']) if totals['impressions'] > 0 else 0

    # Process hourly data (last 24 entries)
    recent_entries = data_entries[-24:] if len(data_entries) > 24 else data_entries
    for i, entry in enumerate(recent_entries):
        metrics = entry['metrics']
        impr = metrics.get('impressions', 0)
        views = metrics.get('views', 0)
        rate = (views / impr * 100) if impr > 0 else 0
        hourly_data['labels'].append(f"Hour {i+1}")
        hourly_data['view_rates'].append(round(rate, 1))

    # Process location performance
    location_stats = {
        'active_count': len(location_data),
        'locations': []
    }
    
    location_performance = {
        'labels': [],
        'impressions': [],
        'views': []
    }

    for loc_id, data in location_data.items():
        view_rate = (data['views'] / data['impressions']) if data['impressions'] > 0 else 0
        location_stats['locations'].append({
            'name': loc_id,
            'impressions': data['impressions'],
            'views': data['views'],
            'view_rate': view_rate,
            'last_seen': data['last_seen'][-8:]  # Show time only
        })
        location_performance['labels'].append(loc_id)
        location_performance['impressions'].append(data['impressions'])
        location_performance['views'].append(data['views'])

    # Dwell time analysis (by age group or time periods)
    dwell_analysis = {
        'labels': ['Morning', 'Afternoon', 'Evening', 'Night'],
        'values': [
            sum(dwell_times[:len(dwell_times)//4]) / max(len(dwell_times)//4, 1),
            sum(dwell_times[len(dwell_times)//4:len(dwell_times)//2]) / max(len(dwell_times)//4, 1),
            sum(dwell_times[len(dwell_times)//2:3*len(dwell_times)//4]) / max(len(dwell_times)//4, 1),
            sum(dwell_times[3*len(dwell_times)//4:]) / max(len(dwell_times)//4, 1)
        ]
    }

    # Clean up data and handle unknown demographics gracefully
    processed_gender = {
        'labels': [k for k, v in gender_dist.items() if v > 0], 
        'values': [v for v in gender_dist.values() if v > 0]
    }
    processed_age = {
        'labels': [k for k, v in age_dist.items() if v > 0], 
        'values': [v for v in age_dist.values() if v > 0]
    }
    
    # If all demographics are unknown, show placeholder message
    if not processed_gender['labels'] or all(label == 'unknown' for label in processed_gender['labels']):
        processed_gender = {'labels': ['No Data'], 'values': [1]}
    if not processed_age['labels'] or all(label == 'unknown' for label in processed_age['labels']):
        processed_age = {'labels': ['No Data'], 'values': [1]}

    return {
        'totals': totals,
        'gender_dist': processed_gender,
        'age_dist': processed_age,
        'timeline': timeline,
        'hourly_data': hourly_data,
        'location_stats': location_stats,
        'location_performance': location_performance,
        'dwell_analysis': dwell_analysis
    }

@app.route('/')
def dashboard():
    """Menampilkan halaman dasbor utama dengan grafik."""
    with g_lock:
        data_copy = list(g_data_store)
    
    dashboard_data = process_data_for_dashboard(data_copy)
    return render_template_string(DASHBOARD_TEMPLATE, **dashboard_data)

@app.route('/api/metrics', methods=['POST'])
def receive_metrics():
    """Endpoint untuk menerima data metrik dari klien."""
    data = request.get_json()
    if not data or 'location_id' not in data or 'metrics' not in data:
        return jsonify({"status": "error", "message": "Invalid data format"}), 400

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {
        "received_at": timestamp,
        "location_id": data['location_id'],
        "metrics": data['metrics']
    }

    print(f"[{timestamp}] Data diterima dari Lokasi: {data['location_id']}")

    # Simpan ke log file dan memori
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(str(log_entry) + '\n')
    except IOError as e:
        print(f"[ERROR] Gagal menulis ke log file: {e}")

    with g_lock:
        g_data_store.append(log_entry)
        if len(g_data_store) > 1000: # Simpan 1000 entri data terakhir
            g_data_store.pop(0)

    return jsonify({"status": "success", "message": "Data received"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f" * Server berjalan di http://0.0.0.0:{port}")
    print(f" * Dasbor pemantauan tersedia di http://127.0.0.1:{port}/")
    app.run(host='0.0.0.0', port=port, debug=False)