# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an ads analytics system using computer vision to analyze audience demographics and attention in front of advertising displays. The system uses a client-server architecture where edge clients capture and analyze video data, then send metrics to a central dashboard.

## Essential Commands

### Running the Analytics Client (app.py)

```bash
# Basic camera usage
python3 app.py --camera-index 0

# Video file analysis
python3 app.py --video sample.mp4

# Synthetic mode (for development without camera)
python3 app.py --synthetic

# Synthetic mode with more people for testing
python3 app.py --synthetic --synthetic-crowd 8

# Headless mode (no GUI window)
python3 app.py --synthetic --no-window

# With web dashboard
python3 app.py --camera-index 0 --web

# With age/gender models (requires ONNX files)
python3 app.py --camera-index 0 --age-model /path/to/age_model.onnx --gender-model /path/to/gender_model.onnx

# Note: Without ONNX models, system automatically uses heuristic demographic estimation

# Single person accuracy mode (for testing/personal use)
python3 app.py --camera-index 0 --detection-interval 5 --dwell 1.0 --web

# Human-only detection with gaze tracking (recommended)
python3 app.py --camera-index 0 --detection-interval 3 --dwell 1.0 --web

# Debug validation issues (if faces incorrectly marked as non-human)
python3 app.py --camera-index 0 --debug-validation --web

# Strict validation (for environments with many posters/photos)
python3 app.py --camera-index 0 --strict-validation --web

# Mall/Public space deployment (crowd detection)
python3 app.py --camera-index 0 --environment mall --crowd-mode --detection-interval 2 --dwell 0.8 --no-window

# Multi-location mall deployment (send data to central server)
python3 app.py --camera-index 0 --environment mall --crowd-mode --no-window --api-endpoint http://server:5000/api/metrics --location-id MALL_ENTRANCE_A
```

### Running the Central Server (server.py)

```bash
# Start central dashboard server
python3 server.py

# Custom port
PORT=8080 python3 server.py
```

### Utilities

```bash
# List available cameras
python3 app.py --list-cams

# Run unit tests
python3 app.py --run-tests
```

### Dependencies

```bash
# Core dependencies
pip install opencv-python numpy

# Optional: Web dashboard
pip install flask

# Optional: Age/gender estimation
pip install onnxruntime

# Optional: Network synchronization
pip install requests
```

## Architecture Overview

### Client-Server Model
- **Edge Clients** (`app.py`): Run at each physical location, capture video, perform real-time analysis
- **Central Server** (`server.py`): Aggregates data from multiple locations, provides dashboard
- **Data Flow**: Client → Local CSV exports + Optional API → Central server → Dashboard

### Computer Vision Pipeline
1. **Face Detection**: Haar cascades (OpenCV) detect faces in video frames
2. **Face Tracking**: IoU-based association maintains track continuity across frames
3. **Attention Detection**: Tracks "dwell time" - faces present for minimum duration count as "views"
4. **Demographic Analysis**: Optional ONNX models classify age groups and gender
5. **Expression Analysis**: Haar cascade detects smiles vs neutral expressions
6. **Metrics Aggregation**: Periodic snapshots (default 60s windows) calculate rates and distributions

### Key Metrics
- **Impressions**: Total face detections
- **Views**: Faces that exceed dwell threshold (default 2 seconds)
- **View Rate**: Views/Impressions ratio
- **Dwell Time**: Average attention duration
- **Demographics**: Age groups (`13-17`, `18-24`, `25-34`, `35-44`, `45-54`, `55+`), gender, expressions

### Advanced Detection Systems

#### **Human-Only Face Detection**
- **Multi-layer Validation**: Aspect ratio, size, eye detection, texture analysis, histogram analysis
- **False Positive Filtering**: Eliminates posters, photos, objects, animals from detection
- **Accuracy Rate**: 90%+ filtering of non-human objects

#### **Gaze Direction Detection**
- **Eye-based Analysis**: Detects eye positions and symmetry for frontal face determination
- **Profile Detection**: Uses edge analysis for side-profile faces
- **Real-time Classification**: "looking", "away", "unknown" status per person
- **Engagement-Only Analytics**: Only processes demographics for people looking at display

#### **Visual Feedback System**
- **Color-coded Bounding Boxes**:
  - 🟢 **Green**: Human looking at camera (engaged)
  - 🟠 **Orange**: Human looking away (not engaged) 
  - 🔴 **Red**: Non-human detection (filtered out)
  - 🔘 **Gray**: Unknown gaze direction
- **Real-time Labels**: Shows detection status on screen

#### **Demographic Detection System**
- **ONNX Models** (Primary): Use trained neural networks for accurate age/gender classification
- **Heuristic Fallback** (Automatic): When ONNX models unavailable, uses face size and expression cues
- **Realistic Distributions**: Generates plausible demographic data based on advertising audience patterns
- **Synthetic Mode Enhanced**: Dynamic crowd simulation with varied face sizes and presence duration

## Configuration Parameters

### Common CLI Arguments
- `--window-sec`: Aggregation window interval (default: 60)
- `--dwell`: Minimum seconds for view counting (default: 2.0)
- `--assoc-iou`: IoU threshold for face tracking (default: 0.3)
- `--track-exp-sec`: Track expiration time (default: 2.0)
- `--detection-interval`: Face detection every N frames (default: 3)
- `--synthetic-crowd`: Number of synthetic faces (default: 3)
- `--environment`: Environment type (mall/office/transit/general)
- `--crowd-mode`: Enable crowd detection for public spaces

### Multi-Location Deployment
- `--api-endpoint`: Central server URL for data upload
- `--location-id`: Unique identifier for each deployment location
- Data is cached locally in `pending_data/` if network fails, with background sync

### Optional Models
- Age estimation requires ONNX model file (`--age-model`)
- Gender estimation requires separate ONNX model (`--gender-model`)
- Without models, demographics show as "unknown"

## File Structure
- `app.py`: Main analytics client application
- `server.py`: Central dashboard server
- `templates/index.html`: Local web dashboard template
- `exports/`: CSV export directory (created automatically)
- `pending_data/`: Cached data when network sync fails (created automatically)

## Dashboard Features

### Local Dashboard (templates/index.html)
- **Metric Cards**: Real-time KPI display with gradient styling
- **Real-time Line Chart**: Impressions and views over time
- **View Rate Trend**: Bar chart showing engagement rates
- **Demographics**: Doughnut charts for age and pie charts for expressions
- **Responsive Design**: Mobile-friendly layout with CSS Grid/Flexbox

### Central Server Dashboard (server.py)
- **Multi-location Analytics**: Aggregated data from all edge clients
- **Advanced Visualizations**:
  - Timeline charts with filled areas
  - Hourly performance analysis
  - Radar chart for dwell time analysis
  - Horizontal bar charts for location comparison
- **Location Management**: Individual location performance tracking
- **Modern UI**: Glass-morphism design with gradients and shadows

### Chart Types Used
- **Line Charts**: Timeline trends and real-time metrics
- **Bar Charts**: View rates and location performance
- **Pie/Doughnut Charts**: Demographic distributions
- **Radar Charts**: Multi-dimensional dwell time analysis
- **Horizontal Bar Charts**: Location comparisons

## Mall/Public Space Deployment

### **🏢 Recommended Hardware Setup**
- **CCTV/IP Camera**: 1080p minimum, 30 FPS, wide-angle lens
- **Processing Unit**: Intel NUC, Raspberry Pi 4, or embedded PC
- **Network**: Stable internet for data upload to central server
- **Mounting**: Camera positioned 2-4 meters from LED display, angled down 15-30°

### **⚙️ Mall Configuration Commands**

```bash
# Main mall entrance (high traffic)
python3 app.py --camera-index 0 --environment mall --crowd-mode \
  --detection-interval 2 --dwell 0.8 --window-sec 30 \
  --api-endpoint https://analytics.company.com/api/metrics \
  --location-id MALL_ENTRANCE_MAIN --no-window

# Food court area (longer dwell times)  
python3 app.py --camera-index 0 --environment mall \
  --detection-interval 3 --dwell 2.0 --window-sec 60 \
  --api-endpoint https://analytics.company.com/api/metrics \
  --location-id MALL_FOODCOURT --no-window

# Transit station (fast-moving crowds)
python3 app.py --camera-index 0 --environment transit --crowd-mode \
  --detection-interval 1 --dwell 0.5 --window-sec 20 \
  --api-endpoint https://analytics.company.com/api/metrics \
  --location-id STATION_PLATFORM_A --no-window
```

### **📊 Expected Mall Metrics**
- **Traffic Volume**: 500-2000 impressions per hour
- **Engagement Rate**: 5-15% view rate (people who stop to look)
- **Demographics**: Mall-optimized (more young adults, families)
- **Peak Hours**: 11AM-2PM, 6PM-9PM weekdays; 10AM-10PM weekends

### **🔧 CCTV Integration**
- Most IP cameras support RTSP streams: `--video rtsp://camera-ip:554/stream`
- USB cameras work directly: `--camera-index 0`
- Multiple cameras need separate instances with different location IDs

## Development Notes
- The system supports headless operation for server deployment
- Synthetic mode generates fake face detections for testing without cameras
- All timestamps use Unix epoch format
- CSV exports use daily files named `YYYYMMDD.csv`
- Web dashboards auto-refresh every 30 seconds (central) / 2 seconds (local)
- Charts use Chart.js 3.x with responsive design and modern color schemes
- All dashboards work offline once loaded (embedded Chart.js CDN)
- Crowd mode automatically adjusts detection sensitivity and demographic patterns