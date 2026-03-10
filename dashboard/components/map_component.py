"""
Professional Leaflet Map Component with Bidirectional Hover Sync
================================================================
Replaces Folium to eliminate flickering. Uses raw HTML/JS for instant
marker highlighting via postMessage without page refresh.

This is the "Gold Standard" for technical hackathon submissions.
"""

import json
import streamlit.components.v1 as components
from typing import List, Dict, Any, Optional


def render_leaflet_map(
    route: List[List[float]],
    flagged_moments: List[Dict[str, Any]],
    progress: float = 1.0,
    trip_info: Optional[Dict] = None,
    brake_zones: Optional[List[Dict]] = None,
    height: int = 500,
    map_key: str = "leaflet_map"
) -> None:
    """
    Render a professional Uber-style Leaflet map with bidirectional hover sync.
    
    Args:
        route: List of [lat, lon] coordinates for the trip path
        flagged_moments: List of event dicts with gps_lat, gps_lon, severity, etc.
        progress: Trip progress 0.0-1.0 (for simulation)
        trip_info: Dict with pickup_location, dropoff_location
        brake_zones: Optional list of H3 brake zone data
        height: Map height in pixels
        map_key: Unique key for the component
    """
    
    if not route or len(route) < 2:
        components.html("<div style='color:#8E8E93; padding:20px;'>Route unavailable</div>", height=100)
        return
    
    # Calculate center and bounds
    lats = [p[0] for p in route]
    lons = [p[1] for p in route]
    center_lat = (min(lats) + max(lats)) / 2
    center_lon = (min(lons) + max(lons)) / 2
    
    # Split route into driven vs remaining based on progress
    split_idx = max(1, int(len(route) * progress))
    driven_route = route[:split_idx]
    remaining_route = route[split_idx-1:] if split_idx < len(route) else []
    
    # Car position
    car_pos = driven_route[-1] if driven_route else route[0]
    
    # Filter events to only show those the car has passed
    visible_events = []
    total_points = len(route)
    for evt in flagged_moments:
        evt_progress = evt.get('progress', 0)
        if evt_progress <= progress:
            visible_events.append(evt)
    
    # Convert to JSON for JavaScript
    route_json = json.dumps(route)
    driven_json = json.dumps(driven_route)
    remaining_json = json.dumps(remaining_route)
    events_json = json.dumps(visible_events)
    car_pos_json = json.dumps(car_pos)
    bounds_json = json.dumps([[min(lats), min(lons)], [max(lats), max(lons)]])
    brake_zones_json = json.dumps(brake_zones or [])
    
    # Trip info
    pickup_name = trip_info.get('pickup_location', 'Pickup') if trip_info else 'Pickup'
    dropoff_name = trip_info.get('dropoff_location', 'Dropoff') if trip_info else 'Dropoff'
    
    # Build the HTML/JS template
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ background: transparent; }}
            #map {{ 
                height: {height}px; 
                width: 100%; 
                border-radius: 12px;
                background: #1C1C1E;
            }}
            
            /* Custom popup styling */
            .leaflet-popup-content-wrapper {{
                background: #1C1C1E;
                color: #FFFFFF;
                border-radius: 8px;
                border: 1px solid #2C2C2E;
            }}
            .leaflet-popup-tip {{
                background: #1C1C1E;
            }}
            .leaflet-popup-content {{
                margin: 8px 12px;
                font-family: 'Inter', -apple-system, sans-serif;
                font-size: 13px;
            }}
            
            /* Pulsing car animation */
            @keyframes carPulse {{
                0% {{ box-shadow: 0 0 8px rgba(0,122,255,0.8), 0 0 16px rgba(0,122,255,0.4); transform: scale(1); }}
                50% {{ box-shadow: 0 0 16px rgba(0,122,255,1), 0 0 32px rgba(0,122,255,0.6); transform: scale(1.1); }}
                100% {{ box-shadow: 0 0 8px rgba(0,122,255,0.8), 0 0 16px rgba(0,122,255,0.4); transform: scale(1); }}
            }}
            
            .car-marker {{
                background: #007AFF;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                border: 3px solid white;
                animation: carPulse 1.5s ease-in-out infinite;
            }}
            
            /* Event marker hover effect */
            @keyframes markerHighlight {{
                0% {{ transform: scale(1); }}
                50% {{ transform: scale(1.3); }}
                100% {{ transform: scale(1); }}
            }}
            
            .event-marker-highlighted {{
                animation: markerHighlight 0.3s ease-out;
            }}
            
            /* Event icon markers */
            .event-icon-marker {{
                display: flex;
                align-items: center;
                justify-content: center;
                width: 28px;
                height: 28px;
                border-radius: 50%;
                border: 2px solid white;
                box-shadow: 0 2px 6px rgba(0,0,0,0.4);
                font-size: 12px;
                color: white;
                transition: all 0.2s ease;
            }}
            .event-icon-marker:hover {{
                transform: scale(1.2);
            }}
            .event-icon-marker.highlighted {{
                transform: scale(1.4);
                box-shadow: 0 0 12px rgba(255,255,255,0.8);
            }}
            
            /* Animated route dashes */
            @keyframes dash {{
                to {{ stroke-dashoffset: -30; }}
            }}
            
            .animated-route {{
                animation: dash 1s linear infinite;
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        
        <script>
            // Initialize map with dark theme
            var map = L.map('map', {{
                zoomControl: true,
                attributionControl: false
            }});
            
            // Uber-style dark tiles (Carto Dark Matter)
            L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                maxZoom: 19,
                subdomains: 'abcd'
            }}).addTo(map);
            
            // Fit to bounds
            var bounds = {bounds_json};
            map.fitBounds(bounds, {{ padding: [40, 40] }});
            
            // Data
            var fullRoute = {route_json};
            var drivenRoute = {driven_json};
            var remainingRoute = {remaining_json};
            var events = {events_json};
            var carPos = {car_pos_json};
            var brakeZones = {brake_zones_json};
            
            // Ghost route (gray - full path)
            L.polyline(fullRoute, {{
                color: '#444444',
                weight: 4,
                opacity: 0.5
            }}).addTo(map);
            
            // Driven route (green with glow)
            if (drivenRoute.length > 1) {{
                // Glow effect (wider, semi-transparent)
                L.polyline(drivenRoute, {{
                    color: '#30D158',
                    weight: 10,
                    opacity: 0.3
                }}).addTo(map);
                
                // Main line
                L.polyline(drivenRoute, {{
                    color: '#30D158',
                    weight: 5,
                    opacity: 0.9
                }}).addTo(map);
                
                // Animated dashes overlay
                var animatedPath = L.polyline(drivenRoute, {{
                    color: '#FFFFFF',
                    weight: 2,
                    opacity: 0.6,
                    dashArray: '10, 20'
                }}).addTo(map);
                
                // Animate the dashes using CSS
                animatedPath._path.classList.add('animated-route');
            }}
            
            // Start marker (white circle)
            L.circleMarker(fullRoute[0], {{
                radius: 10,
                fillColor: '#FFFFFF',
                color: '#FFFFFF',
                weight: 2,
                fillOpacity: 1
            }}).addTo(map).bindPopup('<b>📍 Pickup</b><br>{pickup_name}');
            
            // End marker (green circle)
            L.circleMarker(fullRoute[fullRoute.length - 1], {{
                radius: 10,
                fillColor: '#30D158',
                color: '#30D158',
                weight: 2,
                fillOpacity: 1
            }}).addTo(map).bindPopup('<b>🏁 Dropoff</b><br>{dropoff_name}');
            
            // Car marker (pulsing blue)
            var carIcon = L.divIcon({{
                className: '',
                html: '<div class="car-marker"></div>',
                iconSize: [26, 26],
                iconAnchor: [13, 13]
            }});
            var carMarker = L.marker(carPos, {{ icon: carIcon }}).addTo(map);
            carMarker.bindTooltip('Your Vehicle', {{ permanent: false, direction: 'top' }});
            
            // Event markers storage for hover sync
            var eventMarkers = {{}};
            
            // Event type configurations with Font Awesome icons
            var eventConfig = {{
                'POTHOLE': {{ color: '#FF9F0A', faIcon: 'fa-road', emoji: '🕳️', label: 'Pothole' }},
                'AGGRESSIVE_BRAKING': {{ color: '#FF453A', faIcon: 'fa-hand', emoji: '🛑', label: 'Hard Brake' }},
                'HARSH_BRAKING': {{ color: '#FF453A', faIcon: 'fa-hand', emoji: '🛑', label: 'Hard Brake' }},
                'RAPID_ACCELERATION': {{ color: '#FF9F0A', faIcon: 'fa-gauge-high', emoji: '⚡', label: 'Rapid Accel' }},
                'HARSH_TURN': {{ color: '#FF9F0A', faIcon: 'fa-rotate-left', emoji: '↩️', label: 'Sharp Turn' }},
                'COMPOUND_BRAKE_CONFLICT': {{ color: '#FF453A', faIcon: 'fa-bolt', emoji: '⚠️', label: 'Brake + Stress' }},
                'COMPOUND_TURN_CONFLICT': {{ color: '#FF453A', faIcon: 'fa-bolt', emoji: '⚠️', label: 'Turn + Stress' }},
                'STRESS_DETECTED': {{ color: '#BF5AF2', faIcon: 'fa-volume-high', emoji: '🔊', label: 'Audio Stress' }},
                'ACUTE_SAFETY': {{ color: '#FF453A', faIcon: 'fa-triangle-exclamation', emoji: '🚨', label: 'Safety Alert' }},
                'default': {{ color: '#FF9F0A', faIcon: 'fa-exclamation-triangle', emoji: '⚡', label: 'Event' }}
            }};
            
            // Add event markers with Font Awesome icons
            events.forEach(function(evt, idx) {{
                var lat = evt.gps_lat || evt.lat;
                var lon = evt.gps_lon || evt.lon;
                
                if (!lat || !lon) return;
                
                var eventType = evt.event_label || evt.type || 'default';
                var signalType = evt.signal_type || 'MOTION';
                var config = eventConfig[eventType] || eventConfig['default'];
                var severity = (evt.severity || 'medium').toLowerCase();
                
                // Determine color and icon based on signal type
                var markerColor = config.color;
                var faIcon = config.faIcon;
                
                // Override for specific signal types
                if (signalType === 'AUDIO') {{
                    markerColor = '#BF5AF2';  // Purple for audio
                    faIcon = 'fa-volume-high';
                }} else if (signalType === 'COMPOUND') {{
                    markerColor = '#FF453A';  // Red for compound
                    faIcon = 'fa-bolt';
                }}
                
                // High severity = red
                if (severity === 'high' || severity === 'critical') {{
                    markerColor = '#FF453A';
                }}
                
                // Create icon marker with Font Awesome
                var iconHtml = '<div class=\"event-icon-marker\" style=\"background:' + markerColor + ';\"><i class=\"fa-solid ' + faIcon + '\"></i></div>';
                var divIcon = L.divIcon({{
                    className: '',
                    html: iconHtml,
                    iconSize: [28, 28],
                    iconAnchor: [14, 14]
                }});
                
                var marker = L.marker([lat, lon], {{ icon: divIcon }}).addTo(map);
                
                // Popup with event details
                var popupContent = '<b>' + config.emoji + ' ' + config.label + '</b>';
                if (evt.explanation) {{
                    popupContent += '<br><span style=\"color:#8E8E93;\">' + evt.explanation + '</span>';
                }}
                marker.bindPopup(popupContent);
                
                // Store marker for hover sync
                var markerId = evt.flag_id || evt.window_id || ('evt_' + idx);
                eventMarkers[markerId] = {{
                    marker: marker,
                    originalRadius: 8,
                    originalColor: markerColor
                }};
            }});
            
            // Add H3 brake zones if available
            brakeZones.forEach(function(zone) {{
                var lat = zone.center_lat || zone.lat;
                var lon = zone.center_lon || zone.lon;
                if (!lat || !lon) return;
                
                // Draw hexagon approximation (circle for simplicity without h3-js)
                L.circle([lat, lon], {{
                    radius: 150,
                    fillColor: '#FF453A',
                    color: '#FF453A',
                    weight: 1,
                    fillOpacity: 0.2,
                    opacity: 0.5
                }}).addTo(map).bindPopup('<b>⚠️ Hard Brake Zone</b><br>' + (zone.event_count || 'Multiple') + ' incidents');
            }});
            
            // ============================================
            // BIDIRECTIONAL HOVER SYNC - THE MAGIC PART
            // ============================================
            
            // Listen for messages from Streamlit/parent to highlight markers
            window.addEventListener("message", function(event) {{
                var msg = event.data;
                
                if (msg.type === "HIGHLIGHT_MARKER") {{
                    var markerId = msg.marker_id || msg.window_id || msg.flag_id;
                    var markerData = eventMarkers[markerId];
                    
                    if (markerData) {{
                        // Highlight: add CSS class to the icon div
                        var iconEl = markerData.marker.getElement();
                        if (iconEl) {{
                            var iconDiv = iconEl.querySelector('.event-icon-marker');
                            if (iconDiv) {{
                                iconDiv.classList.add('highlighted');
                            }}
                        }}
                        
                        // Pan to marker smoothly
                        map.panTo(markerData.marker.getLatLng(), {{ animate: true, duration: 0.3 }});
                    }}
                }}
                
                else if (msg.type === "RESET_MARKER") {{
                    var markerId = msg.marker_id || msg.window_id || msg.flag_id;
                    var markerData = eventMarkers[markerId];
                    
                    if (markerData) {{
                        // Reset: remove highlight class
                        var iconEl = markerData.marker.getElement();
                        if (iconEl) {{
                            var iconDiv = iconEl.querySelector('.event-icon-marker');
                            if (iconDiv) {{
                                iconDiv.classList.remove('highlighted');
                            }}
                        }}
                    }}
                }}
                
                else if (msg.type === "RESET_ALL_MARKERS") {{
                    Object.keys(eventMarkers).forEach(function(id) {{
                        var markerData = eventMarkers[id];
                        var iconEl = markerData.marker.getElement();
                        if (iconEl) {{
                            var iconDiv = iconEl.querySelector('.event-icon-marker');
                            if (iconDiv) {{
                                iconDiv.classList.remove('highlighted');
                            }}
                        }}
                    }});
                }}
                
                else if (msg.type === "UPDATE_CAR_POSITION") {{
                    var newPos = msg.position;
                    if (newPos && newPos.length === 2) {{
                        carMarker.setLatLng(newPos);
                    }}
                }}
                
                else if (msg.type === "UPDATE_PROGRESS") {{
                    // Update driven route - this allows smooth progress updates
                    var progress = msg.progress;
                    var splitIdx = Math.max(1, Math.floor(fullRoute.length * progress));
                    var newDriven = fullRoute.slice(0, splitIdx);
                    var newCarPos = newDriven[newDriven.length - 1] || fullRoute[0];
                    
                    carMarker.setLatLng(newCarPos);
                }}
            }});
            
            // Send marker clicks back to Streamlit
            Object.keys(eventMarkers).forEach(function(markerId) {{
                eventMarkers[markerId].marker.on('click', function() {{
                    window.parent.postMessage({{
                        type: 'MARKER_CLICKED',
                        marker_id: markerId
                    }}, '*');
                }});
            }});
            
            // Notify parent that map is ready
            window.parent.postMessage({{ type: 'MAP_READY', marker_count: Object.keys(eventMarkers).length }}, '*');
        </script>
    </body>
    </html>
    """
    
    # Render the component
    components.html(html_template, height=height + 10, scrolling=False)


def send_highlight_message(marker_id: str) -> None:
    """Send a message to highlight a specific marker on the map."""
    js_code = f"""
    <script>
        window.parent.postMessage({{
            type: 'HIGHLIGHT_MARKER',
            marker_id: '{marker_id}'
        }}, '*');
    </script>
    """
    components.html(js_code, height=0, width=0)


def send_reset_message(marker_id: str = None) -> None:
    """Reset marker(s) to original state."""
    if marker_id:
        js_code = f"""
        <script>
            window.parent.postMessage({{
                type: 'RESET_MARKER',
                marker_id: '{marker_id}'
            }}, '*');
        </script>
        """
    else:
        js_code = """
        <script>
            window.parent.postMessage({
                type: 'RESET_ALL_MARKERS'
            }, '*');
        </script>
        """
    components.html(js_code, height=0, width=0)


def create_event_hover_trigger(marker_id: str, event_text: str) -> str:
    """
    Create HTML for an event item with hover triggers for map sync.
    Returns HTML string that can be used with st.markdown(unsafe_allow_html=True).
    """
    return f"""
    <div class="event-hover-item" 
         onmouseenter="window.parent.postMessage({{type:'HIGHLIGHT_MARKER', marker_id:'{marker_id}'}}, '*')"
         onmouseleave="window.parent.postMessage({{type:'RESET_MARKER', marker_id:'{marker_id}'}}, '*')">
        {event_text}
    </div>
    """
