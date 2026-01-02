import streamlit as st
import serial
import serial.tools.list_ports
import numpy as np
import pickle
import time
from datetime import datetime
from pathlib import Path
from virtual_serial_emulator import get_serial_port  # Virtual Arduino support


# Page configuration
st.set_page_config(
    page_title="Real-Time Crop Monitoring & Recommendation",
    layout="wide"
)


# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-optimal {
        color: #    ;
        font-weight: bold;
    }
    .status-low {
        color: #d32f2f;
        font-weight: bold;
    }
    .status-high {
        color: #f57c00;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Load crop requirements from pickle file
@st.cache_resource
def load_crop_requirements():
    """Load crop requirements database"""
    try:
        script_dir = Path(__file__).parent
        crop_req_path = script_dir / "crop_requirements.pkl"
        with open(crop_req_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("crop_requirements.pkl not found! Please generate it first.")
        return {}


# Load prediction model
@st.cache_resource
def load_prediction_model():
    """Load ML model for crop prediction"""
    try:
        script_dir = Path(__file__).parent
        model_path = script_dir / "Model.pkl"
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning("Model.pkl not found. Prediction feature disabled.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("If you see a dtype error, retrain your model with current scikit-learn version.")
        return None


# Get available serial ports
def get_serial_ports():
    """Get list of available COM ports"""
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]


# Read data from Arduino (real or virtual)
def read_arduino_data(port, baudrate=9600, use_virtual=False):
    """
    Read sensor data from Arduino (real or virtual)
    Expected format: N,P,K,temperature,humidity,ph,rainfall
    Example: 85.5,52.3,48.2,24.5,72.0,6.5,120.0
    """
    try:
        ser = get_serial_port(port, baudrate, timeout=1, use_virtual=use_virtual)
        time.sleep(2)

        if ser.in_waiting > 0:
            line = ser.readline().decode("utf-8").strip()
            data = line.split(",")

            if len(data) == 7:
                result = {
                    "N": float(data[0]),
                    "P": float(data[1]),
                    "K": float(data[2]),
                    "temperature": float(data[3]),
                    "humidity": float(data[4]),
                    "ph": float(data[5]),
                    "rainfall": float(data[6]),
                }
                ser.close()
                return result
        ser.close()
    except Exception as e:
        st.error(f"Arduino Error: {e}")
    return None


# Simulate sensor data for testing
def simulate_sensor_data():
    """Generate random sensor data for testing"""
    return {
        "N": np.random.uniform(20, 140),
        "P": np.random.uniform(20, 100),
        "K": np.random.uniform(10, 80),
        "temperature": np.random.uniform(15, 35),
        "humidity": np.random.uniform(50, 95),
        "ph": np.random.uniform(5.0, 8.0),
        "rainfall": np.random.uniform(40, 300),
    }


# Analyze parameters and provide recommendations
def analyze_parameters(data, crop_type, crop_db):
    """Analyze sensor data against optimal ranges"""
    if crop_type not in crop_db:
        return [], {}

    requirements = crop_db[crop_type]
    recommendations = []
    status = {}

    param_info = {
        "N": ("Nitrogen", "urea or ammonium nitrate fertilizer"),
        "P": ("Phosphorus", "DAP/SSP/rock phosphate fertilizer"),
        "K": ("Potassium", "muriate of potash (MOP) or SOP fertilizer"),
        "ph": ("pH Level", "lime (to increase) or sulfur (to decrease)"),
        "temperature": ("Temperature", "mulching, shading, or irrigation adjustment"),
        "humidity": ("Humidity", "irrigation or ventilation adjustment"),
        "rainfall": ("Rainfall/Irrigation", "increase irrigation or improve drainage"),
    }

    for param, value in data.items():
        if param in requirements and isinstance(requirements[param], tuple):
            min_val, max_val = requirements[param]
            param_name, remedy = param_info.get(param, (param, "adjustment"))

            if value < min_val:
                deficit = min_val - value
                status[param] = "low"

                unit = (
                    "°C"
                    if param == "temperature"
                    else ("%" if param == "humidity" else ("mm" if param == "rainfall" else ""))
                )

                recommendations.append(
                    {
                        "type": "warning",
                        "param": param_name,
                        "message": f"{param_name} is LOW: Current {value:.1f}{unit}, "
                                   f"Optimal {min_val:.1f}-{max_val:.1f}{unit}",
                        "action": f"Increase by approximately {deficit:.1f}{unit}. Apply {remedy}.",
                    }
                )

            elif value > max_val:
                excess = value - max_val
                status[param] = "high"

                unit = (
                    "°C"
                    if param == "temperature"
                    else ("%" if param == "humidity" else ("mm" if param == "rainfall" else ""))
                )

                recommendations.append(
                    {
                        "type": "warning",
                        "param": param_name,
                        "message": f"{param_name} is HIGH: Current {value:.1f}{unit}, "
                                   f"Optimal {min_val:.1f}-{max_val:.1f}{unit}",
                        "action": f"Reduce by approximately {excess:.1f}{unit}. {remedy}.",
                    }
                )
            else:
                status[param] = "optimal"

    if not recommendations:
        recommendations.append(
            {
                "type": "success",
                "param": "All Parameters",
                "message": f"All conditions are optimal for {crop_type.upper()}.",
                "action": "Crop conditions are good. Continue current practices.",
            }
        )

    return recommendations, status


# Find best matching crop based on current conditions
def find_best_crop(data, crop_db, top_n=3):
    """Find crops that best match current conditions"""
    scores = {}

    for crop, requirements in crop_db.items():
        score = 0
        total_params = 0

        for param, value in data.items():
            if param in requirements and isinstance(requirements[param], tuple):
                min_val, max_val = requirements[param]

                if min_val <= value <= max_val:
                    score += 100
                else:
                    if value < min_val:
                        distance = (min_val - value) / min_val if min_val > 0 else 0
                    else:
                        distance = (value - max_val) / max_val if max_val > 0 else 0

                    penalty = max(0, 100 - (distance * 100))
                    score += penalty

                total_params += 1

        if total_params > 0:
            scores[crop] = score / total_params

    sorted_crops = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_crops[:top_n]


def render_sidebar(crop_db):
    st.header("System Configuration")

    data_source = st.radio(
        "Data Source:",
        ["Manual Input", "Arduino Sensors", "Simulation Mode"],
        help="Choose how to input sensor data",
    )

    selected_port = None
    baudrate = 9600
    use_virtual_arduino = False

    if data_source == "Arduino Sensors":
        st.subheader("Arduino Setup")

        use_virtual_arduino = st.checkbox(
            "Use Virtual Arduino (No hardware needed)",
            value=False,
            help="Enable this to simulate Arduino without physical device",
        )

        ports = get_serial_ports()

        if ports or use_virtual_arduino:
            if use_virtual_arduino:
                selected_port = st.text_input("Virtual COM Port:", value="COM1")
            else:
                if "COM1" in ports:
                    default_index = ports.index("COM1")
                elif ports:
                    default_index = 0
                else:
                    default_index = 0
                selected_port = st.selectbox("COM Port:", ports, index=default_index) if ports else None

            baudrate = st.number_input("Baud Rate:", value=9600, step=1)
        else:
            st.warning("No serial ports detected.")
            selected_port = None
            use_virtual_arduino = False

        st.info(
            """
Arduino Data Format:
Send comma-separated values:
N,P,K,temp,humidity,ph,rainfall

Example:
85.5,52.3,48.2,24.5,72.0,6.5,120.0

Check "Use Virtual Arduino" for testing without hardware.
"""
        )

    st.divider()

    analysis_mode = st.radio(
        "Analysis Mode:",
        ["Specific Crop Analysis", "Best Crop Recommendation"],
        help="Choose analysis type",
    )

    selected_crop = None
    if analysis_mode == "Specific Crop Analysis" and crop_db:
        selected_crop = st.selectbox(
            "Select Crop:",
            sorted(crop_db.keys()),
            help="Analyze conditions for specific crop",
        )

        if selected_crop in crop_db:
            with st.expander("Crop Info"):
                req = crop_db[selected_crop]
                st.write(f"Samples: {req.get('sample_count', 'N/A')}")
                st.write(req.get("description", ""))

    st.divider()
    st.caption("Powered by ML and IoT")

    return data_source, analysis_mode, selected_crop, selected_port, baudrate, use_virtual_arduino


def render_manual_input():
    with st.form("manual_input_form"):
        input_col1, input_col2 = st.columns(2)

        with input_col1:
            st.markdown("Soil Nutrients")
            N = st.number_input("Nitrogen (N)", 0.0, 200.0, 90.0, 1.0)
            P = st.number_input("Phosphorus (P)", 0.0, 150.0, 50.0, 1.0)
            K = st.number_input("Potassium (K)", 0.0, 210.0, 50.0, 1.0)
            ph = st.number_input("pH Level", 0.0, 14.0, 6.5, 0.1)

        with input_col2:
            st.markdown("Environmental Conditions")
            temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0, 0.5)
            humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0, 1.0)
            rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 120.0, 5.0)

        submitted = st.form_submit_button("Analyze Conditions", use_container_width=True, type="primary")

        if submitted:
            return {
                "N": N,
                "P": P,
                "K": K,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
            }

    return None


def render_sensor_input_section(
    data_source, selected_port, baudrate, use_virtual_arduino
):
    st.subheader("Sensor Data Input")
    sensor_data = None

    if data_source == "Manual Input":
        sensor_data = render_manual_input()

    elif data_source == "Arduino Sensors":
        col_btn1, _ = st.columns(2)
        with col_btn1:
            if st.button("Read from Arduino", type="primary", use_container_width=True):
                if selected_port:
                    with st.spinner("Reading sensor data..."):
                        sensor_data = read_arduino_data(
                            selected_port,
                            baudrate,
                            use_virtual=use_virtual_arduino,
                        )
                        if sensor_data:
                            if use_virtual_arduino:
                                st.success("Data received from virtual Arduino.")
                            else:
                                st.success("Data received from Arduino.")
                        else:
                            st.error("Failed to read data from Arduino.")
                else:
                    st.error("No COM port selected.")

    elif data_source == "Simulation Mode":
        if st.button("Generate Random Sensor Data", type="primary", use_container_width=True):
            sensor_data = simulate_sensor_data()
            st.success("Simulated data generated.")

    if sensor_data:
        st.divider()
        st.subheader("Current Readings")

        metric_cols = st.columns(4)
        metrics = [
            ("N", sensor_data["N"], ""),
            ("P", sensor_data["P"], ""),
            ("K", sensor_data["K"], ""),
            ("pH", sensor_data["ph"], ""),
            ("Temp", sensor_data["temperature"], "°C"),
            ("Humidity", sensor_data["humidity"], "%"),
            ("Rainfall", sensor_data["rainfall"], "mm"),
            ("Time", datetime.now().strftime("%H:%M:%S"), ""),
        ]

        for i, (label, value, unit) in enumerate(metrics):
            with metric_cols[i % 4]:
                if isinstance(value, (int, float)):
                    st.metric(label, f"{value:.1f}{unit}")
                else:
                    st.metric(label, value)

    return sensor_data


def render_analysis_section(
    sensor_data, analysis_mode, selected_crop, crop_db
):
    st.subheader("Analysis Results")

    if sensor_data is None:
        st.info("Please input sensor data to see recommendations.")
        return

    if analysis_mode == "Specific Crop Analysis" and selected_crop:
        st.markdown(f"Analysis for {selected_crop.upper()}")

        recommendations, status = analyze_parameters(sensor_data, selected_crop, crop_db)

        for rec in recommendations:
            if rec["type"] == "success":
                st.success(rec["message"])
                st.info(rec["action"])
            else:
                st.warning(rec["message"])
                st.info(f"Action: {rec['action']}")

        with st.expander("Parameter Status Summary"):
            status_cols = st.columns(2)
            params = ["N", "P", "K", "ph", "temperature", "humidity", "rainfall"]

            for i, param in enumerate(params):
                with status_cols[i % 2]:
                    if param in status:
                        if status[param] == "optimal":
                            st.success(f"{param.upper()}: Optimal")
                        elif status[param] == "low":
                            st.error(f"{param.upper()}: Low")
                        else:
                            st.warning(f"{param.upper()}: High")

    else:
        st.markdown("Best Crop Recommendations")

        best_crops = find_best_crop(sensor_data, crop_db, top_n=5)

        for i, (crop, score) in enumerate(best_crops, 1):
            if score > 90:
                badge = "Excellent match"
            elif score > 75:
                badge = "Good match"
            else:
                badge = "Fair match"

            st.markdown(f"{i}. {crop.upper()} - {badge} ({score:.1f}%)")

            recs, _ = analyze_parameters(sensor_data, crop, crop_db)

            with st.expander(f"View details for {crop}"):
                if len(recs) == 1 and recs[0]["type"] == "success":
                    st.success("All parameters are optimal.")
                else:
                    for rec in recs:
                        if rec["type"] == "warning":
                            st.write(f"- {rec['message']}")


def render_model_prediction(sensor_data, model, crop_db):
    if not (sensor_data and model):
        return

    st.divider()
    st.subheader("ML Model Prediction")

    try:
        input_array = np.array(
            [
                [
                    sensor_data["N"],
                    sensor_data["P"],
                    sensor_data["K"],
                    sensor_data["temperature"],
                    sensor_data["humidity"],
                    sensor_data["ph"],
                    sensor_data["rainfall"],
                ]
            ]
        )

        prediction = model.predict(input_array)[0]

        col_pred1, col_pred2 = st.columns([2, 1])

        with col_pred1:
            st.success(f"Predicted Crop: {prediction.upper()}")

        with col_pred2:
            if prediction in crop_db:
                match_score = find_best_crop(
                    sensor_data, {prediction: crop_db[prediction]}, top_n=1
                )[0][1]
                st.metric("Confidence", f"{match_score:.1f}%")

        if prediction in crop_db:
            with st.expander(f"Condition Analysis for {prediction.upper()}"):
                recs, _ = analyze_parameters(sensor_data, prediction, crop_db)
                for rec in recs:
                    if rec["type"] == "success":
                        st.success(rec["message"])
                    else:
                        st.info(f"{rec['param']}: {rec['action']}")

    except Exception as e:
        st.error(f"Prediction error: {e}")


# Main App
def main():
    crop_db = load_crop_requirements()
    model = load_prediction_model()

    st.markdown(
        '<div class="main-header">Smart Crop Monitoring & Recommendation System</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"Real-time monitoring system with {len(crop_db)} crops in database.")
    st.divider()

    with st.sidebar:
        data_source, analysis_mode, selected_crop, selected_port, baudrate, use_virtual_arduino = render_sidebar(
            crop_db
        )

    col1, col2 = st.columns([3, 2])

    with col1:
        sensor_data = render_sensor_input_section(
            data_source, selected_port, baudrate, use_virtual_arduino
        )

    with col2:
        render_analysis_section(
            sensor_data, analysis_mode, selected_crop, crop_db
        )

    render_model_prediction(sensor_data, model, crop_db)


if __name__ == "__main__":
    main()
