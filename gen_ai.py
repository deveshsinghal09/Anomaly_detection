import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from dotenv import load_dotenv
import google.generativeai as genai
import time
from datetime import datetime

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("❌ Google API key not found. Please add GOOGLE_API_KEY to your .env file.")
    st.info("🔗 Get your free API key from: https://makersuite.google.com/app/apikey")
    st.stop()

# Configure Google Gemini
genai.configure(api_key=google_api_key)

try:
    with open("notebooks/anomaly_model.pkl", "rb") as f:
        anomaly_model = pickle.load(f)

    with open("notebooks/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Error loading pickle files: {e}. Ensure notebooks/anomaly_model.pkl and scaler.pkl exist.")
    st.stop()

def save_results_to_csv(df, filename_prefix="vehicle_anomaly_results"):
    """Save the dataframe with anomaly results to CSV file"""
    try:
        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.csv"
        
        # Create output directory if it doesn't exist
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filepath = os.path.join(output_dir, filename)
        
        # Save the dataframe
        df.to_csv(filepath, index=False)
        return filepath, filename
        
    except Exception as e:
        st.error(f"Error saving CSV file: {e}")
        return None, None

def convert_df_to_csv(df):
    """Convert dataframe to CSV format for download"""
    return df.to_csv(index=False).encode('utf-8')

class VehicleDiagnosticAI:
    def __init__(self):
        try:
            # Initialize Gemini model
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Test the connection
            test_response = self.model.generate_content("Test connection")
            st.success("✅ Google Gemini API connected successfully!")
            
        except Exception as e:
            st.error(f"❌ Error initializing Google Gemini: {e}")
            st.info("Please check your API key and internet connection.")
            st.stop()
    
    def generate_explanation(self, telemetry_data, anomaly_score=None):
        """Generate vehicle diagnostic explanation using Google Gemini"""
        
        telemetry_str = "\n".join([f"- {key.replace('_', ' ').title()}: {value}" 
                                  for key, value in telemetry_data.items()])
        
        prompt = f"""You are an expert vehicle diagnostic AI assistant. Analyze the following vehicle telemetry data where an anomaly has been detected and provide a detailed explanation.

Vehicle Telemetry Data:
{telemetry_str}

Anomaly Status: DETECTED{f' (Score: {anomaly_score:.3f})' if anomaly_score else ''}

Please provide:
1. **Potential Issues**: What could be causing this anomaly?
2. **Risk Assessment**: How serious is this issue?
3. **Recommended Actions**: What should the driver/mechanic do?
4. **Prevention Tips**: How to avoid this in the future?

Keep your response clear, practical, and focused on actionable insights."""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                
                if response.text:
                    return response.text.strip()
                else:
                    return "Unable to generate explanation. Please try again."
                    
            except Exception as e:
                if "RATE_LIMIT_EXCEEDED" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2  # Exponential backoff
                        st.warning(f"⏳ Rate limit exceeded. Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return self._fallback_explanation(telemetry_data, anomaly_score)
                
                elif "SAFETY" in str(e):
                    return "⚠️ Content filtered for safety. Unable to provide specific diagnostic information."
                
                elif attempt < max_retries - 1:
                    st.warning(f"⚠️ API error, retrying... (attempt {attempt + 1})")
                    time.sleep(1)
                    continue
                else:
                    return self._fallback_explanation(telemetry_data, anomaly_score)
        
        return self._fallback_explanation(telemetry_data, anomaly_score)
    
    def _fallback_explanation(self, telemetry_data, anomaly_score):
        """Rule-based fallback explanation when API fails"""
        explanations = []
        
        speed = telemetry_data.get('speed', 0)
        engine_temp = telemetry_data.get('engine_temp', 0)
        fuel_efficiency = telemetry_data.get('fuel_efficiency', 0)
        rpm = telemetry_data.get('rpm', 0)
        brake_events = telemetry_data.get('brake_events', 0)
        ambient_temp = telemetry_data.get('ambient_temp', 0)
        
        # Rule-based analysis
        if engine_temp > 100:
            explanations.append("🔥 **Critical Engine Overheating**: Engine temperature exceeds safe limits. Immediate attention required!")
        elif engine_temp > 90:
            explanations.append("⚠️ **High Engine Temperature**: Monitor cooling system and check coolant levels.")
        
        if speed > 80 and rpm > 4000:
            explanations.append("🏃 **Aggressive Driving Pattern**: High speed with high RPM indicates stress on engine components.")
        
        if fuel_efficiency < 15:
            explanations.append("⛽ **Poor Fuel Economy**: May indicate engine inefficiency, clogged filters, or maintenance issues.")
        
        if brake_events > 15:
            explanations.append("🛑 **Excessive Braking**: High brake frequency suggests aggressive driving or brake system issues.")
        
        if rpm > 5000:
            explanations.append("📈 **Engine Over-revving**: Sustained high RPM can cause premature engine wear.")
        
        if abs(engine_temp - ambient_temp) < 10 and engine_temp > 60:
            explanations.append("🌡️ **Temperature Sensor Issue**: Engine and ambient temperatures too similar - possible sensor malfunction.")
        
        if not explanations:
            explanations.append(f"🔍 **Anomaly Detected**: Unusual vehicle behavior pattern identified (confidence: {anomaly_score:.1%} if anomaly_score else 'High').")
        
        result = "## 🚗 Vehicle Diagnostic Analysis\n\n"
        result += "\n\n".join(explanations)
        result += "\n\n**🔧 Recommended Action**: Schedule a comprehensive vehicle inspection with a qualified mechanic to prevent potential breakdowns."
        
        return result

try:
    diagnostic_ai = VehicleDiagnosticAI()
except:
    st.stop()

st.set_page_config(page_title="Vehicle Anomaly Explainer", layout="wide", page_icon="🚗")
st.title("🚗 Vehicle Anomaly Detection & AI Explanation")
st.markdown("*Powered by Google Gemini AI*")

# Sidebar with information
with st.sidebar:
    st.header("📊 System Status")
    st.success("✅ Google Gemini API: Connected")
    st.success("✅ ML Models: Loaded")
    
    st.header("📋 Required CSV Columns")
    required_cols = ['speed', 'engine_temp', 'fuel_efficiency', 'ambient_temp', 'rpm', 'brake_events']
    for col in required_cols:
        st.text(f"• {col}")
    
    st.header("🔗 Useful Links")
    st.link_button("🔑 Get Google API Key", "https://makersuite.google.com/app/apikey")
    st.link_button("📚 Gemini API Docs", "https://ai.google.dev/docs")

# File upload
uploaded_file = st.file_uploader("📁 Upload CSV file with vehicle sensor data", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("📄 Uploaded Data Preview")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.metric("Total Records", len(df))
            st.metric("Columns", len(df.columns))
            st.metric("Missing Values", df.isnull().sum().sum())

        # Validate required features
        features = ['speed', 'engine_temp', 'fuel_efficiency', 'ambient_temp', 'rpm', 'brake_events']
        missing_features = [f for f in features if f not in df.columns]
        
        if missing_features:
            st.error(f"❌ Missing required columns: {', '.join(missing_features)}")
            st.info(f"📋 Available columns: {', '.join(df.columns.tolist())}")
            st.stop()

        # Handle missing values
        if df[features].isnull().sum().sum() > 0:
            st.warning("⚠️ Found missing values. Filling with median values.")
            df[features] = df[features].fillna(df[features].median())

        # Anomaly detection
        with st.spinner("🔍 Detecting anomalies..."):
            try:
                scaled_input = scaler.transform(df[features])
                predictions = anomaly_model.predict(scaled_input).flatten()
                labels = (predictions > 0.5).astype(int)
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                st.info("Ensure your data format matches the trained model.")
                st.stop()

        # Add results to dataframe
        df["anomaly_detected"] = labels
        df["anomaly_score"] = predictions
        
        # Add timestamp for when analysis was performed
        df["analysis_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Display results summary
        st.subheader("📊 Anomaly Detection Results")
        
        anomaly_count = sum(labels)
        total_count = len(labels)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", total_count)
        with col2:
            st.metric("Anomalies Found", anomaly_count, delta=f"{(anomaly_count/total_count)*100:.1f}%")
        with col3:
            st.metric("Normal Records", total_count - anomaly_count)
        with col4:
            avg_score = predictions.mean()
            st.metric("Avg Score", f"{avg_score:.3f}")

        st.subheader("💾 Save Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download button for complete results
            csv_data = convert_df_to_csv(df)
            st.download_button(
                label="📥 Download Complete Results (CSV)",
                data=csv_data,
                file_name=f"vehicle_anomaly_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download all data with anomaly detection results"
            )
        
        with col2:
            # Download button for anomalies only
            if anomaly_count > 0:
                anomaly_df = df[df["anomaly_detected"] == 1]
                anomaly_csv = convert_df_to_csv(anomaly_df)
                st.download_button(
                    label="🚨 Download Anomalies Only (CSV)",
                    data=anomaly_csv,
                    file_name=f"vehicle_anomalies_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download only the anomalous records"
                )
            else:
                st.info("No anomalies to download")
        
        with col3:
            # Save to local file (server-side)
            if st.button("💾 Save to Server", help="Save results to server's output folder"):
                filepath, filename = save_results_to_csv(df)
                if filepath:
                    st.success(f"✅ Results saved as: {filename}")
                    st.info(f"📁 File location: {filepath}")

        # Display full results
        st.subheader("📋 Detailed Results")
        
        # Add filter options
        col1, col2 = st.columns(2)
        with col1:
            show_filter = st.selectbox(
                "🔍 Filter Results:",
                ["Show All", "Show Anomalies Only", "Show Normal Only"]
            )
        
        with col2:
            sort_by = st.selectbox(
                "📊 Sort By:",
                ["Original Order", "Anomaly Score (High to Low)", "Anomaly Score (Low to High)"]
            )
        
        display_df = df.copy()
        
        if show_filter == "Show Anomalies Only":
            display_df = display_df[display_df["anomaly_detected"] == 1]
        elif show_filter == "Show Normal Only":
            display_df = display_df[display_df["anomaly_detected"] == 0]
        
        if sort_by == "Anomaly Score (High to Low)":
            display_df = display_df.sort_values("anomaly_score", ascending=False)
        elif sort_by == "Anomaly Score (Low to High)":
            display_df = display_df.sort_values("anomaly_score", ascending=True)
        
        # Color-code the anomalies
        def highlight_anomalies(row):
            return ['background-color: #ffcccc' if row['anomaly_detected'] == 1 else '' for _ in row]
        
        styled_df = display_df.style.apply(highlight_anomalies, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Show summary of filtered results
        if show_filter != "Show All":
            st.info(f"Showing {len(display_df)} out of {len(df)} total records")

        # Generate AI explanations for anomalies
        if anomaly_count > 0:
            st.subheader("🤖 AI-Powered Diagnostic Explanations")
            
            anomaly_rows = df[df["anomaly_detected"] == 1]
            
            # Limit explanations for performance
            max_explanations = 10
            if len(anomaly_rows) > max_explanations:
                st.warning(f"⚠️ Showing explanations for first {max_explanations} anomalies out of {len(anomaly_rows)} detected.")
                anomaly_rows = anomaly_rows.head(max_explanations)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_container = st.empty()
            
            for idx, (i, row) in enumerate(anomaly_rows.iterrows()):
                # Update progress
                progress = (idx + 1) / len(anomaly_rows)
                progress_bar.progress(progress)
                status_container.info(f"🔄 Generating explanation {idx + 1} of {len(anomaly_rows)}...")
                
                telemetry = row[features].to_dict()
                anomaly_score = row['anomaly_score']
                
                # Create expandable section
                with st.expander(f"🚨 Anomaly #{idx + 1} - Row {i + 1} (Score: {anomaly_score:.3f})", expanded=(idx < 2)):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("**📊 Vehicle Data:**")
                        for key, value in telemetry.items():
                            formatted_key = key.replace('_', ' ').title()
                            if isinstance(value, (int, float)):
                                st.metric(formatted_key, f"{value:.1f}")
                            else:
                                st.metric(formatted_key, str(value))
                    
                    with col2:
                        st.markdown("**🔍 AI Diagnosis:**")
                        with st.spinner("Analyzing..."):
                            explanation = diagnostic_ai.generate_explanation(telemetry, anomaly_score)
                        st.markdown(explanation)
                
            progress_bar.empty()
            status_container.empty()
            st.success(f"✅ Generated {len(anomaly_rows)} diagnostic explanations!")
            
        else:
            st.success("🎉 Great news! No anomalies detected in your vehicle data.")
            st.balloons()
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV file is properly formatted with the required columns.")

# Instructions section
with st.expander("📖 How to Use This App"):
    st.markdown("""
    ### 🚀 Quick Start Guide:
    
    1. **Get Google API Key** (Free):
       - Visit https://makersuite.google.com/app/apikey
       - Create a new API key
       - Add it to your `.env` file as: `GOOGLE_API_KEY=your_key_here`
    
    2. **Prepare Your Data**:
       - CSV file with vehicle sensor data
       - Must include these columns: `speed`, `engine_temp`, `fuel_efficiency`, `ambient_temp`, `rpm`, `brake_events`
    
    3. **Upload & Analyze**:
       - Upload your CSV file
       - Review anomaly detection results
       - Get AI-powered explanations for each anomaly
       - Download results in CSV format
    
    ### 💾 Save Options:
    - **Complete Results**: Download all data with anomaly detection results
    - **Anomalies Only**: Download only the records flagged as anomalous
    - **Save to Server**: Save results to the server's output folder
    
    ### 📊 Sample Data Format:
    ```csv
    speed,engine_temp,fuel_efficiency,ambient_temp,rpm,brake_events
    65.2,82.1,28.5,22.3,2100,3
    78.5,95.2,22.1,28.7,3200,8
    45.0,88.9,31.5,19.2,1800,2
    ```
    
    ### 🔧 Troubleshooting:
    - **API Key Issues**: Ensure your Google API key is valid and has Generative AI API enabled
    - **Rate Limits**: The app handles rate limits automatically with retry logic
    - **File Format**: Ensure CSV has proper headers and numeric data
    - **Download Issues**: If download doesn't work, try the "Save to Server" option
    """)
st.markdown("---")
st.markdown("🚗 **Vehicle Anomaly Detection System** | Powered by Google Gemini AI | Built with Streamlit")