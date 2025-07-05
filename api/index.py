import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
from datetime import datetime
from io import BytesIO
import time
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Title Screening Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- Default Screening Criteria ---
DEFAULT_INCLUSION_CRITERIA = """
- Empirical-based observational studies (prospective and retrospective cohorts, case-control, cross-sectional)
- Quantitative studies and mixed studies
- Randomized Controlled Trials (RCTs)
- Studies involving E-healthcare systems or similar technology applications
- Studies published from year 2000 onwards
"""

DEFAULT_EXCLUSION_CRITERIA = """
- Review studies (systematic reviews, narrative reviews)
- Conference proceedings, and theses
- Studies in languages other than English
- Studies published before the year 2000
- Opinion-based grey literature (commentaries, editorials, perspectives)
"""

# --- Core Functions ---
def load_dataframe(uploaded_file):
    """Load dataframe from uploaded file with multiple encoding support"""
    if uploaded_file.name.endswith(('.xls', '.xlsx')):
        try:
            return pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            return None

    encodings_to_try = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
    file_bytes = uploaded_file.getvalue()

    for encoding in encodings_to_try:
        try:
            file_buffer = BytesIO(file_bytes)
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(file_buffer, encoding=encoding)
                st.success(f"Successfully read file with '{encoding}' encoding.")
                return df
            elif uploaded_file.name.endswith('.tsv'):
                df = pd.read_csv(file_buffer, sep='\t', encoding=encoding)
                st.success(f"Successfully read file with '{encoding}' encoding.")
                return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            st.error(f"Unexpected error with '{encoding}': {e}")
            return None

    st.error("Could not read the file. All tried encodings failed.")
    return None


def test_api_connection(api_key):
    """Test if the API key and connection are working"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Simple test prompt
        test_response = model.generate_content("Hello, respond with 'API working'")
        return True, "Connection successful"
    except Exception as e:
        return False, str(e)


def get_screening_analysis_from_gemini(title, inclusion_criteria, exclusion_criteria, keywords, model, max_retries=3):
    """Get screening decision from Gemini API with improved error handling"""
    
    keywords_section = ""
    if keywords and keywords.strip():
        keywords_section = f"""
    **Keywords to Consider:**
    {keywords}
    
    Note: Papers containing these keywords may be more likely to be relevant, but still apply the inclusion/exclusion criteria strictly.
    """
    
    prompt = f"""
    You are an expert academic research assistant performing a systematic literature review screening.
    Your task is to screen papers based ONLY on their title.

    **Screening Rules:**
    **Inclusion Criteria:**
    {inclusion_criteria}

    **Exclusion Criteria:**
    {exclusion_criteria}
    
    {keywords_section}

    **Instructions:**
    1. Analyze the Title provided below.
    2. Have a keywords given if inputed.
    3. Based *strictly* on the rules, decide if the paper should be "Include" or "Exclude".
    4. Consider the keywords as additional context, but prioritize the inclusion/exclusion criteria.
    5. Provide ONLY the final decision in the specified JSON format. Do not provide a reason.

    **Output Format:**
    {{
      "final_decision": "Include or Exclude"
    }}

    --- DATA TO ANALYZE ---
    Title: {title}
    """

    for attempt in range(max_retries):
        try:
            # Progressive delay: 2s, 5s, 10s
            delay = 2 ** attempt
            time.sleep(delay)
            
            # Set a shorter timeout for individual requests
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=100,
                )
            )
            
            cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
            data = json.loads(cleaned_text)
            return data.get("final_decision", "Error")

        except json.JSONDecodeError:
            st.warning(f"⚠️ Gemini did not return valid JSON for title: '{title[:50]}...'")
            if hasattr(response, 'text'):
                st.text_area("Gemini's Raw Response:", response.text, height=100)
            return "Error"

        except Exception as e:
            error_str = str(e)
            
            # Handle different types of errors
            if "429" in error_str or "quota" in error_str.lower():
                wait_seconds = 60
                match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                if match:
                    wait_seconds = int(match.group(1))
                st.warning(f"⚠️ Rate limit hit. Waiting {wait_seconds} seconds before retrying... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_seconds)
            elif "503" in error_str or "timeout" in error_str.lower() or "connect" in error_str.lower():
                wait_seconds = 5 * (attempt + 1)
                st.warning(f"⚠️ Connection error. Waiting {wait_seconds} seconds before retrying... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_seconds)
            elif "403" in error_str or "invalid" in error_str.lower():
                st.error(f"❌ API Key error: {e}")
                return "API Key Error"
            else:
                st.error(f"❌ Unexpected API error on attempt {attempt+1}: {e}")
                if attempt == max_retries - 1:
                    break
                time.sleep(2 ** attempt)

    st.error(f"❌ Failed to process title after {max_retries} attempts. Error: Connection/timeout issues.")
    return "Connection Error"


def create_summary_stats(df):
    """Create summary statistics for the screening results"""
    if 'Final Decision' not in df.columns:
        return None
    
    stats = df['Final Decision'].value_counts().to_dict()
    total = len(df)
    
    summary = {
        'Total Papers': total,
        'Included': stats.get('Include', 0),
        'Excluded': stats.get('Exclude', 0),
        'Errors': stats.get('Error', 0) + stats.get('Connection Error', 0) + stats.get('API Key Error', 0),
        'Not Processed': stats.get('Not Processed', 0)
    }
    
    return summary


# --- Streamlit UI ---
st.title("📑 AI-Powered Title Screening Assistant")
st.markdown("Upload a CSV, TSV, or Excel file with your paper titles to perform an initial screening based on your criteria.")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter your Gemini API Key", type="password", help="**Get a Gemini API Key**: Visit [Google AI Studio](https://aistudio.google.com/apikey) to get your free API key",)
    
    if api_key:
        if st.button("Test API Connection"):
            with st.spinner("Testing connection..."):
                success, message = test_api_connection(api_key)
                if success:
                    st.success("✅ API connection successful!")
                else:
                    st.error(f"❌ API connection failed: {message}")

    st.subheader("Keywords")
    keywords = st.text_area(
        "Keywords to Look For (Optional)", 
        placeholder="Enter keywords separated by commas, e.g., machine learning, artificial intelligence, healthcare, telemedicine",
        help="These keywords will be used to help identify relevant papers during screening",
        height=100
    )
    
    st.subheader("Screening Rules")
    inclusion_criteria = st.text_area("Inclusion Criteria", value=DEFAULT_INCLUSION_CRITERIA, height=200)
    exclusion_criteria = st.text_area("Exclusion Criteria", value=DEFAULT_EXCLUSION_CRITERIA, height=200)
    
    st.subheader("Advanced Settings")
    batch_size = st.slider("Batch Size (process every N papers)", 1, 20, 10)
    max_retries = st.slider("Max Retries per Paper", 1, 5, 3)
    
    # Instructions and Tips in sidebar
    with st.expander("📋 Instructions & Tips"):
        st.markdown("""
        ### How to Use:
        1. **Get a Gemini API Key**: Visit [Google AI Studio](https://aistudio.google.com/apikey) to get your free API key
        2. **Upload your file**: CSV, TSV, or Excel files are supported
        3. **Select the title column**: Choose which column contains your paper titles
        4. **Customize criteria**: Modify the inclusion/exclusion criteria as needed
        5. **Add keywords** (optional): Specify relevant keywords to help guide the screening
        6. **Test connection**: Use the "Test API Connection" button to verify your setup
        7. **Start screening**: Click "Begin AI Screening Process" to process all papers
        
        ### Tips:
        - **Keywords are optional** but can help improve screening accuracy
        - **Keywords should be comma-separated** (e.g., "machine learning, AI, healthcare")
        - The app will **resume from where it left off** if interrupted
        - Adjust **batch size** and **max retries** in Advanced Settings for better performance
        - **Connection errors** are automatically retried with exponential backoff
        - Results are automatically saved and can be downloaded as CSV
        
        ### Troubleshooting:
        - **503/Timeout errors**: Usually temporary - the app will retry automatically
        - **Rate limit errors**: The app will wait and retry automatically
        - **API key errors**: Check your API key and connection
        - **File encoding issues**: The app tries multiple encodings automatically
        """)

st.header("Step 1: Upload Your File")
uploaded_file = st.file_uploader("Choose a CSV, TSV, or Excel file", type=['csv', 'tsv', 'xlsx', 'xls'])

if uploaded_file:
    df = load_dataframe(uploaded_file)
    if df is not None:
        st.markdown("### File Preview")
        st.dataframe(df.head())

        st.header("Step 2: Select Your Data Columns")
        columns = df.columns.tolist()
        title_col = st.selectbox("Which column contains the paper titles?", columns, index=0)

        st.header("Step 3: Start Screening")
        
        # Option to resume from where left off
        if 'Final Decision' in df.columns:
            processed_count = len(df[df['Final Decision'] != 'Not Processed'])
            st.info(f"Found {processed_count} already processed papers. Will continue from where left off.")
        
        if st.button("🚀 Begin AI Screening Process"):
                if not api_key:
                    st.error("❌ Please enter your Gemini API Key in the sidebar.")
                else:
                    try:
                        genai.configure(api_key=api_key)
                        # Use a more reliable model
                        model = genai.GenerativeModel("gemma-3n-e4b-it")
                        
                        st.info(f"🚀 Starting the screening process for {len(df)} titles...")

                        if 'Final Decision' not in df.columns:
                            df['Final Decision'] = 'Not Processed'

                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Create metrics display
                        metrics_container = st.empty()

                        processed_count = 0
                        for index, row in df.iterrows():
                            # Skip already processed papers
                            if df.loc[index, 'Final Decision'] not in ['Not Processed', 'Connection Error']:
                                processed_count += 1
                                continue
                                
                            progress = (index + 1) / len(df)
                            progress_bar.progress(progress)

                            title = str(row[title_col]) if pd.notna(row[title_col]) else ""
                            if not title:
                                df.loc[index, 'Final Decision'] = 'Skipped'
                                continue

                            status_text.info(f"Processing ({index+1}/{len(df)}): {title[:80]}...")

                            decision = get_screening_analysis_from_gemini(
                                title=title,
                                inclusion_criteria=inclusion_criteria,
                                exclusion_criteria=exclusion_criteria,
                                keywords=keywords,
                                model=model,
                                max_retries=max_retries
                            )

                            df.loc[index, 'Final Decision'] = decision
                            processed_count += 1
                            
                            # Update metrics every few papers
                            if processed_count % batch_size == 0:
                                summary = create_summary_stats(df)
                                if summary:
                                    with metrics_container.container():
                                        col1, col2, col3, col4 = st.columns(4)
                                        col1.metric("Included", summary['Included'])
                                        col2.metric("Excluded", summary['Excluded'])
                                        col3.metric("Errors", summary['Errors'])
                                        col4.metric("Remaining", summary['Not Processed'])

                        status_text.success("✅ Analysis complete!")

                        st.header("Screening Results")
                        
                        # Display summary statistics
                        summary = create_summary_stats(df)
                        if summary:
                            st.subheader("Summary Statistics")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            col1.metric("Total Papers", summary['Total Papers'])
                            col2.metric("Included", summary['Included'])
                            col3.metric("Excluded", summary['Excluded'])
                            col4.metric("Errors", summary['Errors'])
                            col5.metric("Not Processed", summary['Not Processed'])

                        # Display results table
                        st.dataframe(df)

                        # Download button
                        csv_data = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8')
                        output_filename = f"screened_{uploaded_file.name.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"

                        st.download_button(
                            label="📥 Download Screened Results as CSV",
                            data=csv_data,
                            file_name=output_filename,
                            mime="text/csv",
                        )

                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                        st.info("This could be due to an invalid API key or an issue with the Gemini service.")

st.markdown(
    "<hr style='margin-top:2em;margin-bottom:0.5em;border:1px solid #eee'>"
    "<div style='text-align:center; color:gray; font-size:0.95em;'>"
    "All rights reserved by <a href='https://www.linkedin.com/in/ekram-asif/' style='color:white; text-decoration:none;' target='_blank'>Ekram Asif</a>"
    "</div>",
    unsafe_allow_html=True
)
