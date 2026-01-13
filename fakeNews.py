import streamlit as st
import os
import json
from datetime import datetime
import utils
import analyzer

import pandas as pd
import scikitLearn


st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# Add custom CSS to fix visibility of input fields in dark mode
st.markdown("""
    <style>
        /* General fix for Streamlit text input boxes in dark mode */
        .stTextInput input,
        .stTextInput > div > div > input,
        .stTextArea textarea,
        input[type="text"],
        textarea {
            color: white !important;
            background-color: #333 !important;
            border: 1px solid #555 !important;
        }

        /* Make placeholder text visible too */
        .stTextInput input::placeholder,
        textarea::placeholder {
            color: #bbb !important;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        background-color: #f8f9fa;
    }
            
    .report-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .header-container {
        background: linear-gradient(90deg, #1E3A8A 0%, #3949ab 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .input-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .history-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Fixed button styling for better visibility */
    .stButton>button {
        background-color: #2962ff !important;
        color: white !important;
        border-radius: 5px;
        padding: 0.75rem 1rem;
        font-weight: bold;
        width: 100%;
        font-size: 1.1rem;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1e40af !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transform: translateY(-2px);
    }
    .stButton>button:active {
        transform: translateY(1px);
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .stTextArea>div>div>textarea {
    border-radius: 5px;
    color: white !important;
    background-color: #36393f !important;
    caret-color: white !important; /* This makes the cursor visible */
}

    }
    .stExpander {
        border-radius: 5px;
    }
    .stProgress > div > div > div > div {
        background-color: #1E3A8A;
    }
    .stDataFrame {
        padding: 1rem;
        border-radius: 10px;
    }
    /* Improve tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f2f6;
        border-bottom: 3px solid #1E3A8A;
    }
    /* Fix text visibility */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stText {
        color: #333333 !important;
    }
    .header-container h1, .header-container h2, .header-container h3, 
    .header-container p, .header-container .stMarkdown {
        color: white !important;
    }
    .input-container h1, .input-container h2, .input-container h3, 
    .input-container h4, .input-container h5, .input-container p, 
    .input-container label, .input-container .stMarkdown {
        color: #333333 !important;
    }
    .stRadio > div {
        color: #333333 !important;
    }
    .stRadio > div > label {
        color: #333333 !important;
        background-color: rgba(255, 255, 255, 0.7);
        padding: 5px;
        border-radius: 5px;
    }
    .stRadio [data-baseweb="radio"] {
        background-color: white;
    }
    /* Fix textarea and input text colors */
    .stTextInput input {
        color: white !important;
    background-color: #36393f !important;
    caret-color: white !important;
    border-radius: 5px;
    }
    /* Style info boxes */
    .stAlert {
        background-color: white;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

if not os.path.exists("reports"):
    os.makedirs("reports")

with st.container():
    st.title("🔍 Fake News Detector")
    st.subheader("Analyze news content to detect potential fake news")
    st.markdown('</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📝 Analysis", "📊 Results History", "📂 Reports Management"])

with tab1:
    st.header("Submit Content for Analysis")
    
    input_method = st.radio(
        "Choose input method:",
        ["Text Input", "URL Analysis"], 
        horizontal=True,
        help="Select whether you want to analyze a text snippet or a news article URL"
    )
    
    if input_method == "Text Input":
        st.markdown("#### Enter news text to analyze")
        st.markdown("Paste any news article, social media post, or text content that you want to verify.")
        news_text = st.text_area("News text:", height=200, placeholder="Paste the news text here...")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_button = st.button("🔍 Analyze Text", use_container_width=True)
        
        if analyze_button and news_text:
            try:

                analysis_result = analyzer.analyze_text(news_text)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = f"reports/text_analysis_{timestamp}.json"
                utils.save_report(report_path, {
                    "input_type": "text",
                    "content": news_text,
                    "analysis": analysis_result,
                    "timestamp": datetime.now().isoformat()
                })
                
                st.session_state.history.append({
                    "type": "text",
                    "content": news_text[:100] + "..." if len(news_text) > 100 else news_text,
                    "result": analysis_result,
                    "timestamp": datetime.now().isoformat(),
                    "report_path": report_path
                })
                
                utils.display_analysis_results(analysis_result)
                
            except Exception as e:
                st.error(f"Error analyzing text: {str(e)}")
    
    else:  
        st.markdown("#### Enter a news article URL to analyze")
        st.markdown("Provide a link to any news article you want to verify.")
        news_url = st.text_input("Article URL:", placeholder="https://example.com/news-article")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_url_button = st.button("🔗 Analyze URL", use_container_width=True)
        
        if analyze_url_button and news_url:
            with st.spinner("📥 Fetching article content..."):
                try:
                    article_text = utils.extract_article_content(news_url)
                    
                    if article_text:
        
                        analysis_result = analyzer.analyze_url(news_url, article_text)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        report_path = f"reports/url_analysis_{timestamp}.json"
                        utils.save_report(report_path, {
                            "input_type": "url",
                            "url": news_url,
                            "content": article_text,
                            "analysis": analysis_result,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        st.session_state.history.append({
                            "type": "url",
                            "content": news_url,
                            "result": analysis_result,
                            "timestamp": datetime.now().isoformat(),
                            "report_path": report_path
                        })
                        
                        utils.display_analysis_results(analysis_result)
                    else:
                        st.error("❌ Could not extract content from the provided URL. Please check if the URL is valid and accessible.")
                
                except Exception as e:
                    st.error(f"Error analyzing URL: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ How This Tool Works"):
        st.markdown("""
        ### How the Fake News Detector Works
        
        This tool uses a **Random Forest Machine Learning model** trained on a dataset of real and fake news articles to analyze news content and assess its credibility. The model uses TF-IDF vectorization to extract features from text and makes predictions based on patterns learned from training data.
        
        The analysis is performed across several dimensions:
        
        1. **Factual Accuracy**: ML model evaluates if the content patterns match real or fake news characteristics
        2. **Source Credibility**: Assesses source reliability based on text patterns
        3. **Bias Detection**: Identifies potential bias indicators in the text
        4. **Clickbait Detection**: Detects sensationalist or clickbait patterns
        
        The tool provides an overall credibility score (based on prediction probability) and specific scores for each dimension, along with detailed analysis and recommendations.
        
        **Note:** While this tool provides ML-assisted analysis, it should not be the sole determinant for judging news credibility. Always cross-check information from multiple reliable sources.
        """)

with tab2:
    st.markdown('<div class="history-container">', unsafe_allow_html=True)
    st.header("📚 Analysis History")
    
    if not st.session_state.history:
        st.info("No analysis history yet. Submit content to see results here.")
    else:
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"{item['type'].capitalize()}: {item['content']}", expanded=i==0):
                st.markdown(f"**Analyzed on:** {datetime.fromisoformat(item['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
                
                credibility_score = item['result'].get('credibility_score', 0)
                
                # Determine color based on score (visual indicator only)
                if credibility_score >= 70:
                    verdict_color = "green"
                elif credibility_score >= 40:
                    verdict_color = "orange"
                else:
                    verdict_color = "red"
                
                # Show only the percentage - no redundant labels
                st.markdown(f"<h3 style='text-align: center; color: {verdict_color};'>Credibility Score: {credibility_score}%</h3>", unsafe_allow_html=True)
                st.progress(credibility_score/100)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"📄 View Full Report", key=f"view_{i}"):
                        with open(item['report_path'], 'r') as f:
                            report_data = json.load(f)
                        
                        st.json(report_data)
                
                with col2:
                    with open(item['report_path'], 'r', encoding='utf-8', errors='ignore') as f:
                        report_content = f.read()

                        
                    st.download_button(
                        label="💾 Download Report",
                        data=report_content,
                        file_name=os.path.basename(item['report_path']),
                        mime="application/json",
                        key=f"download_{i}"
                    )
    st.markdown('</div>', unsafe_allow_html=True)


with tab3:
    st.markdown('<div class="history-container">', unsafe_allow_html=True)
    st.header("📂 Reports Management")

    if os.path.exists("reports"):
        report_files = [f for f in os.listdir("reports") if f.endswith(".json")]

        if report_files:
            reports_data = []
            for i, report_file in enumerate(sorted(report_files, reverse=True)):
                file_path = os.path.join("reports", report_file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        reports_data.append({
                            "Filename": report_file,
                            "Type": data["input_type"].capitalize(),
                            "Content": data["url"] if data["input_type"] == "url" else (data["content"][:50] + "..." if len(data["content"]) > 50 else data["content"]),
                            "Timestamp": datetime.fromisoformat(data["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                            "Credibility Score": f"{data['analysis'].get('credibility_score', 0)}%"
                        })
                except Exception as e:
                    st.warning(f"Could not load report {report_file}: {str(e)}")

            if reports_data:
                reports_df = pd.DataFrame(reports_data)
                st.dataframe(reports_df, use_container_width=True)

                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("🗑️ Clear All Reports", use_container_width=True):
                        try:
                            for file in report_files:
                                os.remove(os.path.join("reports", file))
                            st.session_state.history = []
                            st.success("✅ All reports cleared successfully!")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error clearing reports: {str(e)}")
            else:
                st.info("No saved reports found.")

            st.markdown("---")
            st.subheader("🛠 Individual Report Tools")

            for idx, report_file in enumerate(sorted(report_files, reverse=True)):
                file_path = os.path.join("reports", report_file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    with st.expander(f"📄 {report_file}", expanded=False):
                        st.caption(data.get("content", "")[:100] + "...")
                        col1, col2 = st.columns([3, 3])

                        with col1:
                            with open(file_path, "r", encoding="utf-8") as f_download:
                                st.download_button("📥 Download", f_download.read(),
                                    file_name=report_file, mime="application/json",
                                    key=f"dl_{report_file}_{idx}")

                        with col2:
                            new_name = st.text_input("✏️ Rename file to:", value=report_file, key=f"rename_input_{report_file}_{idx}")
                            if new_name != report_file and st.button("✅ Rename", key=f"rename_btn_{report_file}_{idx}"):
                                os.rename(file_path, os.path.join("reports", new_name))
                                st.success(f"Renamed to {new_name}")
                                st.experimental_rerun()

                        st.markdown("---")

                        new_text = st.text_area("📝 Edit Content", value=data.get("content", ""), key=f"edit_text_{report_file}_{idx}")
                        save_col, delete_col = st.columns([1, 1])

                        with save_col:
                            if st.button("💾 Save", key=f"save_btn_{report_file}_{idx}"):
                                data["content"] = new_text
                                with open(file_path, "w", encoding="utf-8") as f:
                                    json.dump(data, f, indent=4)
                                st.success("Changes saved.")
                                st.experimental_rerun()

                        with delete_col:
                            if st.button("🗑 Delete", key=f"delete_btn_{report_file}_{idx}"):
                                os.remove(file_path)
                                st.success(f"Deleted {report_file}")
                                st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error loading {report_file}: {str(e)}")
        else:
            st.info("No saved reports found.")
    else:
        st.info("Reports directory not found.")

    st.markdown('</div>', unsafe_allow_html=True)
