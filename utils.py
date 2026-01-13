import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
import re

# print("🔥 utils.py started")     #(you can check is your utils file worked?)



import requests
from bs4 import BeautifulSoup
import re
from newspaper import Article
import streamlit as st


def extract_article_content(url, use_streamlit=True, max_words=800):
    # ---------- TRY BEAUTIFULSOUP FIRST ----------
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            tag.decompose()

        content = None

        for tag in ['article', 'main', 'section']:
            found = soup.find(tag)
            if found and len(found.get_text(strip=True)) > 200:
                content = found.get_text(" ", strip=True)
                break

        if not content:
            for cls in [
                'content', 'article-content', 'story-content',
                'entry-content', 'post-content', 'td-post-content'
            ]:
                div = soup.find('div', class_=cls)
                if div and len(div.get_text(strip=True)) > 200:
                    content = div.get_text(" ", strip=True)
                    break

        if not content and soup.body:
            content = soup.body.get_text(" ", strip=True)

        if content:
            content = clean_and_limit_text(content, max_words)
            return content

    except Exception as e:
        if not use_streamlit:
            print("BeautifulSoup failed:", e)

    # ---------- FALLBACK: NEWSPAPER3K ----------
    try:
        article = Article(url)
        article.download()
        article.parse()

        if article.text and len(article.text) > 200:
            content = clean_and_limit_text(article.text, max_words)
            return content

    except Exception as e:
        if not use_streamlit:
            print("newspaper3k failed:", e)

    # ---------- BOTH FAILED ----------
    msg = "⚠️ Unable to extract article. This site may block scraping."
    if use_streamlit:
        st.warning(msg)
    else:
        print(msg)

    return None


def clean_and_limit_text(text, max_words):
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return text.strip()



def save_report(file_path, data):
    try:
        with open(file_path, 'w', encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error saving report: {str(e)}")
        return False


def display_analysis_results(analysis):
    ml_label = analysis.get("ml_model_label", "Unknown")
    credibility_score = analysis.get("credibility_score", 0)

    st.markdown('<div class="report-container">', unsafe_allow_html=True)

    if credibility_score >= 70:
        bar_color = 'green'
        emoji = "✅"
    elif credibility_score >= 40:
        bar_color = 'orange'
        emoji = "⚠️"
    else:
        bar_color = 'red'
        emoji = "❌"

    st.markdown(
        f"<h2 style='text-align:center;'>Analysis Results {emoji}</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<h3 style='text-align:center;color:{bar_color};'>Credibility Score: {credibility_score}%</h3>",
        unsafe_allow_html=True
    )

    st.progress(credibility_score / 100)

    st.markdown("### 🧠 ML Model Prediction")
    ml_color = "green" if ml_label == "REAL" else "red"
    st.markdown(
        f"<h4 style='color:{ml_color}; text-align:center;'>Prediction: {ml_label}</h4>",
        unsafe_allow_html=True
    )

    tabs = st.tabs(["📊 Overview", "📚 Factual", "🔍 Source", "⚖️ Bias", "🎯 Clickbait"])

    with tabs[0]:
        st.metric("Factual Accuracy", f"{analysis.get('factual_accuracy', {}).get('score', 0)}%")
        st.metric("Source Credibility", f"{analysis.get('source_credibility', {}).get('score', 0)}%")
        st.metric("Bias Score", f"{analysis.get('bias_detection', {}).get('score', 0)}%")
        st.metric("Clickbait Score", f"{analysis.get('clickbait_detection', {}).get('score', 0)}%")

    with tabs[1]:
        for issue in analysis.get('factual_accuracy', {}).get('factual_issues', []):
            st.warning(f"🔸 {issue}")

    with tabs[2]:
        st.markdown(analysis.get('source_credibility', {}).get('source_analysis', 'N/A'))

    with tabs[3]:
        st.markdown(analysis.get('bias_detection', {}).get('bias_analysis', 'N/A'))

    with tabs[4]:
        st.markdown(analysis.get('clickbait_detection', {}).get('clickbait_analysis', 'N/A'))

    st.markdown('</div>', unsafe_allow_html=True)


# ✅ TERMINAL TEST
if __name__ == "__main__":

    url = r"https://towardsdatascience.com/what-is-natural-language-processing-nlp-3d4fbc6c4f8e"
    text = extract_article_content(url, use_streamlit=False)

    if text:
        print("SUCCESS:", len(text))
        print(text[:300])
    else:
        print("FAILED")
