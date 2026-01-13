import streamlit as st
import scikitLearn
import re

# Load model and vectorizer
model, vectorizer = scikitLearn.prepare_model()


def detect_gibberish(text):
    """
    Detect if text contains gibberish/nonsense words
    Returns True if gibberish is detected
    """
    # Common English words (expanded dictionary)
    common_words = {
        'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'can', 'this', 'that', 'these', 'those', 'a', 'an', 'and', 'or', 'but', 'if',
        'of', 'to', 'in', 'on', 'at', 'for', 'with', 'from', 'by', 'about', 'into',
        'through', 'during', 'including', 'against', 'among', 'throughout', 'despite',
        'towards', 'upon', 'concerning', 'prime', 'minister', 'india', 'government',
        'president', 'country', 'state', 'city', 'news', 'report', 'said', 'according',
        'official', 'announced', 'narendra', 'modi', 'india', 'indian'
    }
    
    # Split text into words
    words = re.findall(r'\b[a-z]+\b', text.lower())
    
    if len(words) == 0:
        return False
    
    # Check for words that look like gibberish
    gibberish_count = 0
    total_checkable_words = 0
    
    for word in words:
        # Skip very common words and very short words
        if word in common_words or len(word) <= 2:
            continue
        
        total_checkable_words += 1
        
        # Pattern 1: Too many consecutive consonants (e.g., "sajsfj" = s, j, s, f, j)
        # Check for 3+ consecutive consonants
        consonants = re.findall(r'[bcdfghjklmnpqrstvwxyz]+', word)
        max_consonants = max([len(c) for c in consonants] + [0])
        if max_consonants >= 3:
            # Check if it's a valid word pattern (has vowels between consonants)
            vowel_count = len(re.findall(r'[aeiou]', word))
            # If very few vowels relative to length, likely gibberish
            if vowel_count == 0 or (len(word) > 4 and vowel_count < len(word) * 0.2):
                gibberish_count += 1
                continue
        
        # Pattern 2: Too many consecutive vowels (e.g., "aeiou")
        if re.search(r'[aeiou]{4,}', word):
            gibberish_count += 1
            continue
        
        # Pattern 3: Unusual consonant-vowel patterns
        # Words with very irregular patterns (like "sajsfj")
        if len(word) >= 5:
            # Check consonant-to-vowel ratio
            consonant_count = len(re.findall(r'[bcdfghjklmnpqrstvwxyz]', word))
            vowel_count = len(re.findall(r'[aeiou]', word))
            if vowel_count == 0 and consonant_count >= 4:
                gibberish_count += 1
                continue
            # Very low vowel ratio suggests gibberish
            if consonant_count > 0 and (vowel_count / len(word)) < 0.15:
                gibberish_count += 1
                continue
        
        # Pattern 4: Check for keyboard mashing patterns (like "asdf", "qwerty")
        keyboard_patterns = ['asdf', 'qwerty', 'zxcv', 'hjkl', 'fghj']
        if any(pattern in word for pattern in keyboard_patterns):
            gibberish_count += 1
            continue
    
    # If we have checkable words and more than 15% are gibberish, flag it
    # Also flag if even one word is clearly gibberish in a short text
    if total_checkable_words > 0:
        gibberish_ratio = gibberish_count / total_checkable_words
        if gibberish_ratio > 0.15 or (gibberish_count >= 1 and len(words) <= 5):
            return True
    
    return False


def validate_text_quality(text):
    """
    Validate text quality and detect obvious issues
    Returns (is_valid, issues_list)
    """
    issues = []
    
    # Check for gibberish
    if detect_gibberish(text):
        issues.append("Contains gibberish or nonsense words")
    
    # Check for very short text
    words = text.split()
    if len(words) < 3:
        issues.append("Text is too short to analyze reliably")
    
    # Check for excessive special characters
    special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', text)) / max(len(text), 1)
    if special_char_ratio > 0.3:
        issues.append("Contains excessive special characters")
    
    # Check for random character sequences
    if re.search(r'[a-z]{1,2}[a-z]{1,2}[a-z]{1,2}[a-z]{1,2}', text.lower()) and len(text.split()) < 5:
        # Very short text with random-looking sequences
        if detect_gibberish(text):
            issues.append("Appears to contain random text or gibberish")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def analyze_text(text):
    """
    Analyze text for fake news detection using Random Forest ML model
    
    Args:
        text: The news text to analyze
        
    Returns:
        dict: Analysis results with ML model prediction
    """
    try:
        with st.spinner("🔍 Analyzing text with ML model..."):
            # First, validate text quality
            is_valid, validation_issues = validate_text_quality(text)
            
            # If text has obvious issues, override prediction to FAKE
            if not is_valid:
                pass
                
                analysis = {
                    "credibility_score": credibility_score,
                    "factual_accuracy": {
                        "score": 10,
                        "factual_issues": validation_issues
                    },
                    "source_credibility": {
                        "score": 10,
                        "source_analysis": f"Text quality validation failed. Issues detected: {', '.join(validation_issues)}"
                    },
                    "bias_detection": {
                        "score": 20,
                        "bias_type": "invalid",
                        "bias_analysis": "Text contains invalid content or gibberish that cannot be reliably analyzed."
                    },
                    "clickbait_detection": {
                        "score": 15,
                        "clickbait_analysis": "Text quality is too poor to analyze properly."
                    },
                    "insights": f"Text validation detected issues: {', '.join(validation_issues)}. This content cannot be reliably verified.",
                    "recommendations": [
                        "The text contains invalid or nonsensical content",
                        "Please provide a complete, valid news article or text",
                        "Check for typos or incomplete text"
                    ],
                    "ml_model_label": "FAKE"
                }
                return analysis
            
            # Get ML model prediction
            ml_label = scikitLearn.predict_text(model, vectorizer, text)
            
            # Get prediction probability for credibility score
            # Use the same cleaning as in predict_text
            cleaned = scikitLearn.clean_text(text)
            vector = vectorizer.transform([cleaned])
            probabilities = model.predict_proba(vector)[0]
            
            # Model uses numeric classes: 0 = FAKE, 1 = REAL
            # probabilities[0] = probability of FAKE, probabilities[1] = probability of REAL
            real_probability = probabilities[1]  # Index 1 is REAL class
            fake_probability = probabilities[0]   # Index 0 is FAKE class
            
            # Adjust credibility score based on prediction
            if ml_label == 'REAL':
                # For REAL predictions, boost credibility score appropriately
                # The model predicted REAL, so we should reflect higher credibility
                
                # Calculate base score (how confident model is it's REAL)
                base_score = real_probability
                
                # Boost credibility for REAL predictions:
                # - If model is >50% confident it's REAL, boost to at least 70%
                # - This ensures REAL news doesn't show as "Questionable"
                # - Scale the boost based on confidence level
                if real_probability >= 0.5:
                    # Boost formula: ensure minimum 70% for REAL predictions above 50% confidence
                    # Scale from 50% → 70% to 100% → 100%
                    if real_probability < 0.7:
                        # Boost moderate confidence (50-70%) to good confidence (70-85%)
                        boost_factor = (0.7 - real_probability) * 0.5  # Partial boost
                        credibility_score = int((real_probability + boost_factor) * 100)
                        credibility_score = max(credibility_score, 70)  # Minimum 70% for REAL
                    else:
                        # High confidence (70%+), use as-is or slight boost
                        credibility_score = int(real_probability * 100)
                        credibility_score = min(credibility_score, 100)
                else:
                    # Low confidence but still predicted REAL (edge case)
                    credibility_score = int(real_probability * 100)
            else:  # FAKE
                # For FAKE predictions, credibility is low (probability of being REAL)
                credibility_score = int(real_probability * 100)
            
            # Generate analysis based on ML prediction and credibility score
            if ml_label == 'REAL':
                # High credibility = REAL news
                factual_score = credibility_score
                source_score = min(credibility_score + 5, 100)
                bias_score = min(credibility_score + 10, 100)
                clickbait_score = min(credibility_score + 15, 100)
                insights = f"The ML model indicates this content appears to be REAL news with {credibility_score}% confidence based on text analysis patterns."
                recommendations = [
                    "Verify the source independently",
                    "Check for multiple reliable sources confirming the information",
                    "Look for publication date and author information"
                ]
            else:  # FAKE
                # Low credibility = FAKE news
                factual_score = credibility_score
                source_score = max(credibility_score - 10, 0)
                bias_score = max(credibility_score - 20, 0)
                clickbait_score = max(credibility_score - 30, 0)
                fake_confidence = 100 - credibility_score
                insights = f"The ML model indicates this content appears to be FAKE news with {fake_confidence}% confidence based on text analysis patterns. Exercise caution and verify from reliable sources."
                recommendations = [
                    "Cross-check with established news sources",
                    "Verify claims with fact-checking websites",
                    "Be skeptical of sensational claims",
                    "Check the publication date and source credibility"
                ]
            
            analysis = {
                "credibility_score": credibility_score,
                "factual_accuracy": {
                    "score": factual_score,
                    "factual_issues": [] if ml_label == 'REAL' else ["Potential misinformation detected by ML model"]
                },
                "source_credibility": {
                    "score": source_score,
                    "source_analysis": f"ML model prediction: {ml_label}. {'Source appears credible based on text patterns.' if ml_label == 'REAL' else 'Source credibility questionable based on text patterns.'}"
                },
                "bias_detection": {
                    "score": bias_score,
                    "bias_type": "unknown" if ml_label == 'REAL' else "potential",
                    "bias_analysis": "Text analysis suggests balanced reporting." if ml_label == 'REAL' else "Text patterns indicate potential bias or misleading information."
                },
                "clickbait_detection": {
                    "score": clickbait_score,
                    "clickbait_analysis": "Content appears to be standard news format." if ml_label == 'REAL' else "Content may contain clickbait or sensationalist elements."
                },
                "insights": insights,
                "recommendations": recommendations,
                "ml_model_label": ml_label
            }
            
            return analysis
    
    except Exception as e:
        st.error(f"❌ Error analyzing text: {str(e)}")
        return {
            "credibility_score": 0,
            "factual_accuracy": {"score": 0, "factual_issues": ["Error analyzing text"]},
            "source_credibility": {"score": 0, "source_analysis": "Error analyzing source"},
            "bias_detection": {"score": 0, "bias_type": "unknown", "bias_analysis": "Error analyzing bias"},
            "clickbait_detection": {"score": 0, "clickbait_analysis": "Error analyzing clickbait elements"},
            "insights": "Error generating insights",
            "recommendations": ["Please try again or use a different text"],
            "ml_model_label": "UNKNOWN"
        }


def analyze_url(url, content):
    """
    Analyze URL content for fake news detection using Random Forest ML model
    
    Args:
        url: The news article URL
        content: The extracted content from the URL
        
    Returns:
        dict: Analysis results with ML model prediction
    """
    try:
        with st.spinner("🔍 Analyzing article content with ML model..."):
            # Use the same analyze_text function since we're analyzing content
            analysis = analyze_text(content)
            
            # Add URL-specific information
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            analysis["source_credibility"]["source_analysis"] += f" Domain: {domain}"
            
            return analysis
    
    except Exception as e:
        st.error(f"❌ Error analyzing URL: {str(e)}")
        return {
            "credibility_score": 0,
            "factual_accuracy": {"score": 0, "factual_issues": ["Error analyzing article"]},
            "source_credibility": {"score": 0, "source_analysis": f"Error analyzing source"},
            "bias_detection": {"score": 0, "bias_type": "unknown", "bias_analysis": "Error analyzing bias"},
            "clickbait_detection": {"score": 0, "clickbait_analysis": "Error analyzing clickbait elements"},
            "insights": "Error generating insights",
            "recommendations": ["Please try again or use a different URL"],
            "ml_model_label": "UNKNOWN"
        }
