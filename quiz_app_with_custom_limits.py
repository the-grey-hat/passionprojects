import streamlit as st
import json
import os
import requests
from dotenv import load_dotenv
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
import base64
from streamlit_option_menu import option_menu

SCHEDULER_DATA_FILE = "scheduler_data.json"

load_dotenv()  # Load all the environment variables from .env file

# ╭─ OpenRouter config ───────────────────────────────────────────╮
OPENROUTER_API_KEY = (
    st.secrets.get("OPENROUTER_API_KEY")     # 1️⃣  primary source (cloud)
    or os.getenv("OPENROUTER_API_KEY")        # 2️⃣  local fallback
)

if not OPENROUTER_API_KEY:
    st.error(
        "⚠️  OpenRouter API key not found.\n"
        "• In Streamlit Cloud, add it in **Edit Secrets**.\n"
        "• Locally, put it in a `.env` file."
    )
    st.stop()   # bail early so the rest of the script doesn't crash

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "meta-llama/llama-3.3-70b-instruct"
# ╰───────────────────────────────────────────────────────────────╯

# Test API connection on startup
def test_api_connection():
    """Test the OpenRouter API connection"""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello, who are you?"}],
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            OPENROUTER_BASE_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return True, "API connection successful"
    except requests.exceptions.RequestException as e:
        return False, f"API connection failed: {e}"

class LlamaQuizGenerator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",  # Optional: helps with rate limits
        }
    
    def generate_quiz(self, text_content, quiz_level, num_questions=10):
        """Generate quiz using Llama 3.3 via OpenRouter with customizable question count"""
        
        # Response format template
        response_format = {
            "mcqs": [
                {
                    "mcq": "multiple choice question",
                    "options": {
                        "a": "choice here",
                        "b": "choice here", 
                        "c": "choice here",
                        "d": "choice here"
                    },
                    "correct": "correct choice option in the form of a, b, c or d"
                }
            ]
        }
        
        # Optimized prompt for Llama 3.3 with customizable question count
        system_prompt = f"""You are an expert quiz generator. Create exactly {num_questions} multiple choice questions based on the provided text. 

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{{
  "mcqs": [
    {{
      "mcq": "question text here",
      "options": {{
        "a": "first option",
        "b": "second option", 
        "c": "third option",
        "d": "fourth option"
      }},
      "correct": "a"
    }}
  ]
}}

Rules:
- Generate exactly {num_questions} questions
- Questions must be based on the provided text
- Each question must have 4 options (a, b, c, d)
- The "correct" field must contain only the letter (a, b, c, or d)
- No repeated questions
- Match the specified difficulty level"""

        user_prompt = f"""Text to create quiz from:
{text_content}

Difficulty level: {quiz_level}

Create {num_questions} multiple choice questions based on this text at {quiz_level} difficulty level. Return only valid JSON."""

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 2000,  # Increased for more questions
            "top_p": 0.9
        }
        
        try:
            response = requests.post(
                OPENROUTER_BASE_URL, 
                headers=self.headers, 
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            # Extract JSON if wrapped in code blocks
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            elif content.startswith('```'):
                content = content.replace('```', '').strip()
            
            # Parse the JSON response
            quiz_data = json.loads(content)
            return quiz_data.get("mcqs", [])
            
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {str(e)}")
            return []
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse response as JSON. Please try again.")
            st.write("Raw response:", content if 'content' in locals() else "No response")
            return []
        except KeyError as e:
            st.error(f"Unexpected response format: {str(e)}")
            return []
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return []

@st.cache_data
def fetch_questions(text_content, quiz_level, num_questions=10):
    """Cached wrapper for quiz generation with customizable question count"""
    if not OPENROUTER_API_KEY:
        st.error("Please set OPENROUTER_API_KEY in your .env file")
        return []
    
    generator = LlamaQuizGenerator(OPENROUTER_API_KEY)
    return generator.generate_quiz(text_content, quiz_level, num_questions)

def validate_input(text_content):
    """Validate user input"""
    if not text_content or len(text_content.strip()) < 50:
        return False, "Please provide at least 50 characters of text content."
    return True, ""

def display_quiz_results(questions, selected_options):
    """Display quiz results with scoring"""
    if not questions or not selected_options:
        return
    
    marks = 0
    st.header("📊 Quiz Results")
    
    for i, question in enumerate(questions):
        selected_option = selected_options[i]
        correct_option = question["options"][question["correct"]]
        
        # Create expandable section for each question
        with st.expander(f"Question {i+1}: {question['mcq'][:50]}..."):
            st.write(f"**Question:** {question['mcq']}")
            
            if selected_option:
                if selected_option == correct_option:
                    st.success(f"✅ Correct! You selected: {selected_option}")
                    marks += 1
                else:
                    st.error(f"❌ Incorrect. You selected: {selected_option}")
                    st.info(f"💡 Correct answer: {correct_option}")
            else:
                st.warning("⚠️ No answer selected")
                st.info(f"💡 Correct answer: {correct_option}")
    
    # Overall score
    percentage = (marks / len(questions)) * 100
    st.subheader(f"🎯 Final Score: {marks}/{len(questions)} ({percentage:.1f}%)")
    
    # Performance feedback
    if percentage >= 80:
        st.success("🎉 Excellent work!")
    elif percentage >= 60:
        st.info("👍 Good job!")
    else:
        st.warning("📚 Keep studying!")
    
    return marks, len(questions), percentage

def quiz_generator_tab():
    """Quiz Generator Tab Content with Customizable Question Limits"""
    st.header("🧠 Subomi's AI Quiz Generator")
    st.markdown("*Powered by Llama 3.3 via OpenRouter*")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input
        text_content = st.text_area(
            "📝 Paste your text content here:",
            height=150,
            placeholder="Enter the text you want to create a quiz from..."
        )
        
        # Input validation
        if text_content:
            word_count = len(text_content.split())
            char_count = len(text_content)
            st.caption(f"📊 {word_count} words, {char_count} characters")
    
    with col2:
        # Quiz settings
        st.subheader("⚙️ Quiz Settings")
        quiz_level = st.selectbox(
            "🎯 Select difficulty:",
            ["Easy", "Medium", "Hard"],
            help="Easy: Basic comprehension, Medium: Analysis, Hard: Critical thinking"
        )
        
        # 🔥 CUSTOMIZABLE QUESTION LIMIT FEATURE 🔥
        num_questions = st.selectbox(
            "📊 Number of questions:",
            [10, 20, 30, 40, 50],
            help="Select how many questions you want in your quiz"
        )
        
        # Show selected count
        st.info(f"🎯 You selected: **{num_questions} questions**")
        
        # Model info
        st.info(f"🤖 Model: {MODEL_NAME.split('/')[-1]}")

    # Validate input
    is_valid, error_msg = validate_input(text_content)
    if text_content and not is_valid:
        st.error(error_msg)
        return

    # Initialize session state
    if 'quiz_generated' not in st.session_state:
        st.session_state.quiz_generated = False
    if 'questions' not in st.session_state:
        st.session_state.questions = []

    # Generate Quiz button
    if st.button("🚀 Generate Quiz", disabled=not is_valid, type="primary"):
        if is_valid:
            with st.spinner(f"🔄 Generating {num_questions} questions with Llama 3.3 engine - subomi..."):
                st.session_state.questions = fetch_questions(text_content, quiz_level.lower(), num_questions)
                st.session_state.quiz_generated = True if st.session_state.questions else False

    # Display quiz if generated
    if st.session_state.quiz_generated and st.session_state.questions:
        st.divider()
        st.header(f"📝 Your {len(st.session_state.questions)}-Question Quiz")
        
        # Show quiz statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Questions Generated", len(st.session_state.questions))
        with col2:
            st.metric("Difficulty Level", quiz_level)
        with col3:
            st.metric("Total Points", len(st.session_state.questions))
        
        st.divider()
        
        # Display questions with radio buttons
        selected_options = []
        
        for i, question in enumerate(st.session_state.questions):
            st.subheader(f"Question {i+1} of {len(st.session_state.questions)}")
            options = list(question["options"].values())
            
            selected_option = st.radio(
                question["mcq"],
                options,
                index=None,
                key=f"q_{i}",
                help=f"Choose the best answer for question {i+1}"
            )
            selected_options.append(selected_option)
            
            # Add some spacing between questions
            if i < len(st.session_state.questions) - 1:
                st.markdown("---")

        # Submit button
        st.divider()
        if st.button("✅ Submit Quiz", type="secondary", use_container_width=True):
            marks, total, percentage = display_quiz_results(st.session_state.questions, selected_options)
            
            # Enhanced results display
            st.balloons()  # Celebration effect
            
            # Results summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Questions Answered", len([opt for opt in selected_options if opt is not None]))
            with col2:
                st.metric("Correct Answers", marks)
            with col3:
                st.metric("Final Percentage", f"{percentage:.1f}%")
    
    elif st.session_state.quiz_generated and not st.session_state.questions:
        st.error("❌ Failed to generate quiz. Please try again with different text or check your API key.")

def main():
    st.set_page_config(
        page_title="Baptist Academy Smart School AI (BASS A.I) - Custom Quiz Limits",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Baptist Academy Smart School AI (BASS A.I)")
    st.markdown("*Complete Educational Tool with **Customizable Quiz Limits** (10-50 Questions)*")
    
    # Feature highlight
    st.info("🆕 **NEW FEATURE**: Choose between 10, 20, 30, 40, or 50 questions for your quiz!")
    
    # Test API connection on startup
    api_status, api_message = test_api_connection()
    if not api_status:
        st.warning(f"⚠️ API Connection Issue: {api_message}")
    
    # Main quiz generator
    quiz_generator_tab()

if __name__ == "__main__":
    main()