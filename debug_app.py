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
    
    def generate_quiz(self, text_content, quiz_level):
        """Generate quiz using Llama 3.3 via OpenRouter"""
        
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
        
        # Optimized prompt for Llama 3.3
        system_prompt = """You are an expert quiz generator. Create exactly 10 multiple choice questions based on the provided text. 

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{
  "mcqs": [
    {
      "mcq": "question text here",
      "options": {
        "a": "first option",
        "b": "second option", 
        "c": "third option",
        "d": "fourth option"
      },
      "correct": "a"
    }
  ]
}

Rules:
- Generate exactly 10 questions
- Questions must be based on the provided text
- Each question must have 4 options (a, b, c, d)
- The "correct" field must contain only the letter (a, b, c, or d)
- No repeated questions
- Match the specified difficulty level"""

        user_prompt = f"""Text to create quiz from:
{text_content}

Difficulty level: {quiz_level}

Create 10 multiple choice questions based on this text at {quiz_level} difficulty level. Return only valid JSON."""

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1500,
            "top_p": 0.9
        }
        
        try:
            response = requests.post(  # Fixed typo: was "poist"
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

class ResultAnalyzer:
    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state for result tracking"""
        if 'exam_records' not in st.session_state:
            st.session_state.exam_records = []
        if 'subjects' not in st.session_state:
            st.session_state.subjects = []
        if 'grade_weights' not in st.session_state:
            st.session_state.grade_weights = {}

    def add_exam_record(self, subject, exam_type, score, max_score, date, weight=1.0):
        """Add a new exam record"""
        record = {
            'subject': subject,
            'exam_type': exam_type,
            'score': score,
            'max_score': max_score,
            'percentage': (score / max_score) * 100,
            'date': date,
            'weight': weight,  # This weight is not used in calculations, keeping for potential future use
            'timestamp': datetime.now()
        }
        st.session_state.exam_records.append(record)

        # Update subjects list and default weights
        if subject not in st.session_state.subjects:
            st.session_state.subjects.append(subject)
            st.session_state.grade_weights[subject] = {'test': 30, 'exam': 70}

    def get_subject_stats(self, subject):
        """Get statistics for a specific subject"""
        subject_records = [r for r in st.session_state.exam_records if r['subject'] == subject]

        if not subject_records:
            return None

        all_scores = [r['percentage'] for r in subject_records]

        test_scores = [r['percentage'] for r in subject_records if r['exam_type'].lower() in ['test', 'quiz', 'assignment']]
        exam_scores = [r['percentage'] for r in subject_records if r['exam_type'].lower() not in ['test', 'quiz', 'assignment']]

        avg_test_score = np.mean(test_scores) if test_scores else 0
        avg_exam_score = np.mean(exam_scores) if exam_scores else 0

        weights = st.session_state.grade_weights.get(subject, {'test': 30, 'exam': 70})
        test_weight = weights['test'] / 100
        exam_weight = weights['exam'] / 100

        # Calculate weighted average based on component averages
        weighted_average = (avg_test_score * test_weight) + (avg_exam_score * exam_weight)

        # Adjust weighted average if only one type of assessment exists
        if not test_scores and exam_scores:
            weighted_average = avg_exam_score
        elif not exam_scores and test_scores:
            weighted_average = avg_test_score
        elif not test_scores and not exam_scores:
            weighted_average = 0

        return {
            'average': np.mean(all_scores) if all_scores else 0,
            'weighted_average': weighted_average,
            'highest': max(all_scores) if all_scores else 0,
            'lowest': min(all_scores) if all_scores else 0,
            'count': len(all_scores),
            'recent_scores': all_scores[-5:],
            'trend': self.calculate_trend(all_scores)
        }

    def calculate_trend(self, scores):
        """Calculate trend direction"""
        if len(scores) < 2:
            return "Insufficient data"

        # Use linear regression for a more robust trend analysis
        x = np.arange(len(scores))
        slope, _ = np.polyfit(x, scores, 1)

        if slope > 2:
            return "Improving"
        elif slope < -2:
            return "Declining"
        else:
            return "Stable"

    def predict_required_score(self, subject, target_percentage, current_weights):
        """Predict required score for the next exam to reach a target grade."""
        subject_records = [r for r in st.session_state.exam_records if r['subject'] == subject]

        test_scores = [r['percentage'] for r in subject_records if r['exam_type'].lower() in ['test', 'quiz', 'assignment']]
        exam_scores = [r['percentage'] for r in subject_records if r['exam_type'].lower() not in ['test', 'quiz', 'assignment']]

        test_weight = current_weights['test'] / 100
        exam_weight = current_weights['exam'] / 100

        avg_test_score = np.mean(test_scores) if test_scores else 0

        # Current state
        current_exam_avg = np.mean(exam_scores) if exam_scores else 0
        num_exams_taken = len(exam_scores)

        # Calculate the current overall weighted average
        current_weighted_avg = (avg_test_score * test_weight) + (current_exam_avg * exam_weight)

        # If exam weight is zero, prediction is not possible
        if exam_weight <= 0:
            return {
                'required_score': float('inf'),
                'current_average': current_weighted_avg,
                'feasible': False
            }

        # Formula to find required score 'X' on the next exam:
        # X = (((target - (avg_test_score * test_weight)) / exam_weight) * (num_exams_taken + 1)) - (current_exam_avg * num_exams_taken)
        if num_exams_taken == 0:
             # Simplified formula if no exams taken yet: target = test_avg*test_weight + X*exam_weight
             required_score = (target_percentage - (avg_test_score * test_weight)) / exam_weight
        else:
            required_score = (((target_percentage - (avg_test_score * test_weight)) / exam_weight) * (num_exams_taken + 1)) - (current_exam_avg * num_exams_taken)

        return {
            'required_score': required_score,
            'current_average': current_weighted_avg,
            'feasible': 0 <= required_score <= 100
        }

    def generate_insights(self, subject_stats):
        """Generate AI-powered insights"""
        insights = []

        for subject, stats in subject_stats.items():
            if stats:
                if stats['trend'] == 'Improving':
                    insights.append(f"📈 {subject}: You're on an upward trajectory! Keep up the good work.")
                elif stats['trend'] == 'Declining':
                    insights.append(f"📉 {subject}: Consider reviewing your study strategy for this subject.")

                if stats['weighted_average'] >= 85:
                    insights.append(f"⭐ {subject}: Excellent performance! You're mastering this subject.")
                elif stats['weighted_average'] < 60:
                    insights.append(f"⚠️ {subject}: This subject needs more attention. Consider additional practice.")

        return insights

@st.cache_data
def fetch_questions(text_content, quiz_level):
    """Cached wrapper for quiz generation"""
    if not OPENROUTER_API_KEY:
        st.error("Please set OPENROUTER_API_KEY in your .env file")
        return []
    
    generator = LlamaQuizGenerator(OPENROUTER_API_KEY)
    return generator.generate_quiz(text_content, quiz_level)

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

def create_performance_chart(analyzer):
    """Create performance visualization"""
    if not st.session_state.exam_records:
        return None
    
    df = pd.DataFrame(st.session_state.exam_records)
    
    # Time series chart
    fig = px.line(df, x='date', y='percentage', color='subject', 
                  title='Performance Trends Over Time',
                  labels={'percentage': 'Score (%)', 'date': 'Date'})
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 100]),
        height=400
    )
    
    return fig

def create_subject_comparison_chart(analyzer):
    """Create subject comparison chart"""
    if not st.session_state.subjects:
        return None
    
    subjects = []
    averages = []
    
    for subject in st.session_state.subjects:
        stats = analyzer.get_subject_stats(subject)
        if stats:
            subjects.append(subject)
            averages.append(stats['average'])
    
    if not subjects:
        return None
    
    fig = go.Figure(data=[
        go.Bar(x=subjects, y=averages, 
               marker_color=["#FF1B1B" if avg < 60 else "#2DEBDF" if avg < 80 else "#00ABD1" for avg in averages])
    ])
    
    fig.update_layout(
        title='Subject Performance Comparison',
        xaxis_title='Subjects',
        yaxis_title='Average Score (%)',
        yaxis=dict(range=[0, 100]),
        height=400
    )
    
    return fig

def generate_pdf_report(analyzer):
    """Generate PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#2E86AB')
    )
    story.append(Paragraph("Academic Performance Report", title_style))
    story.append(Spacer(1, 20))
    
    # Summary statistics
    story.append(Paragraph("Performance Summary", styles['Heading2']))
    
    summary_data = []
    for subject in st.session_state.subjects:
        stats = analyzer.get_subject_stats(subject)
        if stats:
            summary_data.append([
                subject,
                f"{stats['average']:.1f}%",
                f"{stats['highest']:.1f}%",
                f"{stats['lowest']:.1f}%",
                stats['trend']
            ])
    
    if summary_data:
        summary_table = Table([
            ['Subject', 'Average', 'Highest', 'Lowest', 'Trend']
        ] + summary_data)
        
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
    
    # Generate insights
    subject_stats = {subject: analyzer.get_subject_stats(subject) for subject in st.session_state.subjects}
    insights = analyzer.generate_insights(subject_stats)
    
    if insights:
        story.append(Spacer(1, 20))
        story.append(Paragraph("Key Insights", styles['Heading2']))
        for insight in insights:
            story.append(Paragraph(f"• {insight}", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def quiz_generator_tab():
    """Quiz Generator Tab Content"""
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
            with st.spinner("🔄 Generating quiz with Llama 3.3 engine - subomi..."):
                st.session_state.questions = fetch_questions(text_content, quiz_level.lower())
                st.session_state.quiz_generated = True if st.session_state.questions else False

    # Display quiz if generated
    if st.session_state.quiz_generated and st.session_state.questions:
        st.divider()
        st.header("📝 Your Quiz")
        
        # Display questions with radio buttons
        selected_options = []
        
        for i, question in enumerate(st.session_state.questions):
            st.subheader(f"Question {i+1}")
            options = list(question["options"].values())
            
            selected_option = st.radio(
                question["mcq"],
                options,
                index=None,
                key=f"q_{i}",
                help=f"Choose the best answer for question {i+1}"
            )
            selected_options.append(selected_option)

        # Submit button
        if st.button("✅ Submit Quiz", type="secondary"):
            marks, total, percentage = display_quiz_results(st.session_state.questions, selected_options)
            
            # Option to save to records
            st.subheader("💾 Save to Records")
            with st.form("save_quiz_result"):
                col1, col2 = st.columns(2)
                with col1:
                    subject = st.text_input("Subject", placeholder="e.g., Mathematics")
                with col2:
                    exam_type = st.selectbox("Type", ["Quiz", "Test", "Exam"])
                
                if st.form_submit_button("Save Result"):
                    if subject:
                        analyzer = ResultAnalyzer()
                        analyzer.add_exam_record(
                            subject=subject,
                            exam_type=exam_type,
                            score=marks,
                            max_score=total,
                            date=datetime.now().date()
                        )
                        st.success(f"✅ Result saved for {subject}!")
                        st.rerun()
    
    elif st.session_state.quiz_generated and not st.session_state.questions:
        st.error("❌ Failed to generate quiz. Please try again with different text or check your API key.")

def result_analyzer_tab():
    """Result Analyzer Tab Content"""
    st.header("📈 Result Analyzer & Target Predictor")
    
    analyzer = ResultAnalyzer()
    
    # Add new exam record
    with st.expander("➕ Add New Exam Result", expanded=False):
        with st.form("add_exam"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                subject = st.text_input("Subject*", placeholder="e.g., Mathematics")
                exam_type = st.selectbox("Exam Type*", ["Test", "Quiz", "Midterm", "Final", "Assignment"])
            
            with col2:
                score = st.number_input("Score*", min_value=0.0, step=0.1)
                max_score = st.number_input("Max Score*", min_value=0.1, step=0.1, value=100.0)
            
            with col3:
                exam_date = st.date_input("Date*", value=datetime.now().date())
                weight = st.number_input("Weight", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            
            if st.form_submit_button("Add Record", type="primary"):
                if subject and score >= 0 and max_score > 0:
                    analyzer.add_exam_record(subject, exam_type, score, max_score, exam_date, weight)
                    st.success(f"✅ Added {exam_type} result for {subject}")
                    st.rerun()
                else:
                    st.error("Please fill all required fields correctly")

    # Display current records
    if st.session_state.exam_records:
        st.subheader("📊 Performance Dashboard")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_exams = len(st.session_state.exam_records)
            st.metric("Total Exams", total_exams)
        
        with col2:
            avg_score = np.mean([r['percentage'] for r in st.session_state.exam_records])
            st.metric("Overall Average", f"{avg_score:.1f}%")
        
        with col3:
            subjects_count = len(st.session_state.subjects)
            st.metric("Subjects", subjects_count)
        
        with col4:
            recent_scores = [r['percentage'] for r in st.session_state.exam_records[-5:]]
            recent_avg = np.mean(recent_scores) if recent_scores else 0
            st.metric("Recent Average", f"{recent_avg:.1f}%")
        
        # Charts
        st.subheader("📊 Performance Visualization")
        
        chart_type = st.selectbox("Select Chart Type", ["Performance Trends", "Subject Comparison"])
        
        if chart_type == "Performance Trends":
            fig = create_performance_chart(analyzer)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            fig = create_subject_comparison_chart(analyzer)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Subject Analysis
        st.subheader("🔍 Detailed Subject Analysis")
        
        for subject in st.session_state.subjects:
            stats = analyzer.get_subject_stats(subject)
            if stats:
                with st.expander(f"📚 {subject} Analysis"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Average Score", f"{stats['average']:.1f}%")
                        st.metric("Highest Score", f"{stats['highest']:.1f}%")
                        st.metric("Lowest Score", f"{stats['lowest']:.1f}%")
                    
                    with col2:
                        st.metric("Total Exams", stats['count'])
                        st.metric("Trend", stats['trend'])
                        st.metric("Weighted Average", f"{stats['weighted_average']:.1f}%")
        
        # Target Predictor
        st.subheader("🎯 Target Score Predictor")
        
        with st.form("target_predictor"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                target_subject = st.selectbox("Select Subject", st.session_state.subjects)
            
            with col2:
                target_percentage = st.number_input("Target Grade (%)", min_value=0.0, max_value=100.0, value=85.0)
            
            with col3:
                st.write("**Grade Weights**")
                if target_subject in st.session_state.grade_weights:
                    test_weight = st.number_input("Test Weight (%)", 
                                                value=st.session_state.grade_weights[target_subject]['test'],
                                                min_value=0, max_value=100)
                    exam_weight = 100 - test_weight
                    st.write(f"Exam Weight: {exam_weight}%")
                    
                    # Update weights
                    st.session_state.grade_weights[target_subject] = {
                        'test': test_weight,
                        'exam': exam_weight
                    }
                else:
                    test_weight = st.number_input("Test Weight (%)", value=30, min_value=0, max_value=100)
                    exam_weight = 100 - test_weight
            
            if st.form_submit_button("Calculate Required Score"):
                if target_subject:
                    current_weights = st.session_state.grade_weights.get(target_subject, {'test': 30, 'exam': 70})
                    prediction = analyzer.predict_required_score(target_subject, target_percentage, current_weights)
                    
                    if prediction:
                        st.subheader("🔮 Prediction Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Current Average", f"{prediction['current_average']:.1f}%")
                            st.metric("Target Grade", f"{target_percentage:.1f}%")
                        
                        with col2:
                            required_score = prediction['required_score']
                            
                            if required_score > 1000:  # Handle impossibly high scores
                                display_score = "Over 1000%"
                                st.metric("Required Next Score", display_score)
                                st.error("❌ Target is not realistically achievable.")
                            else:
                                st.metric("Required Next Score", f"{required_score:.1f}%")

                            if prediction['feasible']:
                                st.success("✅ Target is achievable!")
                            elif required_score > 100:
                                st.warning("⚠️ Target is achievable, but requires over 100%.")
                            else:  # This case now means required_score is negative
                                st.error("❌ Target is not achievable, your grade is already too high.")
        
        # Insights
        st.subheader("💡 AI Insights")
        subject_stats = {subject: analyzer.get_subject_stats(subject) for subject in st.session_state.subjects}
        insights = analyzer.generate_insights(subject_stats)
        
        if insights:
            for insight in insights:
                st.info(insight)
        
        # Export options
        st.subheader("📄 Export Reports")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Generate PDF Report"):
                pdf_buffer = generate_pdf_report(analyzer)
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_buffer.getvalue(),
                    file_name=f"academic_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
        
        with col2:
            if st.button("📈 Export Data (CSV)"):
                df = pd.DataFrame(st.session_state.exam_records)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"exam_records_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    else:
        st.info("📝 No exam records found. Add your first exam result to get started!")

# --- SMART SCHEDULER CODE ---

def load_scheduler_data():
    """Loads scheduler data from the JSON file."""
    try:
        with open(SCHEDULER_DATA_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"subjects": [], "timetable": {}, "homework": [], "mode": "school"}

def save_scheduler_data(data):
    """Saves scheduler data to the JSON file."""
    with open(SCHEDULER_DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)
    st.cache_data.clear() # Clear cache after saving

def scheduler_tab():
    """Smart Scheduler Tab Content"""
    st.header("📅 Smart Scheduler")

    # --- Session State Initialization ---
    if 'scheduler_data' not in st.session_state:
        st.session_state.scheduler_data = load_scheduler_data()
    if 'last_notification_check' not in st.session_state:
        # Set to a past time to ensure the first check runs
        st.session_state.last_notification_check = datetime.now() - timedelta(seconds=61)

    data = st.session_state.scheduler_data

    # --- Notification Logic ---
    def check_for_notifications():
        now = datetime.now()
        # Only check for notifications once per minute to avoid spamming
        if now - st.session_state.last_notification_check < timedelta(seconds=60):
            return

        st.session_state.last_notification_check = now
        
        # Period End Notifications
        current_day = now.strftime("%A")
        current_time = now.strftime("%H:%M")
        if data.get("mode") == "school" and current_day in data.get("timetable", {}):
            for period in data["timetable"][current_day]:
                if period["end"] == current_time:
                    st.toast(f"🔔 Period Over! Your {period['subject']} class has ended.", icon='🔔')

        # Homework Due Notifications
        today = now.date()
        for hw in data.get("homework", []):
            if not hw.get("done", False):
                due_date = datetime.strptime(hw["due_date"], "%Y-%m-%d").date()
                if due_date == today:
                    st.toast(f"❗ Homework Due Today: {hw['task']} ({hw['subject']})", icon='❗')

    check_for_notifications()

    # --- UI Components ---
    st.subheader("🏠 Daily Dashboard")
    
    mode = data.get("mode", "school")
    if st.button(f"Current Mode: {mode.capitalize()}. Click to Toggle."):
        data["mode"] = "weekend" if mode == "school" else "school"
        save_scheduler_data(data)
        st.success(f"Switched to {data['mode'].capitalize()} Mode.")
        st.rerun()

    if mode == 'school':
        # Upcoming Class Display
        now = datetime.now()
        current_day = now.strftime("%A")
        current_time = now.strftime("%H:%M")
        upcoming_class = "No more classes today."
        if current_day in data.get("timetable", {}):
            for period in sorted(data["timetable"][current_day], key=lambda x: x["start"]):
                if current_time < period["end"]:
                    if current_time < period["start"]:
                        upcoming_class = f"Next: {period['subject']} at {period['start']}"
                    else:
                        upcoming_class = f"Ongoing: {period['subject']} until {period['end']}"
                    break
        st.metric("Upcoming Class", upcoming_class)

        # Due Homework Display
        st.subheader("❗ Due Homework")
        today = datetime.now().date()
        due_soon_alerts = [
            f"**{hw['task']}** ({hw['subject']}) - Due: {hw['due_date']}"
            for hw in data.get("homework", [])
            if not hw.get("done", False) and datetime.strptime(hw["due_date"], "%Y-%m-%d").date() <= today + timedelta(days=1)
        ]
        if due_soon_alerts:
            for alert in due_soon_alerts:
                st.warning(alert)
        else:
            st.info("No homework due soon.")
    else:
        st.info("It's the weekend! No classes scheduled.")

    # --- Management Sections in Expanders ---
    with st.expander("📚 ➕ Add New Subject"):
        with st.form("add_subject_form"):
            new_subject = st.text_input("Subject Name")
            if st.form_submit_button("Add Subject"):
                if new_subject and new_subject not in data["subjects"]:
                    data["subjects"].append(new_subject)
                    save_scheduler_data(data)
                    st.success(f"Subject '{new_subject}' added.")
                    st.rerun()
                else:
                    st.error("Subject name cannot be empty or already exists.")

    with st.expander("🕐 Set Timetable"):
        if not data["subjects"]:
            st.warning("Please add subjects first.")
        else:
            with st.form("set_timetable_form"):
                cols = st.columns(4)
                day = cols[0].selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                start_time = cols[1].time_input("Start Time", value=datetime.now().time())
                end_time = cols[2].time_input("End Time", value=(datetime.now() + timedelta(hours=1)).time())
                subject = cols[3].selectbox("Subject", data["subjects"])
                
                if st.form_submit_button("Add Period"):
                    if day not in data["timetable"]:
                        data["timetable"][day] = []
                    data["timetable"][day].append({"start": start_time.strftime("%H:%M"), "end": end_time.strftime("%H:%M"), "subject": subject})
                    save_scheduler_data(data)
                    st.success(f"Timetable updated for {day}.")
                    st.rerun()

    with st.expander("📝 Add Homework"):
        if not data["subjects"]:
            st.warning("Please add subjects first.")
        else:
            with st.form("add_homework_form"):
                cols = st.columns(3)
                hw_subject = cols[0].selectbox("Subject", data["subjects"], key="hw_subject")
                hw_task = cols[1].text_input("Task Description")
                hw_due_date = cols[2].date_input("Due Date", min_value=datetime.now().date())

                if st.form_submit_button("Add Homework"):
                    if hw_task:
                        data["homework"].append({"subject": hw_subject, "task": hw_task, "due_date": hw_due_date.strftime("%Y-%m-%d"), "done": False})
                        save_scheduler_data(data)
                        st.success("Homework added.")
                        st.rerun()
                    else:
                        st.error("Task description cannot be empty.")

    # --- Data Viewing and Management Tabs ---
    st.subheader("📊 All Data")
    tab1, tab2, tab3 = st.tabs(["Timetable", "Homework", "Subjects"])

    with tab1:
        timetable_list = []
        timetable_data = data.get("timetable", {})
        # Sort days of the week
        sorted_days = sorted(timetable_data.keys(), key=lambda day: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day))
        
        for day in sorted_days:
            periods = sorted(timetable_data[day], key=lambda x: x['start'])
            for period in periods:
                timetable_list.append([day, period['start'], period['end'], period['subject']])

        if timetable_list:
            df = pd.DataFrame(timetable_list, columns=["Day", "Start Time", "End Time", "Subject"])
            st.dataframe(df.reset_index(drop=True), use_container_width=True)
        else:
            st.info("No timetable set yet.")

    with tab2:
        homework_df = pd.DataFrame(data.get("homework", []))
        if not homework_df.empty:
            st.dataframe(homework_df)
            tasks_to_mark = [f"{i}: {hw['task']}" for i, hw in enumerate(data['homework']) if not hw.get('done')]
            if tasks_to_mark:
                task_to_toggle = st.selectbox("Mark homework as done/undone", options=tasks_to_mark)
                if st.button("Update Status"):
                    task_index = int(task_to_toggle.split(':')[0])
                    data['homework'][task_index]['done'] = not data['homework'][task_index].get('done', False)
                    save_scheduler_data(data)
                    st.success("Homework status updated.")
                    st.rerun()
        else:
            st.info("No homework yet.")

    with tab3:
        st.write(data.get("subjects", []))

    # --- Delete Data Section ---
    with st.expander("🗑️ Delete Data"):
        # Delete Subject
        st.subheader("Delete a Subject")
        if data.get("subjects"):
            subject_to_delete = st.selectbox("Select subject to delete", options=[""] + data["subjects"], key="del_subject")
            if st.button("Delete Subject and All Related Data"):
                if subject_to_delete:
                    # Remove subject
                    data["subjects"].remove(subject_to_delete)
                    
                    # Remove related timetable entries
                    for day, periods in list(data.get("timetable", {}).items()):
                        data["timetable"][day] = [p for p in periods if p["subject"] != subject_to_delete]
                        if not data["timetable"][day]:
                            del data["timetable"][day]

                    # Remove related homework
                    data["homework"] = [hw for hw in data.get("homework", []) if hw["subject"] != subject_to_delete]

                    save_scheduler_data(data)
                    st.success(f"Subject '{subject_to_delete}' and all its data have been deleted.")
                    st.rerun()
                else:
                    st.warning("Please select a subject to delete.")
        else:
            st.info("No subjects to delete.")

        st.divider()

        # Delete Timetable Period
        st.subheader("Delete a Timetable Period")
        period_options = {f"{day}-{i}": f"{day}: {p['start']}-{p['end']} ({p['subject']})"
                        for day, periods in data.get("timetable", {}).items()
                        for i, p in enumerate(periods)}
        
        if period_options:
            period_key_to_delete = st.selectbox("Select period to delete", options=[""] + list(period_options.keys()), format_func=lambda k: period_options.get(k, "Select a period"), key="del_period")
            if st.button("Delete Selected Period"):
                if period_key_to_delete:
                    day, index_str = period_key_to_delete.split('-')
                    index = int(index_str)
                    
                    del data["timetable"][day][index]
                    if not data["timetable"][day]:
                        del data["timetable"][day]
                    
                    save_scheduler_data(data)
                    st.success("Period deleted.")
                    st.rerun()
                else:
                    st.warning("Please select a period to delete.")
        else:
            st.info("No periods to delete.")

        st.divider()

        # Delete Homework
        st.subheader("Delete a Homework Assignment")
        homework_options = {i: f"{hw['task']} ({hw['subject']})"
                          for i, hw in enumerate(data.get("homework", []))}

        if homework_options:
            homework_index_to_delete = st.selectbox("Select homework to delete", options=[""] + list(homework_options.keys()), format_func=lambda k: homework_options.get(k, "Select homework"), key="del_hw")
            if st.button("Delete Selected Homework"):
                if isinstance(homework_index_to_delete, int):
                    del data["homework"][homework_index_to_delete]
                    save_scheduler_data(data)
                    st.success("Homework deleted.")
                    st.rerun()
                else:
                    st.warning("Please select a homework assignment to delete.")
        else:
            st.info("No homework to delete.")

def main():
    st.set_page_config(
        page_title="Baptist Academy Smart School AI (BASS A.I)",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Baptist Academy Smart School AI (BASS A.I)")
    st.markdown("*Complete Educational Tool with Quiz Generation, Result Analysis & Target Prediction by Baptist Academy*")
    
    # Test API connection on startup
    api_status, api_message = test_api_connection()
    if not api_status:
        st.warning(f"⚠️ API Connection Issue: {api_message}")
    
    # Top navigation
    selected_tab = option_menu(
        menu_title=None,
        options=["Quiz Generator", "Result Analyzer", "Smart Scheduler"],
        icons=["question-circle", "graph-up", "calendar-check"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

    # Main content based on selected tab
    if selected_tab == "Quiz Generator":
        quiz_generator_tab()
    elif selected_tab == "Result Analyzer":
        result_analyzer_tab()
    else:
        scheduler_tab()

if __name__ == "__main__":
    main()