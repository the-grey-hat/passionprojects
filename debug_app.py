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
import uuid
from pathlib import Path
import mimetypes

SCHEDULER_DATA_FILE = "scheduler_data.json"
NOTES_DATA_FILE = "notes_data.json"
FEE_DATA_FILE = "fee_data.json"

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

# --- NOTE SHARING & READER APP CLASSES ---

class NotesManager:
    def __init__(self):
        self.initialize_notes_state()

    def initialize_notes_state(self):
        """Initialize session state for notes"""
        if 'notes_data' not in st.session_state:
            st.session_state.notes_data = self.load_notes_data()

    def load_notes_data(self):
        """Load notes data from JSON file"""
        try:
            with open(NOTES_DATA_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "notes": [],
                "subjects": [],
                "classes": [],
                "comments": [],
                "ratings": {}
            }

    def save_notes_data(self):
        """Save notes data to JSON file"""
        with open(NOTES_DATA_FILE, "w") as f:
            json.dump(st.session_state.notes_data, f, indent=4, default=str)

    def add_note(self, title, content, subject, class_name, file_type, author, audio_transcript=None):
        """Add a new note"""
        note_id = str(uuid.uuid4())
        note = {
            'id': note_id,
            'title': title,
            'content': content,
            'subject': subject,
            'class': class_name,
            'file_type': file_type,
            'author': author,
            'upload_date': datetime.now().isoformat(),
            'views': 0,
            'downloads': 0,
            'audio_transcript': audio_transcript
        }
        
        st.session_state.notes_data['notes'].append(note)
        
        # Add to subjects and classes if new
        if subject not in st.session_state.notes_data['subjects']:
            st.session_state.notes_data['subjects'].append(subject)
        if class_name not in st.session_state.notes_data['classes']:
            st.session_state.notes_data['classes'].append(class_name)
        
        # Initialize rating
        st.session_state.notes_data['ratings'][note_id] = {'total': 0, 'count': 0, 'average': 0}
        
        self.save_notes_data()
        return note_id

    def add_comment(self, note_id, author, comment):
        """Add a comment to a note"""
        comment_data = {
            'id': str(uuid.uuid4()),
            'note_id': note_id,
            'author': author,
            'comment': comment,
            'timestamp': datetime.now().isoformat()
        }
        st.session_state.notes_data['comments'].append(comment_data)
        self.save_notes_data()

    def add_rating(self, note_id, rating):
        """Add a rating to a note"""
        if note_id not in st.session_state.notes_data['ratings']:
            st.session_state.notes_data['ratings'][note_id] = {'total': 0, 'count': 0, 'average': 0}
        
        ratings = st.session_state.notes_data['ratings'][note_id]
        ratings['total'] += rating
        ratings['count'] += 1
        ratings['average'] = ratings['total'] / ratings['count']
        
        self.save_notes_data()

    def get_notes_by_filter(self, subject=None, class_name=None):
        """Get notes filtered by subject and/or class"""
        notes = st.session_state.notes_data['notes']
        
        if subject and subject != "All":
            notes = [n for n in notes if n['subject'] == subject]
        if class_name and class_name != "All":
            notes = [n for n in notes if n['class'] == class_name]
        
        return notes

    def get_comments_for_note(self, note_id):
        """Get all comments for a specific note"""
        return [c for c in st.session_state.notes_data['comments'] if c['note_id'] == note_id]

    def increment_views(self, note_id):
        """Increment view count for a note"""
        for note in st.session_state.notes_data['notes']:
            if note['id'] == note_id:
                note['views'] += 1
                break
        self.save_notes_data()

    def increment_downloads(self, note_id):
        """Increment download count for a note"""
        for note in st.session_state.notes_data['notes']:
            if note['id'] == note_id:
                note['downloads'] += 1
                break
        self.save_notes_data()

# --- SCHOOL FEE TRACKER CLASSES ---

class FeeTracker:
    def __init__(self):
        self.initialize_fee_state()

    def initialize_fee_state(self):
        """Initialize session state for fee tracking"""
        if 'fee_data' not in st.session_state:
            st.session_state.fee_data = self.load_fee_data()

    def load_fee_data(self):
        """Load fee data from JSON file"""
        try:
            with open(FEE_DATA_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "students": [],
                "fee_structures": [],
                "payments": [],
                "receipts": []
            }

    def save_fee_data(self):
        """Save fee data to JSON file"""
        with open(FEE_DATA_FILE, "w") as f:
            json.dump(st.session_state.fee_data, f, indent=4, default=str)

    def add_student(self, name, student_id, class_name, parent_contact):
        """Add a new student"""
        student = {
            'id': str(uuid.uuid4()),
            'name': name,
            'student_id': student_id,
            'class': class_name,
            'parent_contact': parent_contact,
            'registration_date': datetime.now().isoformat()
        }
        st.session_state.fee_data['students'].append(student)
        self.save_fee_data()
        return student['id']

    def add_fee_structure(self, class_name, term, tuition, books, uniform, transport, meals, other, total):
        """Add a fee structure"""
        fee_structure = {
            'id': str(uuid.uuid4()),
            'class': class_name,
            'term': term,
            'breakdown': {
                'tuition': tuition,
                'books': books,
                'uniform': uniform,
                'transport': transport,
                'meals': meals,
                'other': other
            },
            'total': total,
            'academic_year': datetime.now().year
        }
        st.session_state.fee_data['fee_structures'].append(fee_structure)
        self.save_fee_data()
        return fee_structure['id']

    def add_payment(self, student_id, amount, payment_date, payment_method, term, description):
        """Add a payment record"""
        payment = {
            'id': str(uuid.uuid4()),
            'student_id': student_id,
            'amount': amount,
            'payment_date': payment_date,
            'payment_method': payment_method,
            'term': term,
            'description': description,
            'timestamp': datetime.now().isoformat()
        }
        st.session_state.fee_data['payments'].append(payment)
        self.save_fee_data()
        return payment['id']

    def upload_receipt(self, payment_id, receipt_data, filename):
        """Upload a receipt for a payment"""
        receipt = {
            'id': str(uuid.uuid4()),
            'payment_id': payment_id,
            'filename': filename,
            'data': receipt_data,
            'upload_date': datetime.now().isoformat()
        }
        st.session_state.fee_data['receipts'].append(receipt)
        self.save_fee_data()
        return receipt['id']

    def get_student_balance(self, student_id, term):
        """Calculate student's balance for a term"""
        student = next((s for s in st.session_state.fee_data['students'] if s['id'] == student_id), None)
        if not student:
            return None

        # Get fee structure for student's class and term
        fee_structure = next((fs for fs in st.session_state.fee_data['fee_structures'] 
                            if fs['class'] == student['class'] and fs['term'] == term), None)
        
        if not fee_structure:
            return None

        # Calculate total payments for this student and term
        total_paid = sum(p['amount'] for p in st.session_state.fee_data['payments'] 
                        if p['student_id'] == student_id and p['term'] == term)

        balance = fee_structure['total'] - total_paid
        
        return {
            'total_fee': fee_structure['total'],
            'total_paid': total_paid,
            'balance': balance,
            'fee_breakdown': fee_structure['breakdown']
        }

    def get_upcoming_due_dates(self, days_ahead=30):
        """Get upcoming fee due dates"""
        # This is a simplified implementation
        # In a real system, you'd have specific due dates for each term
        today = datetime.now().date()
        upcoming_dates = []
        
        # Sample due dates (you can customize this based on your school's schedule)
        term_due_dates = {
            'First Term': datetime(today.year, 9, 30).date(),
            'Second Term': datetime(today.year, 1, 31).date(),
            'Third Term': datetime(today.year, 4, 30).date()
        }
        
        for term, due_date in term_due_dates.items():
            if today <= due_date <= today + timedelta(days=days_ahead):
                upcoming_dates.append({
                    'term': term,
                    'due_date': due_date,
                    'days_remaining': (due_date - today).days
                })
        
        return upcoming_dates

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
        
        # Customizable question limit
        num_questions = st.selectbox(
            "📊 Number of questions:",
            [10, 20, 30, 40, 50],
            help="Select how many questions you want in your quiz"
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
            with st.spinner(f"🔄 Generating {num_questions} questions with Llama 3.3 engine - subomi..."):
                st.session_state.questions = fetch_questions(text_content, quiz_level.lower(), num_questions)
                st.session_state.quiz_generated = True if st.session_state.questions else False

    # Display quiz if generated
    if st.session_state.quiz_generated and st.session_state.questions:
        st.divider()
        st.header(f"📝 Your {len(st.session_state.questions)}-Question Quiz")
        
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

# --- NOTE SHARING & READER APP TAB ---

def notes_sharing_tab():
    """Note Sharing & Reader App Tab Content"""
    st.header("📚 Note Sharing & Reader App")
    st.markdown("*Share and access quality study materials*")
    
    notes_manager = NotesManager()
    
    # Sidebar for filtering
    with st.sidebar:
        st.subheader("🔍 Filter Notes")
        subjects = ["All"] + st.session_state.notes_data.get('subjects', [])
        classes = ["All"] + st.session_state.notes_data.get('classes', [])
        
        selected_subject = st.selectbox("Subject", subjects)
        selected_class = st.selectbox("Class", classes)
        
        # Sort options
        sort_by = st.selectbox("Sort by", ["Upload Date", "Views", "Rating", "Title"])
        sort_order = st.radio("Order", ["Descending", "Ascending"])

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📖 Browse Notes", "📤 Upload Notes", "📊 My Statistics"])
    
    with tab1:
        # Browse and read notes
        st.subheader("📖 Available Study Notes")
        
        # Get filtered notes
        filtered_notes = notes_manager.get_notes_by_filter(
            subject=selected_subject if selected_subject != "All" else None,
            class_name=selected_class if selected_class != "All" else None
        )
        
        # Sort notes
        if sort_by == "Upload Date":
            filtered_notes.sort(key=lambda x: x['upload_date'], reverse=(sort_order == "Descending"))
        elif sort_by == "Views":
            filtered_notes.sort(key=lambda x: x['views'], reverse=(sort_order == "Descending"))
        elif sort_by == "Rating":
            filtered_notes.sort(key=lambda x: st.session_state.notes_data['ratings'].get(x['id'], {}).get('average', 0), reverse=(sort_order == "Descending"))
        else:  # Title
            filtered_notes.sort(key=lambda x: x['title'], reverse=(sort_order == "Descending"))
        
        if filtered_notes:
            for note in filtered_notes:
                with st.expander(f"📄 {note['title']} - {note['subject']} ({note['class']})"):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**Author:** {note['author']}")
                        st.write(f"**Subject:** {note['subject']}")
                        st.write(f"**Class:** {note['class']}")
                        st.write(f"**Upload Date:** {note['upload_date'][:10]}")
                        
                        # Rating display
                        rating_data = st.session_state.notes_data['ratings'].get(note['id'], {})
                        avg_rating = rating_data.get('average', 0)
                        rating_count = rating_data.get('count', 0)
                        st.write(f"**Rating:** {'⭐' * int(avg_rating)} ({avg_rating:.1f}/5.0 from {rating_count} ratings)")
                    
                    with col2:
                        st.metric("👁️ Views", note['views'])
                        st.metric("📥 Downloads", note['downloads'])
                    
                    with col3:
                        # Read button
                        if st.button(f"📖 Read", key=f"read_{note['id']}"):
                            notes_manager.increment_views(note['id'])
                            st.session_state[f"reading_{note['id']}"] = True
                            st.rerun()
                        
                        # Download button
                        if st.button(f"📥 Download", key=f"download_{note['id']}"):
                            notes_manager.increment_downloads(note['id'])
                            st.download_button(
                                label="💾 Save File",
                                data=note['content'],
                                file_name=f"{note['title']}.txt",
                                mime="text/plain"
                            )
                    
                    # Reading view
                    if st.session_state.get(f"reading_{note['id']}", False):
                        st.divider()
                        st.subheader(f"📖 Reading: {note['title']}")
                        
                        # Content display
                        st.text_area("Content", note['content'], height=300, disabled=True)
                        
                        # Audio notes (if available)
                        if note.get('audio_transcript'):
                            st.subheader("🎵 Audio Notes Transcript")
                            st.text_area("Audio Transcript", note['audio_transcript'], height=150, disabled=True)
                        
                        # Rating system
                        st.subheader("⭐ Rate this Note")
                        user_rating = st.select_slider(
                            "Your Rating",
                            options=[1, 2, 3, 4, 5],
                            value=3,
                            key=f"rating_{note['id']}"
                        )
                        
                        if st.button(f"Submit Rating", key=f"rate_{note['id']}"):
                            notes_manager.add_rating(note['id'], user_rating)
                            st.success("✅ Rating submitted!")
                            st.rerun()
                        
                        # Comments section
                        st.subheader("💬 Comments")
                        
                        # Add comment
                        with st.form(f"comment_{note['id']}"):
                            comment_author = st.text_input("Your Name")
                            comment_text = st.text_area("Comment")
                            
                            if st.form_submit_button("Post Comment"):
                                if comment_author and comment_text:
                                    notes_manager.add_comment(note['id'], comment_author, comment_text)
                                    st.success("✅ Comment posted!")
                                    st.rerun()
                        
                        # Display comments
                        comments = notes_manager.get_comments_for_note(note['id'])
                        if comments:
                            for comment in sorted(comments, key=lambda x: x['timestamp'], reverse=True):
                                st.write(f"**{comment['author']}** - {comment['timestamp'][:19]}")
                                st.write(comment['comment'])
                                st.divider()
                        else:
                            st.info("No comments yet. Be the first to comment!")
                        
                        # Close reading view
                        if st.button(f"❌ Close", key=f"close_{note['id']}"):
                            st.session_state[f"reading_{note['id']}"] = False
                            st.rerun()
        else:
            st.info("📝 No notes found matching your criteria. Try adjusting the filters or upload some notes!")
    
    with tab2:
        # Upload new notes
        st.subheader("📤 Upload Study Notes")
        
        with st.form("upload_note"):
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("Note Title*", placeholder="e.g., Introduction to Calculus")
                author = st.text_input("Your Name*", placeholder="Your name")
                subject = st.text_input("Subject*", placeholder="e.g., Mathematics")
            
            with col2:
                class_name = st.text_input("Class/Grade*", placeholder="e.g., Grade 12")
                file_type = st.selectbox("File Type", ["Text", "PDF", "Word Document", "PowerPoint"])
            
            # Content input
            content = st.text_area("Note Content*", height=200, placeholder="Enter your notes here...")
            
            # Audio notes
            st.subheader("🎵 Audio Notes (Optional)")
            audio_transcript = st.text_area("Audio Transcript", height=100, 
                                          placeholder="If you have audio notes, provide the transcript here...")
            
            # File upload (placeholder for future implementation)
            uploaded_file = st.file_uploader("Upload File (Optional)", 
                                           type=['txt', 'pdf', 'docx', 'pptx'])
            
            if st.form_submit_button("📤 Upload Note", type="primary"):
                if title and author and subject and class_name and content:
                    # Handle file upload if provided
                    if uploaded_file:
                        file_content = uploaded_file.read()
                        if file_type == "Text":
                            content = file_content.decode('utf-8')
                    
                    note_id = notes_manager.add_note(
                        title=title,
                        content=content,
                        subject=subject,
                        class_name=class_name,
                        file_type=file_type,
                        author=author,
                        audio_transcript=audio_transcript if audio_transcript else None
                    )
                    
                    st.success(f"✅ Note '{title}' uploaded successfully!")
                    st.info(f"📝 Note ID: {note_id}")
                    st.rerun()
                else:
                    st.error("❌ Please fill all required fields!")
    
    with tab3:
        # Statistics and analytics
        st.subheader("📊 Notes Statistics")
        
        notes = st.session_state.notes_data.get('notes', [])
        
        if notes:
            # Overall stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📚 Total Notes", len(notes))
            
            with col2:
                total_views = sum(note['views'] for note in notes)
                st.metric("👁️ Total Views", total_views)
            
            with col3:
                total_downloads = sum(note['downloads'] for note in notes)
                st.metric("📥 Total Downloads", total_downloads)
            
            with col4:
                avg_rating = np.mean([
                    st.session_state.notes_data['ratings'].get(note['id'], {}).get('average', 0)
                    for note in notes
                ])
                st.metric("⭐ Avg Rating", f"{avg_rating:.1f}")
            
            # Subject distribution
            st.subheader("📊 Notes by Subject")
            subject_counts = {}
            for note in notes:
                subject_counts[note['subject']] = subject_counts.get(note['subject'], 0) + 1
            
            if subject_counts:
                fig = px.pie(
                    values=list(subject_counts.values()),
                    names=list(subject_counts.keys()),
                    title="Distribution of Notes by Subject"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Most popular notes
            st.subheader("🔥 Most Popular Notes")
            popular_notes = sorted(notes, key=lambda x: x['views'], reverse=True)[:5]
            
            for i, note in enumerate(popular_notes, 1):
                st.write(f"{i}. **{note['title']}** ({note['subject']}) - {note['views']} views")
        else:
            st.info("📝 No notes uploaded yet. Upload some notes to see statistics!")

# --- SCHOOL FEE TRACKER TAB ---

def fee_tracker_tab():
    """School Fee Tracker Tab Content"""
    st.header("💰 School Fee Tracker")
    st.markdown("*Never miss a fee deadline or overpay again*")
    
    fee_tracker = FeeTracker()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["👥 Students", "💳 Fee Structure", "💰 Payments", "📊 Dashboard", "⚠️ Alerts"])
    
    with tab1:
        # Student Management
        st.subheader("👥 Student Management")
        
        # Add new student
        with st.expander("➕ Add New Student"):
            with st.form("add_student"):
                col1, col2 = st.columns(2)
                
                with col1:
                    student_name = st.text_input("Student Name*")
                    student_id = st.text_input("Student ID*")
                
                with col2:
                    class_name = st.text_input("Class/Grade*")
                    parent_contact = st.text_input("Parent Contact")
                
                if st.form_submit_button("Add Student", type="primary"):
                    if student_name and student_id and class_name:
                        fee_tracker.add_student(student_name, student_id, class_name, parent_contact)
                        st.success(f"✅ Student '{student_name}' added successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Please fill all required fields!")
        
        # Display students
        students = st.session_state.fee_data.get('students', [])
        if students:
            st.subheader("📋 All Students")
            
            # Create DataFrame for better display
            students_df = pd.DataFrame(students)
            students_df['registration_date'] = pd.to_datetime(students_df['registration_date']).dt.date
            
            st.dataframe(
                students_df[['name', 'student_id', 'class', 'parent_contact', 'registration_date']],
                use_container_width=True
            )
        else:
            st.info("👥 No students registered yet. Add your first student above!")
    
    with tab2:
        # Fee Structure Management
        st.subheader("💳 Fee Structure Management")
        
        # Add fee structure
        with st.expander("➕ Add Fee Structure"):
            with st.form("add_fee_structure"):
                col1, col2 = st.columns(2)
                
                with col1:
                    fee_class = st.text_input("Class/Grade*")
                    term = st.selectbox("Term*", ["First Term", "Second Term", "Third Term"])
                
                with col2:
                    academic_year = st.number_input("Academic Year", 
                                                  min_value=2020, 
                                                  max_value=2030, 
                                                  value=datetime.now().year)
                
                st.subheader("💰 Fee Breakdown")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    tuition = st.number_input("Tuition Fee", min_value=0.0, step=100.0)
                    books = st.number_input("Books & Materials", min_value=0.0, step=50.0)
                
                with col2:
                    uniform = st.number_input("Uniform", min_value=0.0, step=50.0)
                    transport = st.number_input("Transport", min_value=0.0, step=100.0)
                
                with col3:
                    meals = st.number_input("Meals", min_value=0.0, step=50.0)
                    other = st.number_input("Other Fees", min_value=0.0, step=50.0)
                
                total = tuition + books + uniform + transport + meals + other
                st.metric("💰 Total Fee", f"${total:,.2f}")
                
                if st.form_submit_button("Add Fee Structure", type="primary"):
                    if fee_class and term:
                        fee_tracker.add_fee_structure(
                            fee_class, term, tuition, books, uniform, 
                            transport, meals, other, total
                        )
                        st.success(f"✅ Fee structure for {fee_class} - {term} added!")
                        st.rerun()
                    else:
                        st.error("❌ Please fill required fields!")
        
        # Display fee structures
        fee_structures = st.session_state.fee_data.get('fee_structures', [])
        if fee_structures:
            st.subheader("📋 Current Fee Structures")
            
            for structure in fee_structures:
                with st.expander(f"💳 {structure['class']} - {structure['term']} ({structure['academic_year']})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Fee Breakdown:**")
                        for item, amount in structure['breakdown'].items():
                            st.write(f"• {item.title()}: ${amount:,.2f}")
                    
                    with col2:
                        st.metric("💰 Total Fee", f"${structure['total']:,.2f}")
        else:
            st.info("💳 No fee structures defined yet. Add fee structures above!")
    
    with tab3:
        # Payment Management
        st.subheader("💰 Payment Management")
        
        students = st.session_state.fee_data.get('students', [])
        
        if not students:
            st.warning("👥 Please add students first before recording payments.")
            return
        
        # Record new payment
        with st.expander("➕ Record New Payment"):
            with st.form("add_payment"):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Student selection
                    student_options = {s['id']: f"{s['name']} ({s['student_id']})" for s in students}
                    selected_student_id = st.selectbox(
                        "Select Student*",
                        options=list(student_options.keys()),
                        format_func=lambda x: student_options[x]
                    )
                    
                    amount = st.number_input("Payment Amount*", min_value=0.01, step=10.0)
                    payment_method = st.selectbox("Payment Method*", 
                                                ["Cash", "Bank Transfer", "Check", "Credit Card", "Mobile Money"])
                
                with col2:
                    payment_date = st.date_input("Payment Date*", value=datetime.now().date())
                    term = st.selectbox("Term*", ["First Term", "Second Term", "Third Term"])
                    description = st.text_area("Description/Notes", placeholder="Optional payment description...")
                
                # Receipt upload
                st.subheader("📄 Upload Receipt (Optional)")
                receipt_file = st.file_uploader("Upload Receipt", type=['jpg', 'jpeg', 'png', 'pdf'])
                
                if st.form_submit_button("Record Payment", type="primary"):
                    if selected_student_id and amount > 0 and payment_method and term:
                        payment_id = fee_tracker.add_payment(
                            selected_student_id, amount, payment_date.isoformat(), 
                            payment_method, term, description
                        )
                        
                        # Handle receipt upload
                        if receipt_file:
                            receipt_data = base64.b64encode(receipt_file.read()).decode()
                            fee_tracker.upload_receipt(payment_id, receipt_data, receipt_file.name)
                        
                        st.success(f"✅ Payment of ${amount:,.2f} recorded successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Please fill all required fields!")
        
        # Display payments
        payments = st.session_state.fee_data.get('payments', [])
        if payments:
            st.subheader("📋 Payment History")
            
            # Create payments DataFrame with student names
            payments_display = []
            for payment in payments:
                student = next((s for s in students if s['id'] == payment['student_id']), None)
                if student:
                    payments_display.append({
                        'Student': f"{student['name']} ({student['student_id']})",
                        'Amount': f"${payment['amount']:,.2f}",
                        'Date': payment['payment_date'],
                        'Method': payment['payment_method'],
                        'Term': payment['term'],
                        'Description': payment.get('description', '')
                    })
            
            if payments_display:
                payments_df = pd.DataFrame(payments_display)
                st.dataframe(payments_df, use_container_width=True)
        else:
            st.info("💰 No payments recorded yet. Record your first payment above!")
    
    with tab4:
        # Dashboard and Analytics
        st.subheader("📊 Fee Tracking Dashboard")
        
        students = st.session_state.fee_data.get('students', [])
        payments = st.session_state.fee_data.get('payments', [])
        fee_structures = st.session_state.fee_data.get('fee_structures', [])
        
        if not students or not fee_structures:
            st.warning("📊 Please add students and fee structures to see the dashboard.")
            return
        
        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Total Students", len(students))
        
        with col2:
            total_collected = sum(p['amount'] for p in payments)
            st.metric("💰 Total Collected", f"${total_collected:,.2f}")
        
        with col3:
            st.metric("💳 Payment Records", len(payments))
        
        with col4:
            current_month_payments = [
                p for p in payments 
                if datetime.fromisoformat(p['payment_date']).month == datetime.now().month
            ]
            monthly_total = sum(p['amount'] for p in current_month_payments)
            st.metric("📅 This Month", f"${monthly_total:,.2f}")
        
        # Student balances
        st.subheader("💰 Student Balances")
        
        term_filter = st.selectbox("Select Term for Balance View", 
                                 ["First Term", "Second Term", "Third Term"])
        
        balance_data = []
        for student in students:
            balance_info = fee_tracker.get_student_balance(student['id'], term_filter)
            if balance_info:
                status = "✅ Paid" if balance_info['balance'] <= 0 else "⚠️ Outstanding"
                balance_data.append({
                    'Student': f"{student['name']} ({student['student_id']})",
                    'Class': student['class'],
                    'Total Fee': f"${balance_info['total_fee']:,.2f}",
                    'Paid': f"${balance_info['total_paid']:,.2f}",
                    'Balance': f"${balance_info['balance']:,.2f}",
                    'Status': status
                })
        
        if balance_data:
            balance_df = pd.DataFrame(balance_data)
            st.dataframe(balance_df, use_container_width=True)
            
            # Payment status chart
            st.subheader("📊 Payment Status Overview")
            paid_count = len([b for b in balance_data if "✅" in b['Status']])
            outstanding_count = len(balance_data) - paid_count
            
            fig = px.pie(
                values=[paid_count, outstanding_count],
                names=['Fully Paid', 'Outstanding Balance'],
                title=f"Payment Status - {term_filter}",
                color_discrete_map={'Fully Paid': '#00cc44', 'Outstanding Balance': '#ff6b6b'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"📊 No balance data available for {term_filter}. Check fee structures and payments.")
        
        # Fee Calculator
        st.subheader("🧮 Fee Payment Calculator")
        
        with st.form("fee_calculator"):
            col1, col2 = st.columns(2)
            
            with col1:
                calc_student_id = st.selectbox(
                    "Select Student",
                    options=[s['id'] for s in students],
                    format_func=lambda x: next(f"{s['name']} ({s['student_id']})" for s in students if s['id'] == x)
                )
                calc_term = st.selectbox("Term", ["First Term", "Second Term", "Third Term"])
            
            with col2:
                payment_amount = st.number_input("Payment Amount", min_value=0.0, step=10.0)
            
            if st.form_submit_button("Calculate Balance"):
                balance_info = fee_tracker.get_student_balance(calc_student_id, calc_term)
                if balance_info:
                    new_balance = balance_info['balance'] - payment_amount
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Balance", f"${balance_info['balance']:,.2f}")
                    with col2:
                        st.metric("Payment Amount", f"${payment_amount:,.2f}")
                    with col3:
                        st.metric("New Balance", f"${new_balance:,.2f}")
                    
                    if new_balance <= 0:
                        st.success("✅ This payment will clear the balance!")
                        if new_balance < 0:
                            st.info(f"💰 Overpayment: ${abs(new_balance):,.2f}")
                    else:
                        st.warning(f"⚠️ Remaining balance: ${new_balance:,.2f}")
    
    with tab5:
        # Alerts and Due Dates
        st.subheader("⚠️ Fee Alerts & Due Dates")
        
        # Upcoming due dates
        st.subheader("📅 Upcoming Due Dates")
        upcoming_dues = fee_tracker.get_upcoming_due_dates()
        
        if upcoming_dues:
            for due in upcoming_dues:
                days_left = due['days_remaining']
                if days_left <= 7:
                    alert_type = "error"
                    icon = "🚨"
                elif days_left <= 14:
                    alert_type = "warning"
                    icon = "⚠️"
                else:
                    alert_type = "info"
                    icon = "📅"
                
                getattr(st, alert_type)(
                    f"{icon} **{due['term']}** due in {days_left} days ({due['due_date']})"
                )
        else:
            st.info("📅 No upcoming due dates in the next 30 days.")
        
        # Outstanding balances alert
        st.subheader("💰 Outstanding Balance Alerts")
        
        students = st.session_state.fee_data.get('students', [])
        current_term = st.selectbox("Check Term", ["First Term", "Second Term", "Third Term"])
        
        outstanding_students = []
        for student in students:
            balance_info = fee_tracker.get_student_balance(student['id'], current_term)
            if balance_info and balance_info['balance'] > 0:
                outstanding_students.append({
                    'name': student['name'],
                    'student_id': student['student_id'],
                    'balance': balance_info['balance'],
                    'contact': student.get('parent_contact', 'N/A')
                })
        
        if outstanding_students:
            st.warning(f"⚠️ {len(outstanding_students)} students have outstanding balances for {current_term}")
            
            for student in outstanding_students:
                with st.expander(f"💰 {student['name']} ({student['student_id']}) - ${student['balance']:,.2f}"):
                    st.write(f"**Outstanding Amount:** ${student['balance']:,.2f}")
                    st.write(f"**Parent Contact:** {student['contact']}")
                    
                    # Quick actions
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"📞 Send Reminder", key=f"remind_{student['student_id']}"):
                            st.info(f"📱 Reminder sent to {student['contact']}")
                    
                    with col2:
                        if st.button(f"💰 Record Payment", key=f"pay_{student['student_id']}"):
                            st.info("💳 Redirect to payment recording...")
        else:
            st.success(f"✅ All students have cleared their fees for {current_term}!")
        
        # Fee collection summary
        st.subheader("📊 Collection Summary")
        
        payments = st.session_state.fee_data.get('payments', [])
        if payments:
            # Monthly collection chart
            monthly_collections = {}
            for payment in payments:
                payment_date = datetime.fromisoformat(payment['payment_date'])
                month_key = payment_date.strftime("%Y-%m")
                monthly_collections[month_key] = monthly_collections.get(month_key, 0) + payment['amount']
            
            if monthly_collections:
                months = list(monthly_collections.keys())
                amounts = list(monthly_collections.values())
                
                fig = px.bar(
                    x=months,
                    y=amounts,
                    title="Monthly Fee Collections",
                    labels={'x': 'Month', 'y': 'Amount Collected ($)'}
                )
                st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(
        page_title="Baptist Academy Smart School AI (BASS A.I)",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Baptist Academy Smart School AI (BASS A.I)")
    st.markdown("*Complete Educational Tool with Quiz Generation, Result Analysis, Smart Scheduling, Note Sharing & Fee Tracking by Baptist Academy*")
    
    # Test API connection on startup
    api_status, api_message = test_api_connection()
    if not api_status:
        st.warning(f"⚠️ API Connection Issue: {api_message}")
    
    # Top navigation
    selected_tab = option_menu(
        menu_title=None,
        options=["Quiz Generator", "Result Analyzer", "Smart Scheduler", "Note Sharing", "Fee Tracker"],
        icons=["question-circle", "graph-up", "calendar-check", "book", "credit-card"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

    # Main content based on selected tab
    if selected_tab == "Quiz Generator":
        quiz_generator_tab()
    elif selected_tab == "Result Analyzer":
        result_analyzer_tab()
    elif selected_tab == "Smart Scheduler":
        scheduler_tab()
    elif selected_tab == "Note Sharing":
        notes_sharing_tab()
    else:  # Fee Tracker
        fee_tracker_tab()

if __name__ == "__main__":
    main()