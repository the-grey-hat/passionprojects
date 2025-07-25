# 🤖 Baptist Academy Smart School AI (BASS A.I)

A comprehensive educational tool with AI-powered quiz generation, result analysis, target prediction, and smart scheduling.

## ✨ Features

### 🧠 AI Quiz Generator
- **Powered by Llama 3.3** via OpenRouter API
- Generate 10 multiple-choice questions from any text
- Three difficulty levels: Easy, Medium, Hard
- Automatic scoring and performance feedback
- Save results to Result Analyzer

### 📈 Result Analyzer & Target Predictor
- Track exam scores across multiple subjects
- Visual performance charts and trends
- Weighted grade calculations
- **Target Score Prediction**: Calculate what you need on your next exam
- Export reports as PDF or CSV
- AI-powered insights and recommendations

### 📅 Smart Scheduler
- Manage class timetables
- Track homework assignments with due dates
- Real-time notifications for class periods and due homework
- School/Weekend mode toggle
- Complete data management (add, edit, delete)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenRouter API key (get from [openrouter.ai](https://openrouter.ai/))

### Installation

1. **Clone or download the files**
   ```bash
   # If you have git
   git clone <repository-url>
   cd bass-ai
   
   # Or download the files directly
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Copy `.env.example` to `.env` (or create `.env`)
   - Add your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_actual_api_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run debug_app.py
   ```

## 🔧 What Was Fixed

### Major Bug Fixes:
1. **API Request Typo**: Fixed `requests.poist()` → `requests.post()`
2. **Duplicate Configuration**: Removed redundant OpenRouter API setup code
3. **Clean Code Structure**: Better organization and error handling
4. **Session State Management**: Proper initialization of all variables
5. **UI Improvements**: Enhanced user experience with better emojis and feedback

### Key Improvements:
- ✅ Robust error handling for API calls
- ✅ Startup API connection testing
- ✅ Cleaner configuration management
- ✅ Better user feedback and notifications
- ✅ Consistent UI styling throughout

## 📊 Usage Guide

### Quiz Generator
1. Paste your study text (minimum 50 characters)
2. Select difficulty level
3. Click "Generate Quiz"
4. Answer questions and submit
5. View detailed results and save to records

### Result Analyzer
1. Add exam results manually or from quiz generator
2. View performance dashboard with charts
3. Use Target Predictor to calculate required scores
4. Export reports and data

### Smart Scheduler
1. Add subjects first
2. Set up your class timetable
3. Add homework assignments
4. Toggle between school/weekend modes
5. Receive automatic notifications

## 🛠️ Configuration

### OpenRouter API Setup
1. Visit [openrouter.ai](https://openrouter.ai/)
2. Sign up/login and get your API key
3. Add to `.env` file or Streamlit secrets

### Streamlit Cloud Deployment
If deploying to Streamlit Cloud:
1. Add `OPENROUTER_API_KEY` in **Settings** → **Secrets**
2. Format: `OPENROUTER_API_KEY = "your_key_here"`

## 📁 File Structure
```
bass-ai/
├── debug_app.py          # Main debugged application
├── requirements.txt      # Python dependencies
├── .env                 # Environment variables (create this)
├── README.md            # This file
└── scheduler_data.json  # Auto-created scheduler data
```

## 🔍 Troubleshooting

### Common Issues:

1. **API Key Error**
   - Ensure `.env` file exists with correct API key
   - For Streamlit Cloud, check secrets configuration

2. **Module Import Errors**
   - Activate virtual environment
   - Install all requirements: `pip install -r requirements.txt`

3. **Quiz Generation Fails**
   - Check internet connection
   - Verify API key is valid
   - Ensure text input is at least 50 characters

4. **Charts Not Displaying**
   - Clear browser cache
   - Refresh the page
   - Check browser console for errors

## 🆘 Support

If you encounter issues:
1. Check the error messages in the app
2. Verify your API key is working
3. Ensure all dependencies are installed
4. Check the browser console for JavaScript errors

## 🎯 Tips for Best Results

### Quiz Generator:
- Use well-structured, informative text
- Longer texts (200+ words) produce better questions
- Try different difficulty levels for varied question types

### Result Analyzer:
- Input accurate scores for better predictions
- Use consistent subject naming
- Regularly update your exam records

### Smart Scheduler:
- Set up all subjects before creating timetables
- Use specific homework descriptions
- Check notifications regularly

## 🔒 Privacy & Data

- All data is stored locally in your browser session
- Scheduler data is saved to `scheduler_data.json`
- No personal data is sent to external services except for quiz generation
- API calls to OpenRouter only include the text for quiz generation

---

**Developed by Baptist Academy** 🎓

*Empowering education through AI technology*
