# 📋 Engosoft Sales Call QA Platform (Arabic QA Scorecard & AI Coaching)

An incredibly professional, modern, and production-ready **Sales Call Quality Assurance (QA) Dashboard** built with Streamlit. Designed specifically for auditing and coaching sales departments under Engosoft's strict quality rules, featuring a complete Right-to-Left (RTL) layout, Google Font 'Tajawal' typography, isolated call session histories, and a dedicated AI Coaching chatbot for each call.

Developed for seamless deployment to the **Streamlit Community Cloud** and optimized for GitHub version control.

---

## 🌟 Key System Features

1. **Session & History Management (Sidebar)**
   - Powered by `st.session_state` to handle multiple distinct call audit sessions.
   - **➕ مكالمة جديدة (New Chat)** button to immediately reset the UI and analyze a new call transcript or upload an audio recording.
   - **Interactive Call History List** featuring visual status emojis (🟢 for final score >= 80%, 🔴 for final score < 80%) along with timestamps.
   - Clicking any historical entry swaps the main screen context to display its specific scorecard evaluation metrics and isolated coaching chatbot history.

2. **Strict QA Scorecard Compliance (Arabic Only)**
   - Renders quality audits strictly conforming to the predefined Arabic QA matrix format:
     - `🎧 نص المكالمة`: Speaker-tagged dialogue translated to formal sales Arabic.
     - `📌 نوع المكالمة`: Fresh Lead or Follow-up categorization.
     - `📊 ملخص السكور`: Percentage final score and Auto-Fail indicator.
     - `❌ الأخطاء`: Bulleted list detailing compliance infractions and point deductions.
     - `📋 جدول التقييم`: Structured markdown matrix detailing category, item, status (TRUE/FALSE), and score impact.
     - `🧠 ملاحظات الكواليتي (Coaching)`: Actionable feedback split into Strengths, Main Errors, and an Improvement Plan.
     - `🤖 تحليل ذكي`: Buyer temperature type, buying signals, missed opportunities, and converting probability.

3. **Isolated Chatbot Per Call (AI Coaching Assistant)**
   - Positioned beneath each call's evaluation scorecard.
   - Chat histories are isolated and saved per call (`st.session_state.sessions[session_id]['chat_history']`).
   - Acts as an **AI Coaching Assistant** to help auditors and sales managers practice objection handling, brainstorm alternative sales pitches, and generate high-converting scripts based on the call transcript.
   - Smart offline mock responses trigger if no API key is supplied (e.g. providing real-world scripts for objections when asked for alternatives).

4. **UI/UX & Arabic RTL Support**
   - Custom CSS injected using `unsafe_allow_html=True` to enforce a gorgeous RTL (Right-to-Left) reading experience.
   - Professional Arabic font family **Tajawal** imported dynamically from Google Fonts.
   - Curated navy/slate color palette with custom background styles for chat bubbles and score metric cards.

5. **API & Offline Safe Mode**
   - Runs in **Demo / Mock Mode** out-of-the-box using preloaded high and low scoring compliance audits.
   - New transcripts pasted in Demo Mode are dynamically audited using local regex/keyword matrices to output realistic evaluations.
   - Paste a `GEMINI_API_KEY` into the sidebar to activate the live Gemini 2.5 Flash reasoning engine.

---

## 🛠️ Local Installation & Setup

Ensure you have Python 3.9 or higher installed on your computer.

1. **Clone the Repository:**
   ```bash
   git clone <your-repository-url>
   cd <repository-directory>
   ```

2. **Set Up Python Environment:**
   It is recommended to run in a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables (Optional):**
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```

5. **Run the Application:**
   Start the Streamlit development server:
   ```bash
   streamlit run app.py
   ```
   Open your browser and navigate to `http://localhost:8501`.

---

## 🚀 Step-by-Step Deployment to Streamlit Community Cloud

Streamlit Community Cloud is the best platform to host this application. Follow these instructions:

1. **Push Code to GitHub:**
   Ensure all files (`app.py`, `requirements.txt`, `packages.txt`, `.gitignore`) are committed and pushed to your public or private GitHub repository.

2. **Sign In to Streamlit Community Cloud:**
   Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.

3. **Deploy a New App:**
   - Click the **"New app"** button.
   - Select your Repository, Branch (e.g., `main`), and set the Main file path to `app.py`.
   - Click **"Deploy!"**.

4. **Configure Environment Secrets (For Gemini Integration):**
   - Once the application is deployed, click on the **"Settings"** icon in the bottom-right corner of the Streamlit dashboard screen.
   - In the settings pop-up, navigate to the **"Secrets"** tab.
   - Enter your API Key in TOML format:
     ```toml
     GEMINI_API_KEY = "your_actual_gemini_api_key_here"
     ```
   - Click **"Save"**. Streamlit will automatically restart your app with the secret active.

---

## 📂 Project Architecture

```
├── app.py              # Main dashboard application code (RTL styling, session flow, schemas)
├── main.py             # Console testing runner (Pydantic schema definitions, system instructions)
├── requirements.txt    # Python dependencies for Streamlit Cloud deployment
├── packages.txt        # Linux compilers required on Streamlit server
├── .env                # Environment secrets configuration (Excluded from git)
└── README.md           # Professional project documentation
```
