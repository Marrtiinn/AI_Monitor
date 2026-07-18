import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load hidden environment variables and initialize Gemini Client
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found! Please check your .env file setup.")

client = genai.Client(api_key=api_key)

# ==========================================
# 2. Structural Schema Definition (Pydantic)
# ==========================================

class EvaluationRow(BaseModel):
    category: str = Field(description="The evaluation block (e.g., Introduction, Company Profiling, Needs Discovery...)")
    item: str = Field(description="The targeted item criteria evaluated (e.g., Greeting, Ice Breaking, Original vs Discounted Price...)")
    status: bool = Field(description="Set to True if completely and accurately fulfilled, False if missed or flawed")
    impact: str = Field(description="The final metric penalty score calculated for this single rule (e.g., '0%', '-5%', '-2%', '+15%')")

class QualityReport(BaseModel):
    transcription: str = Field(description="Fully cleaned, optimized, speaker-tagged dialogue mapping text strictly translated to formal sales Arabic.")
    call_type: str = Field(description="Context categorization evaluated strictly as 'Fresh Lead' or 'Follow-up'")
    final_score: str = Field(description="Calculated final percentage string based on a 100% baseline (e.g., '85%')")
    auto_fail: bool = Field(description="Boolean verification flag stating whether any compliance breach triggered an immediate 0% auto-fail score status")
    errors_list: List[str] = Field(description="Detailed compliance report specifying each recorded violation along with its numeric impact penalty breakdown")
    evaluation_table: List[EvaluationRow] = Field(description="Linear array tracking the evaluation outcome state across all metrics")
    strengths: List[str] = Field(description="Core strategic advantages, proper phrase usage, or strong behavioral observations logged from the employee")
    main_errors: List[str] = Field(description="Core structural errors, missing information statements, or critical process errors committed")
    improvement_plan: List[str] = Field(description="Direct corrective training paths split strictly into actionable scripts, enhanced closing patterns, and objection frameworks")
    client_type: str = Field(description="Inferred buyer interest classification profile: Hot / Warm / Cold")
    buying_signals: List[str] = Field(description="Recorded client buying indicators, underlying product interest points, or payment alignment queries")
    missed_opportunities: List[str] = Field(description="Unexploited deal accelerators, cross/up-selling gaps, or value reinforcement points left unsaid")
    closing_probability: str = Field(description="Calculated logical evaluation of conversion success written as a percentage string (e.g., '75%')")

# ==========================================
# 3. Quality Assurance Engine Logic
# ==========================================

SYSTEM_PROMPT = """
You are a highly precise and uncompromising automated Quality Assurance (QA) Specialist analyzing sales calls for Engosoft.
Your objective is to evaluate calls strictly according to the following 4-stage pipeline guidelines:

STAGE 1 & 2: TRANSCRIPTION, CLEANING, AND TRANSLATION
- If the input is audio, process and transcribe it. If text, clean and structure it.
- Log conversational streams using absolute Speaker Diarization formatting (الموظف: ... / العميل: ...).
- MANDATORY CRITERIA: Regardless of input language (English, slang, or local dialects), the output transcript must be translated and cleanly recorded entirely in formal, high-impact sales Arabic.
- Strip all verbal filler components, structural stutters, and operational tokens (e.g., "um", "uh", "ah") to yield optimized readability.

STAGE 3: CALL CONTEXT CLASSIFICATION
- Prior to evaluation, classify the dialogue context profile as:
  * Fresh Lead: First-time engagement with no established communication record. Focus areas: Greeting mechanics, structural profiling, identity verification, and implicit need discovery.
  * Follow-up: Re-engagement loop with an active proposal on record. Focus areas: Deal velocity, objection neutralizing frameworks, alternative payment processing, and final closing execution.
- Apply contextual fairness rules. Never penalize a Fresh Lead for lacking closing aggression or long-term follow-up setups. Never over-penalize a Follow-up for bypassing a long-form company profile intro.

STAGE 4: QUALITY ASSURANCE (QA) COMPLIANCE RULES
- Base Score initialization: 100%. Apply negative point adjustments per infraction.
- Strict Objective Testing: Do not extrapolate or assume missing execution parameters. If not explicitly verified in the script, log as FALSE.

DEDUCTION MATRIX RATINGS:
1. Introduction (الافتتاحية)
   - Greeting (الترحيب) [-2%]
   - Ice Breaking (كسر الجليد) [-1%]
   - Establishing Reason for Call (تحديد سبب الاتصال) [-2%]
2. Company Profiling (التعريف)
   - Profile Failure [-5%] (Must explicitly mention all 3: 'Engosoft' + '13+ years of experience' + 'Location: Riyadh, Al-Yarmouk District, Rajia Street'). Missing any element triggers the full deduction.
3. Needs Discovery & Analysis (تحليل الاحتياج)
   - Technical Questions [-2%] (e.g., current job title, current workflow, tenure length).
   - Intent/Need Questions [-3%] (e.g., reason for choosing this course, professional objective).
   - Investment/Background Questions [-5%] (e.g., prior certifications held, history of training courses taken).
4. Product Knowledge (معرفة المنتج)
   - Content Coverage [-1%], Software Used [-1%], Coding/Tools Used [-1%], Duration [-1%], Attendance Type/Format [-1%], Certifications Provided [-1%], Trainer Bio/Name (At least name) [-1%].
5. Additional Sales (البيع الإضافي)
   - Cross-Selling Failure [-2%]
   - Up-Selling Failure [-3%]
6. Language & Etiquette (اللغة)
   - Blaming the Customer [-1%], Unprofessional Language Usage [-2%]
7. Business Critical Errors (أخطاء حرجة)
   - Call Control: Active Listening/Non-interruption [-3%]
   - Negotiation: Presenting Center-specific Benefits [-2%], Defining exact ROI Value Proposition to customer [-1%]
   - Pricing Mechanics: Bypassing Original vs. Discounted Price comparative display [-2%]
   - Payment Methods: Missing active options promotion (Cash, Transfer, Tabby, Tamara installment payment frameworks) [-2%]
   - Objection Handling: Price Objections [-1%], Time/Schedule Objections [-1%], Content Objections [-2%], Brand Objections [-1%]
   - Closing Execution: Setting concrete Follow-up Time [-2%], Commitment to Send Price Quotation [-3%], Clear Deadline Definition [-2%], Total Follow-up Protocol Neglect [-3%]
8. Auto Fail Conditions [Instant 0% Total Score Assignment]
   - Misleading/False declarations, Incomplete critical statements, Aggressive behavioral posture, Evading corporate responsibility.
9. Bonus Structure [+15% Addition]
   - Client Recommendation/Referrals: Explicitly log whether prompted by the employee or suggested independently by the client, specifying who initiated it.

OUTPUT FORMAT STANDARD:
Generate the entire response object strictly mapping to the predefined JSON structure, maintaining standard terminology output formatting guidelines.
"""

def analyze_call_with_gemini(input_data: str, is_audio_file: bool = False) -> QualityReport:
    """Uploads audio or parses text input to compute compliance scoring through Gemini."""
    contents = []

    if is_audio_file:
        print("⏳ Uploading native media asset file directly to Google AI Studio storage...")
        audio_file = client.files.upload(file=input_data)
        contents.append(audio_file)
        contents.append("Transcribe the attached file, perform structural auditing, and compile the report details.")
    else:
        contents.append(f"Perform a strict QA evaluation and audit on the following conversational script text:\n\n{input_data}")

    print("🧠 Initiating processing via Gemini reasoning engine core...")
    
    # Execute API call with forced structured schema validation
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=QualityReport,
            temperature=0.1,  # Low variance to maintain compliance auditing stability
        ),
    )

    # Re-validate JSON string structural conversion payload back to a strict Pydantic Object representation
    report = QualityReport.model_validate_json(response.text)
    return report

def display_report(report: QualityReport):
    """Formats and renders the structured Pydantic compliance dataset cleanly onto the active console."""
    table_rows = ""
    for row in report.evaluation_table:
        status_str = "TRUE" if row.status else "FALSE"
        table_rows += f"| {row.category} | {row.item} | {status_str} | {row.impact} |\n"

    print("\n" + "="*70)
    print(f"🎧 **Transcription Text (Arabic Translation):**\n{report.transcription}\n")
    print(f"📌 **Categorized Call Type Context:**\n{report.call_type}\n")
    print(f"📊 **Scoring Analytics Summary Metrics:**")
    print(f"* Final Score Rating Calculation: {report.final_score}")
    print(f"* Strict Compliance Auto-Fail Condition Triggered: {'YES' if report.auto_fail else 'NO'}\n")
    
    print("❌ **Recorded Audited Infractions List Breakdown:**")
    for error in report.errors_list:
        print(f"* {error}")
        
    print(f"\n📋 **Compliance Auditing Cross-Reference Matrix:**\n")
    print("| Category | Target Item Evaluation Criterion | State | Score Impact |")
    print("| :--- | :--- | :---: | :---: |")
    print(table_rows)
    
    print("🧠 **Quality Assurance Actionable Coaching Notes:**")
    print("* Internal Observations & Strengths Identified:\n  - " + "\n  - ".join(report.strengths))
    print("* Critical System Mistakes & Vulnerabilities:\n  - " + "\n  - ".join(report.main_errors))
    print("* Professional Performance Development & Remediation Path:\n  - " + "\n  - ".join(report.improvement_plan))
    
    print("\n🤖 **Downstream Conversational Predictive Analytics Logic:**")
    print(f"* Prospect Interest Engagement Classification: {report.client_type}")
    print(f"* Identified Buying Intent Indicators: {', '.join(report.buying_signals)}")
    print(f"* High-Value Revenue Opportunities Forfeited: {', '.join(report.missed_opportunities)}")
    print(f"* Algorithmic Expected Close Probability Rate: {report.closing_probability}")
    print("="*70 + "\n")

# ==========================================
# 4. Runtime System Testing Pipeline Execution
# ==========================================
if __name__ == "__main__":
    # Standard static dialogue input mock-up sequence simulation
    sample_text = """
    Employee: Welcome, this is Ahmed calling from Engosoft here in our Riyadh headquarters located at Al-Yarmouk District on Rajia Street. We bring over 13 years of technical excellence to the market. How can I guide your engineering learning choices today?
    Client: Hello Ahmed. I am looking to get more technical details regarding your Advanced Corporate Software Architecture and Development Training Course track.
    Employee: Excellent choice! That complete curriculum covers Python system designs, cloud pipelines, and microservices patterns. It spans a flat duration of 8 consecutive weeks, delivered online via live interactive sessions. Our lead engineering instructor for this pathway is Dr. Martin. The total standard investment price is 5,000 SAR, but if we process your enrollment today, I can activate a special promotion dropping it to 4,000 SAR. We also support full flexible payment splits using Tabby or Tamara network options.
    Client: That sounds exactly like what my team needs. I'll take it.
    """
    
    # Process text input sequence
    report_data = analyze_call_with_gemini(sample_text, is_audio_file=False)
    
    # To process raw multi-modal audio recordings instead, comment the line above and uncomment below:
    
    # report_data = analyze_call_with_gemini("call_record.mp3", is_audio_file=True)
    
    display_report(report_data)