import os
import faiss
#from script import Resume_Chunks
from dotenv import load_dotenv
from groq import Groq
import streamlit as st
#from google import genai
#from test_real_chunks import questions
from sentence_transformers import SentenceTransformer
import streamlit as st

Resume_Chunks = [
    # CONTACT
    {"id": "c1", "Section": "Contact",
     "Content": "Contact Vishnu directly via LinkedIn at linkedin.com/in/nadendla-v for scheduling interviews or discussing opportunities"},
    {"id": "c1a", "Section": "Portfolio",
     "Content": "Explore Vishnu’s technical projects, Python scripts, and Data Engineering repositories on GitHub at github.com/vishnunadendla"},

    # SUMMARY
    {"id": "sum1", "Section": "Summary",
     "Content": "Vishnu has 4+ years of data analytics experience across marketing (Epsilon 2025-2026), healthcare (TCS 2021-2023), and retail domains. Expert in SQL, Python, Power BI, dbt, and Airflow."},

    # EDUCATION
    {"id": "c2", "Section": "Education",
     "Content": "Masters in Computer Science, Rowan University (2023-2025). Bachelor of Computer Science, Hindustan University, India."},

    # SKILLS
    {"id":"s0", "Section":"Skills Summary",
     "Content":"Vishnu's technical skills and tools: Power BI, Tableau, SQL, Python, dbt, Apache Airflow, Snowflake, BigQuery, Azure Databricks, Soda Core, Docker, Git, Microsoft Fabric. Expert in data engineering and analytics tools"},
    {"id": "s1", "Section": "BI & Visualization",
     "Content": "BI Tools: Power BI, Tableau, Looker Studio, Google Analytics, Adobe Analytics, DAX, Power Query."},
    {"id": "s2", "Section": "Data Engineering",
     "Content": "Data Engineering Stack: Apache Airflow, dbt, Soda Core, Docker, Git, Claude Code."},
    {"id": "s3", "Section": "Databases",
     "Content": "Databases: MySQL, PostgreSQL, BigQuery, Snowflake, Azure Databricks, SQL Server."},
    {"id": "s4", "Section": "Programming",
     "Content": "Programming: Python (Pandas, NumPy, Matplotlib, Scikit-learn). Statistics: A/B Testing, Regression, Time Series."},
    {"id": "s5", "Section": "SQL",
     "Content": "Advanced SQL: CTEs, window functions, complex joins, query optimization, large dataset analysis."},
    {"id": "s6", "Section": "Cloud",
     "Content": "Cloud: GCP (BigQuery, Cloud Storage, Dataflow), AWS (S3, Lambda familiar), Azure Databricks."},

    # EPSILON
    {"id": "e1", "Section": "Epsilon Experience",
     "Content": "At Epsilon (Jan 2025 - Mar 2026), built 10+ Power BI and Tableau dashboards tracking campaign performance, customer behavior, and operational KPIs across marketing and operations."},
    {"id": "e2", "Section": "Epsilon Experience",
     "Content": "At Epsilon, engineered automated ETL pipelines using Apache Airflow and Python for large-scale marketing datasets, eliminating manual processing and uncovering churn patterns."},
    {"id": "e3", "Section": "Epsilon Experience",
     "Content": "At Epsilon, served as ETL Team Lead coordinating workloads in Jira, managing project milestones, and reporting delivery status directly to stakeholders."},
    {"id": "e4", "Section": "Epsilon Experience",
     "Content": "At Epsilon, designed Star Schema dimensional models and maintained dbt transformation workflows to standardize production analytics and KPI definitions."},
    {"id": "e5", "Section": "Epsilon Experience",
     "Content": "At Epsilon, implemented data quality checks and validation routines using Soda Core ensuring accuracy and reliability of production dashboards."},

    # TCS
    {"id": "t1", "Section": "TCS Experience",
     "Content": "At TCS (May 2021 - Jul 2023), led healthcare analytics delivery building Power BI dashboards for revenue cycle workflows and operational performance metrics."},
    {"id": "t2", "Section": "TCS Experience",
     "Content": "At TCS, wrote complex SQL for cleansing and transforming large-scale healthcare datasets supporting weekly client leadership reporting cycles."},
    {"id": "t3", "Section": "TCS Experience",
     "Content": "At TCS, implemented Soda Core data quality gates and Azure Databricks ETL workflows ensuring audit-ready accuracy for healthcare clients."},
    {"id": "t4", "Section": "TCS Experience",
     "Content": "At TCS, supported retail and capital markets accounts automating recurring reports using SQL and Python for multiple enterprise clients."},

    # PROJECTS
    {"id": "p1", "Section": "GCP Project",
     "Content": "Project (GCP): Built containerized ELT platform using Airflow, dbt, Soda Core on Google Cloud Platform with end-to-end lineage tracking via Cosmos."},
    {"id": "p2", "Section": "ML Project",
     "Content": "Project (ML): Built retail customer segmentation using K-Means clustering, RFM analysis, t-SNE dimensionality reduction, and Seaborn visualization."},
    {"id": "p3", "Section": "Microsoft Fabric",
     "Content": "Self-study: Applied Medallion Architecture (Bronze/Silver/Gold) using Microsoft Fabric, Dataflow Gen2, Lakehouse design, and semantic models."},
    {"id": "p4", "Section": "Hotel Forecasting Project",
     "Content": "Project: Built hotel revenue forecasting model for Pebblebrook Hotel Trust analyzing booking velocity, golden rows, event impact with distance decay across Boston and Santa Monica properties."},

    # SOFT SKILLS
    {"id": "soft1", "Section": "Process & Collaboration",
     "Content": "Experienced in Agile methodologies using Jira for sprint planning. Regularly translated complex data findings into clear business insights for non-technical stakeholders."},

    # CERTIFICATIONS & STATUS
    {"id": "st1", "Section": "Certifications",
     "Content": "Vishnu is a Microsoft Certified: Power BI Data Analyst Associate, demonstrating professional expertise in data modeling and visualization. Currently pursuing DP-600 Microsoft Fabric certification."},
    {"id": "st2", "Section": "Availability",
     "Content": "Available immediately for full-time roles. Vishnu is currently based in Philadelphia, Pennsylvania, and is open to relocation for the right opportunity across the United States."},
    {"id": "st3", "Section": "Job Preferences",
     "Content": "Vishnu is looking for roles as Data Analyst, BI Analyst, Analytics Engineer, or Data Engineer. Target salary $85,000-$120,000. Interested in AI-forward companies using modern data stack like dbt, Airflow, and Snowflake"},
    {"id": "st4", "Section": "Visa Status",
     "Content": "Authorized to work in USA on STEM OPT. Requires H1B sponsorship. Strong sponsorship profile — 3+ years experience, Master's degree, PL-300 certified."},
]
contents = [
    f"{chunk['Section']}: {chunk['Content']}"
    for chunk in Resume_Chunks
]
questions = [
    "Does Vishnu have Power BI experience?",
    "What did he do at Epsilon?",
    "Is he open to relocation?",
    "Does he need visa sponsorship?"
]

@st.cache_resource
def load_models():
    encoder=  SentenceTransformer('all-MiniLM-L6-v2')
    contents = [
        f"{chunk['Section']}: {chunk['Content']}"
        for chunk in Resume_Chunks
    ]
    Resume_Chunks1=encoder.encode(contents)
    index=faiss.IndexFlatL2(384)
    index.add(Resume_Chunks1)
    return encoder, index

encoder, index = load_models()

load_dotenv()

client = Groq(api_key=st.secrets["GROQ_KEY"])

def answer_question(question,index,Resume_Chunks):

    question_vector =  encoder.encode(question)
    question_vector =  question_vector.reshape(1,-1)
    distance,indices =  index.search(question_vector,k=3)

    retrieved_chunks=[]
    for i in indices[0]:
        chunk = Resume_Chunks[i]
        retrieved_chunks.append(
            f"[{chunk['Section']}]: {chunk['Content']}"
        )
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
        ROLE:
        You are "Vardhan-Index," a high-performance AI Career Representative for Vishnu Vardhan. 
        Your goal is to bridge the gap between Vishnu's technical data engineering expertise and the recruiter's business needs.

        STRICT RULES:
        1. DATA INTEGRITY: Use ONLY the provided CONTEXT. If the answer isn't there, say: "I'll have Vishnu follow up on that specific detail directly."
        2. FORMATTING: Use **bolding** for technical tools (e.g., **dbt**, **Airflow**, **SQL**) to make the answer scannable.
        3. SOURCE ATTRIBUTION: Always end the response by mentioning the relevant section not in between (e.g., "Source: [Epsilon Experience]").
        4. NEVER say "not a fit" or "lacks experience" Instead say "transferable skill" or "can ramp up quickly based on [evidence]"

        TONE & PERSONALITY:
        - Professional, analytical, and proactive.
        - Instead of just saying "Yes," explain the **how** and the **outcome** based on the context.
        - If asked about a skill Vishnu has, frame it as a solution to a business problem..
        - Never negative. Always find the transferable angle.Maximum 4 sentences per answer.

        CONTEXT:
        {context}

        RECRUITER QUESTION:
        {question}

        YOUR ANSWER:
        """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return "Vishnu is currently pushing a hotfix to my reasoning engine. Try again in a few seconds!"


'''def analyze_jd(jd_text):
    all_resume = "\n".join([
        f"[{chunk['Section']}]: {chunk['Content']}"
        for chunk in Resume_Chunks
    ])

    prompt = f"""
    You are analyzing job fit for Vishnu Vardhan.

    VISHNU'S COMPLETE PROFILE:
    {all_resume}

    JOB DESCRIPTION:
    {jd_text}

    Return EXACTLY this format:

    ✅ STRONG MATCH:
    → [skill]: [evidence from profile]

    ⚡ PARTIAL MATCH:
    → [skill]: [transferable explanation]

    ❌ GAP:
    → [skill]: [honest acknowledgment]

    MATCH SCORE: X/100

    RECOMMENDATION: [one positive line]
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
st.divider()
st.subheader("📋 Paste JD — See Instant Match Score")

jd_input = st.text_area(
    "Paste job description:",
    height=200,
    placeholder="Paste any job description here..."
)

if st.button("🔍 Analyze My Fit"):
    question = st.chat_input("Chek How Fit is Vishnu for your role")
    def analyze_jd(jd_text):
        all_resume = "\n".join([
            f"[{chunk['Section']}]: {chunk['Content']}"
            for chunk in Resume_Chunks
        ])

        prompt = f"""
        You are analyzing job fit for Vishnu Vardhan.

        VISHNU'S COMPLETE PROFILE:
        {all_resume}

        JOB DESCRIPTION:
        {jd_text}

        Return EXACTLY this format:

        ✅ STRONG MATCH:
        → [skill]: [evidence from profile]

        ⚡ PARTIAL MATCH:
        → [skill]: [transferable explanation]

        ❌ GAP:
        → [skill]: [honest acknowledgment]

        MATCH SCORE: X/100

        RECOMMENDATION: [one positive line]
        """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content'''

'''if st.button("🔍 Analyze My Fit"):
    if jd_input:
        with st.spinner("Analyzing fit..."):
            result = analyze_jd(jd_input)
        st.markdown(result)
    else:
        st.warning("Please paste a job description!")'''

st.title("Chat with Vishnu's AI Data Insight Engine")

# Get question from user
question = st.chat_input("Ask me anything about Vishnu:")

with st.sidebar:
    st.title("Vishnu Vardhan")
    st.write("Data Analyst | 4+ years")
    st.write("📍 Philadelphia, PA")
    st.markdown("[🔗 LinkedIn](https://linkedin.com/in/nadendla-v)")
    st.markdown("[💻 GitHub](https://github.com/vishnunadendla)")
    #st.markdown("[📄 Download Resume](https://raw.githubusercontent.com/vishnunadendla/AI-Data-Insight-Engine/main/data/vishnu_resume.pdf)")

col1, col2 = st.columns(2)

with col1:
    if st.button("💼 Work experience?"):
        question = "How many of years experinece vihsnu have and what are they"

with col2:
    if st.button("🛠️ Technical skills?"):
        question = "What are Vishnu's technical skills?"

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = answer_question(
                question, index, Resume_Chunks
            )

        st.write(answer)