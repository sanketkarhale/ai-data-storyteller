import os
from typing import Optional

import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import plotly.express as px
import pdfplumber
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


# ----------------- CONFIG -----------------

st.set_page_config(
    page_title="AI-Powered Data Storyteller",
    layout="wide"
)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ----------------- PDF → DataFrame (RESULT SHEET) -----------------

def pdf_result_to_df(file):
    """
    Try to read a result sheet PDF as a table.
    Works ONLY if the PDF contains text tables (not scanned images).
    - Reads all pages.
    - Assumes first row of each page is header.
    Returns a pandas DataFrame or None.
    """
    # Make sure it's a bytes-like object
    if hasattr(file, "read"):
        file_bytes = file.read()
        file_obj = io.BytesIO(file_bytes)
    else:
        file_obj = file

    all_rows = []
    header = None

    try:
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if not table:
                    continue

                # First row is header
                if header is None:
                    header = [str(h).strip() if h is not None else "" for h in table[0]]

                # Remaining rows are data
                for row in table[1:]:
                    row_clean = [str(x).strip() if x is not None else "" for x in row]
                    # pad/trim row to match header length
                    if len(row_clean) < len(header):
                        row_clean += [""] * (len(header) - len(row_clean))
                    elif len(row_clean) > len(header):
                        row_clean = row_clean[:len(header)]
                    all_rows.append(row_clean)
    except Exception:
        # If pdfplumber fails, return None and let caller fallback
        return None

    if not all_rows or header is None:
        return None

    df = pd.DataFrame(all_rows, columns=header)
    return df


def pdf_to_text(file_bytes: bytes, max_pages: int = 20) -> str:
    """Extract plain text from a PDF (no rows/columns).

    Returns an empty string if no text could be extracted.
    """
    text_chunks = []

    file_obj = io.BytesIO(file_bytes)

    try:
        with pdfplumber.open(file_obj) as pdf:
            total_pages = len(pdf.pages)
            pages_to_read = min(max_pages, total_pages)

            for i in range(pages_to_read):
                page = pdf.pages[i]
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_chunks.append(page_text.strip())
    except Exception:
        return ""

    if not text_chunks:
        return ""

    return "\n\n".join(text_chunks)


def categorize_result_row(grade_value):
    """
    Convert a grade (like 'O', 'A', 'B+', 'FF') into:
      - pass_fail: 'Pass' or 'Fail'
      - category: currently mirrors pass/fail but can be extended.
    """
    if pd.isna(grade_value):
        return "Unknown", "Unknown"

    grade = str(grade_value).strip().upper()

    # Fail grades
    if grade in ["FF", "F", "FAIL"]:
        return "Fail", "Fail"

    # Everything else treated as Pass
    return "Pass", "Pass"


def df_to_pdf_bytes(df: pd.DataFrame, title: str = "Students") -> bytes:
    """
    Convert a DataFrame into a simple PDF table and return the PDF as bytes.
    Used for download buttons.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph(title, styles["Heading1"]))

    # Data for table (header + rows)
    data = [list(df.columns)] + df.astype(str).values.tolist()

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
    ]))

    elements.append(table)
    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# ----------------- APP HEADER -----------------

st.title("📊 AI-Powered Data Storyteller (Result + CSV)")

st.write(
    """
Upload your **result sheet** in **PDF or CSV** format.

This app will:
1. Load the data into a table  
2. Show
 pass/fail separation and let you download separate files  
3. (If API key is set) Generate an AI-based story using Groq (LLaMA-3)
"""
)

# ----------------- FILE UPLOAD -----------------

uploaded_file = st.file_uploader(
    "Upload your result file (CSV or PDF)",
    type=["csv", "pdf"]
)

df = None
doc_text = ""  # for non-tabular PDFs

if uploaded_file is not None:
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        # ----- your existing CSV logic -----
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ CSV file loaded successfully!")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    elif filename.endswith(".pdf"):
        # ----- NEW: unified PDF handling -----

        # 1) Read file into bytes ONCE
        pdf_bytes = uploaded_file.read()

        # 2) First, try your EXISTING table-extraction code
        try:
            # use BytesIO so the same bytes can be reused by pdf_result_to_df
            df = pdf_result_to_df(io.BytesIO(pdf_bytes))
        except Exception as e:
            st.warning(f"Error while trying to extract table from PDF: {e}")
            df = None

        # 3) If table not found → fallback to text-only mode
        if df is None or (hasattr(df, "empty") and df.empty):
            st.info(
                "No clear table detected in this PDF. "
                "Switching to text-only mode (no rows/columns)."
            )

            doc_text = pdf_to_text(pdf_bytes, max_pages=20)

            if not doc_text.strip():
                st.error(
                    "Could not read any text from this PDF. "
                    "It may be a scanned image (photo)."
                )
            else:
                st.success("✅ Extracted text from PDF (without rows/columns).")
        else:
            st.success(f"✅ PDF result loaded! Rows: {len(df)}, Columns: {len(df.columns)}")
else:
    st.info("Please upload a CSV or PDF result file to begin.")
    st.stop()

# ----------------- DATA PREVIEW / TEXT-ONLY PDF MODE -----------------

if df is not None:
    st.subheader("👀 Data Preview (first 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)


    # ----------------------------------------------------------
    #        RESULT: PASS / FAIL SEPARATION + DOWNLOADS
    # ----------------------------------------------------------

    st.header("🎓 Result Analysis – Pass / Fail Separation")

    # Try to detect the grade column (case-insensitive)
    grade_col = None
    possible_grade_names = [
        "grade", "grd", "result_grade", "overall_grade", "final_grade",
        "sem_grade", "sgpi_grade", "spi_grade"
    ]

    for col in df.columns:
        col_normalized = col.strip().lower()
        if col_normalized in possible_grade_names:
            grade_col = col
            break

    if grade_col is None:
        st.info(
            "No grade/result column detected (like 'Grade', 'GRD'). "
            "Pass/Fail separation is only available for proper result sheets."
        )
    else:
        st.success(f"Using grade column: `{grade_col}`")

        # Apply categorize_result_row to each grade
        pass_fail_list = []
        category_list = []

        for g in df[grade_col]:
            pf, cat = categorize_result_row(g)
            pass_fail_list.append(pf)
            category_list.append(cat)

        df["Pass/Fail"] = pass_fail_list
        df["Result Category"] = category_list

        # Separate DataFrames
        pass_df = df[df["Pass/Fail"] == "Pass"].copy()
        fail_df = df[df["Pass/Fail"] == "Fail"].copy()

        # Show quick counts
        st.subheader("Overall Pass / Fail Count")
        pf_counts = df["Pass/Fail"].value_counts().reset_index()
        pf_counts.columns = ["Status", "Count"]
        st.table(pf_counts)

        # -------- Preview Tables --------
        st.subheader("✅ Pass Students (Preview)")
        if pass_df.empty:
            st.info("No Pass students found.")
        else:
            st.dataframe(pass_df.head(30), use_container_width=True)

        st.subheader("❌ Fail Students (Preview)")
        if fail_df.empty:
            st.info("No Fail students found.")
        else:
            st.dataframe(fail_df.head(30), use_container_width=True)

        # -------- Download Buttons: CSV --------
        st.subheader("⬇ Download as CSV")

        if not pass_df.empty:
            pass_csv = pass_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download PASS students CSV",
                data=pass_csv,
                file_name="pass_students.csv",
                mime="text/csv"
            )

        if not fail_df.empty:
            fail_csv = fail_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download FAIL students CSV",
                data=fail_csv,
                file_name="fail_students.csv",
                mime="text/csv"
            )

        # -------- Download Buttons: PDF --------
        st.subheader("⬇ Download as PDF")

        if not pass_df.empty:
            pass_pdf = df_to_pdf_bytes(pass_df, title="Pass Students")
            st.download_button(
                label="Download PASS students PDF",
                data=pass_pdf,
                file_name="pass_students.pdf",
                mime="application/pdf"
            )

        if not fail_df.empty:
            fail_pdf = df_to_pdf_bytes(fail_df, title="Fail Students")
            st.download_button(
                label="Download FAIL students PDF",
                data=fail_pdf,
                file_name="fail_students.pdf",
                mime="application/pdf"
            )

    # ----------------- Simple Visualization -----------------
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        st.header("📈 Visualize a Numeric Column")
        col_to_plot = st.selectbox("Select a numeric column", numeric_cols, key="num_col")
        bins = st.slider("Number of bins", min_value=5, max_value=100, value=20, key="bins")
        fig = px.histogram(df, x=col_to_plot, nbins=bins, title=f"Histogram of {col_to_plot}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns found for plotting.")

    # ----------------------------------------------------------
    #      ADVANCED ANALYSIS FOR NON-RESULT (GENERAL DATA)
    # ----------------------------------------------------------

    st.header("📊 Advanced Data Analysis (For Company / Sales / General Data)")

    # Check if this dataset is NOT a student result sheet
    is_result_dataset = "Pass/Fail" in df.columns

    if not is_result_dataset:

        st.success("General dataset detected — enabling advanced analytics 🚀")

        # ----------------------------------------------------------
        # 1) NUMERIC & CATEGORICAL COLUMN DETECTION
        # ----------------------------------------------------------
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        st.subheader("📌 Detected Columns")
        st.write("**Numeric columns:**", numeric_cols)
        st.write("**Categorical columns:**", categorical_cols)

        # ----------------------------------------------------------
        # 2) HISTOGRAM (NUMERIC COLUMN)
        # ----------------------------------------------------------
        import matplotlib.pyplot as plt
        if numeric_cols:
            st.subheader("📈 Histogram")

            num_col = st.selectbox("Select a numeric column for histogram", numeric_cols)
            bins = st.slider("Bins", min_value=5, max_value=50, value=20)

            fig, ax = plt.subplots()
            ax.hist(df[num_col].dropna(), bins=bins)
            ax.set_title(f"Histogram of {num_col}")
            ax.set_xlabel(num_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        else:
            st.info("No numeric columns found for histogram.")

        # ----------------------------------------------------------
        # 3) BAR CHART (CATEGORICAL)
        # ----------------------------------------------------------
        if categorical_cols:
            st.subheader("📊 Bar Chart (Category Counts)")

            cat_col = st.selectbox("Select a categorical column", categorical_cols)
            st.bar_chart(df[cat_col].value_counts())

        # ----------------------------------------------------------
        # 4) PIE CHART (CATEGORICAL)
        # ----------------------------------------------------------
        if categorical_cols:
            st.subheader("🥧 Pie Chart")

            pie_col = st.selectbox("Select a column for Pie Chart", categorical_cols)

            fig, ax = plt.subplots()
            df[pie_col].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)

        # ----------------------------------------------------------
        # 5) CORRELATION HEATMAP (NUMERIC)
        # ----------------------------------------------------------
        if len(numeric_cols) >= 2:
            st.subheader("🔥 Correlation Heatmap")

            corr = df[numeric_cols].corr()

            fig, ax = plt.subplots(figsize=(8, 5))
            cax = ax.matshow(corr, cmap='coolwarm')
            fig.colorbar(cax)
            ax.set_xticks(range(len(numeric_cols)))
            ax.set_yticks(range(len(numeric_cols)))
            ax.set_xticklabels(numeric_cols, rotation=45)
            ax.set_yticklabels(numeric_cols)
            st.pyplot(fig)

        # ----------------------------------------------------------
        # 6) AI STORY FOR GENERAL DATASET
        # ----------------------------------------------------------
        st.subheader("🤖 AI Summary for General Dataset")

        if client is None:
            st.info("Groq API key not found — AI summary disabled.")
        else:
            if st.button("✨ Generate AI Summary for This Dataset"):
                with st.spinner("Generating AI Summary..."):
                    try:
                        summary_stats = df.describe(include="all").transpose().reset_index()
                        summary_text = summary_stats.to_string()

                        prompt = (
                            "You are a senior data analyst. Analyze the following dataset summary:\n\n"
                            f"{summary_text}\n\n"
                            "Write a clear business-level summary including:\n"
                            "- Key insights and trends\n"
                            "- Outliers or unusual patterns\n"
                            "- Business impact or meaning\n"
                            "- 3 bullet recommendations\n"
                            "Explain in simple language."
                        )

                        response = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[
                                {"role": "system", "content": "Explain clearly, simply, professionally."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=900,
                            temperature=0.6
                        )

                        st.subheader("📘 AI-Generated Summary")
                        st.write(response.choices[0].message.content)

                    except Exception as e:
                        st.error(f"Error generating AI summary: {e}")

    else:
        st.info("This dataset is identified as a RESULT SHEET. Advanced company analysis disabled.")

    # ----------------- AI STORY (GROQ) for tabular data -----------------
    st.header("🧠 AI Data Story (Groq - LLaMA-3)")

    if client is None:
        st.info(
            "Groq client not configured. Set GROQ_API_KEY in a `.env` file "
            "to enable AI-powered storytelling."
        )
    else:
        st.write(
            "Click the button below to generate an AI-written summary about your dataset."
        )

    if st.button("✨ Generate AI Story (Groq)"):
        if client is None:
            st.error("Groq client not configured. Check GROQ_API_KEY in `.env`.")
        else:
            with st.spinner("Generating AI story using Groq..."):
                try:
                    summary_stats = df.describe(include="all").transpose().reset_index()
                    summary_text = summary_stats.to_string()

                    prompt = (
                        "You are a helpful data analyst. You are given summary statistics "
                        "from a dataset. Write a clear, simple story about the data in 3–6 paragraphs. "
                        "Focus on key trends, outliers, and comparisons. Then provide:\n"
                        "- 3 bullet-point insights\n"
                        "- 2 suggestions for further analysis or decisions.\n\n"
                        f"Here are the summary statistics:\n{summary_text}"
                    )

                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": "Explain data in very simple, clear language."},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=900,
                        temperature=0.6,
                    )

                    story = response.choices[0].message.content
                    st.subheader("📖 AI-Generated Story (Groq)")
                    st.write(story)

                except Exception as e:
                    st.error(f"Error while generating AI story: {e}")

elif doc_text:
    # -------- TEXT-ONLY PDF MODE (NO ROWS / COLUMNS) --------

    st.header("📄 PDF Text (No Rows / Columns Detected)")

    # Show the extracted text in a scrollable box
    st.text_area(
        "Extracted Result Text",
        value=doc_text,
        height=300
    )

    # Optional: let user download the extracted text
    st.download_button(
        label="⬇ Download extracted text",
        data=doc_text,
        file_name="result_text.txt",
        mime="text/plain"
    )

    # ---------- AI SUMMARY ON RAW PDF TEXT ----------

    st.subheader("🧠 AI Summary of Result PDF (Text Mode)")

    if client is None:
        st.info(
            "Groq client not configured. Set GROQ_API_KEY in a `.env` file "
            "to enable AI summary for text-only PDFs."
        )
    else:
        if st.button("✨ Generate AI Summary from PDF Text"):
            with st.spinner("Generating AI summary from PDF text..."):
                try:
                    # (Optional) limit text length to avoid too-long prompts
                    text_for_ai = doc_text[:6000]

                    prompt = (
                        "You are an exam result analyst. You are given raw result text from a university PDF.\n"
                        "The text may not be in a clean table format.\n"
                        "From this text, write a clear summary including:\n"
                        "- Total number of students (approximate if needed)\n"
                        "- How many PASS / FAIL (if visible)\n"
                        "- General performance observations\n"
                        "- Any notable grades or patterns.\n\n"
                        "Here is the raw result text:\n\n"
                        f"{text_for_ai}"
                    )

                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {
                                "role": "system",
                                "content": "Explain exam results in simple, clear language."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            },
                        ],
                        max_tokens=900,
                        temperature=0.6,
                    )

                    story = response.choices[0].message.content
                    st.subheader("📖 AI Summary of PDF Result")
                    st.write(story)

                except Exception as e:
                    st.error(f"Error while generating AI summary: {e}")

else:
    st.info("No data loaded.")
    st.stop()
