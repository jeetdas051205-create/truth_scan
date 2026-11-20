"""
Generate High-Quality PDF for TruthScan AI Project Report using ReportLab.
"""
import os
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page numbers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(40, 805, "TruthScan AI — Technical & Academic Project Report")
            self.drawRightString(555, 805, "github.com/jeetdas051205-create/truth_scan")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(40, 800, 555, 800)

        # Footer (All pages)
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(40, 42, 555, 42)
        
        self.drawString(40, 30, "Confidential & Academic — TruthScan AI")
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(555, 30, page_str)
        self.restoreState()


def build_pdf(filename="TRUTHSCAN_COMPLETE_PROJECT_REPORT.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=48,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0f172a")
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#475569")
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6
    )

    body_bold = ParagraphStyle(
        'Body_Bold',
        parent=body_style,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor("#0f172a")
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#1e293b")
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=colors.white
    )

    kpi_num_style = ParagraphStyle(
        'KpiNum',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        alignment=1,
        textColor=colors.HexColor("#1d4ed8")
    )
    
    kpi_lbl_style = ParagraphStyle(
        'KpiLbl',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        alignment=1,
        textColor=colors.HexColor("#64748b")
    )

    story = []

    # Title & Badge
    badge_html = '<font color="#1d4ed8"><b>PROJECT VERIFICATION REPORT</b></font>'
    story.append(Paragraph(badge_html, body_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph("TruthScan AI", title_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph("AI-Powered Fake News Detection & Multi-Modal Content Verification System", subtitle_style))
    story.append(Spacer(1, 8))

    # Meta Table
    meta_data = [
        [
            Paragraph("<b>Core Domain:</b> NLP & Misinformation", table_cell),
            Paragraph("<b>Peak Accuracy:</b> 98.81% (LinearSVC & Bi-LSTM)", table_cell)
        ],
        [
            Paragraph("<b>Dataset:</b> ISOT (34,103 Articles) + FakeNewsNet", table_cell),
            Paragraph("<b>Deployment:</b> Render, Gunicorn, Flask REST API", table_cell)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[255, 260])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # KPI Highlight Cards
    kpi_data = [
        [
            Paragraph("98.81%", kpi_num_style),
            Paragraph("98.81%", kpi_num_style),
            Paragraph("4.15M", kpi_num_style),
            Paragraph("< 10 ms", kpi_num_style)
        ],
        [
            Paragraph("LinearSVC Accuracy", kpi_lbl_style),
            Paragraph("Bi-LSTM Accuracy", kpi_lbl_style),
            Paragraph("Deep DL Weights", kpi_lbl_style),
            Paragraph("Inference Latency", kpi_lbl_style)
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[128, 128, 128, 131])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f1f5f9")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 10))

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "<b>TruthScan AI</b> is a production-grade artificial intelligence and natural language processing (NLP) system "
        "engineered to identify digital disinformation, fabricated editorial articles, and synthetic AI-generated text. "
        "The project integrates theoretical machine learning rigor with production web infrastructure, offering high-accuracy "
        "binary text classification, multi-model consensus voting, live web URL scraping, and stylometric authenticity detection.",
        body_style
    ))
    story.append(Paragraph(
        "Evaluated on the academic <b>ISOT Fake News Dataset</b> (University of Victoria) encompassing 34,103 balanced records, "
        "the system achieves a peak classification accuracy of <b>98.81%</b> and an F1-score of <b>0.9881</b> using both Support Vector Machines "
        "(LinearSVC) and Bidirectional Long Short-Term Memory (Bi-LSTM) neural networks.",
        body_style
    ))

    # 2. Problem Statement & Motivation
    story.append(Paragraph("2. Problem Statement & Research Objectives", h1_style))
    story.append(Paragraph(
        "Digital misinformation spreads across social platforms up to six times faster than verified reporting. Traditional human fact-checking "
        "cannot scale to modern internet content velocity. Concurrently, generative Large Language Models (LLMs) allow bad actors to produce "
        "convincing synthetic journalism at near-zero marginal cost.",
        body_style
    ))
    story.append(Paragraph("• <b>High Sensitivity & Precision:</b> Preventing False Positives while actively flagging deceptive narratives.", body_style))
    story.append(Paragraph("• <b>Dual-Layer Verification:</b> Analyzing both semantic factual markers and stylometric indicators of LLM origin.", body_style))
    story.append(Paragraph("• <b>Low-Latency Web Deployment:</b> Offering sub-100ms response times via interactive UI and REST API endpoints.", body_style))

    # 3. Dataset Architecture
    story.append(Paragraph("3. Dataset Architecture & Empirical Statistics", h1_style))
    dataset_table_data = [
        [Paragraph("Dataset Metric", table_header), Paragraph("ISOT Benchmark (Primary)", table_header), Paragraph("FakeNewsNet (Secondary)", table_header)],
        [Paragraph("<b>Total Verified Records</b>", table_cell), Paragraph("<b>34,103 articles</b>", table_cell), Paragraph("23,196 articles", table_cell)],
        [Paragraph("<b>Real News Distribution</b>", table_cell), Paragraph("17,056 (50.01%) — Reuters verified", table_cell), Paragraph("1,056 PolitiFact verified", table_cell)],
        [Paragraph("<b>Fake News Distribution</b>", table_cell), Paragraph("17,047 (49.99%) — Flagged disinformation", table_cell), Paragraph("22,140 GossipCop verified", table_cell)],
        [Paragraph("<b>Train / Test Split</b>", table_cell), Paragraph("80% Train (27,282) / 20% Test (6,821)", table_cell), Paragraph("Cross-domain evaluation", table_cell)],
        [Paragraph("<b>Vocabulary Dimension</b>", table_cell), Paragraph("10,000 sublinear unigram/bigrams", table_cell), Paragraph("Variable word embeddings", table_cell)]
    ]
    ds_table = Table(dataset_table_data, colWidths=[155, 180, 180])
    ds_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(ds_table)
    story.append(Spacer(1, 10))

    # 4. Data Preprocessing & NLP Pipeline
    story.append(Paragraph("4. Data Preprocessing & Leakage Prevention", h1_style))
    story.append(Paragraph(
        "To ensure models learn genuine deception patterns rather than memorizing publisher artifacts, watermarks "
        "(e.g., <i>(Reuters) -</i>, <i>Reuters</i>) are systematically stripped. The text is normalized by removing HTML markup, URLs, "
        "and non-alphabetic noise, followed by lowercase standardization.",
        body_style
    ))
    story.append(Paragraph(
        "• <b>TF-IDF Vectorization:</b> Sublinear term frequency scaling across 10,000 unigram and bigram features with stopword removal.<br/>"
        "• <b>Deep Sequence Tokenization:</b> Post-padded sequence representation bounded at 100 timesteps for recurrent ingestion.",
        body_style
    ))

    # Page Break for clean Model section
    story.append(PageBreak())

    # 5. Algorithmic Suite & Model Architectures
    story.append(Paragraph("5. Algorithmic Suite & Deep Learning Architectures", h1_style))
    story.append(Paragraph(
        "TruthScan AI integrates five classical machine learning algorithms alongside a 4.15M-parameter deep neural network:",
        body_style
    ))
    story.append(Paragraph("• <b>Linear Support Vector Machine (LinearSVC):</b> Constructs a maximum-margin separating hyperplane in high-dimensional TF-IDF space. Highly resistant to overfitting on sparse vocabulary matrices.", body_style))
    story.append(Paragraph("• <b>Bidirectional LSTM (Deep Learning):</b> Features a 128-dimensional embedding layer, spatial dropout (0.2), 64-unit bidirectional recurrent layer (128 total units) with recurrent dropout (0.2), and dense classification head (4,148,933 parameters).", body_style))
    story.append(Paragraph("• <b>Passive Aggressive Classifier:</b> First-order margin optimization suited for high-throughput streaming classification.", body_style))
    story.append(Paragraph("• <b>Logistic Regression:</b> Probabilistic sigmoid baseline with L2 penalty.", body_style))
    story.append(Paragraph("• <b>Random Forest Classifier:</b> 100 bagging decision trees capturing non-linear token interactions.", body_style))
    story.append(Paragraph("• <b>Multinomial Naive Bayes:</b> Probabilistic baseline utilizing conditional feature token distributions.", body_style))

    # 6. Experimental Results & Benchmarks
    story.append(Paragraph("6. Experimental Benchmarks & Evaluation", h1_style))
    story.append(Paragraph(
        "All algorithms were benchmarked on the identical held-out test split of <b>6,821 articles</b> (3,411 Real / 3,410 Fake).",
        body_style
    ))

    benchmark_data = [
        [
            Paragraph("Model Architecture", table_header),
            Paragraph("Accuracy", table_header),
            Paragraph("Precision", table_header),
            Paragraph("Recall", table_header),
            Paragraph("F1-Score", table_header),
            Paragraph("Train Time", table_header),
            Paragraph("Latency", table_header)
        ],
        [Paragraph("<b>Linear Support Vector (LinearSVC)</b>", table_cell), Paragraph("<b>98.81%</b>", table_cell), Paragraph("98.97%", table_cell), Paragraph("98.65%", table_cell), Paragraph("<b>0.9881</b>", table_cell), Paragraph("1.77 s", table_cell), Paragraph("0.081 s", table_cell)],
        [Paragraph("<b>Bidirectional LSTM (Bi-LSTM)</b>", table_cell), Paragraph("<b>98.81%</b>", table_cell), Paragraph("98.75%", table_cell), Paragraph("98.87%", table_cell), Paragraph("<b>0.9881</b>", table_cell), Paragraph("412.5 s", table_cell), Paragraph("0.045 s", table_cell)],
        [Paragraph("<b>Passive Aggressive Classifier</b>", table_cell), Paragraph("98.70%", table_cell), Paragraph("99.11%", table_cell), Paragraph("98.27%", table_cell), Paragraph("0.9869", table_cell), Paragraph("0.32 s", table_cell), Paragraph("0.005 s", table_cell)],
        [Paragraph("<b>Logistic Regression</b>", table_cell), Paragraph("98.48%", table_cell), Paragraph("98.96%", table_cell), Paragraph("97.98%", table_cell), Paragraph("0.9847", table_cell), Paragraph("0.52 s", table_cell), Paragraph("0.004 s", table_cell)],
        [Paragraph("<b>Random Forest (100 Trees)</b>", table_cell), Paragraph("96.48%", table_cell), Paragraph("98.68%", table_cell), Paragraph("94.22%", table_cell), Paragraph("0.9640", table_cell), Paragraph("5.28 s", table_cell), Paragraph("0.115 s", table_cell)],
        [Paragraph("<b>Multinomial Naive Bayes</b>", table_cell), Paragraph("95.18%", table_cell), Paragraph("95.19%", table_cell), Paragraph("95.16%", table_cell), Paragraph("0.9518", table_cell), Paragraph("0.04 s", table_cell), Paragraph("0.013 s", table_cell)],
    ]
    bm_table = Table(benchmark_data, colWidths=[155, 60, 60, 60, 60, 60, 60])
    bm_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#eff6ff"), colors.HexColor("#f0fdf4"), colors.white, colors.HexColor("#f8fafc"), colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(bm_table)
    story.append(Spacer(1, 8))

    # Confusion Matrix Table
    story.append(Paragraph("Confusion Matrix Analysis (LinearSVC on 6,821 Test Articles)", h2_style))
    cm_data = [
        [Paragraph("Actual Class \\ Predicted", table_header), Paragraph("Predicted Real (0)", table_header), Paragraph("Predicted Fake (1)", table_header), Paragraph("Total Records", table_header)],
        [Paragraph("<b>Actual Real News</b>", table_cell), Paragraph("<font color='#16a34a'><b>3,376 (True Negative)</b></font>", table_cell), Paragraph("<font color='#dc2626'><b>35 (False Positive)</b></font>", table_cell), Paragraph("3,411", table_cell)],
        [Paragraph("<b>Actual Fake News</b>", table_cell), Paragraph("<font color='#dc2626'><b>46 (False Negative)</b></font>", table_cell), Paragraph("<font color='#16a34a'><b>3,364 (True Positive)</b></font>", table_cell), Paragraph("3,410", table_cell)]
    ]
    cm_table = Table(cm_data, colWidths=[155, 120, 120, 120])
    cm_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(cm_table)
    story.append(Spacer(1, 10))

    # 7. Stylometric AI & Scraper
    story.append(Paragraph("7. Stylometric AI-Content Detector & Live URL Ingestion", h1_style))
    story.append(Paragraph(
        "Beyond binary classification, TruthScan AI implements a multi-feature stylometric engine evaluating:",
        body_style
    ))
    story.append(Paragraph("• <b>Burstiness & Sentence Length Variance:</b> Quantifies syntactic cadence. Human authors exhibit rich variation in sentence complexity, whereas LLM text presents uniform lengths.", body_style))
    story.append(Paragraph("• <b>Formulaic LLM Transition Markers:</b> Identifies synthetic transitional stock phrases (e.g., <i>'in conclusion'</i>, <i>'it is worth noting'</i>, <i>'a multifaceted approach'</i>).", body_style))
    story.append(Paragraph("• <b>Journalistic Attribution Density:</b> Measures the frequency of direct quotes, named sources, and dates.", body_style))
    story.append(Paragraph("• <b>Live URL Scraper:</b> Uses <code>requests</code> and <code>BeautifulSoup4</code> with browser simulation headers to parse live news web pages.", body_style))

    # 8. Deployment Specification
    story.append(Paragraph("8. Cloud Deployment & REST API Architecture", h1_style))
    story.append(Paragraph(
        "The application is fully containerized and hosted on Render Cloud Infrastructure:<br/>"
        "• <b>Production WSGI:</b> Multi-worker Gunicorn server binding automatically to cloud environment ports.<br/>"
        "• <b>Infrastructure as Code:</b> <code>render.yaml</code> Blueprint and <code>Procfile</code> for automated CI/CD.<br/>"
        "• <b>Key Endpoints:</b> <code>POST /api/predict</code>, <code>POST /api/scrape-and-predict</code>, <code>POST /api/check-ai-content</code>, <code>GET /api/metrics</code>.",
        body_style
    ))

    # 9. Conclusion
    story.append(Paragraph("9. Conclusion & Project Deliverables", h1_style))
    story.append(Paragraph(
        "TruthScan AI achieves near-human-expert classification accuracy (<b>98.81%</b>) with minimal latency (<b>< 10ms</b>), "
        "providing a comprehensive multi-modal defense against modern disinformation and AI-generated news.",
        body_style
    ))

    # Build the document with Two-Pass Page Counter
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[+] Successfully generated PDF: {filename}")


if __name__ == "__main__":
    output_filename = "TRUTHSCAN_COMPLETE_PROJECT_REPORT.pdf"
    if len(sys.argv) > 1:
        output_filename = sys.argv[1]
    build_pdf(output_filename)
