from io import BytesIO
from datetime import datetime
import os
import html

# Prefer WeasyPrint for correct Kannada (complex script) rendering; fall back to ReportLab
_use_weasyprint = None


def _can_use_weasyprint():
    global _use_weasyprint
    if _use_weasyprint is not None:
        return _use_weasyprint
    try:
        from weasyprint import HTML, CSS
        from weasyprint.fonts import FontConfiguration
        _use_weasyprint = True
    except Exception:
        _use_weasyprint = False
    return _use_weasyprint


def _project_base():
    """Project root (cur_pro)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _font_dir():
    return os.path.join(_project_base(), "fonts")


def _kannada_font_path():
    p = os.path.join(_font_dir(), "NotoSansKannada-Regular.ttf")
    return p if os.path.exists(p) else None


def _build_pdf_weasyprint(case: dict) -> bytes:
    """Build PDF from HTML so Kannada renders with proper shaping (WeasyPrint)."""
    from weasyprint import HTML, CSS
    from weasyprint.fonts import FontConfiguration

    base_dir = _project_base()
    font_path = _kannada_font_path()
    # base_url for resolving relative URLs (e.g. fonts/NotoSansKannada-Regular.ttf)
    base_url = "file:///" + base_dir.replace("\\", "/").replace("//", "/").strip("/") + "/"

    # Optional @font-face for Kannada; use path relative to project root
    if font_path:
        font_css = """
        @font-face {
            font-family: 'NotoSansKannada';
            src: url('fonts/NotoSansKannada-Regular.ttf') format('truetype');
        }
        .kannada { font-family: 'NotoSansKannada', sans-serif; font-size: 11pt; line-height: 1.5; }
        """
    else:
        font_css = """
        .kannada { font-family: sans-serif; font-size: 11pt; line-height: 1.5; }
        """

    style = f"""
    body {{ font-family: Helvetica, sans-serif; font-size: 11pt; margin: 1in; }}
    h1 {{ font-size: 18pt; }}
    h2 {{ font-size: 14pt; margin-top: 14pt; }}
    h3 {{ font-size: 12pt; margin-top: 10pt; }}
    table {{ border-collapse: collapse; margin: 8pt 0; }}
    th, td {{ border: 1px solid #333; padding: 6px 10px; text-align: left; }}
    th {{ background: #eee; }}
    p {{ margin: 6pt 0; }}
    {font_css}
    """

    def esc(s):
        return html.escape(str(s or ""), quote=True)

    created = case.get("created_at", datetime.utcnow())
    date_str = created.strftime("%Y-%m-%d %H:%M:%S") if hasattr(created, "strftime") else str(created)
    prediction = case.get("prediction", {})
    disease_name = prediction.get("disease_name", "")
    disease_code = prediction.get("disease_code", "")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"/><title>CaseWise Report</title></head>
    <body>
    <h1>CaseWise – AI Healthcare Report</h1>
    <table>
    <tr><th>Patient Name</th><td>{esc(case.get("patient_name", ""))}</td></tr>
    <tr><th>Patient ID</th><td>{esc(case.get("patient_id", ""))}</td></tr>
    <tr><th>City</th><td>{esc(case.get("city", ""))}</td></tr>
    <tr><th>Country</th><td>{esc(case.get("country", ""))}</td></tr>
    <tr><th>Date</th><td>{esc(date_str)}</td></tr>
    </table>
    <h2>Disease: {esc(disease_name)} ({esc(disease_code)})</h2>
    <h3>Explanation (English):</h3>
    <p>{esc(case.get("explanation_en", ""))}</p>
    <h3>Explanation (Kannada):</h3>
    <p class="kannada">{esc(case.get("explanation_kn", ""))}</p>
    <h3>Diet Recommendation (English):</h3>
    <p>{esc(case.get("diet_en", ""))}</p>
    <h3>Diet Recommendation (Kannada):</h3>
    <p class="kannada">{esc(case.get("diet_kn", ""))}</p>
    """

    doctors = case.get("doctor_suggestions") or []
    if doctors:
        html_content += "<h3>Suggested Specialists (Mysore):</h3>"
        for d in doctors:
            line = f"{esc(d.get('name', ''))} – {esc(d.get('specialization', ''))}, {esc(d.get('hospital', ''))} (Contact: {esc(d.get('contact', ''))})"
            html_content += f"<p>{line}</p>"

    html_content += "</body></html>"

    buffer = BytesIO()
    font_config = FontConfiguration()
    # Resolve URLs relative to project root (font path in CSS is absolute in the @font-face we use)
    html_doc = HTML(string=html_content, base_url=base_url)
    css_doc = CSS(string=style, base_url=base_url)
    html_doc.write_pdf(buffer, stylesheets=[css_doc], font_config=font_config)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data


def _build_pdf_reportlab(case: dict) -> bytes:
    """Fallback: ReportLab (Kannada may not render cleanly due to no complex script shaping)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    def register_kannada_font():
        try:
            font_path = _kannada_font_path()
            if font_path:
                pdfmetrics.registerFont(TTFont("NotoSansKannada", font_path))
                return "NotoSansKannada"
        except Exception:
            pass
        return None

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    kannada_font_name = register_kannada_font()

    if kannada_font_name:
        kn_style = ParagraphStyle(
            "Kannada",
            parent=styles["Normal"],
            fontName=kannada_font_name,
            fontSize=11,
        )
    else:
        kn_style = styles["Normal"]

    elements = []
    elements.append(Paragraph("<b>CaseWise</b> – AI Healthcare Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    meta_data = [
        ["Patient Name", case.get("patient_name", "")],
        ["Patient ID", case.get("patient_id", "")],
        ["City", case.get("city", "")],
        ["Country", case.get("country", "")],
        ["Date", case.get("created_at", datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S")],
    ]
    meta_table = Table(meta_data, hAlign="LEFT")
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
        ])
    )
    elements.append(meta_table)
    elements.append(Spacer(1, 16))

    prediction = case.get("prediction", {})
    disease_name = prediction.get("disease_name", "")
    disease_code = prediction.get("disease_code", "")
    elements.append(Paragraph(f"Disease: <b>{disease_name} ({disease_code})</b>", styles["Heading2"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Explanation (English):", styles["Heading3"]))
    elements.append(Paragraph(case.get("explanation_en", ""), styles["Normal"]))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("Explanation (Kannada):", styles["Heading3"]))
    elements.append(Paragraph(case.get("explanation_kn", ""), kn_style))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Diet Recommendation (English):", styles["Heading3"]))
    elements.append(Paragraph(case.get("diet_en", ""), styles["Normal"]))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("Diet Recommendation (Kannada):", styles["Heading3"]))
    elements.append(Paragraph(case.get("diet_kn", ""), kn_style))
    elements.append(Spacer(1, 12))

    doctors = case.get("doctor_suggestions") or []
    if doctors:
        elements.append(Paragraph("Suggested Specialists (Mysore):", styles["Heading3"]))
        for d in doctors:
            line = f"{d.get('name', '')} – {d.get('specialization', '')}, {d.get('hospital', '')} (Contact: {d.get('contact', '')})"
            elements.append(Paragraph(line, styles["Normal"]))
            elements.append(Spacer(1, 4))

    doc.build(elements)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data


def build_pdf_report(case: dict) -> bytes:
    """Build PDF report. Uses WeasyPrint when available for clean Kannada rendering."""
    if _can_use_weasyprint():
        return _build_pdf_weasyprint(case)
    return _build_pdf_reportlab(case)
