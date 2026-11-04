import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="Report/Technical_Report.md")
    p.add_argument("--output",default="Report/Technical_Report.pdf")
    a=p.parse_args()
    try:
        import markdown
        from weasyprint import HTML
    except ImportError:
        raise SystemExit("Install with: pip install markdown weasyprint")
    md=Path(a.input).read_text(encoding="utf-8")
    body=markdown.markdown(md,extensions=["tables","fenced_code"])
    css="<style>body{font-family:DejaVu Sans,sans-serif;margin:32px;line-height:1.45}h1,h2{color:#26364a}table{border-collapse:collapse;width:100%}td,th{border:1px solid #aaa;padding:6px}</style>"
    HTML(string=f"<html><head>{css}</head><body>{body}</body></html>").write_pdf(a.output)
    print("Created",a.output)
if __name__=="__main__": main()
