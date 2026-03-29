# Fonts for PDF reports (Kannada)

For **clean Kannada text** in the generated PDF report, place a Kannada-capable font here.

## Noto Sans Kannada (recommended)

1. Download **Noto Sans Kannada** from Google Fonts:
   - https://fonts.google.com/noto/specimen/Noto+Sans+Kannada
   - Click "Download family" and unzip.

2. Copy **NotoSansKannada-Regular.ttf** into this folder:
   ```
   fonts/
     NotoSansKannada-Regular.ttf
   ```

The app uses **WeasyPrint** to generate the PDF when available, so Kannada conjuncts and vowel signs render correctly. If the font is missing, the report still generates but Kannada may fall back to a system font or render less clearly.
