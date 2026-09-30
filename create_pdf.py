from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Proposal: Integrating Audio for Low-Light Anomaly Detection', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 7, body)
        self.ln()

pdf = PDF()
pdf.add_page()

pdf.chapter_title('1. Introduction')
body1 = (
    "While our Spatial-Temporal Graph Convolutional Network (STGCN) model performs well for visual anomaly detection, "
    "it faces practical limitations in low-light environments, such as hospital wards at night. To ensure continuous "
    "and reliable monitoring, we propose integrating an audio analysis pipeline. This will allow the system to detect "
    "anomalies, such as falls or distress calls, through sound when video visibility is poor."
)
pdf.chapter_body(body1)

pdf.chapter_title('2. Audio Feature Extraction')
body2 = (
    "We will analyze the audio extracted from the video feeds using two primary methods:\n\n"
    "Volume Thresholding: By measuring the Root Mean Square (RMS) energy or decibel (dB) levels, we can quickly "
    "detect sudden, loud noises (e.g., a crash or a shout) that stand out from the quiet baseline of a night ward.\n\n"
    "Sound Classification: To distinguish between normal loud noises (like a door closing) and actual anomalies "
    "(like a person falling), we will extract Mel-Frequency Cepstral Coefficients (MFCCs). These features act as "
    "acoustic signatures, helping us classify the specific type of sound."
)
pdf.chapter_body(body2)

pdf.chapter_title('3. Proposed Technology Stack')
body3 = (
    "We will use established open-source tools to implement this efficiently:\n\n"
    "FFmpeg: A reliable tool to extract the audio tracks from our existing video datasets. This allows us to use "
    "our current data without requiring new hardware.\n\n"
    "Librosa: A standard Python library for audio processing. We will use it to compute decibel levels and extract "
    "MFCC features from the audio files."
)
pdf.chapter_body(body3)

pdf.chapter_title('4. Multimodal Integration Strategy')
body4 = (
    "To combine the new audio system with our existing STGCN model, we recommend a 'Late Fusion' approach. "
    "This ensures our current video model remains unchanged while acting as a safety net.\n\n"
    "1. Independent Processing: The STGCN model will generate a visual anomaly score, while the audio pipeline "
    "will independently generate an audio anomaly score.\n\n"
    "2. Dynamic Weighting: We will implement a decision-making layer that assesses the lighting conditions. "
    "If the video frame is too dark, the system will dynamically increase the weight of the audio score to make "
    "the final prediction."
)
pdf.chapter_body(body4)

pdf.chapter_title('5. Implementation Steps')
body5 = (
    "We plan to execute this integration through the following steps:\n\n"
    "Step 1 - Data Extraction: Use FFmpeg to isolate audio files from the current video dataset.\n\n"
    "Step 2 - Baseline Analysis: Use Librosa to analyze decibel levels of normal versus anomalous events to "
    "establish initial volume thresholds.\n\n"
    "Step 3 - Model Training (Optional): If volume thresholding is insufficient, train a lightweight classifier "
    "(such as a Random Forest) on the MFCC features to improve sound classification accuracy.\n\n"
    "Step 4 - System Fusion: Develop the logic to combine the STGCN visual scores and the audio scores into a "
    "final, reliable prediction.\n\n"
    "This approach offers a practical, robust solution to the low-light limitation without overhauling our "
    "current architecture."
)
pdf.chapter_body(body5)

pdf.output('e:\\Capstone_Project\\audio_integration_report.pdf', 'F')
