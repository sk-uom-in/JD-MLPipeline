import pymupdf4llm
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy

# Download necessary NLTK data if not already available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Extract text using PyMuPDF4LLM
md_text = pymupdf4llm.to_markdown("guidelines/ONR-Report.pdf")

with open("extracted.txt", "w", encoding="utf-8") as file:
    file.write(md_text)


def clean_text(text):
    """Cleans extracted text by removing unwanted artifacts, normalizing, and filtering."""
    
    # Remove dashed separators like "-----"
    text = re.sub(r'-{3,}', '', text)

    # Remove reference numbers (e.g., D/1321/165002/2)
    text = re.sub(r'D/\d+/\d+/\d+', '', text)

    # Remove version history (e.g., "v3.0 FINAL 92")
    text = re.sub(r'v\d+\.\d+\s*(FINAL\s*\d*)?', '', text)

    # Remove standalone dates (e.g., '24 February, 2021')
    text = re.sub(r'\d{1,2} \w+, \d{4}', '', text)

    # Remove "Page X/Y" numbering
    text = re.sub(r'Page\s*\d+/\d+', '', text, flags=re.IGNORECASE)

    # Remove Table of Contents-style entries (e.g., "4.1 Developing an AI framework 48")
    text = re.sub(r'^\s*\d+(\.\d+)*\s+.*?\d+\s*$', '', text, flags=re.MULTILINE)

    # Remove "Appendix", "Figure", "Table" listings
    text = re.sub(r'^(Figure|Table|Appendix)\s+\d+:.*?\d+\s*$', '', text, flags=re.MULTILINE)

    # Remove lines that contain at least 5 consecutive dots anywhere in the line
    text = re.sub(r'^.*\.{5,}.*$', '', text, flags=re.MULTILINE)

    # Remove lines that contain only special characters (dots, dashes, underscores, etc.)
    text = re.sub(r'^[^\w\s]+$', '', text, flags=re.MULTILINE)

    # Remove multiple consecutive newlines
    text = re.sub(r'\n+', '\n', text).strip()

    try:
        tokens = word_tokenize(text.lower())  # Use word_tokenize safely
    except LookupError:
        print("Punkt tokenizer not found! Downloading now...")
        nltk.download('punkt')  # Download again if missing
        tokens = text.split()  # Retry tokenization

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word.lower() for word in tokens if word not in stop_words and word not in string.punctuation]

    return ' '.join(tokens)


def segment_text_into_sections(text, max_words=150):
    """
    Splits the text into sections of approximately `max_words` words,
    ensuring that new paragraphs start a new chunk.
    """
    nlp = spacy.load("en_core_web_sm")
    
    # Split text into paragraphs based on new lines
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sections = []
    chunk = ""
    current_word_count = 0
    
    for sentence in paragraphs:
        sections.append(sentence.strip())

    return sections

# Process text
# text_o = clean_text(md_text)
text_oo = segment_text_into_sections(md_text, max_words=150)

text_o = [clean_text(chunk).strip() for chunk in text_oo if len(re.findall(r'\w+', chunk)) >= 20]

print(f"Number of sections created: {len(text_oo)}")
print(f"Number of sections created: {len(text_o)}")

# Save cleaned output
# Save cleaned output (each section on a new line)
with open("checklist.txt", "w", encoding="utf-8") as file:
    for section in text_o:

        file.write(section + "\n\n")  # Add double newlines for readability
