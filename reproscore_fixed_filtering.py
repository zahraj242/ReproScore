```python
import streamlit as st
import requests
from Bio import Entrez
from geopy.geocoders import Nominatim
import json
import pandas as pd
import plotly.express as px
import time
import threading
from queue import Queue
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import re
import math
import datetime
import stripe
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import random
import string
from dotenv import load_dotenv
import logging
import sys

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis_logs.txt'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)
for handler in logger.handlers:
    handler.flush = lambda: sys.stdout.flush()

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID")

col1, col2 = st.columns([1, 6])
with col1:
    st.image("reproscore.jpg", width=80)
with col2:
    st.title("ReproScore - Reproducibility Analysis App")
    st.subheader("Validate Candidate Reliability and Build on Reproducible Foundations for Research Excellence")

geolocator = Nominatim(user_agent="repro_app")

ncbi_api_key = st.text_input("Optional NCBI API key (for higher rate limits)", type="password")
user_email = st.text_input("Optional Email for Results (recommended for long analyses)")
Entrez.email = os.getenv("NCBI_EMAIL")
if ncbi_api_key:
    Entrez.api_key = ncbi_api_key

gemini_api_key = os.getenv("GEMINI_API_KEY")

sender_email = os.getenv("EMAIL_SENDER")  # Must be a Gmail address (e.g., yourgmail@gmail.com)
sender_password = os.getenv("EMAIL_PASSWORD")  # Must be a 16-char Gmail app password

impact_factors_db = {}
try:
    with open('impact_factors.json', 'r', encoding='utf-8') as f:
        impact_factors_db = json.load(f)
    logger.info(f"✅ Loaded {len(impact_factors_db)} journals from impact_factors.json")
except Exception as e:
    logger.warning(f"⚠️ Could not load impact_factors.json: {e}")

doi = st.text_input("Enter DOI")
inquired_result = st.text_input("Enter Result Description (optional - leave empty to analyze all results)")
progress_bar = st.progress(0)
status_text = st.empty()
info_text = st.empty()
estimated_time_text = st.empty()
advice_text = st.info("Enter DOI and result to start. For highly cited papers, analysis may take 10-60 minutes. Do not close the page during analysis, or opt for email results.")

def generate_captcha_text(length=6):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def generate_captcha_image(text):
    img = Image.new('RGB', (150, 30), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    d.text((10, 5), text, fill=(255, 255, 0), font=font)
    for _ in range(100):
        x = random.randint(0, 149)
        y = random.randint(0, 29)
        d.point((x, y), fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf

if 'captcha_text' not in st.session_state:
    st.session_state.captcha_text = generate_captcha_text()
    st.session_state.captcha_image = generate_captcha_image(st.session_state.captcha_text)

st.image(st.session_state.captcha_image, caption="Enter the text in the image above (case-sensitive)", width='stretch')
captcha_input = st.text_input("CAPTCHA Verification", key="captcha_input")

if st.button("Regenerate CAPTCHA"):
    st.session_state.captcha_text = generate_captcha_text()
    st.session_state.captcha_image = generate_captcha_image(st.session_state.captcha_text)
    st.rerun()

def send_email(to_email, subject, body):
    if not sender_email or not sender_password or not to_email:
        return False
    # Uses Gmail SMTP server (smtp.gmail.com:587) with TLS
    # Ensure sender_email is a Gmail address and sender_password is a 16-char app password
    # Generate app password at: Google Account > Security > 2FA > App passwords > Select "Mail" > "Other (Custom)" > Name "ReproScore"
    msg = MIMEMultipart()
    from email.utils import formataddr
    msg['From'] = formataddr(("ReproScore Analysis", sender_email))
    msg['To'] = to_email
    msg['Subject'] = subject
    msg['Reply-To'] = "noreply@reproscore.app"
    msg.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587, timeout=10)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        logger.info(f"✅ Email sent to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Email send error: {str(e)}")
        return False

def send_final_results_email(to_email, doi, repro_score, basic_score, geo_score, category_scores, num_supporting, num_contradicting, num_citations):
    if not to_email:
        return False
    subject = f"ReproScore Analysis Complete - Score: {repro_score}/100"
    body = f"""Dear Researcher,

Your ReproScore analysis is complete!

DOI: {doi}
Overall ReproScore: {repro_score}/100
Basic Reproducibility: {basic_score}/100
Geographic Independence: {geo_score}/100

Category Scores:
{category_scores}

Summary:
• Total papers analyzed: {num_citations}
• Supporting evidence: {num_supporting}
• Contradicting evidence: {num_contradicting}

You can view the full analysis and