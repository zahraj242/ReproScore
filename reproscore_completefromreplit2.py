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

sender_email = os.getenv("EMAIL_SENDER")
sender_password = os.getenv("EMAIL_PASSWORD")

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

You can view the full analysis and geographic distribution in your browser session.

Thank you for using ReproScore!

Best regards,
The ReproScore Team

---
This is an automated message. Please do not reply to this email.
"""
    return send_email(to_email, subject, body)

def send_immediate_confirmation_email(to_email, doi):
    if not to_email:
        return False
    subject = "Thank You for Using ReproScore - Analysis In Progress"
    body = f"""Dear Researcher,
Thank you for using ReproScore to analyze the reproducibility of your research finding!
Your analysis has been successfully started for:
DOI: {doi}
We are currently processing your request, which includes:
• Retrieving the original paper from PubMed
• Finding all citing papers and relevant research
• Analyzing geographic distribution of replication studies
• Using AI to classify supporting evidence
• Calculating reproducibility scores
Expected Processing Time: 10-60 minutes
(depending on the number of citations and relevant papers)
You will receive a follow-up email with the complete analysis results once processing is complete. You can also keep this browser window open to see real-time progress.
Best regards,
The ReproScore Team
---
This is an automated message. Please do not reply to this email.
"""
    return send_email(to_email, subject, body)

def analyze_thread(queue, doi, inquired_result):
    logger.info(f"========== STARTING ANALYSIS FOR DOI: {doi} ==========")

    # Step 1: Get PubMed ID from DOI
    logger.info("STEP 1: Getting PubMed ID from DOI...")
    try:
        handle = Entrez.esearch(db="pubmed", term=doi, retmax=1)
        record = Entrez.read(handle)
        handle.close()
        time.sleep(0.3)
        if record["IdList"]:
            pmid = record["IdList"][0]
            logger.info(f"✅ STEP 1 SUCCESS: Found PMID {pmid}")
        else:
            logger.error(f"❌ STEP 1 FAILED: DOI not found in PubMed")
            queue.put(("error", "DOI not found in PubMed."))
            return
    except Exception as e:
        logger.error(f"❌ STEP 1 FAILED: PubMed search error: {e}")
        queue.put(("error", f"PubMed search error: {e}"))
        return

    # Step 2: Fetch original paper metadata and abstract
    logger.info("STEP 2: Fetching original paper metadata and abstract...")
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
        record = Entrez.read(handle)
        handle.close()
        time.sleep(0.3)
        article = record["PubmedArticle"][0]
        title = article["MedlineCitation"]["Article"]["ArticleTitle"]
        author_list = article["MedlineCitation"]["Article"].get("AuthorList", [])
        original_authors = set()
        for au in author_list:
            if 'CollectiveName' in au:
                original_authors.add(au['CollectiveName'].lower().strip())
            else:
                fore = au.get('ForeName', '')
                last = au.get('LastName', '')
                if fore or last:
                    original_authors.add(f"{fore} {last}".lower().strip())
        journal = article["MedlineCitation"]["Article"]["Journal"]["Title"]
        original_affil = "Unknown"
        if author_list:
            affil_info = author_list[0].get("AffiliationInfo", [])
            if affil_info:
                original_affil = affil_info[0].get("Affiliation", "Unknown")
        original_abstract = ''.join([str(text) for text in article["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", ["No abstract"])])
        pub_date = article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"].get("Year", "2000")
        original_date = f"{pub_date}-01-01"

        try:
            original_location = geolocator.geocode(original_affil, timeout=10)
            original_coords = (original_location.latitude, original_location.longitude) if original_location else None
            logger.info(f"✅ Geocoding successful: {original_coords}")
        except Exception as e:
            logger.warning(f"⚠️ Geocoding failed: {e}")
            original_coords = None

        sentences = original_abstract.split('. ')
        key_sentences = [sentence.strip() for sentence in sentences if any(word in sentence.lower() for word in ['result', 'show', 'found', 'demonstrate', 'indicate', 'suggest', 'reveal', 'discover', 'conclude'])]
        if key_sentences:
            explain_output = f"Key findings: {'. '.join(key_sentences[:3])}."
        else:
            explain_output = f"Study findings: {sentences[0].strip()}. {sentences[-1].strip()}" if len(sentences) >= 2 else original_abstract[:200]
        queue.put(("original_info", (title, ', '.join(original_authors), original_affil, original_abstract, explain_output)))
        queue.put(("original_findings", explain_output))
        logger.info(f"✅ STEP 2 SUCCESS: Retrieved paper '{title[:80]}...'")
    except Exception as e:
        logger.error(f"❌ STEP 2 FAILED: PubMed fetch error: {e}")
        queue.put(("error", f"PubMed fetch error: {e}"))
        return

    # Step 3: Fetch citing papers
    logger.info("STEP 3: Fetching citing papers...")
    citing_pmids = []
    seen_pmids = set()
    retstart = 0
    batch_size = 200
    iteration = 0
    while True:
        iteration += 1
        try:
            logger.info(f"STEP 3.{iteration}: Calling PubMed elink API (offset={retstart}, batch_size={batch_size})...")
            handle = Entrez.elink(dbfrom="pubmed", db="pubmed", id=pmid, linkname="pubmed_pubmed_citedin", retstart=retstart, retmax=batch_size)
            links = Entrez.read(handle)
            handle.close()
            time.sleep(0.3)
            batch_ids = [link["Id"] for link_set in links[0]["LinkSetDb"] for link in link_set["Link"]] if links[0]["LinkSetDb"] else []
            if not batch_ids:
                break
            new_papers = [pid for pid in batch_ids if pid not in seen_pmids]
            if not new_papers:
                break
            citing_pmids.extend(new_papers)
            seen_pmids.update(new_papers)
            retstart += batch_size
        except Exception as e:
            logger.error(f"❌ STEP 3.{iteration} ERROR: {type(e).__name__}: {e}")
            break
    logger.info(f"✅ STEP 3 SUCCESS: Found {len(citing_pmids)} citing papers total")

    # Step 4: Search for uncited relevant papers
    logger.info("STEP 4: Searching for related papers...")
    uncited_pmids = []
    search_query_base = inquired_result.strip() if inquired_result and inquired_result.strip() else explain_output[:200]
    if search_query_base:
        search_query = f"({search_query_base}) AND (replication OR confirmation OR contradiction OR related) NOT pmid:{pmid}"
        retstart = 0
        step4_iteration = 0
        while True:
            step4_iteration += 1
            try:
                handle = Entrez.esearch(db="pubmed", term=search_query, datetype="pdat", mindate=original_date, retstart=retstart, retmax=batch_size)
                record = Entrez.read(handle)
                handle.close()
                time.sleep(0.3)
                batch_ids = record["IdList"]
                if not batch_ids:
                    break
                uncited_pmids.extend(batch_ids)
                retstart += batch_size
            except Exception as e:
                logger.error(f"❌ STEP 4.{step4_iteration} ERROR: {type(e).__name__}: {e}")
                break
    logger.info(f"✅ STEP 4 SUCCESS: Found {len(uncited_pmids)} related papers total")

    all_pmids = list(set(citing_pmids + uncited_pmids))
    num_papers = len(all_pmids)
    logger.info(f"TOTAL: {num_papers} papers to analyze ({len(citing_pmids)} citing + {len(uncited_pmids)} related)")
    queue.put(("stage", f"📚 Found {len(citing_pmids)} citing papers and {len(uncited_pmids)} related papers"))
    queue.put(("num_papers", num_papers))
    estimated_time = num_papers * 0.05
    queue.put(("estimated_time", estimated_time))
    queue.put(("stage", f"🔍 AI-filtering papers..."))

    logger.info("STEP 5: AI-filtering and processing papers...")
    filtered_citations = []
    batch_num = 0
    ai_batch_size = 5
    for start in range(0, len(all_pmids), batch_size):
        batch_num += 1
        batch = all_pmids[start:start + batch_size]
        try:
            handle = Entrez.efetch(db="pubmed", id=",".join(batch), retmode="xml")
            citing_records = Entrez.read(handle)
            handle.close()
            time.sleep(0.3)

            candidate_citations = []
            for citing_article in citing_records["PubmedArticle"]:
                citing_title = citing_article["MedlineCitation"]["Article"]["ArticleTitle"]
                author_list = citing_article["MedlineCitation"]["Article"].get("AuthorList", [])
                citing_authors_set = set()
                for au in author_list:
                    if 'CollectiveName' in au:
                        citing_authors_set.add(au['CollectiveName'].lower().strip())
                    else:
                        fore = au.get('ForeName', '')
                        last = au.get('LastName', '')
                        if fore or last:
                            citing_authors_set.add(f"{fore} {last}".lower().strip())
                citing_journal = citing_article["MedlineCitation"]["Article"]["Journal"]["Title"]
                journal_issn = None
                try:
                    issn_data = citing_article["MedlineCitation"]["Article"]["Journal"].get("ISSN", {})
                    if isinstance(issn_data, dict):
                        journal_issn = issn_data.get("content", issn_data.get("#text", "")).strip()
                    elif isinstance(issn_data, str):
                        journal_issn = issn_data.strip()
                except:
                    pass
                citing_abstract = ''.join([str(text) for text in citing_article["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", ["No abstract"])])
                citing_affils = [affil.get("Affiliation", "Unknown") for author in author_list for affil in author.get("AffiliationInfo", [])]
                affiliation = citing_affils[0] if citing_affils else "Unknown"
                year = citing_article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"].get("Year", "N/A")
                publication_types = [str(pt) for pt in citing_article["MedlineCitation"]["Article"].get("PublicationTypeList", [])]

                candidate_citations.append({
                    'title': citing_title,
                    'authors_set': citing_authors_set,
                    'authors': ', '.join(citing_authors_set),
                    'year': year,
                    'abstract': citing_abstract,
                    'affiliation': affiliation,
                    'affils': citing_affils,
                    'journal_title': citing_journal,
                    'journal_issn': journal_issn,
                    'publication_types': ' '.join(publication_types)
                })

            # AI filtering in sub-batches
            for ai_start in range(0, len(candidate_citations), ai_batch_size):
                ai_batch = candidate_citations[ai_start:ai_start + ai_batch_size]
                prompts = []
                for cite in ai_batch:
                    metadata = f"Title: {cite['title']}\nJournal: {cite['journal_title']}\nAuthors: {cite['authors']}\nPublication Types: {cite['publication_types']}\nAbstract: {cite['abstract'][:2000]}"
                    prompt = f"Is this paper a valid empirical study relevant to the original finding '{explain_output}'? Exclude reviews, meta-analyses, systematic reviews, self-citations (if authors overlap with original: {', '.join(original_authors)}), and preprints. Respond with JSON: {{'include': true/false, 'reason': 'brief reason'}}"
                    prompts.append(prompt)

                payload = {
                    "system_instruction": {"parts": [{"text": "You are a scientific paper filter. Respond with JSON only, one per prompt separated by ---."}]},
                    "contents": [{"parts": [{"text": "\n---\n".join(prompts)}]}]
                }
                try:
                    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
                    response = requests.post(endpoint, params={"key": gemini_api_key}, json=payload)
                    response.raise_for_status()
                    api_result = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                    classifications = api_result.split("---")
                    for j, cls in enumerate(classifications):
                        try:
                            result = json.loads(cls.strip())
                            if result.get('include', False):
                                cite = ai_batch[j]
                                # Calculate geo_score for included
                                geo_score = 1
                                countries = set()
                                cities = set()
                                institutes = set()
                                for aff in cite['affils']:
                                    if aff == "Unknown":
                                        continue
                                    parts = [p.strip() for p in aff.split(',')]
                                    if len(parts) >= 3:
                                        institute = parts[0].lower()
                                        city = parts[-2].lower() if len(parts) > 2 else ''
                                        cntry = parts[-1].lower()
                                        institutes.add(institute)
                                        if city:
                                            cities.add(city)
                                        countries.add(cntry)
                                original_parts = [p.strip() for p in original_affil.split(',')]
                                original_institutes = set([original_parts[0].lower()] if original_parts else [])
                                original_cities = set([original_parts[-2].lower()] if len(original_parts) > 2 else [])
                                original_countries = set([original_parts[-1].lower()] if original_parts else [])

                                if institutes & original_institutes:
                                    geo_score = 1
                                elif (cities & original_cities) and (countries & original_countries):
                                    geo_score = 2
                                elif countries & original_countries:
                                    geo_score = 3
                                else:
                                    geo_score = 9

                                # Impact factor
                                impact_factor = 1.0
                                if_is_real = False
                                if cite['journal_issn']:
                                    try:
                                        ooir_url = f"https://ooir.org/j.php?issn={cite['journal_issn']}"
                                        resp = requests.get(ooir_url, timeout=2)
                                        if resp.status_code == 200:
                                            match = re.search(r'Impact Factor,\s*(\d+\.\d+)', resp.text, re.IGNORECASE)
                                            if match:
                                                impact_factor = float(match.group(1))
                                                if_is_real = True
                                    except:
                                        pass
                                if not if_is_real and impact_factors_db:
                                    journal_name_normalized = cite['journal_title'].lower().strip()
                                    if journal_name_normalized in impact_factors_db:
                                        impact_factor = impact_factors_db[journal_name_normalized]
                                        if_is_real = True

                                filtered_citations.append({
                                    'title': cite['title'],
                                    'authors': cite['authors'],
                                    'year': cite['year'],
                                    'abstract': cite['abstract'],
                                    'affiliation': cite['affiliation'],
                                    'geo_score': geo_score,
                                    'country': list(countries)[0] if countries else "Unknown",
                                    'journal_title': cite['journal_title'],
                                    'journal_issn': cite['journal_issn'],
                                    'impact_factor': impact_factor,
                                    'if_is_real': if_is_real
                                })
                        except:
                            pass
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"❌ AI filtering error: {e}")

            if len(all_pmids) > 0:
                progress = min(len(filtered_citations) / len(all_pmids), 1)
                queue.put(("progress", progress))

        except Exception as e:
            logger.error(f"❌ Error processing batch: {e}")
            queue.put(("error", str(e)))

    logger.info(f"✅ STEP 5 SUCCESS: Filtered to {len(filtered_citations)} valid papers")
    queue.put(("stage", f"✅ Filtered to {len(filtered_citations)} valid papers for AI analysis"))
    if filtered_citations:
        citation_list = "\n".join([f" • {cite['title'][:80]}... ({cite['year']})" for cite in filtered_citations[:10]])
        if len(filtered_citations) > 10:
            citation_list += f"\n ... and {len(filtered_citations) - 10} more papers"
        queue.put(("citations_list", citation_list))

    queue.put(("result", filtered_citations))
    logger.info(f"========== ANALYSIS COMPLETE FOR DOI: {doi} ==========")

if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False
if 'analysis_queue' not in st.session_state:
    st.session_state.analysis_queue = None
if 'analysis_thread' not in st.session_state:
    st.session_state.analysis_thread = None
if 'filtered_citations' not in st.session_state:
    st.session_state.filtered_citations = []
if 'original_findings' not in st.session_state:
    st.session_state.original_findings = ""
if 'analysis_error' not in st.session_state:
    st.session_state.analysis_error = None
if 'analyses_today' not in st.session_state:
    st.session_state.analyses_today = {}
if 'paid' not in st.session_state:
    st.session_state.paid = False

today = datetime.date.today().isoformat()
if today not in st.session_state.analyses_today:
    st.session_state.analyses_today[today] = 0

if st.button("Analyze"):
    if captcha_input != st.session_state.captcha_text:
        st.error("❌ CAPTCHA verification failed. Please try again.")
        st.stop()

    if not doi or not doi.strip():
        st.error("❌ Please enter a DOI to start the analysis!")
        st.stop()
    if not gemini_api_key:
        st.error("❌ Gemini API Key not set in environment!")
        st.stop()

    if st.session_state.analyses_today[today] >= 1 and not st.session_state.paid:
        st.warning("You've used your 1 free analysis today. Pay $2 for another?")
        if st.button("Pay $2 via Stripe"):
            try:
                session = stripe.checkout.Session.create(
                    payment_method_types=['card'],
                    line_items=[{'price': STRIPE_PRICE_ID, 'quantity': 1}],
                    mode='payment',
                    success_url=st.query_params.get('session_id', [''])[0] + '?success=true' if st.query_params.get('session_id') else '?success=true',
                    cancel_url=st.query_params.get('session_id', [''])[0] + '?canceled=true' if st.query_params.get('session_id') else '?canceled=true',
                )
                st.markdown(f"[Pay Now]({session.url})", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Payment setup error: {e}")
        st.stop()
    else:
        st.session_state.analyses_today[today] += 1

    if user_email and sender_email and sender_password:
        try:
            email_sent = send_immediate_confirmation_email(user_email, doi)
            if email_sent:
                logger.info(f"✅ Initial confirmation email sent to {user_email}")
            else:
                logger.warning(f"⚠️ Initial email failed to send to {user_email}")
        except Exception as e:
            logger.error(f"❌ Initial email exception: {e}")

    st.session_state.analysis_running = True
    st.session_state.filtered_citations = []
    st.session_state.original_findings = ""
    st.session_state.analysis_error = None
    st.session_state.analysis_queue = Queue()
    st.session_state.user_email = user_email
    st.session_state.doi = doi
    st.session_state.analysis_thread = threading.Thread(target=analyze_thread, args=(st.session_state.analysis_queue, doi, inquired_result))
    st.session_state.analysis_thread.daemon = True
    st.session_state.analysis_thread.start()
    st.rerun()

query_params = dict(st.query_params)
if 'success' in query_params:
    st.session_state.paid = True
    st.success("Payment successful! You can now run more analyses.")

if st.session_state.analysis_running:
    st.success("🚀 Analysis Running! Processing your request...")

    st.markdown("---")
    st.markdown("### 📊 Analysis Progress")
    st.markdown("**Status:** Analyzing...")
    progress_bar.progress(0)
    status_text.text("Processing...")
    info_text.text("Fetching data...")
    estimated_time_text.text("Calculating time...")
    st.markdown("---")

    queue = st.session_state.analysis_queue
    filtered_citations = st.session_state.filtered_citations
    original_findings = st.session_state.original_findings

    messages_processed = 0
    while not queue.empty() and messages_processed < 200:
        try:
            msg_type, msg = queue.get_nowait()
            messages_processed += 1
            logger.info(f"UI received message: {msg_type}")

            if msg_type == "progress":
                progress_bar.progress(msg)
                status_text.text(f"🔍 Processing papers... {int(msg * 100)}% complete")
            elif msg_type == "stage":
                status_text.text(msg)
            elif msg_type == "citations_list":
                st.write("### 📑 Papers Found for Analysis:")
                st.code(msg, language="")
            elif msg_type == "num_papers":
                info_text.text(f"📄 Found {msg} papers to analyze")
            elif msg_type == "estimated_time":
                estimated_time_text.text(f"⏱️ Estimated time: {msg:.1f} minutes")
            elif msg_type == "original_info":
                title, authors, affil, abstract, explain = msg
                st.write("## 📋 Original Paper Analysis")
                st.write(f"**Title:** {title}")
                st.write(f"**Authors:** {authors}")
                st.write(f"**Affiliation:** {affil}")
            elif msg_type == "original_findings":
                st.session_state.original_findings = msg
                original_findings = msg
            elif msg_type == "error":
                logger.error(f"Analysis error received: {msg}")
                st.session_state.analysis_error = msg
                st.session_state.analysis_running = False
                st.error(f"❌ Error: {msg}")
            elif msg_type == "result":
                logger.info(f"✅ Analysis result received: {len(msg)} citations")
                st.session_state.filtered_citations = msg
                st.session_state.analysis_running = False
                filtered_citations = msg
        except:
            break

    if st.session_state.analysis_thread and not st.session_state.analysis_thread.is_alive():
        logger.info("✅ Thread finished, draining final messages...")
        final_messages = 0
        while not queue.empty() and final_messages < 100:
            try:
                msg_type, msg = queue.get_nowait()
                final_messages += 1
                logger.info(f"Final message: {msg_type}")
                if msg_type == "result":
                    st.session_state.filtered_citations = msg
                    st.session_state.analysis_running = False
                    filtered_citations = msg
                    logger.info(f"✅ Final result captured: {len(msg)} citations")
                    break
                elif msg_type == "error":
                    st.session_state.analysis_error = msg
                    st.session_state.analysis_running = False
                    logger.error(f"Final error: {msg}")
                    break
            except:
                break

    if st.session_state.analysis_running:
        time.sleep(0.5)
        st.rerun()

    filtered_citations = st.session_state.filtered_citations
    original_findings = st.session_state.original_findings

elif st.session_state.filtered_citations:
    filtered_citations = st.session_state.filtered_citations
    original_findings = st.session_state.original_findings
else:
    filtered_citations = []
    original_findings = ""

if filtered_citations:
    st.write(f"**Filtered Citations Found:** {len(filtered_citations)}")

    status_text.text("🤖 Analyzing abstracts with Google Gemini...")
    st.info("📖 Now analyzing paper abstracts using Gemini AI classification...")

    abstracts_with_indices = [(i, cite['abstract']) for i, cite in enumerate(filtered_citations) if cite['abstract'] != 'No abstract']

    analyses = []
    batch_size = 5
    for batch_start in range(0, len(abstracts_with_indices), batch_size):
        batch = abstracts_with_indices[batch_start:batch_start + batch_size]
        prompts = []
        for i, abstract in batch:
            prompt = f"Classify if this abstract supports (1), contradicts (-1), or is neutral (0) to the original finding: '{original_findings}'. Also detect category: 'in_vitro', 'in_vivo_mouse', 'in_vivo_rat', 'in_vivo_unspecified', 'human_clinical', 'human_patients'. Default 'in_vitro'. Respond with JSON: {{'category': 1/-1/0, 'category': 'category'}}"
            prompts.append(prompt)

        payload = {
            "system_instruction": {"parts": [{"text": "You are a scientific classifier. Respond with JSON only, one per line for each prompt."}]},
            "contents": [{"parts": [{"text": "\n\n".join(prompts)}]}]
        }
        try:
            endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
            response = requests.post(endpoint, params={"key": gemini_api_key}, json=payload)
            response.raise_for_status()
            api_result = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            classifications = [json.loads(cls.strip()) for cls in api_result.split("\n") if cls.strip()]
            for j, cls in enumerate(classifications):
                filtered_citations[batch[j][0]]['ai_category'] = cls.get('category', 0)
                filtered_citations[batch[j][0]]['ai_study_category'] = cls.get('category', 'in_vitro')
            time.sleep(0.5)
        except Exception as e:
            st.error(f"❌ AI Analysis Failed: {e}")
            st.session_state.analysis_running = False
            st.stop()

    # Calculate scores
    num_analyses = len(filtered_citations)
    num_confirms = sum(1 for cite in filtered_citations if cite.get('ai_category', 0) == 1)
    num_contradicts = sum(1 for cite in filtered_citations if cite.get('ai_category', 0) == -1)
    num_neutral = num_analyses - num_confirms - num_contradicts

    basic_score = (num_confirms / num_analyses * 100) if num_analyses > 0 else 0

    confirms = [cite for cite in filtered_citations if cite.get('ai_category', 0) == 1]
    unrelated_sum = sum(cite['geo_score'] * math.log(cite['impact_factor'] + 1) for cite in confirms)
    max_unrelated = sum(9 * math.log(cite['impact_factor'] + 1) for cite in confirms)
    geo_score = (unrelated_sum / max_unrelated * 100) if max_unrelated > 0 else 0

    from collections import defaultdict
    category_counts = defaultdict(lambda: {'confirms': 0, 'total': 0})
    for cite in filtered_citations:
        cat = cite.get('ai_study_category', 'in_vitro')
        category_counts[cat]['total'] += 1
        if cite.get('ai_category', 0) == 1:
            category_counts[cat]['confirms'] += 1
    category_scores = "\n".join([f"• {cat}: {(counts['confirms'] / counts['total'] * 100):.1f}/100 ({counts['confirms']}/{counts['total']})" for cat, counts in category_counts.items() if counts['total'] > 0])

    total_support = sum(cite['geo_score'] * cite['impact_factor'] for cite in filtered_citations if cite.get('ai_category', 0) == 1)
    total_contradict = sum(cite['geo_score'] * cite['impact_factor'] for cite in filtered_citations if cite.get('ai_category', 0) == -1)
    repro_score = max(0, min(100, (total_support - total_contradict) / num_analyses * 100)) if num_analyses > 0 else 0

    st.success(f"✅ Analysis Complete! ReproScore: {repro_score:.1f}/100")
    st.write(f"Basic Reproducibility Score: {basic_score:.1f}/100")
    st.write(f"Geographic Independence Score: {geo_score:.1f}/100")
    st.write("Category-Specific Scores:")
    st.write(category_scores)
    st.write(f"Supporting: {num_confirms} | Contradicting: {num_contradicts} | Neutral: {num_neutral}")

    if 'user_email' in st.session_state and st.session_state.user_email and 'final_email_sent' not in st.session_state:
        try:
            email_sent = send_final_results_email(st.session_state.user_email, st.session_state.get('doi', 'Unknown'), int(repro_score), basic_score, geo_score, category_scores, num_confirms, num_contradicts, num_analyses)
            if email_sent:
                logger.info(f"✅ Final results email sent to {st.session_state.user_email}")
                st.session_state.final_email_sent = True
            else:
                logger.warning(f"⚠️ Final results email failed to send")
        except Exception as e:
            logger.error(f"❌ Final email exception: {e}")

    df = pd.DataFrame(filtered_citations)
    country_to_iso3 = {
        'iran': 'IRN', 'usa': 'USA', 'united states': 'USA', 'china': 'CHN', 'japan': 'JPN', 'germany': 'DEU', 'france': 'FRA', 'uk': 'GBR', 'united kingdom': 'GBR',
        'canada': 'CAN', 'australia': 'AUS', 'india': 'IND', 'brazil': 'BRA', 'italy': 'ITA', 'spain': 'ESP', 'netherlands': 'NLD', 'sweden': 'SWE',
        'switzerland': 'CHE', 'south korea': 'KOR', 'korea': 'KOR', 'israel': 'ISR', 'belgium': 'BEL', 'austria': 'AUT', 'denmark': 'DNK', 'norway': 'NOR',
        'poland': 'POL', 'russia': 'RUS', 'turkey': 'TUR', 'mexico': 'MEX', 'argentina': 'ARG', 'south africa': 'ZAF', 'saudi arabia': 'SAU', 'egypt': 'EGY',
        'thailand': 'THA', 'singapore': 'SGP', 'malaysia': 'MYS', 'pakistan': 'PAK', 'bangladesh': 'BGD', 'indonesia': 'IDN', 'vietnam': 'VNM',
        'philippines': 'PHL', 'greece': 'GRC', 'portugal': 'PRT', 'finland': 'FIN', 'new zealand': 'NZL', 'ireland': 'IRL', 'chile': 'CHL', 'colombia': 'COL',
        'unknown': None
    }
    if 'country' in df.columns:
        df['country_iso3'] = df['country'].str.lower().str.strip().map(country_to_iso3)
        unmapped = df[df['country_iso3'].isna()]['country'].unique()
        if len(unmapped) > 0:
            logger.info(f"Unmapped countries: {unmapped.tolist()}")
        df_with_countries = df[df['country_iso3'].notna()]
        if not df_with_countries.empty:
            fig = px.scatter_geo(df_with_countries, locations="country_iso3", locationmode="ISO-3", hover_name="title", size="impact_factor", color="ai_category")
            st.plotly_chart(fig)
        else:
            all_countries = df['country'].unique()
            logger.warning(f"No valid ISO-3 mapping for countries: {all_countries.tolist()}")
            st.info(f"📍 Found papers from: {', '.join(all_countries)}")
            st.warning("Geographic visualization unavailable - country names need ISO-3 code mapping.")
    else:
        st.warning("No country data available for geographic visualization.")

    if user_email:
        subject = "ReproScore Analysis Results"
        body = f"DOI: {doi}\nScore: {repro_score}\nDetails: [summary here]"
```