Python 3.14.0 (v3.14.0:ebf955df7a8, Oct  7 2025, 08:20:14) [Clang 16.0.0 (clang-1600.0.26.6)] on darwin
Enter "help" below or click "Help" above for more information.
>>> import streamlit as st
... import requests
... from Bio import Entrez
... from geopy.geocoders import Nominatim
... from geopy.distance import geodesic
... import json
... import pandas as pd
... import plotly.express as px
... from bs4 import BeautifulSoup as bs
... import time
... import threading
... from queue import Queue
... import smtplib
... from email.mime.text import MIMEText
... from email.mime.multipart import MIMEMultipart
... import os
... import re
... import math
... import datetime
... import stripe  # Add this for Stripe payments
... from PIL import Image, ImageDraw, ImageFont
... from io import BytesIO
... import random
... import string
... from dotenv import load_dotenv
... 
... load_dotenv()  # Load environment variables from .env file for local testing
... 
... # Stripe setup - replace with your keys
... stripe.api_key = os.getenv("STRIPE_SECRET_KEY")  # Set in Vercel env vars
... STRIPE_PRICE_ID = "price_12345"  # Create a $2 one-time product in Stripe dashboard
... # Add logo and title
... col1, col2 = st.columns([1, 6])
... with col1:
...     st.image("reproscore.jpg", width=80)
... with col2:
    st.title("ReproScore - Reproducibility Analysis App")
    st.subheader("Validate Candidate Reliability and Build on Reproducible Foundations for Research Excellence")
# Geopy setup
geolocator = Nominatim(user_agent="repro_app")
# Optional NCBI API key and email for results
ncbi_api_key = st.text_input("Optional NCBI API key (for higher rate limits)", type="password")
user_email = st.text_input("Optional Email for Results (recommended for long analyses)")
Entrez.email = os.getenv("NCBI_EMAIL")  # Set in Vercel env vars or .env file
if ncbi_api_key:
    Entrez.api_key = ncbi_api_key
# xAI API key input
xai_api_key = st.text_input("xAI API Key (required for AI analysis)", type="password")
# Sender email creds from Secrets
sender_email = os.getenv("EMAIL_SENDER")
sender_password = os.getenv("EMAIL_PASSWORD")
# User inputs
doi = st.text_input("Enter DOI")
inquired_result = st.text_input("Enter Result Description (optional - leave empty to analyze all results)")
progress_bar = st.progress(0)
status_text = st.empty()
info_text = st.empty()
estimated_time_text = st.empty()
advice_text = st.info("Enter DOI and result to start. For highly cited papers, analysis may take 10-60 minutes. Do not close the page during analysis, or opt for email results.")

# CAPTCHA setup using PIL
def generate_captcha_text(length=6):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def generate_captcha_image(text):
    img = Image.new('RGB', (150, 30), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    d.text((10, 5), text, fill=(255, 255, 0), font=font)
    # Add noise
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

st.image(st.session_state.captcha_image, caption="Enter the text in the image above (case-sensitive)", use_column_width='auto')
captcha_input = st.text_input("CAPTCHA Verification", key="captcha_input")

if st.button("Regenerate CAPTCHA"):
    st.session_state.captcha_text = generate_captcha_text()
    st.session_state.captcha_image = generate_captcha_image(st.session_state.captcha_text)
    st.rerun()

def send_email(to_email, subject, body):
    """Send email silently - failures don't block analysis"""
    if not sender_email or not sender_password or not to_email:
        return False
    msg = MIMEMultipart()
    # Use professional display name - hide personal email
    from email.utils import formataddr
    msg['From'] = formataddr(("ReproScore Analysis", sender_email))
    msg['To'] = to_email
    msg['Subject'] = subject
    msg['Reply-To'] = "noreply@reproscore.app"  # Discourage replies
    msg.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587, timeout=10)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        return True
    except:
        return False  # Silent failure - email is optional
def send_immediate_confirmation_email(to_email, doi):
    """Send immediate thank you email upon starting analysis"""
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
    # Step 1: Get PubMed ID from DOI
    try:
        handle = Entrez.esearch(db="pubmed", term=doi, retmax=1)
        record = Entrez.read(handle)
        handle.close()
        time.sleep(0.3)  # Rate limit
        if record["IdList"]:
            pmid = record["IdList"][0]
        else:
            queue.put(("error", "DOI not found in PubMed."))
            return
    except Exception as e:
        queue.put(("error", f"PubMed search error: {e}"))
        return
    # Step 2: Fetch original paper metadata and abstract
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
        record = Entrez.read(handle)
        handle.close()
        time.sleep(0.3)  # Rate limit
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
        # Safe affiliation fetch
        original_affil = "Unknown"
        if author_list:
            affil_info = author_list[0].get("AffiliationInfo", [])
            if affil_info:
                original_affil = affil_info[0].get("Affiliation", "Unknown")
        original_abstract = ''.join([str(text) for text in article["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", ["No abstract"])])
       
        # Get original publication date
        pub_date = article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"].get("Year", "2000")
        original_date = f"{pub_date}-01-01"  # Approximate if only year
       
        # Geocode original
        try:
            original_location = geolocator.geocode(original_affil)
            original_coords = (original_location.latitude, original_location.longitude) if original_location else None
        except:
            original_coords = None
       
        # Extract key sentences manually (faster, more reliable than AI for summaries)
        sentences = original_abstract.split('. ')
        key_sentences = []
        for sentence in sentences:
            if any(word in sentence.lower() for word in ['result', 'show', 'found', 'demonstrate', 'indicate', 'suggest', 'reveal', 'discover', 'conclude']):
                key_sentences.append(sentence.strip())
       
        if key_sentences:
            explain_output = f"Key findings: {'. '.join(key_sentences[:3])}."
        else:
            explain_output = f"Study findings: {sentences[0].strip()}. {sentences[-1].strip()}" if len(sentences) >= 2 else original_abstract[:200]
        queue.put(("original_info", (title, ', '.join(original_authors), original_affil, original_abstract, explain_output)))
       
        # Store original findings for AI classification (used when no specific result provided)
        queue.put(("original_findings", explain_output))
    except Exception as e:
        queue.put(("error", f"PubMed fetch error: {e}"))
        return
    # Step 3: Fetch citing papers (all, paginate)
    citing_pmids = []
    retstart = 0
    batch_size = 200
    while True:
        try:
            handle = Entrez.elink(dbfrom="pubmed", db="pubmed", id=pmid, linkname="pubmed_pubmed_citedin", retstart=retstart, retmax=batch_size)
            links = Entrez.read(handle)
            handle.close()
            time.sleep(0.3)  # Rate limit
            batch_ids = [link["Id"] for link_set in links[0]["LinkSetDb"] for link in link_set["Link"]] if links[0]["LinkSetDb"] else []
            if not batch_ids:
                break
            citing_pmids.extend(batch_ids)
            retstart += batch_size
        except Exception as e:
            break
    # Step 4: Search for uncited relevant papers (after original date) - always run, fallback to original_findings if no inquired_result
    uncited_pmids = []
    search_query_base = inquired_result.strip() if inquired_result and inquired_result.strip() else explain_output[:200]  # Fallback to original findings
    if search_query_base:
        search_query = f"({search_query_base}) AND (replication OR confirmation OR contradiction OR related) NOT pmid:{pmid}"
        retstart = 0
        while True:
            try:
                handle = Entrez.esearch(db="pubmed", term=search_query, datetype="pdat", mindate=original_date, retstart=retstart, retmax=batch_size)
                record = Entrez.read(handle)
                handle.close()
                time.sleep(0.3)  # Rate limit
                batch_ids = record["IdList"]
                if not batch_ids:
                    break
                uncited_pmids.extend(batch_ids)
                retstart += batch_size
            except Exception as e:
                break
    all_pmids = list(set(citing_pmids + uncited_pmids))  # Combine and dedup
    num_papers = len(all_pmids)
    queue.put(("stage", f"📚 Found {len(citing_pmids)} citing papers and {len(uncited_pmids)} related papers"))
    queue.put(("num_papers", num_papers))
    estimated_time = num_papers * 0.05  # 0.05 min per paper
    queue.put(("estimated_time", estimated_time))
    queue.put(("stage", f"🔍 Filtering papers (excluding reviews, meta-analyses, and self-citations)..."))
   
    filtered_citations = []
    for start in range(0, len(all_pmids), batch_size):
        batch = all_pmids[start:start+batch_size]
        try:
            handle = Entrez.efetch(db="pubmed", id=",".join(batch), retmode="xml")
            citing_records = Entrez.read(handle)
            handle.close()
            time.sleep(0.3)  # Rate limit
            for citing_article in citing_records["PubmedArticle"]:
                # Exclude reviews/meta
                publication_types = [str(pt) for pt in citing_article["MedlineCitation"]["Article"].get("PublicationTypeList", [])]
                if any(term in ' '.join(publication_types) for term in ['Review', 'Meta-Analysis', 'Systematic Review']):
                    continue
               
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
                if original_authors & citing_authors_set:
                    continue
               
                citing_journal = citing_article["MedlineCitation"]["Article"]["Journal"]["Title"]
                if "preprint" in citing_journal.lower() or "arxiv" in citing_journal.lower() or "biorxiv" in citing_journal.lower():
                    continue
               
                # Extract ISSN for impact factor lookup (handle all PubMed variants)
                journal_issn = None
                try:
                    issn_data = citing_article["MedlineCitation"]["Article"]["Journal"].get("ISSN", {})
                    if isinstance(issn_data, dict):
                        journal_issn = issn_data.get("content", issn_data.get("#text", "")).strip()
                    elif isinstance(issn_data, str):
                        journal_issn = issn_data.strip()
                    # Do NOT remove hyphens: OOIR expects them
                except:
                    pass
               
                citing_abstract = ''.join([str(text) for text in citing_article["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", ["No abstract"])])
                # All affiliations for geo check
                citing_affils = [affil.get("Affiliation", "Unknown") for author in author_list for affil in author.get("AffiliationInfo", [])]
                affiliation = citing_affils[0] if citing_affils else "Unknown"
               
                # Updated geo score
                geo_score = 1  # Default same institute
                countries = set()
                cities = set()
                institutes = set()
                for aff in citing_affils:
                    if aff == "Unknown":
                        continue
                    # Parse affiliation (assume format: Institute, City, Country)
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
               
                # Impact factor lookup using OOIR.org (default 1.0 for missing journals)
                impact_factor = 1.0
                if_is_real = False  # Track if we got real data vs default
               
                # Try OOIR API if we have ISSN (fast timeout for speed)
                if journal_issn:
                    try:
                        ooir_url = f"https://ooir.org/j.php?issn={journal_issn}"
                        response = requests.get(ooir_url, timeout=2)  # Faster timeout
                        if response.status_code == 200:
                            text_content = response.text
                            match = re.search(r'Impact Factor,\s*(\d+\.\d+)', text_content, re.IGNORECASE)
                            if match:
                                impact_factor = float(match.group(1))
                                if_is_real = True
                        time.sleep(0.05)  # Minimal rate limiting for speed
                    except:
                        pass
               
                filtered_citations.append({
                    'title': citing_title,
                    'authors': ', '.join(citing_authors_set),
                    'year': citing_article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"].get("Year", "N/A"),
                    'abstract': citing_abstract,
                    'affiliation': affiliation,
                    'geo_score': geo_score,
                    'country': list(countries)[0] if countries else "Unknown",
                    'journal_title': citing_journal,
                    'journal_issn': journal_issn,
                    'impact_factor': impact_factor,
                    'if_is_real': if_is_real
                })
               
                if len(all_pmids) > 0:
                    progress = min(len(filtered_citations) / len(all_pmids), 1)
                    queue.put(("progress", progress))
                   
        except Exception as e:
            queue.put(("error", str(e)))
    # Send citation list to display
    queue.put(("stage", f"✅ Filtered to {len(filtered_citations)} valid papers for AI analysis"))
    if filtered_citations:
        citation_list = "\n".join([f" • {cite['title'][:80]}... ({cite['year']})" for cite in filtered_citations[:10]])
        if len(filtered_citations) > 10:
            citation_list += f"\n ... and {len(filtered_citations) - 10} more papers"
        queue.put(("citations_list", citation_list))
   
    queue.put(("result", filtered_citations))
# Initialize session state for analysis and usage tracking
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
    st.session_state.analyses_today = {}  # Simple dict: date -> count (per session approximation)
if 'paid' not in st.session_state:
    st.session_state.paid = False
# Check daily limit (approximate with session state)
today = datetime.date.today().isoformat()
if today not in st.session_state.analyses_today:
    st.session_state.analyses_today[today] = 0
# Start analysis in thread
if st.button("Analyze"):
    # CAPTCHA verification
    if captcha_input != st.session_state.captcha_text:
        st.error("❌ CAPTCHA verification failed. Please try again.")
        st.stop()
    
    # Validate inputs
    if not doi or not doi.strip():
        st.error("❌ Please enter a DOI to start the analysis!")
        st.stop()
    if not xai_api_key:
        st.error("❌ Please enter your xAI API Key!")
        st.stop()
   
    # Check limit
    if st.session_state.analyses_today[today] >= 1 and not st.session_state.paid:
        st.warning("You've used your 1 free analysis today. Pay $2 for another?")
        if st.button("Pay $2 via Stripe"):
            try:
                session = stripe.checkout.Session.create(
                    payment_method_types=['card'],
                    line_items=[{
                        'price': STRIPE_PRICE_ID,  # Your $2 product price ID
                        'quantity': 1,
                    }],
                    mode='payment',
                    success_url=st.experimental_get_query_params().get('session_id', [None])[0] + '?success=true' if st.experimental_get_query_params().get('session_id') else '?success=true',  # Redirect back
                    cancel_url=st.experimental_get_query_params().get('session_id', [None])[0] + '?canceled=true' if st.experimental_get_query_params().get('session_id') else '?canceled=true',
                )
                st.markdown(f"[Pay Now]({session.url})", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Payment setup error: {e}")
        st.stop()
    else:
        st.session_state.analyses_today[today] += 1
   
    # Send immediate confirmation email if user provided email (silently fail if issues)
    if user_email and sender_email and sender_password:
        try:
            send_immediate_confirmation_email(user_email, doi)
            # Silent success - don't clutter UI
        except:
            pass  # Email is optional, don't block analysis
   
    # Reset state and start analysis
    st.session_state.analysis_running = True
    st.session_state.filtered_citations = []
    st.session_state.original_findings = ""
    st.session_state.analysis_error = None
    st.session_state.analysis_queue = Queue()
    st.session_state.analysis_thread = threading.Thread(
        target=analyze_thread,
        args=(st.session_state.analysis_queue, doi, inquired_result)
    )
    st.session_state.analysis_thread.daemon = True
    st.session_state.analysis_thread.start()
    st.rerun()
# Handle payment callback (simple check)
query_params = st.experimental_get_query_params()
if 'success' in query_params:
    st.session_state.paid = True
    st.success("Payment successful! You can now run more analyses.")
# Display analysis progress if running
if st.session_state.analysis_running:
    st.success("🚀 Analysis Running! Processing your request...")
   
    # Create prominent progress section
    st.markdown("---")
    st.markdown("### 📊 Analysis Progress")
    st.markdown("**Status:** Analyzing...")
    progress_bar.progress(0)
    status_text.text("Processing...")
    info_text.text("Fetching data...")
    estimated_time_text.text("Calculating time...")
    st.markdown("---")
   
    # Process queue messages
    queue = st.session_state.analysis_queue
    filtered_citations = st.session_state.filtered_citations
    original_findings = st.session_state.original_findings
   
    messages_processed = 0
    while not queue.empty() and messages_processed < 50:  # Limit messages per rerun
        msg_type, msg = queue.get()
        messages_processed += 1
       
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
            st.session_state.analysis_error = msg
            st.session_state.analysis_running = False
            st.error(f"❌ Error: {msg}")
        elif msg_type == "result":
            st.session_state.filtered_citations = msg
            st.session_state.analysis_running = False
            filtered_citations = msg
   
    # Check if thread is still alive
    if st.session_state.analysis_thread and not st.session_state.analysis_thread.is_alive():
        # Thread finished, check for final results
        if not st.session_state.filtered_citations and not st.session_state.analysis_error:
            while not queue.empty():
                msg_type, msg = queue.get()
                if msg_type == "result":
                    st.session_state.filtered_citations = msg
                    st.session_state.analysis_running = False
                    filtered_citations = msg
                    break
                elif msg_type == "error":
                    st.session_state.analysis_error = msg
                    st.session_state.analysis_running = False
   
    # Auto-refresh while running
    if st.session_state.analysis_running:
        time.sleep(0.5)
        st.rerun()
   
    filtered_citations = st.session_state.filtered_citations
    original_findings = st.session_state.original_findings
   
elif st.session_state.filtered_citations:
    # Analysis complete, show results
    filtered_citations = st.session_state.filtered_citations
    original_findings = st.session_state.original_findings
else:
    filtered_citations = []
    original_findings = ""
if filtered_citations:
    st.write(f"**Filtered Citations Found:** {len(filtered_citations)}")
   
    # Process with AI-powered text analysis (using xAI Grok API)
    status_text.text("🤖 Analyzing abstracts with xAI Grok...")
    st.info("📖 Now analyzing paper abstracts using Grok AI classification...")
   
    # Create parallel lists to maintain citation-abstract mapping
    abstracts_with_indices = []
    for i, cite in enumerate(filtered_citations):
        if cite['abstract'] != 'No abstract':
            abstracts_with_indices.append((i, cite['abstract']))
   
    analyses = []
   
    # Test xAI API
    try:
        headers = {
            "Authorization": f"Bearer {xai_api_key}",
            "Content-Type": "application/json"
        }
        test_payload = {
            "model": "grok-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ]
        }
        # Test API connectivity first
        test_response = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=test_payload)
        if test_response.status_code != 200:
            raise Exception(f"xAI API test failed: {test_response.text}")

    except Exception as e:
        st.error(f"❌ xAI API error: {e}")
        st.session_state.analysis_running = False
        return  # Or queue.put(("error", str(e)))

    # Now analyze abstracts in batches (e.g., 5 at a time) to avoid rate limits
    batch_size = 5
    for batch_start in range(0, len(abstracts_with_indices), batch_size):
        batch = abstracts_with_indices[batch_start:batch_start + batch_size]
        prompts = []
        for i, abstract in batch:
            prompt = f"Classify if this abstract supports (1), contradicts (-1), or is neutral (0) to the original finding: '{original_findings}'. Abstract: {abstract[:2000]}"  # Truncate if too long
            prompts.append(prompt)

        # Batch prompt (or send individually if API doesn't support multi-messages)
        payload = {
            "model": "grok-4",
            "messages": [{"role": "system", "content": "You are a scientific classifier. Respond with only: support (1), contradict (-1), or neutral (0)."},
                         {"role": "user", "content": "\n\n".join(prompts)}]  # If batching; else loop individually
        }
        try:
            response = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            api_result = response.json()['choices'][0]['message']['content']
            # Parse results (assume API returns "1\n-1\n0" for batch)
            classifications = [int(cls.strip()) for cls in api_result.split("\n") if cls.strip() in ['1', '-1', '0']]
            for j, cls in enumerate(classifications):
                filtered_citations[batch[j][0]]['ai_category'] = cls  # Add to citation dict
            time.sleep(0.5)  # Rate limit
        except Exception as e:
            st.error(f"AI batch error: {e}")
            continue

    # Calculate reproducibility score
    total_support = sum(cite['geo_score'] * cite['impact_factor'] for cite in filtered_citations if cite.get('ai_category', 0) == 1)
    total_contradict = sum(cite['geo_score'] * cite['impact_factor'] for cite in filtered_citations if cite.get('ai_category', 0) == -1)
    total_neutral = len([cite for cite in filtered_citations if cite.get('ai_category', 0) == 0])
    repro_score = max(0, min(100, (total_support - total_contradict) / max(1, len(filtered_citations)) * 100))  # Normalize 0-100

    # Display results
    st.success(f"✅ Analysis Complete! Reproducibility Score: {repro_score:.1f}/100")
    st.write(f"Supporting: {total_support} | Contradicting: {total_contradict} | Neutral: {total_neutral}")
    
    # Geo-map visualization (example)
    df = pd.DataFrame(filtered_citations)
    fig = px.scatter_geo(df, locations="country", locationmode="country names", hover_name="title",
                         size="impact_factor", color="ai_category")
    st.plotly_chart(fig)

    # Send results email
    if user_email:
        subject = "ReproScore Analysis Results"
        body = f"DOI: {doi}\nScore: {repro_score}\nDetails: [summary here]"
