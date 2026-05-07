"""
Highlights :-
*) Uses a combination of Chunking, Parallel Processing + Queuing & Streaming to reduce the perceived latency, while maintaining byte sequence (voice is streamed in chunks)
*) 3 step AI process - STT -> Text-to-Text (T2T) -> TTS
*) "Client Server Architecture" based Server App.
*) Integrated with WebSockets, WebRTCVAD, Multithreading, AsyncIO, Transformers (for LLMs)
*) Using OpenAI completions API for finetuned responses, and OpenAI speech API for voice.
*) Flexibility to integrate individual on-premise LLMs, eg: Gemini (T2T), Kokoro (TTS), IndicParler (TTS), CoQUI (TTS), Bark (TTS), etc.
*) TTFB of "voice bytes" ~2 seconds.
*) Capability for noise reduction.
"""


"""
App Setup Description followed by Commands :-
1) Download Anaconda : wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
2) Run the binary to set it up : sh Anaconda3-2024.10-1-Linux-x86_64.sh
3) Load conda in the terminal env : source ~/.bashrc
4) Create conda environment with python 3.12 : conda create --name myenv python=3.12
5) Activate conda environment : conda activate myenv
6) Install required packages : pip install torch indic-num2words torchaudio rapidfuzz asyncio websockets webrtcvad torch wave numpy transformers noisereduce soundfile openai chromadb sentence-transformers elevenlabs
7) Mention the Port under "Websocket Configuration"
8) Run the file : python hindi_bot.py
9) Service is ready to connect from the client via WebSockets
"""


import asyncio
import queue
import threading
import websockets
import webrtcvad
import torch
import torchaudio
import traceback
import time
import noisereduce as nr
import numpy as np
import re
import wave
import json
import requests
from rapidfuzz import process
from pathlib import Path
from torch import nn

from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from vectordb_mgmt import *
import urllib.parse
import collections
from datetime import datetime
import time
from rapidfuzz import process, fuzz
import re
import asyncio, threading, queue
import numpy as np
import torchaudio
import noisereduce as nr
import io
import soundfile as sf
from omnivoice import OmniVoice
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
from groq import Groq


###########################
# WEBSOCKET CONFIGURATION
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8579"))
###########################

###########################
# API CLIENTS

GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
visit_id          = os.getenv("VISIT_ID", "")

# Groq client — text-to-text (conversation + slot extraction)
groq_client = Groq(api_key=GROQ_API_KEY)

# Groq client — speech-to-text (whisper-large-v3 via API)
groq_stt_client = Groq(api_key=os.getenv("GROQ_STT_API_KEY", ""))

# Chroma — embeddings computed locally via SentenceTransformer, no API key needed
chroma_client = chromadb.Client()
intent_collection = chroma_client.create_collection(name="intent_collection")


sentenceTransformerModel = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
intent_dataset = [
    # --- INTERESTED / AP PARTNERSHIP (Rupeezy - Hindi + English) ---
    ("Haan, sunao", "interested"),
    ("Haan ji, batao", "interested"),
    ("Haan, mere paas 2 minute hain", "interested"),
    ("Main sun raha hoon", "interested"),
    ("I'm interested", "interested"),
    ("Sounds interesting, tell me more", "interested"),
    ("How do I sign up?", "interested"),
    ("I want to join as AP partner", "interested"),
    ("Let me know more about this", "interested"),
    ("Yes, please continue", "interested"),
    ("Okay, go ahead", "interested"),
    ("I'd like to partner with Rupeezy", "interested"),
    ("Tell me more about this opportunity", "interested"),
    ("I'm open to hearing more", "interested"),
    ("This sounds like a good opportunity", "interested"),
    ("How can I become an AP partner?", "interested"),
    ("I want to register as an AP", "interested"),
    ("Sign me up", "interested"),
    ("Let's proceed", "interested"),
    ("I'm ready to join", "interested"),
    ("yes", "interested"),
    ("yeah", "interested"),
    ("sure", "interested"),
    ("okay", "interested"),
    ("ok", "interested"),
    ("alright", "interested"),
    ("absolutely", "interested"),
    ("of course", "interested"),
    ("go on", "interested"),
    ("tell me", "interested"),
    ("हाँ", "interested"),
    ("हाँ जी", "interested"),
    ("जी हाँ", "interested"),
    ("हाँ, सुनाओ", "interested"),
    ("हाँ जी, बताओ", "interested"),
    ("हाँ, मेरे पास 2 मिनट हैं", "interested"),
    ("मैं सुन रहा हूँ", "interested"),
    ("AP partner बनना चाहता हूँ", "interested"),
    ("Rupeezy के साथ join करना चाहता हूँ", "interested"),
    ("Sign up करना चाहता हूँ", "interested"),
    ("ठीक है, बताओ", "interested"),
    ("हाँ, interesting लगता है", "interested"),
    ("Partner program में interested हूँ", "interested"),
    ("हाँ, partner बनूंगा", "interested"),
    ("मुझे partner program के बारे में बताइए", "interested"),
    ("आगे बढ़ते हैं", "interested"),
    ("Registration करना है", "interested"),
    ("बिल्कुल, बताइए", "interested"),
    ("Main join karna chahta hoon", "interested"),
    ("Main partner banna chahta hoon", "interested"),
    ("Rupeezy AP program mein interested hoon", "interested"),
    ("mujhe yeh opportunity pasand aayi", "interested"),
    ("Haan, try karte hain", "interested"),
    ("Haan theek hai", "interested"),
    ("Let's do it", "interested"),
    ("मैं तैयार हूँ", "interested"),
    ("चलिए शुरू करते हैं", "interested"),
    ("bilkul", "interested"),
    ("ji", "interested"),
    ("haan", "interested"),
    ("haan ji", "interested"),
    ("ha", "interested"),
    ("acha", "interested"),
    ("accha", "interested"),
    ("thik hai", "interested"),
    ("theek hai", "interested"),
    ("Mujhe join karna hai", "interested"),
    ("Partner program ke details batao", "interested"),
    ("Earning opportunity sunna chahta hoon", "interested"),

    # --- OBJECTIONS (Rupeezy AP - Hindi + English) ---
    ("Main pehle se Zerodha ke saath hoon", "obj_already_broker"),
    ("Mera Groww account hai pehle se", "obj_already_broker"),
    ("Main already ek dusre broker ke saath kaam kar raha hoon", "obj_already_broker"),
    ("Main Upstox use karta hoon", "obj_already_broker"),
    ("I already work with another broker", "obj_already_broker"),
    ("I have an account on Zerodha", "obj_already_broker"),
    ("I'm already registered with a brokerage", "obj_already_broker"),
    ("मैं पहले से Zerodha के साथ हूँ", "obj_already_broker"),
    ("मेरा Groww account है", "obj_already_broker"),
    ("मैं already किसी broker के साथ काम कर रहा हूँ", "obj_already_broker"),
    ("मेरा Upstox पर account है", "obj_already_broker"),
    ("मैं पहले से किसी platform पर हूँ", "obj_already_broker"),

    ("Mere paas bahut kam contacts hain", "obj_no_contacts"),
    ("Mera network bahut chhota hai", "obj_no_contacts"),
    ("Main naya hoon is field mein", "obj_no_contacts"),
    ("Zyada log nahi hain mere contacts mein", "obj_no_contacts"),
    ("I don't have many clients", "obj_no_contacts"),
    ("My network is very small", "obj_no_contacts"),
    ("I'm new to this field", "obj_no_contacts"),
    ("मेरे पास बहुत कम contacts हैं", "obj_no_contacts"),
    ("मेरा network बहुत छोटा है", "obj_no_contacts"),
    ("मैं नया हूँ इस field में", "obj_no_contacts"),
    ("ज़्यादा लोग नहीं हैं मेरे contacts में", "obj_no_contacts"),

    ("Support kaisi milegi?", "obj_support"),
    ("Client ko problem aane pe kya hoga?", "obj_support"),
    ("Mujhe technical cheezein samajh nahi aati", "obj_support"),
    ("Training kab milegi?", "obj_support"),
    ("Who will handle my client issues?", "obj_support"),
    ("Is there a helpline I can call?", "obj_support"),
    ("support कैसे मिलेगी?", "obj_support"),
    ("client को problem आने पर क्या होगा?", "obj_support"),
    ("technical cheezein samajh nahi aati", "obj_support"),
    ("मुझे training कब मिलेगी?", "obj_support"),

    ("Rupeezy ka kya bharosa?", "obj_trust"),
    ("Yeh company kitni reliable hai?", "obj_trust"),
    ("SEBI mein registered hai?", "obj_trust"),
    ("Paise safe hain?", "obj_trust"),
    ("Yeh scam toh nahi hai?", "obj_trust"),
    ("Is Rupeezy a genuine company?", "obj_trust"),
    ("How long has Rupeezy been around?", "obj_trust"),
    ("Can I trust this platform?", "obj_trust"),
    ("Rupeezy का क्या भरोसा?", "obj_trust"),
    ("यह company कितनी reliable है?", "obj_trust"),
    ("SEBI में registered है?", "obj_trust"),
    ("पैसे safe हैं?", "obj_trust"),
    ("यह scam तो नहीं है?", "obj_trust"),

    ("Socha hoon, baad mein batata hoon", "obj_think_later"),
    ("Thoda time chahiye mujhe", "obj_think_later"),
    ("Pehle soch lun", "obj_think_later"),
    ("Apni family se baat karke batata hoon", "obj_think_later"),
    ("Let me think about it", "obj_think_later"),
    ("I'll get back to you later", "obj_think_later"),
    ("Give me some time to decide", "obj_think_later"),
    ("I need to discuss with my partner", "obj_think_later"),
    ("सोच लूंगा, बाद में बताऊंगा", "obj_think_later"),
    ("थोड़ा time चाहिए मुझे", "obj_think_later"),
    ("पहले सोच लूं", "obj_think_later"),
    ("बाद में बताता हूँ", "obj_think_later"),
    ("Family से discuss करके बताता हूँ", "obj_think_later"),

    # --- ENQUIRY ABOUT RUPEEZY AP PROGRAM ---
    ("How does the AP partner program work?", "enquiry"),
    ("How much can I earn as an AP?", "enquiry"),
    ("Is there any joining fee?", "enquiry"),
    ("How do daily payouts work?", "enquiry"),
    ("What is the RISE portal?", "enquiry"),
    ("What is the brokerage share percentage?", "enquiry"),
    ("Do I get any training?", "enquiry"),
    ("How many clients do I need?", "enquiry"),
    ("What is the registration process for AP?", "enquiry"),
    ("What is Rupeezy exactly?", "enquiry"),
    ("Is Rupeezy SEBI registered?", "enquiry"),
    ("Is this program legitimate?", "enquiry"),
    ("What is the payout schedule?", "enquiry"),
    ("What is the commission rate?", "enquiry"),
    ("What are the benefits of being an AP partner?", "enquiry"),
    ("When was Rupeezy established?", "enquiry"),
    ("How much commission do I get?", "enquiry"),
    ("How much can I earn monthly?", "enquiry"),
    ("What does 100% brokerage share mean?", "enquiry"),
    ("Is there training for new partners?", "enquiry"),
    ("What support do I get as a partner?", "enquiry"),
    ("Who is my dedicated RM?", "enquiry"),
    ("Can I do this alongside my current job?", "enquiry"),
    ("How do my clients benefit?", "enquiry"),
    ("What products can my clients invest in?", "enquiry"),
    ("How long does registration take?", "enquiry"),
    ("Do I need a license to be an AP?", "enquiry"),
    ("Are there any documents required?", "enquiry"),
    ("Is this only for financial advisors?", "enquiry"),
    ("Can anyone join the AP program?", "enquiry"),
    ("Tell me about Rupeezy", "enquiry"),
    ("What makes Rupeezy different from others?", "enquiry"),
    ("How does this compare to other broker programs?", "enquiry"),
    ("I need more information about the program", "enquiry"),
    ("What is the onboarding process?", "enquiry"),
    ("Are there performance targets for APs?", "enquiry"),
    ("Can I run this from home?", "enquiry"),
    ("AP program kaise kaam karta hai?", "enquiry"),
    ("Kitna earn kar sakta hoon?", "enquiry"),
    ("Joining fee kitni hai?", "enquiry"),
    ("Daily payout kaise milta hai?", "enquiry"),
    ("RISE portal kya hai?", "enquiry"),
    ("Brokerage share kaise hota hai?", "enquiry"),
    ("Training milti hai?", "enquiry"),
    ("Kitne clients chahiye?", "enquiry"),
    ("Registration process kya hai?", "enquiry"),
    ("Rupeezy kya company hai?", "enquiry"),
    ("SEBI registered hai?", "enquiry"),
    ("Commission rate kya hai?", "enquiry"),
    ("Monthly kitna earn ho sakta hai?", "enquiry"),
    ("100% brokerage share matlab kya?", "enquiry"),
    ("Partner ke liye training hai?", "enquiry"),
    ("RM support kaisi hoti hai?", "enquiry"),
    ("पार्टनर प्रोग्राम कैसे काम करता है?", "enquiry"),
    ("Joining fee है क्या?", "enquiry"),
    ("कितना earn हो सकता है?", "enquiry"),
    ("RISE portal क्या है?", "enquiry"),
    ("daily payout कैसे मिलता है?", "enquiry"),
    ("100% brokerage share का मतलब क्या है?", "enquiry"),
    ("Rupeezy SEBI registered है?", "enquiry"),
    ("Registration process क्या है?", "enquiry"),
    ("Partner को training मिलती है?", "enquiry"),
    ("AP partner बनने के फायदे क्या हैं?", "enquiry"),
    ("monthly income कितनी हो सकती है?", "enquiry"),
    ("commission rate क्या है?", "enquiry"),
    ("payout कब मिलता है?", "enquiry"),
    ("onboarding process क्या है?", "enquiry"),
    ("RM support 24x7 मिलती है?", "enquiry"),
    ("minimum कितने clients चाहिए?", "enquiry"),
    ("क्या existing broker के साथ continue कर सकते हैं?", "enquiry"),
    ("Rupeezy के बारे में बताइए", "enquiry"),
    ("यह program legal है?", "enquiry"),
    ("मुझे और information चाहिए", "enquiry"),
    ("payout schedule क्या है?", "enquiry"),
    ("partner support कैसे मिलती है?", "enquiry"),
    ("Rupeezy kitne salon se hai?", "enquiry"),
    ("kya ghar se kaam kar sakte hain?", "enquiry"),



    # --- CANCEL (70) ---
    ("I am not interested", "cancel"),
    ("No, I don't want it", "cancel"),
    ("Cancel the process", "cancel"),
    ("I changed my mind", "cancel"),
    ("Don't proceed", "cancel"),
    ("Stop the purchase", "cancel"),
    ("I don't want insurance now", "cancel"),
    ("Maybe later", "cancel"),
    ("Not required", "cancel"),
    ("I will decide later", "cancel"),
    ("Hold it for now", "cancel"),
    ("Pause the process", "cancel"),
    ("Leave it", "cancel"),
    ("Drop the plan", "cancel"),
    ("Let's not continue", "cancel"),
    ("No thanks", "cancel"),
    ("Cancel my request", "cancel"),
    ("I don't need it now", "cancel"),
    ("Stop it", "cancel"),
    ("मुझे AP program में join नहीं करना", "cancel"),
    ("अभी नहीं चाहिए", "cancel"),
    ("रद्द कर दीजिए", "cancel"),
    ("प्रक्रिया रोक दें", "cancel"),
    ("मुझे अभी नहीं करना है", "cancel"),
    ("कैंसिल कर दीजिए", "cancel"),
    ("मैं Rupeezy join नहीं करना चाहता", "cancel"),
    ("मुझे partner program नहीं चाहिए", "cancel"),
    ("नहीं, interested नहीं हूँ", "cancel"),
    ("अभी कुछ नहीं चाहिए", "cancel"),
    ("बाद में देखेंगे", "cancel"),
    ("अभी नहीं खरीदना", "cancel"),
    ("रुक जाइए", "cancel"),
    ("छोड़ दीजिए", "cancel"),
    ("मुझे अब नहीं चाहिए", "cancel"),
    ("फिलहाल नहीं", "cancel"),
    ("बीमा रद्द कर दीजिए", "cancel"),
    ("अभी रोक दीजिए", "cancel"),
    ("अभी खरीदारी न करें", "cancel"),
    ("रद्द करना है", "cancel"),
    ("नहीं चाहिए", "cancel"),
    ("अभी नहीं करना है", "cancel"),
    ("प्रक्रिया स्थगित करें", "cancel"),
    ("थोड़ी देर बाद", "cancel"),
    ("बाद में निर्णय लेंगे", "cancel"),
    ("अभी नहीं चाहिए", "cancel"),
    ("फिलहाल स्थगित करें", "cancel"),
    ("रद्द करो", "cancel"),
    ("अभी खरीदारी रुकवाइए", "cancel"),
    ("नहीं खरीदना", "cancel"),
    ("छोड़ दो", "cancel"),
    ("बाद में खरीदेंगे", "cancel"),
    ("प्रक्रिया कैंसल करें", "cancel"),
    ("स्टॉप करो", "cancel"),
    ("अभी नहीं खरीदना है", "cancel"),
    ("मुझे अब बीमा नहीं लेना", "cancel"),
    ("अभी नहीं करना चाहता", "cancel"),
    ("अभी ज़रूरत नहीं है", "cancel"),
    ("अभी चर्चा नहीं करना है", "cancel"),
    ("यह मेरे लिए नहीं है", "cancel"),
    ("इतना काफी है", "cancel"),
    ("मुझे partner program नहीं लेना","cancel"),
    ("Rupeezy AP program cancel karna hai","cancel"),
    ("अब और जानकारी नहीं चाहिए", "cancel"),
    ("फिर कभी बात करेंगे", "cancel"),
    ("मैं अभी तैयार नहीं हूँ", "cancel"),
    ("अब नहीं करना", "cancel"),
    ("प्रक्रिया रोक दो", "cancel"),
    ("कृपया बंद करें", "cancel"),
    ("मुझे बाहर निकलना है", "cancel"),
    ("कुछ भी नहीं चाहिए", "cancel"),
    ("मेरे पास समय नहीं है", "cancel"),
    ("मैं जारी नहीं रखना चाहता", "cancel"),
    ("कृपया इसे यहीं समाप्त करें", "cancel"),
    ("अभी कोई निर्णय नहीं लेना", "cancel"),
    ("मुझे कोई रुचि नहीं है", "cancel"),
    ("इन्शुरन्स की जरुरत नहीं है","cancel"),
    ("आगे न बढ़ाएं", "cancel"),
    ("I don't want to continue", "cancel"),
    ("Let's stop here", "cancel"),
    ("This isn't the right time", "cancel"),
    ("I'll get back later", "cancel"),
    ("I don't want to give details", "cancel"),
    ("I'll check it myself", "cancel"),
    ("This is not for me", "cancel"),
    ("Don't ask further", "cancel"),
    ("That's enough", "cancel"),
    ("I'm backing out", "cancel"),
    ("Maybe next time", "cancel"),
    ("I'm not convinced", "cancel"),
    ("End the process", "cancel"),
    ("I'm not ready yet", "cancel"),
    ("Put this on hold", "cancel"),
    ("This is not needed", "cancel"),
    ("I want to exit", "cancel"),
    ("Please stop", "cancel"),
    ("I don't wish to proceed", "cancel"),
    ("Cancel everything", "cancel"),
    ("नहीं","cancel"),



    # --- GREETINGS (70) ---
    ("Hello", "greetings"),
    ("Hi", "greetings"),
    ("Hey", "greetings"),
    ("Good morning", "greetings"),
    ("Good evening", "greetings"),
    ("Good afternoon", "greetings"),
    ("How are you?", "greetings"),
    ("Howdy!", "greetings"),
    ("Nice to meet you", "greetings"),
    ("Hope you are doing well", "greetings"),
    ("Hey there", "greetings"),
    ("Hi there", "greetings"),
    ("Greetings!", "greetings"),
    ("Good night", "greetings"),
    ("Hope you are fine", "greetings"),
    ("Nice to connect", "greetings"),
    ("नमस्ते", "greetings"),
    ("सुप्रभात", "greetings"),
    ("शुभ संध्या", "greetings"),
    ("शुभ रात्रि", "greetings"),
    ("आप कैसे हैं?", "greetings"),
    ("कैसे हो?", "greetings"),
    ("नमस्कार", "greetings"),
    ("राम राम", "greetings"),
    ("सलाम", "greetings"),
    ("आपका स्वागत है", "greetings"),
    ("अदाब", "greetings"),
    ("क्या हाल है?", "greetings"),
    ("कैसे चल रहा है?", "greetings"),

    ("सुबह की शुभकामनाएँ", "greetings"),
    ("संध्या की शुभकामनाएँ", "greetings"),
    ("शुभ दोपहर", "greetings"),
    ("आपसे मिलकर खुशी हुई", "greetings"),
    ("आपसे मिलकर अच्छा लगा", "greetings"),
    ("नमस्कार मित्र", "greetings"),
    ("राम राम जी", "greetings"),
    ("जय श्रीराम", "greetings"),
    ("शुभकामनाएँ", "greetings"),
    ("भाई नमस्ते", "greetings"),
    ("मित्र नमस्कार", "greetings"),
    ("स्वागत है", "greetings"),




    ("समझ नहीं आया","confused"),
    ("मुझे समझ नहीं आया","confused"),
    ("फिर से बताओ","confused"),
    ("दोहराइए","confused"),

    ("नहीं","need_context"),
    ("चालिस साल","need_context"),

    ("सवाल वापीस बताईये", "confused"),
    ("आप क्या कह रहे थे?", "confused"),
    ("नहीं","unknown"),

    # --- RECALL (Rupeezy AP fields) ---
    ("मेरा नाम क्या बताया था मैंने", "recall_name"),
    ("मैंने नाम क्या दर्ज करवाया था", "recall_name"),
    ("मेरा रजिस्टर्ड नाम क्या है", "recall_name"),
    ("आपने मेरा नाम क्या स्टोर किया", "recall_name"),
    ("can you tell me my registered name", "recall_name"),
    ("what name do you have on file", "recall_name"),
    ("repeat the name I gave you", "recall_name"),
    ("मैंने profession क्या बताया था", "recall_profession"),
    ("मेरा profession क्या है आपके records में", "recall_profession"),
    ("what profession did I mention", "recall_profession"),
    ("what do you have as my profession", "recall_profession"),
    ("मेरी city क्या बताई थी", "recall_city"),
    ("मैं किस शहर का हूँ आपके records में", "recall_city"),
    ("which city did I say I'm from", "recall_city"),
    ("what city is in my records", "recall_city"),
    ("मेरा WhatsApp number क्या था", "recall_phone"),
    ("मैंने जो phone number दिया था वो क्या था", "recall_phone"),
    ("what phone number did I give", "recall_phone"),
    ("what's my registered WhatsApp number", "recall_phone"),
    ("मेरा network size क्या था", "recall_network"),
    ("मैंने कितने contacts बताए थे", "recall_network"),
    ("how many contacts did I mention", "recall_network"),
    ("what network size did I tell you", "recall_network"),

    # ── TAMIL (ta) ────────────────────────────────────────────────────────────
    ("நான் partner ஆக சேர விரும்புகிறேன்", "interested"),
    ("Rupeezy-ல் register செய்ய விரும்புகிறேன்", "interested"),
    ("சரி, சேருகிறேன்", "interested"),
    ("Sign up பண்ண விரும்புகிறேன்", "interested"),
    ("கூட்டணி திட்டத்தில் சேர விரும்புகிறேன்", "interested"),
    ("ஆமாம், முன்னேறுங்கள்", "interested"),

    ("நான் ஏற்கனவே Zerodha-வுடன் இருக்கிறேன்", "obj_already_broker"),
    ("வேற broker use பண்றேன்", "obj_already_broker"),
    ("தற்போது வேறொரு platform-ல் இருக்கிறேன்", "obj_already_broker"),

    ("என்னிடம் போதுமான contacts இல்லை", "obj_no_contacts"),
    ("என் network சிறியது", "obj_no_contacts"),
    ("அதிகம் பேரை தெரியாது", "obj_no_contacts"),

    ("client-களுக்கு பிரச்சனை வந்தால் யார் பார்ப்பார்கள்?", "obj_support"),
    ("support எப்படி இருக்கும்?", "obj_support"),

    ("Rupeezy நம்பகமானதா?", "obj_trust"),
    ("இது SEBI registered-ஆ?", "obj_trust"),
    ("இந்த company பற்றி கேள்விப்படவில்லை", "obj_trust"),

    ("யோசித்து சொல்கிறேன்", "obj_think_later"),
    ("பிறகு பேசலாம்", "obj_think_later"),
    ("கொஞ்சம் நேரம் தேவை", "obj_think_later"),

    ("partner program எப்படி வேலை செய்கிறது?", "enquiry"),
    ("எவ்வளவு சம்பாதிக்கலாம்?", "enquiry"),
    ("joining fee இருக்கிறதா?", "enquiry"),

    ("வேண்டாம்", "cancel"),
    ("நிறுத்துங்கள்", "cancel"),
    ("இல்லை", "cancel"),

    ("வணக்கம்", "greetings"),
    ("நமஸ்காரம்", "greetings"),

    ("புரியவில்லை", "confused"),
    ("மீண்டும் சொல்லுங்கள்", "confused"),

    # ── TELUGU (te) ───────────────────────────────────────────────────────────
    ("నేను partner గా చేరాలనుకుంటున్నాను", "interested"),
    ("Sign up చేయాలనుకుంటున్నాను", "interested"),
    ("సరే, చేరుతాను", "interested"),
    ("Rupeezy partner అవ్వాలనుకుంటున్నాను", "interested"),
    ("ముందుకు వెళ్దాం", "interested"),

    ("నేను ఇప్పటికే Zerodha తో ఉన్నాను", "obj_already_broker"),
    ("వేరే broker use చేస్తున్నాను", "obj_already_broker"),
    ("మరో platform తో పని చేస్తున్నాను", "obj_already_broker"),

    ("నాకు తగినంత contacts లేవు", "obj_no_contacts"),
    ("నా network చిన్నది", "obj_no_contacts"),
    ("ఎక్కువ మందిని తెలియదు", "obj_no_contacts"),

    ("clients కి సమస్య వస్తే ఎవరు చూస్తారు?", "obj_support"),
    ("support ఎలా ఉంటుంది?", "obj_support"),

    ("Rupeezy నమ్మకమైనదా?", "obj_trust"),
    ("SEBI registered అయినదా?", "obj_trust"),
    ("ఈ company గురించి వినలేదు", "obj_trust"),

    ("ఆలోచిస్తాను", "obj_think_later"),
    ("తర్వాత మాట్లాడదాం", "obj_think_later"),
    ("కొంత సమయం కావాలి", "obj_think_later"),

    ("partner program ఎలా పని చేస్తుంది?", "enquiry"),
    ("ఎంత సంపాదించవచ్చు?", "enquiry"),
    ("joining fee ఉందా?", "enquiry"),

    ("వద్దు", "cancel"),
    ("ఆపండి", "cancel"),
    ("అవసరం లేదు", "cancel"),

    ("నమస్కారం", "greetings"),
    ("హలో", "greetings"),

    ("అర్థం కాలేదు", "confused"),
    ("మళ్ళీ చెప్పండి", "confused"),

    # ── MARATHI (mr) ──────────────────────────────────────────────────────────
    ("मला partner म्हणून सामील व्हायचे आहे", "interested"),
    ("Sign up करायचे आहे", "interested"),
    ("हो, सामील होतो", "interested"),
    ("Rupeezy partner बनायचे आहे", "interested"),
    ("पुढे जाऊया", "interested"),

    ("मी आधीच Zerodha सोबत आहे", "obj_already_broker"),
    ("दुसरा broker वापरतो", "obj_already_broker"),
    ("दुसऱ्या platform वर आहे मी", "obj_already_broker"),

    ("माझ्याकडे पुरेसे contacts नाहीत", "obj_no_contacts"),
    ("माझे network लहान आहे", "obj_no_contacts"),
    ("जास्त लोक माहित नाहीत", "obj_no_contacts"),

    ("clients ना प्रश्न आला तर कोण बघेल?", "obj_support"),
    ("support कसे असेल?", "obj_support"),

    ("Rupeezy विश्वासार्ह आहे का?", "obj_trust"),
    ("SEBI registered आहे का?", "obj_trust"),
    ("या company बद्दल ऐकले नाही", "obj_trust"),

    ("विचार करतो", "obj_think_later"),
    ("नंतर बोलूया", "obj_think_later"),
    ("थोडा वेळ हवा", "obj_think_later"),

    ("partner program कसे काम करते?", "enquiry"),
    ("किती कमावता येईल?", "enquiry"),
    ("joining fee आहे का?", "enquiry"),

    ("नको", "cancel"),
    ("थांबा", "cancel"),
    ("नाही", "cancel"),

    ("नमस्कार", "greetings"),
    ("नमस्ते", "greetings"),

    ("समजले नाही", "confused"),
    ("परत सांगा", "confused"),

    # ── GUJARATI (gu) ─────────────────────────────────────────────────────────
    ("મારે partner તરીકે જોડાવું છે", "interested"),
    ("Sign up કરવું છે", "interested"),
    ("હા, જોડાઉ છું", "interested"),
    ("Rupeezy partner બનવું છે", "interested"),
    ("આગળ ચાલો", "interested"),

    ("હું પહેલેથી Zerodha સાથે છું", "obj_already_broker"),
    ("બીજો broker use કરું છું", "obj_already_broker"),
    ("બીજા platform પર છું", "obj_already_broker"),

    ("મારી પાસે પૂરતા contacts નથી", "obj_no_contacts"),
    ("મારું network નાનું છે", "obj_no_contacts"),
    ("વધારે લોકો ઓળખતો નથી", "obj_no_contacts"),

    ("clients ને problem આવે તો કોણ સંભાળશે?", "obj_support"),
    ("support કેવું છે?", "obj_support"),

    ("Rupeezy ભરોસાપાત્ર છે?", "obj_trust"),
    ("SEBI registered છે?", "obj_trust"),
    ("આ company વિશે સાંભળ્યું નથી", "obj_trust"),

    ("વિચારું છું", "obj_think_later"),
    ("પછી વાત કરીએ", "obj_think_later"),
    ("થોડો સમય જોઈએ", "obj_think_later"),

    ("partner program કેવી રીતે કામ કરે?", "enquiry"),
    ("કેટલું કમાઈ શકાય?", "enquiry"),
    ("joining fee છે?", "enquiry"),

    ("ના", "cancel"),
    ("બંધ કરો", "cancel"),
    ("નથી જોઈતું", "cancel"),

    ("નમસ્તે", "greetings"),
    ("કેમ છો", "greetings"),

    ("સમજ ન પડ્યું", "confused"),
    ("ફરીથી કહો", "confused"),

    # ── BENGALI (bn) ──────────────────────────────────────────────────────────
    ("আমি partner হিসেবে যোগ দিতে চাই", "interested"),
    ("Sign up করতে চাই", "interested"),
    ("হ্যাঁ, যোগ দেব", "interested"),
    ("Rupeezy partner হতে চাই", "interested"),
    ("এগিয়ে যান", "interested"),

    ("আমি ইতিমধ্যে Zerodha-র সাথে আছি", "obj_already_broker"),
    ("অন্য broker ব্যবহার করি", "obj_already_broker"),
    ("অন্য platform-এ আছি", "obj_already_broker"),

    ("আমার যথেষ্ট contacts নেই", "obj_no_contacts"),
    ("আমার network ছোট", "obj_no_contacts"),
    ("বেশি মানুষ চিনি না", "obj_no_contacts"),

    ("clients-দের সমস্যা হলে কে দেখবে?", "obj_support"),
    ("support কেমন?", "obj_support"),

    ("Rupeezy বিশ্বাসযোগ্য?", "obj_trust"),
    ("এটা SEBI registered?", "obj_trust"),
    ("এই company সম্পর্কে শুনিনি", "obj_trust"),

    ("ভাবব", "obj_think_later"),
    ("পরে কথা বলব", "obj_think_later"),
    ("একটু সময় দরকার", "obj_think_later"),

    ("partner program কীভাবে কাজ করে?", "enquiry"),
    ("কত আয় করা যাবে?", "enquiry"),
    ("joining fee আছে?", "enquiry"),

    ("না", "cancel"),
    ("বন্ধ করুন", "cancel"),
    ("দরকার নেই", "cancel"),

    ("নমস্কার", "greetings"),
    ("হ্যালো", "greetings"),

    ("বুঝলাম না", "confused"),
    ("আবার বলুন", "confused"),
]


# Add documents to the Chroma collection
intent_collection.add(
    documents=[txt for txt, intent in intent_dataset],
    embeddings=[sentenceTransformerModel.encode(txt, normalize_embeddings=True).tolist() for txt, intent in intent_dataset],  # Ensure list type if required
    metadatas=[{"intent": intent} for txt, intent in intent_dataset],
    ids=[f"intent_{i}" for i in range(len(intent_dataset))]
)

# Function to detect intent from user text input
def detect_intent_with_chroma(text: str) -> str:
    embedding = sentenceTransformerModel.encode(text, normalize_embeddings=True).tolist()  # Ensure it's a list
    result = intent_collection.query(query_embeddings=[embedding], n_results=1)
    printLogs(f"result: {result}")
    
    try:
        if result["distances"][0][0] <= 0.45:     # Only return values which match, otherwise unknown
            return result["metadatas"][0][0]["intent"]
    except (IndexError, KeyError, TypeError):
        return "unknown"
    return "unknown"



###########################

###########################
# CONSTANTS
SAMPLE_RATE = 16000
VAD_MODE = 3  # 0–3, higher = more aggressive about detecting speech
FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * (FRAME_MS / 1000.0) * 2)  # 2 bytes per sample (16‑bit mono)
PAUSE_THRESHOLD_FRAMES = 10000
###########################

###########################
# AI MODELS
textToTextModelId    = os.getenv("GROQ_T2T_MODEL_ID",   "llama-3.3-70b-versatile")
extractModelId       = os.getenv("EXTRACT_MODEL_ID",    "llama-3.3-70b-versatile")
ELEVENLABS_VOICE_ID  = os.getenv("ELEVENLABS_VOICE_ID", "1qEiC6qsybMkmnNdVMbK")
ELEVENLABS_TTS_MODEL = os.getenv("ELEVENLABS_TTS_MODEL","eleven_multilingual_v2")

###########################
# QUESTIONNAIRE (inline – can be moved to questions.json)
QUESTIONNAIRE = [
    {
        "id": "name",
        "prompt": "शुरुआत करते हैं — आपका पूरा नाम बताइए?",
        "type": "string",
        "required": True,
        "short": "नाम"
    },
    {
        "id": "profession",
        "prompt": "आप professionally क्या करते हैं — job, business, या कुछ और?",
        "type": "string",
        "required": True,
        "short": "profession"
    },
    {
        "id": "network_size",
        "prompt": "आपके contacts में roughly कितने लोग हैं जो invest करते हैं या करना चाहते हैं?",
        "type": "number",
        "required": True,
        "short": "network size",
        "min": 1
    },
    {
        "id": "city",
        "prompt": "आप किस शहर में हैं?",
        "type": "string",
        "required": True,
        "short": "शहर"
    },
    {
        "id": "current_broker",
        "prompt": "क्या आप पहले से किसी broker platform के साथ काम कर रहे हैं?",
        "type": "string",
        "required": False,
        "short": "current broker"
    },
    {
        "id": "phone",
        "prompt": "आपका WhatsApp number दीजिए — हम sign-up link वहाँ भेज देंगे।",
        "type": "phone",
        "required": True,
        "short": "WhatsApp number"
    },
]

# ── Multilingual question prompts ──────────────────────────────────────────────
QUESTIONNAIRE_LANG = {
    "en": {
        "name":           "Let's start — what's your full name?",
        "profession":     "What do you do professionally — job, business, or something else?",
        "network_size":   "Roughly how many people in your contacts invest or want to invest?",
        "city":           "Which city are you in?",
        "current_broker": "Are you already working with any broker platform?",
        "phone":          "What's your WhatsApp number? We'll send you the sign-up link there.",
    },
    "hi": {
        "name":           "शुरुआत करते हैं — आपका पूरा नाम बताइए?",
        "profession":     "आप professionally क्या करते हैं — job, business, या कुछ और?",
        "network_size":   "आपके contacts में roughly कितने लोग हैं जो invest करते हैं या करना चाहते हैं?",
        "city":           "आप किस शहर में हैं?",
        "current_broker": "क्या आप पहले से किसी broker platform के साथ काम कर रहे हैं?",
        "phone":          "आपका WhatsApp number दीजिए — हम sign-up link वहाँ भेज देंगे।",
    },
    "ta": {
        "name":           "ஆரம்பிக்கலாம் — உங்கள் முழு பெயர் என்ன?",
        "profession":     "நீங்கள் தொழில் ரீதியாக என்ன செய்கிறீர்கள் — வேலை, தொழில், அல்லது வேறு ஏதாவது?",
        "network_size":   "உங்கள் contacts-ல் தோராயமாக எத்தனை பேர் முதலீடு செய்கிறார்கள் அல்லது செய்ய விரும்புகிறார்கள்?",
        "city":           "நீங்கள் எந்த நகரத்தில் இருக்கிறீர்கள்?",
        "current_broker": "நீங்கள் ஏற்கனவே எந்த broker platform-உடன் வேலை செய்கிறீர்களா?",
        "phone":          "உங்கள் WhatsApp number என்ன? நாங்கள் sign-up link அனுப்புவோம்.",
    },
    "te": {
        "name":           "మొదలుపెడదాం — మీ పూర్తి పేరు చెప్పండి?",
        "profession":     "మీరు professionally ఏమి చేస్తారు — job, business, లేదా మరొకటి?",
        "network_size":   "మీ contacts లో దాదాపు ఎంత మంది invest చేస్తున్నారు లేదా చేయాలనుకుంటున్నారు?",
        "city":           "మీరు ఏ నగరంలో ఉన్నారు?",
        "current_broker": "మీరు ఇప్పటికే ఏదైనా broker platform తో పని చేస్తున్నారా?",
        "phone":          "మీ WhatsApp number చెప్పండి — మేము sign-up link పంపుతాము.",
    },
    "mr": {
        "name":           "सुरुवात करूया — तुमचं पूर्ण नाव सांगा?",
        "profession":     "तुम्ही professionally काय करता — नोकरी, व्यवसाय, किंवा इतर काही?",
        "network_size":   "तुमच्या contacts मध्ये साधारण किती लोक invest करतात किंवा करायला इच्छुक आहेत?",
        "city":           "तुम्ही कोणत्या शहरात आहात?",
        "current_broker": "तुम्ही आधीच कोणत्या broker platform सोबत काम करत आहात का?",
        "phone":          "तुमचा WhatsApp number द्या — आम्ही sign-up link तिथे पाठवू.",
    },
    "gu": {
        "name":           "શરૂ કરીએ — તમારું પૂરું નામ બતાવો?",
        "profession":     "તમે professionally શું કરો છો — નોકરી, ધંધો, અથવા બીજું?",
        "network_size":   "તમારા contacts માં લગભગ કેટલા લોકો invest કરે છે અથવા કરવા ઇચ્છે છે?",
        "city":           "તમે કઈ city માં છો?",
        "current_broker": "શું તમે પહેલેથી કોઈ broker platform સાથે કામ કરો છો?",
        "phone":          "તમારો WhatsApp number આપો — અમે sign-up link ત્યાં મોકલીશું.",
    },
    "bn": {
        "name":           "শুরু করি — আপনার পুরো নাম বলুন?",
        "profession":     "আপনি পেশাগতভাবে কী করেন — চাকরি, ব্যবসা, নাকি অন্য কিছু?",
        "network_size":   "আপনার contacts-এ প্রায় কতজন মানুষ invest করেন বা করতে চান?",
        "city":           "আপনি কোন শহরে আছেন?",
        "current_broker": "আপনি কি ইতিমধ্যে কোনো broker platform-এর সাথে কাজ করছেন?",
        "phone":          "আপনার WhatsApp number দিন — আমরা sign-up link সেখানে পাঠাব।",
    },
}

def qprompt(q: dict, lang: str) -> str:
    """Return the question prompt in the session language, falling back to Hindi."""
    return QUESTIONNAIRE_LANG.get(lang, QUESTIONNAIRE_LANG["hi"]).get(q["id"], q["prompt"])

# ── Language metadata ──────────────────────────────────────────────────────────
LANG_NAMES = {
    "hi": "Hindi", "en": "English", "ta": "Tamil",
    "te": "Telugu", "mr": "Marathi", "gu": "Gujarati", "bn": "Bengali",
}

# Intro texts in all supported languages (name placeholder = {name})
INTRO_TEXTS = {
    "hi": (
        "Namaste{name}! Main Priya bol rahi hoon, Rupeezy se. "
        "Kya aapke paas 2 minute hain? "
        "Main aapko ek bahut achhi earning opportunity ke baare mein batana chahti hoon."
    ),
    "en": (
        "Hi{name}! This is Priya calling from Rupeezy. "
        "Do you have a couple of minutes? "
        "I wanted to share a partner opportunity that I think would be a great fit for you."
    ),
    "ta": (
        "Vanakkam{name}! Naan Priya, Rupeezy-il irunthu pesugiren. "
        "Ungalukku 2 nimisham irukka? "
        "Ungalukkaga oru nalla earning opportunity pattri sollanam."
    ),
    "te": (
        "Namaskaram{name}! Nenu Priya, Rupeezy nundi meerto matladutunna. "
        "Meeru 2 nimishalu cheppagalara? "
        "Meeru ki manchidi ayye partner opportunity gurinchi cheppalanikuntunna."
    ),
    "mr": (
        "Namaskar{name}! Mi Priya bolte, Rupeezy-madhun. "
        "Tumhala 2 minute aahet ka? "
        "Ek khup changle earning opportunity sanga."
    ),
    "gu": (
        "Namaste{name}! Hu Priya bol rahi chhu, Rupeezy tharathi. "
        "Tamara paas 2 minute chhe? "
        "Ek khub saari earning opportunity vishe janawu chhu."
    ),
    "bn": (
        "Namaskar{name}! Ami Priya bolchi, Rupeezy theke. "
        "Apnar kache 2 minute achhe? "
        "Apnar jonyo ektai khub bhalo earning opportunity-r bishoye bolte chai."
    ),
}

# Language confirmation question — sent immediately after intro
LANG_CONFIRM_Q = {
    "hi": (
        "Aur ek baat — main aapko abhi Hindi mein baat kar rahi hoon. "
        "Kya yeh theek hai? Ya aap English, Tamil, Telugu, Marathi, Gujarati, "
        "ya Bengali mein prefer karte hain?"
    ),
    "en": (
        "Also — I'm speaking in English right now. Is that comfortable for you? "
        "Or would you prefer Hindi, Tamil, Telugu, Marathi, Gujarati, or Bengali?"
    ),
    "ta": (
        "Oru vishayam — naan ippo Tamil-il pesugiren. Ungalukkaga sari thana? "
        "Illai English, Hindi, Telugu, Marathi, Gujarati, Bengali vidambu?"
    ),
    "te": (
        "Okka vishayam — nenu ipudu Telugu-lo matladutunna. Meeru okay-na? "
        "Leka English, Hindi, Tamil, Marathi, Gujarati, Bengali prefer chestarara?"
    ),
    "mr": (
        "Ek goshta — mi ata Marathi mein bolte. Tumhala theek ahe na? "
        "Ki tumhi English, Hindi, Tamil, Telugu, Gujarati, Bengali prefer karnar?"
    ),
    "gu": (
        "Ek vaat — hu ahhi Gujarati ma vaat karu chhu. Tamne thaik chhe? "
        "Ke tamne English, Hindi, Tamil, Telugu, Marathi, Bengali pasand chhe?"
    ),
    "bn": (
        "Ekta byapar — ami ekhon Bangla-te bolchi. Eta ki apnar subidhajanak? "
        "Nahole English, Hindi, Tamil, Telugu, Marathi, Gujarati prefer koren?"
    ),
}

# Switch acknowledgement after language change
LANG_SWITCH_ACK = {
    "hi": "Bilkul! Ab main Hindi mein baat karta hoon.",
    "en": "Of course! Continuing in English.",
    "ta": "Sari! Naan Tamil-il thodarvugiren.",
    "te": "Sari! Nenu Telugu-lo continue chestanu.",
    "mr": "Theek aahe! Mi Marathi mein pudhey bolto.",
    "gu": "Saru! Hu Gujarati ma aage vaat karis.",
    "bn": "Thik achhe! Ami Bangla-te continue korchi.",
}

# Language confirmation ack (no switch needed)
LANG_CONFIRM_ACK = {
    "hi": "Achha! Chaliye aage badhte hain.",
    "en": "Great! Let's continue.",
    "ta": "Nandru! Thodaruvom.",
    "te": "Manchidi! Mundukundam.",
    "mr": "Chhan! Pudhe jaau.",
    "gu": "Saras! Aage vadhiye.",
    "bn": "Bhalo! Egiye jai.",
}

def detect_language_preference(text: str) -> str | None:
    """
    Parse a user's language-preference response.
    Returns an ISO code if they want a specific language, or None to keep current.
    """
    t = text.lower().strip()

    # Explicit language keywords → map to code
    lang_keywords: dict[str, list[str]] = {
        "en": ["english", "angrezi", "angrejee", "in english", "english mein", "english me"],
        "hi": ["hindi", "hindi mein", "hindi me", "हिंदी", "in hindi"],
        "ta": ["tamil", "tamizh", "தமிழ்", "in tamil", "tamil mein"],
        "te": ["telugu", "తెలుగు", "in telugu", "telugu mein"],
        "mr": ["marathi", "मराठी", "in marathi", "marathi mein"],
        "gu": ["gujarati", "ગુજરાતી", "in gujarati", "gujarati mein"],
        "bn": ["bengali", "bangla", "বাংলা", "in bengali", "bangali"],
    }
    for code, keywords in lang_keywords.items():
        for kw in keywords:
            if kw in t:
                return code

    # Generic positive confirmations → keep current language (return None)
    positive = {
        "yes", "ok", "okay", "fine", "good", "correct", "right", "sure", "alright",
        "haan", "haan ji", "ha", "theek hai", "thik hai", "bilkul", "accha", "acha",
        "ji", "ji haan", "haa", "yeah", "yep", "yup",
        "हाँ", "ठीक है", "बिल्कुल", "हाँ जी", "हां",
        "sari", "seri", "aamam", "avunu", "ho", "hoy", "aaho",
        "নমস্কার", "ঠিক আছে",
    }
    words = set(t.split())
    if words & positive:
        return None  # Confirmed current language

    return None  # Default — keep current language

# ── Multilingual UI strings ────────────────────────────────────────────────────
STRINGS = {
    "en": {
        "thanks_name":     "Great, {}! Let's continue.",
        "cancel_bye":      "Thank you for your time. Have a great day!",
        "repeat_q":        "No problem, let me repeat: {}",
        "resume_journey":  "Let's continue from where we left off. {}",
        "resume_confused": "Let me repeat that — {}",
        "not_ready":       "No worries! Reach out whenever you're ready.",
        "no_info_yet":     "I don't have your {} on file yet.",
        "phone_reask":     "I need a 10-digit WhatsApp number — could you share it again?",
        "network_reask":   "Could you give me a rough number for your network?",
        "noted":           "Got it!",
    },
    "hi": {
        "thanks_name":     "बढ़िया, {} जी! आगे बढ़ते हैं।",
        "cancel_bye":      "अपना समय देने के लिए धन्यवाद। आपका दिन शुभ हो।",
        "repeat_q":        "कोई बात नहीं, मैं सवाल दोहराती हूं: {}",
        "resume_journey":  "चलिए अधूरी यात्रा को शुरू करते हैं। {}",
        "resume_confused": "अगर फिर से शुरू करें तो — {}",
        "not_ready":       "ठीक है, जब आप तैयार हों तब वापस आइए।",
        "no_info_yet":     "मुझे अभी तक आपका {} नहीं मिला है।",
        "phone_reask":     "10 अंकों का WhatsApp number चाहिए — एक बार फिर बताइए।",
        "network_reask":   "Roughly कितने contacts हैं? अनुमान भी बताइए।",
        "noted":           "नोट कर लिया!",
    },
    "ta": {
        "thanks_name":     "நன்றி, {}! தொடரலாம்.",
        "cancel_bye":      "உங்கள் நேரத்திற்கு நன்றி. நல்ல நாள்!",
        "repeat_q":        "பரவாயில்லை, மீண்டும் கேட்கிறேன்: {}",
        "resume_journey":  "நிறுத்திய இடத்திலிருந்து தொடரலாம். {}",
        "resume_confused": "மீண்டும் சொல்கிறேன் — {}",
        "not_ready":       "பரவாயில்லை! தயாரானபோது தொடர்பு கொள்ளுங்கள்.",
        "no_info_yet":     "இன்னும் உங்கள் {} கிடைக்கவில்லை.",
        "phone_reask":     "10 இலக்க WhatsApp number தேவை — மீண்டும் சொல்லுங்கள்.",
        "network_reask":   "தோராயமாக எத்தனை contacts இருக்கிறார்கள்?",
        "noted":           "குறித்துக் கொண்டேன்!",
    },
    "te": {
        "thanks_name":     "ధన్యవాదాలు, {}! కొనసాగిద్దాం.",
        "cancel_bye":      "మీ సమయానికి ధన్యవాదాలు. మంచి రోజు!",
        "repeat_q":        "పర్వాలేదు, మళ్ళీ అడుగుతాను: {}",
        "resume_journey":  "ఆగిన చోటు నుండి కొనసాగిద్దాం. {}",
        "resume_confused": "మళ్ళీ చెప్తాను — {}",
        "not_ready":       "పర్వాలేదు! సిద్ధంగా ఉన్నప్పుడు సంప్రదించండి.",
        "no_info_yet":     "ఇంకా మీ {} అందలేదు.",
        "phone_reask":     "10 అంకెల WhatsApp number కావాలి — మళ్ళీ చెప్పండి.",
        "network_reask":   "దాదాపు ఎంత మంది contacts ఉన్నారు?",
        "noted":           "గుర్తు పెట్టుకున్నాను!",
    },
    "mr": {
        "thanks_name":     "धन्यवाद, {}! पुढे जाऊया.",
        "cancel_bye":      "वेळ दिल्याबद्दल धन्यवाद. शुभ दिवस!",
        "repeat_q":        "ठीक आहे, पुन्हा विचारतो: {}",
        "resume_journey":  "थांबलो होतो तिथून पुढे जाऊया. {}",
        "resume_confused": "पुन्हा सांगतो — {}",
        "not_ready":       "ठीक आहे! तयार झाल्यावर संपर्क करा.",
        "no_info_yet":     "अजून तुमचे {} मिळाले नाही.",
        "phone_reask":     "10 अंकी WhatsApp number हवा — पुन्हा सांगा.",
        "network_reask":   "साधारण किती contacts आहेत?",
        "noted":           "नोंद घेतली!",
    },
    "gu": {
        "thanks_name":     "આભાર, {}! ચાલુ રાખીએ.",
        "cancel_bye":      "સમય આપવા માટે આભાર. સારો દિવસ!",
        "repeat_q":        "ઠીક છે, ફરી પૂછું છું: {}",
        "resume_journey":  "અટક્યા ત્યાંથી ચાલુ રાખીએ. {}",
        "resume_confused": "ફરીથી કહું — {}",
        "not_ready":       "ઠીક છે! તૈયાર થાઓ ત્યારે સંપર્ક કરો.",
        "no_info_yet":     "હજી તમારું {} મળ્યું નથી.",
        "phone_reask":     "10 આંકડાનો WhatsApp number જોઈએ — ફરી કહો.",
        "network_reask":   "આશરે કેટલા contacts છે?",
        "noted":           "નોંધ લીધી!",
    },
    "bn": {
        "thanks_name":     "ধন্যবাদ, {}! চলুন এগিয়ে যাই।",
        "cancel_bye":      "আপনার সময়ের জন্য ধন্যবাদ। শুভ দিন!",
        "repeat_q":        "ঠিক আছে, আবার বলছি: {}",
        "resume_journey":  "যেখানে ছিলাম সেখান থেকে শুরু করি। {}",
        "resume_confused": "আবার বলছি — {}",
        "not_ready":       "ঠিক আছে! প্রস্তুত হলে যোগাযোগ করুন।",
        "no_info_yet":     "এখনো আপনার {} পাইনি।",
        "phone_reask":     "10 সংখ্যার WhatsApp number দরকার — আবার বলুন।",
        "network_reask":   "মোটামুটি কতজন contacts আছেন?",
        "noted":           "নোট করা হয়েছে!",
    },
}

def S(key: str, lang: str, *args) -> str:
    """Fetch a UI string in the given language, format with args, fall back to Hindi."""
    text = STRINGS.get(lang, STRINGS["hi"]).get(key, STRINGS["hi"].get(key, key))
    return text.format(*args) if args else text

###########################
## Prepare Vector DB
#COLLECTION_OBJECT = prepareChromaDB()
###########################

###########################
# # SESSION STORE (one entry per websocket)
SESSIONS: dict[str, dict] = {}
SESSION_CONVERSATION: dict[str, list[dict]] = {}

# ── LEADS DATABASE (demo leads for RM dashboard) ─────────────────────────────
LEADS_DB: list[dict] = [
    {"id": "lead_001", "name": "Rajesh Kumar",  "phone": "9876543210", "language": "hi", "gender": "male",   "status": "pending", "city": "Delhi",      "profession": "Financial Advisor",        "summary": None},
    {"id": "lead_002", "name": "Priya Sharma",  "phone": "9123456780", "language": "hi", "gender": "female", "status": "pending", "city": "Mumbai",     "profession": "Mutual Fund Distributor",  "summary": None},
    {"id": "lead_003", "name": "Ankit Mehta",   "phone": "9988776655", "language": "en", "gender": "male",   "status": "pending", "city": "Bangalore",  "profession": "Insurance Agent",          "summary": None},
    {"id": "lead_004", "name": "Sunita Reddy",  "phone": "8877665544", "language": "hi", "gender": "female", "status": "pending", "city": "Hyderabad",  "profession": "Finance Influencer",       "summary": None},
    {"id": "lead_005", "name": "Vikram Patel",  "phone": "7766554433", "language": "hi", "gender": "male",   "status": "pending", "city": "Ahmedabad",  "profession": "Sub-Broker",               "summary": None},
    {"id": "lead_006", "name": "Meena Joshi",   "phone": "9955112233", "language": "hi", "gender": "female", "status": "pending", "city": "Jaipur",     "profession": "CA / Tax Consultant",      "summary": None},
    {"id": "lead_007", "name": "Arun Nair",     "phone": "9944556677", "language": "en", "gender": "male",   "status": "pending", "city": "Kochi",      "profession": "Stock Sub-Broker",         "summary": None},
]
SESSION_TO_LEAD: dict[str, str] = {}   # sessionId → leadId
###########################

###########################

def printLogs(content, exc: Exception | None = None):
    now = datetime.now()
    print(f"{now.strftime('%Y-%m-%d %H:%M:%S.')}{int(now.microsecond / 1000):03d} [{threading.current_thread().ident}] : {content}")
    if exc:
        print(exc)

def detect_language(text: str) -> str:
    """
    Infer language from Unicode script counts in the transcribed text.
    Returns an ISO 639-1 code: hi, en, ta, te, mr, gu, bn.
    Hinglish (Devanagari + Latin mix) is treated as 'hi'.
    Marathi shares Devanagari with Hindi; 'mr' is returned only when
    known Marathi-specific markers are present.
    """
    tamil     = sum(1 for c in text if '஀' <= c <= '௿')
    telugu    = sum(1 for c in text if 'ఀ' <= c <= '౿')
    gujarati  = sum(1 for c in text if '઀' <= c <= '૿')
    bengali   = sum(1 for c in text if 'ঀ' <= c <= '৿')
    devanagari = sum(1 for c in text if 'ऀ' <= c <= 'ॿ')
    latin     = sum(1 for c in text if c.isascii() and c.isalpha())

    # Regional scripts are unambiguous — pick the strongest one
    script_scores = {"ta": tamil, "te": telugu, "gu": gujarati, "bn": bengali}
    top_regional = max(script_scores, key=script_scores.get)
    if script_scores[top_regional] > 0:
        return top_regional

    # Devanagari: distinguish Marathi by common Marathi function words
    if devanagari > 0:
        marathi_markers = {"आहे", "नाही", "करतो", "करते", "सांगा", "द्या", "आहात", "तुम्ही"}
        words = set(text.split())
        if marathi_markers & words:
            return "mr"
        return "hi"  # Hindi or Hinglish (Latin mixed in is fine)

    if latin > 0:
        return "en"

    return "hi"  # Default fallback

def get_session(ws):
    return SESSIONS.setdefault(ws.data["sessionId"], {
        "unanswered_required": [q["id"] for q in QUESTIONNAIRE if q.get("required")],
        "unanswered_optional": [q["id"] for q in QUESTIONNAIRE if not q.get("required")],
        "answers": {},
        "intent_confirmed": False,
        "lang_confirmed": False,
        "last_asked_qid": None,
    })

def get_next_question(session):
    # Priority 1: Required questions
    if session["unanswered_required"]:
        next_qid = session["unanswered_required"][0]
    # Priority 2: Optional questions
    elif session["unanswered_optional"]:
        next_qid = session["unanswered_optional"][0]
    else:
        return None
    
    return next(q for q in QUESTIONNAIRE if q["id"] == next_qid)


def get_next_k_questions(session, count=3):
    next_ids = []

    printLogs(f"unanswered_required : {session.get('unanswered_required', [])}")
    printLogs(f"unanswered_optional : {session.get('unanswered_optional', [])}")

    next_ids.extend(session.get("unanswered_required", [])[:count])

    if len(next_ids) < count:
        remaining = count - len(next_ids)
        next_ids.extend(session.get("unanswered_optional", [])[:remaining])

    questions = [q for q in QUESTIONNAIRE if q["id"] in next_ids]

    return questions


#######################################################################################
#-------------------------NOISE REDUCTION---------------------------------------------
def reduce_noise_for_whisper(utterance_bytes):
    # Convert raw audio bytes (int16) → float32
    audio_np = np.frombuffer(utterance_bytes, dtype=np.int16).astype(np.float32)

    # Apply spectral noise reduction
    reduced = nr.reduce_noise(
        y=audio_np,
        sr=SAMPLE_RATE,
        prop_decrease=1.0  # full noise reduction (can be tuned)
    )

    # Convert back to int16 PCM
    clipped = np.clip(reduced, -32768, 32767)
    reduced_int16 = clipped.astype(np.int16)
    return reduced_int16.tobytes()
#######################################################################################

def extract_answer_using_gpt(QUESTIONNAIRE, user_input, sess, intent=""):
    """
    Enhanced version with better handling of combined answers
    """

    ## Base instruction template
    instruction = f"""
    You are extracting information from a lead during an outbound sales call for Rupeezy's AP (Authorized Person) Partner Program.

    RULES:
    1. FIRST check if the response contains answers to ANY of the qualification questions
    2. If the response contains a question-answer pair, extract THAT
    3. Otherwise, try to infer the answer from context
    4. If it contains both data and a question, return both
    """

    if "intent_confirmed" in sess and sess["intent_confirmed"]:
        instruction += """
            - The extracted value, if user's input is a potential answer to any of the questions, in the provided format.
            - If the user's input contains an answer AND is also a general question about the AP program, return BOTH. First the answer in '||' format, then on a NEW LINE, the keyword 'general_enquiry'.
        """

    instruction += """
    1. If no answer fits or contains any of the below conditions, return one keyword such as:
        - "intent_confirmed" (if the user shows interest in joining as AP partner)
        - "general_enquiry" (if user is asking questions about the Rupeezy AP program, earnings, process, etc.)
        - "invalid" if what user said has even a single invalid, unknown or nonsense word
        - "not_relevant" (if what user said is not related to the Rupeezy AP partner program, and is neither an answer to the questions below)
        - "cancel" (if user wants to stop or is not interested)
        - "update_request:field_id" if user wants to UPDATE a previously given answer
        - "confused" (if user sounds confused or wants to repeat the prompt — says things like "repeat", "samjh nahi aaya")
        - "continue_journey" (if user explicitly asks to repeat the question or continue)
    2. SPECIAL CASES for data extraction:
        - Names: Convert Hindi names to English. Correct spellings for Hindi names/surnames if possible. If not found, return not_found.
        - network_size: Convert words to numbers (e.g., "pachaas"→50, "ek sau"→100). Return the closest integer. If not found, return not_found.
        - Phone: Must be exactly 10 digits (strip country code if present). If invalid, return not_found.
        - current_broker: Normalize broker names (Zerodha, Groww, Upstox, Angel One, 5paisa, etc.). If none/no, return "none".

        Questions are as follows, in the format "QuestionId|Question|Type|Required_or_not":
        """
    out = []

    ## Questionnaire addition to the code
    for q in QUESTIONNAIRE:
        out.append(f"||{q['id']}|{q['prompt']}|{q['type']}|{q['required']}||")
    result = "\n".join(out)

    if "intent_confirmed" in sess and sess["intent_confirmed"]:
        instruction += result + "\n"

    ## Addition of unanswered questions

    ## Prompt for assessing user input
    
    prompt = f"""
    USER INPUT: {user_input}
    """
    if "intent_confirmed" in sess and sess["intent_confirmed"]:
        if "last_asked_text" in sess:
            prompt += f"Find out if user input is an answer to the question : {sess['last_asked_text']} . If it doesn't match, find the answer as per other instructions"

    if "intent_confirmed" in sess and sess["intent_confirmed"]:
        prompt += """
        EXTRACT ALL ANSWERS AND PRESENT IN FOLLOWING FORMAT, ALL LOWERCASE :-
        ||QuestionId|Answer|Type||
        """
        prompt += f"""
        ANSWER (ONLY the extracted value or special keyword):"""
    else:
        prompt += "Answer the question according to questions"
    if not sess["intent_confirmed"]:
        prompt += """
        The user has NOT confirmed buying intent yet.

        Do BOTH of the following:
        1. If the input contains any details listed below, extract it in this format (all lowercase):
              ||questionId|answer|type||                          Example: ||age|25|number||
           SPECIAL CASES:
        - Names: Convert hindi names to English. Correct spellings as per hindi names and surnames if possible. If not found, return not_found.
        - Age: Convert words to numbers (e.g., "इक्कीस"→21). If the person specifies his/her age, try and figure the number which is closest to what user said in hindi. If still not found, return not_found.
        - Pincode: Must be digits
        - Gender: male/female only. If not matching, return not_found.
        - Diseases: Map to standard categories carefully even for statements like " मुझे शुगर है", "मुझे कोलेस्ट्रॉल है","मुझे बुखार है","मेरा सिर दर्द हो रहा है" and put them in proper categories of 
                        options are as follows → No existing disease | Diabetes | BP/Hypertension | Heart Disease| Asthma | Thyroid Disorder | Any other Disease - User response can be yes or no, or other language equivalents. Choose your answer from the options accordingly.
        2. If no answer fits, return one keyword such as:
        - "intent_confirmed" (if user shows interest in joining as AP partner)
        - "general_enquiry" (if user is asking questions about the Rupeezy AP program)
        - "invalid" if what user said has an invalid, unknown or nonsense word
        - "not_relevant" (if what user said is not related to the Rupeezy AP partner program, and is not an answer)
        - "cancel" (if user wants to stop or is not interested)
        - "update_request:field_id" if user wants to UPDATE a previously given answer
        - "confused" (if user sounds confused or wants to repeat the prompt — says things like "repeat", "samjh nahi aaya")
        """

    printLogs("-----------------------------------------------------------------")
    printLogs("-----------------------------------------------------Instructions")
    printLogs(instruction)
    printLogs("-----------------------------------------------------------------")
    printLogs("-----------------------------------------------------------Prompt")
    printLogs(prompt)
    printLogs("-----------------------------------------------------------------")
    
    try:
        response = groq_client.chat.completions.create(
            model=extractModelId,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        broker_normalise = {
                "zerodha": "Zerodha", "groww": "Groww", "upstox": "Upstox",
                "angel one": "Angel One", "angel": "Angel One",
                "5paisa": "5paisa", "motilal": "Motilal Oswal",
                "hdfc sec": "HDFC Securities", "icici direct": "ICICI Direct",
                "none": "none", "no": "none", "nahi": "none", "नहीं": "none",
            }
        
        extractedString = response.choices[0].message.content.strip()
        printLogs(f"T2T GPT returned : {extractedString}")

        # Handle cases where the *entire* response is a keyword
        keyword_only_responses = [
            "intent_confirmed", "cancel", "confused", "general enquiry", "not_insurance",
            "not_relevant", "invalid", "general_enquiry", "not_an_answer", "not found", "not_found"
        ]
        if extractedString in keyword_only_responses:
            printLogs(f"Returning default value : {extractedString}")
            return {"default": extractedString.replace(" ", "_")}

        extractedData = {}

        matches = re.findall(r"\|\|([^|]+)\|([^|]+)\|([^|]+)\|\|", extractedString)

        default_keywords = [
            "intent_confirmed", "general enquiry", "general_enquiry", "invalid",
            "not_insurance", "not_relevant", "cancel", "confused", "continue_journey"
        ]
        

        string_to_search_keywords = extractedString
        if matches:
            for match in matches:
                full_match_string = f"||{match[0]}|{match[1]}|{match[2]}||"
                string_to_search_keywords = string_to_search_keywords.replace(full_match_string, "")
        
        for keyword in default_keywords:
            if keyword in string_to_search_keywords.lower():
                # Normalize the keyword (e.g., "general enquiry" -> "general_enquiry")
                extractedData["default"] = keyword.replace(" ", "_")
                break # Stop after finding the first keyword
        
        # --- END OF FIX ---

        if not matches and "default" not in extractedData:
            print(f"No valid triples found in GPT response: {extractedString}")

        for qid, raw_value, dtype in matches:
            printLogs(f"extracted values : {qid}, {raw_value}, {dtype}")
            
            # --- core casting ---------------------------------------------------
            if dtype == "number":
                if raw_value != "not_found":
                    try:
                        raw_value = int(raw_value) if raw_value.isdigit() else float(raw_value)
                    except (ValueError, TypeError):
                        raw_value = "not_found"

            # --- special-case handling -----------------------------------------
            elif qid == "current_broker":
                raw_value = broker_normalise.get(raw_value.lower(), raw_value)

            elif qid == "phone":
                digits = re.sub(r"\D", "", str(raw_value))
                if len(digits) == 12 and digits.startswith("91"):
                    digits = digits[2:]
                raw_value = digits if len(digits) == 10 else "not_found"

            # -------------------------------------------------------------------
            if raw_value != "not_found":
                extractedData[qid] = raw_value

        printLogs(f"Prepared data : {extractedData}")

        return extractedData

    except Exception as e:
        printLogs(f"GPT extraction error: {e}")
        traceback.print_exc()
        return {"default": "not_found"}


FIELD_NAMES_HINDI = {
    "name":           "नाम",
    "profession":     "profession",
    "network_size":   "network size",
    "city":           "शहर",
    "current_broker": "current broker",
    "phone":          "WhatsApp number",
}

async def recall_and_confirm(field: str, value: str, q_queue: queue.Queue):
    """Generate recall response for stored Rupeezy AP answers"""
    recalls = {
        "name":           f"आपका नाम मैंने {value} जी के रूप में नोट किया था",
        "profession":     f"आपने बताया था कि आप {value} हैं",
        "network_size":   f"आपने बताया था कि आपके contacts में roughly {value} लोग हैं",
        "city":           f"आपका शहर {value} दर्ज है",
        "current_broker": f"आपने बताया था: current broker — {value}",
        "phone":          f"आपका WhatsApp number {value} रिकॉर्ड किया गया है",
    }
    q_queue.put(recalls.get(field, "मैंने यह जानकारी रिकॉर्ड की है"))


# ---------------------------------------------------------------------------
# Objection rebuttals
# ---------------------------------------------------------------------------
OBJECTION_REBUTTALS = {
    "obj_already_broker": (
        "Bilkul samajh sakte hain! But Rupeezy ke saath aap apna existing network leverage kar sakte hain — "
        "zero joining fee, 100% brokerage share, aur daily payouts. "
        "Dono platforms simultaneously run kar sakte hain — koi conflict nahi."
    ),
    "obj_no_contacts": (
        "Koi baat nahi! Start small — even 5-10 contacts jo invest karna chahte hain kaafi hain. "
        "Rupeezy ka dedicated RM aapko har step pe guide karega. "
        "Sign-up ke baad training bhi milti hai contacts expand karne ke liye."
    ),
    "obj_support": (
        "Aapki concern valid hai. Isliye Rupeezy ne dedicated RM support diya hai — "
        "ek real person jo aapke saath hai, 24x7 helpline nahi. "
        "Onboarding se lekar first payout tak full hand-holding milegi."
    ),
    "obj_trust": (
        "Rupeezy ek SEBI-registered broker hai — fully regulated. "
        "Aapka paisa aur aapke clients ka paisa completely safe hai. "
        "Thousands of partners already earning daily payouts — yeh verified program hai."
    ),
    "obj_think_later": (
        "Sure, time lo! But ek baat batao — joining fee toh hai hi nahi, toh risk bhi zero hai. "
        "Aaj sirf registration kar lo, start kabhi bhi karo jab ready ho."
    ),
}


def compute_lead_score(sess: dict) -> str:
    score = 0
    answers = sess.get("answers", {})

    # Network size
    try:
        n = int(answers.get("network_size", 0))
        if n >= 50:
            score += 3
        elif n >= 20:
            score += 2
        elif n >= 5:
            score += 1
    except (ValueError, TypeError):
        pass

    if sess.get("intent_confirmed"):
        score += 3
    if answers.get("phone"):
        score += 2

    objections_raised   = set(sess.get("objections_raised", []))
    objections_resolved = set(sess.get("objections_resolved", []))
    unresolved = objections_raised - objections_resolved
    score -= len(unresolved)
    if "obj_think_later" in objections_raised:
        score -= 1

    if score >= 6:
        return "Hot"
    if score >= 3:
        return "Warm"
    return "Cold"


def generate_post_call_summary(sess: dict, session_id: str) -> dict:
    answers = sess.get("answers", {})
    lead_score = compute_lead_score(sess)

    if lead_score == "Hot":
        recommended_action = "Immediate RM handoff — high intent + large network."
    elif lead_score == "Warm":
        recommended_action = "Send WhatsApp follow-up with sign-up link within 1 hour."
    else:
        recommended_action = "Add to nurture pipeline — re-contact after 7 days."

    return {
        "session_id":          session_id,
        "lead_name":           answers.get("name", ""),
        "city":                answers.get("city", ""),
        "network_size":        answers.get("network_size", ""),
        "current_broker":      answers.get("current_broker", ""),
        "phone":               answers.get("phone", ""),
        "profession":          answers.get("profession", ""),
        "lang":                sess.get("lang", "hi"),
        "lead_score":          lead_score,
        "objections_raised":   sess.get("objections_raised", []),
        "objections_resolved": sess.get("objections_resolved", []),
        "intent_confirmed":    sess.get("intent_confirmed", False),
        "recommended_action":  recommended_action,
        "topics_covered":      list(answers.keys()),
        "transcript":          [],
    }


async def handle_intent(user_text: str, ws, sess: dict, q_queue: queue.Queue) -> bool:
    intent = detect_intent_with_chroma(user_text)
    printLogs(f"Intent detected: {intent}")

    if intent.startswith("recall_"):
        field = intent.replace("recall_", "")
        if field in sess.get("answers", {}):
            await recall_and_confirm(field, sess["answers"][field], q_queue)
        else:
            q_queue.put(S("no_info_yet", sess.get("lang", "hi")))
        return True

    if intent == "interested":
        existing_answers = sess.get("answers", {})
        existing_unanswered_required = sess.get("unanswered_required", [])
        existing_unanswered_optional = sess.get("unanswered_optional", [])
        sess.update({
            "intent_confirmed": True,
            "cancel": False,
            "unanswered_required": existing_unanswered_required or [q["id"] for q in QUESTIONNAIRE if q.get("required")],
            "unanswered_optional": existing_unanswered_optional or [q["id"] for q in QUESTIONNAIRE if not q.get("required")],
            "answers": existing_answers,
            "objections_raised": sess.get("objections_raised", []),
            "objections_resolved": sess.get("objections_resolved", []),
        })
        await ws.send("PLAY_CONGRATS")
        return True

    if intent.startswith("obj_"):
        lang = sess.get("lang", "hi")
        if intent not in sess.get("objections_raised", []):
            sess.setdefault("objections_raised", []).append(intent)
        rebuttal = OBJECTION_REBUTTALS.get(intent)
        if rebuttal:
            q_queue.put(rebuttal)
        # Treat any objection as implicit engagement — enable questionnaire after rebuttal
        sess["intent_confirmed"] = True
        existing_unanswered_required = sess.get("unanswered_required", [])
        existing_unanswered_optional = sess.get("unanswered_optional", [])
        sess.setdefault("unanswered_required", existing_unanswered_required or [q["id"] for q in QUESTIONNAIRE if q.get("required")])
        sess.setdefault("unanswered_optional", existing_unanswered_optional or [q["id"] for q in QUESTIONNAIRE if not q.get("required")])
        return True

    if intent == "cancel":
        lang = sess.get("lang", "hi")
        session_id = ws.data.get("sessionId")
        if session_id in SESSIONS:
            del SESSIONS[session_id]
        q_queue.put(S("cancel_bye", lang))
        await ws.send("FILLER:flow_complete")
        await ws.send("FLOW_CANCELLED")
        return True

    if user_text.strip().lower() in ["हाँ", "haan", "yes", "haan ji", "ji haan", "bilkul", "ok", "okay", "ji", "ha", "haa", "acha", "accha", "theek hai", "thik hai"]:
        sess["flow_active"] = True
        sess["intent_confirmed"] = True
        sess["retry_count"] = 0
        lang = sess.get("lang", "hi")
        current_q_idx = sess.get("q_idx", 0)
        q = QUESTIONNAIRE[current_q_idx] if current_q_idx < len(QUESTIONNAIRE) else QUESTIONNAIRE[0]
        prompt = qprompt(q, lang)
        q_queue.put(prompt)
        sess["last_asked_text"] = prompt
        return True

    ## TODO: Why is this needed ?
    elif user_text.strip().lower() in ["नहीं", "nahin", "no"]:
        sess["retry_count"] = 0
        await ws.send("ठीक है, जब आप तैयार हों तब वापस आइए।")
        return True

    return False

# -------------------------------------------------------------------------------------------------------------
# --------------------------
# Model initialisation
# --------------------------
printLogs("[BOOT] Using Groq API for STT (whisper-large-v3).")

REF_AUDIO_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'intro.mp3')
REF_TEXT       = os.getenv("REF_TEXT", "")  # transcript of intro.mp3; empty = auto-detect
TTS_DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"
TTS_DTYPE      = torch.float16 if TTS_DEVICE.startswith("cuda") else torch.float32
printLogs(f"[BOOT] Loading OmniVoice on {TTS_DEVICE} ({TTS_DTYPE})...")
omnivoice_model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map=TTS_DEVICE,
    dtype=TTS_DTYPE,
)
printLogs("[BOOT] OmniVoice ready.")

# --------------------------
# TTS worker thread
# --------------------------

def tts_consumer(q: queue.Queue, main_loop, ws):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        printLogs("waiting for element in queue")
        text_chunk = q.get()
        if text_chunk is None:
            break
        stream_tts(text_chunk, ws, main_loop)
    printLogs("[TTS] consumer exit")

# --------------------------
# TTS synthesis
# --------------------------

def stream_tts(text: str, ws, main_loop):
    printLogs(f"[TTS] {text}")

    if not ws.connOpen:
        printLogs("WebSocket is not open. Skipping TTS.")
        return

    try:
        audio_list = omnivoice_model.generate(
            text      = text,
            ref_audio = REF_AUDIO_PATH,
            ref_text  = REF_TEXT or None,
            instruct  = None,
        )
        if not audio_list:
            printLogs("[TTS] No audio returned by model.generate(), skipping.")
            return
        audio = audio_list[0]
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        if not isinstance(audio, np.ndarray):
            audio = np.asarray(audio)
        buf = io.BytesIO()
        sf.write(buf, audio, 16000, format='WAV')
        audio_bytes = buf.getvalue()
        fut = asyncio.run_coroutine_threadsafe(ws.send(audio_bytes), main_loop)
        fut.result()
    except websockets.exceptions.ConnectionClosed:
        printLogs("Websocket closed. Skipping.")
    except Exception as e:
        printLogs(f"Error in TTS or WebSocket send: {e}")


# --------------------------
# Main utterance processor
# --------------------------

# ----------------------------------------------------------------------
# Constants you can pull to your config file if you like
SAMPLE_RATE = 16_000              # Whisper loves 16 kHz mono

playContinueJourney = False

# ----------------------------------------------------------------------


async def process_utterance(utter_bytes: bytes, ws, q_queue):
    """
    • Streams TTS replies through `tts_consumer` in a background thread.
    • Runs Whisper‑large‑v3 once per utterance with robust decode settings.
    • Drives the buy‑flow questionnaire or falls back to the normal LLM path.
    """
    global playContinueJourney
    printLogs("[START] process_utterance")

    # with wave.open(f"test_{time.time()}.wav", 'wb') as wf:
    #     wf.setnchannels(1)
    #     wf.setsampwidth(2)
    #     wf.setframerate(16000)
    #     wf.writeframes(utter_bytes)

    # Noise reduction + Groq Whisper STT
    start_time = time.time()
    pcm = np.frombuffer(utter_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    pcm = nr.reduce_noise(y=pcm, sr=16_000)
    pcm_int16 = (pcm * 32768.0).clip(-32768, 32767).astype(np.int16)
    end_time = time.time()
    printLogs(f"Noise-reduction time: {(end_time-start_time):.2f}s")

    try:
        whisper_start = time.time()
        wav_buf = io.BytesIO()
        # with wave.open(wav_buf, 'wb') as wf:
        #     wf.setnchannels(1)
        #     wf.setsampwidth(2)
        #     wf.setframerate(16_000)
        #     wf.writeframes(pcm_int16.tobytes())
        wav_buf.seek(0)
        transcription = groq_stt_client.audio.transcriptions.create(
            file=("audio.wav", wav_buf.read(), "audio/wav"),
            model="whisper-large-v3",
        )
        whisper_end = time.time()
        printLogs(f"WHISPER (Groq API) time: {(whisper_end-whisper_start):.2f}s")
        user_text = transcription.text.strip()
    except Exception as e:
        printLogs(f"[ASR-ERR] {e}")
        traceback.print_exc()
        q_queue.put("माफ़ कीजिए, बोल कर दोबारा बताइए।")
        return

    sess = get_session(ws)

    # Detect and lock the session language on first turn (if not pre-set from lead data)
    if "lang" not in sess:
        sess["lang"] = detect_language(user_text)
    lang = sess["lang"]
    printLogs(f"[LANG] Session language: {lang}")

    # ── Language preference gate (first utterance only) ───────────────────────
    if not sess.get("lang_confirmed", False):
        preferred = detect_language_preference(user_text)
        if preferred is not None and preferred != lang:
            # Lead wants a different language — switch and ack
            sess["lang"] = preferred
            lang = preferred
            q_queue.put(LANG_SWITCH_ACK.get(lang, LANG_SWITCH_ACK["hi"]))
            printLogs(f"[LANG] Switched to {lang} per lead preference")
        else:
            q_queue.put(LANG_CONFIRM_ACK.get(lang, LANG_CONFIRM_ACK["hi"]))

        sess["lang_confirmed"] = True

        # Also check for cancel at this stage
        quick_intent = detect_intent_with_chroma(user_text)
        if quick_intent == "cancel":
            q_queue.put(S("cancel_bye", lang))
            await ws.send("FLOW_CANCELLED")
            return

        # Proceed directly to first qualifying question (intro already asked for 2 min)
        sess["intent_confirmed"] = True
        next_q = get_next_question(sess)
        if next_q:
            prompt = qprompt(next_q, lang)
            sess["last_asked_qid"] = next_q["id"]
            sess["last_asked_text"] = prompt
            q_queue.put(prompt)
        return
    # ─────────────────────────────────────────────────────────────────────────

    ##### ---- DUMMY CODE to bypass whisper. TODO: REMOVE THIS SECTION ----
    ##user_text = "मेरा नाम आदित्य है, मैं छत्तीस साल का हूं और मैं 110085 में रहता हूं।"
    ##sess["intent_confirmed"] = True
    ##### -----------------------------------------------
    printLogs(f"[PROCESS_UTTERANCE] Received {len(utter_bytes)} bytes of audio")
    printLogs(f"[USER] Whisper detected: {user_text}")
    await ws.send(f"TRANSCRIPT:USER:{user_text}")

      # Intent handling
    intent = detect_intent_with_chroma(user_text)
    
    ##############################################
    # Handle recall intents first
    if intent.startswith("recall_"):
        field = intent.replace("recall_", "")
        if field in sess.get("answers", {}):
            await recall_and_confirm(field, sess["answers"][field], q_queue)
        else:
            q_queue.put(f"मुझे अभी तक आपका {FIELD_NAMES_HINDI.get(field, 'यह')} जानकारी नहीं मिली है")
        return
    ##############################################

    # Intent handling
    if not sess["intent_confirmed"]:
        intent_start = time.perf_counter()
        isIntent = await handle_intent(user_text, ws, sess, q_queue)
        printLogs(f"[LATENCY] handle_intent: {time.perf_counter() - intent_start:.2f}s")
        if isIntent:
            intent = detect_intent_with_chroma(user_text)
            if intent=="cancel":
                session_id = ws.data.get("sessionId")
                if session_id in SESSIONS:
                    del SESSIONS[session_id]
                sess.clear()
                # Clear queue before cancellation message
                while not q_queue.empty():
                    q_queue.get()
                q_queue.put("अपना समय देने के लिए धन्यवाद। आपका दिन शुभ हो।")
                await ws.send("FLOW_CANCELLED")

                # Clear session data
                session_id = ws.data.get("sessionId")
                if session_id in SESSIONS:
                    del SESSIONS[session_id]
                sess.clear()

                # Clear queue
                try:
                    with q_queue.mutex:
                        q_queue.queue.clear()
                        q_queue.all_tasks_done.notify_all()
                        q_queue.unfinished_tasks = 0
                except AttributeError:
                    while not q_queue.empty():
                        try:
                            q_queue.get_nowait()
                        except queue.Empty:
                            break

                # Close WebSocket
                #await ws.close()
                printLogs("WebSocket connection closed after cancel intent.")
                return
            # elif intent=="interested":
            else:
                printLogs("[INFO] Intent detected after free text.")
                #next_q = get_next_question(sess)
                next_k_q = get_next_k_questions(sess)
                printLogs(f"next_k_q : {next_k_q}")
                next_ids = [q["id"] for q in next_k_q]
                shorterNames = [q["short"] for q in QUESTIONNAIRE if q["id"] in next_ids]
                lastQuestion = next_k_q[-1]
                sess["last_asked_qid"] = lastQuestion["id"]
                printLogs(f"shorterNames : {shorterNames}")

                shortNamesString = ", ".join(shorterNames)
                lang = sess.get("lang", "hi")
                batch_q = {
                    "en": f"Can I get your {shortNamesString}?",
                    "hi": f"आपका {shortNamesString} क्या हैं?",
                    "ta": f"உங்கள் {shortNamesString} சொல்லுங்கள்?",
                    "te": f"మీ {shortNamesString} చెప్పండి?",
                    "mr": f"तुमचे {shortNamesString} सांगा?",
                    "gu": f"તમારું {shortNamesString} શું છે?",
                    "bn": f"আপনার {shortNamesString} বলুন?",
                }.get(lang, f"आपका {shortNamesString} क्या हैं?")
                sess["last_asked_text"] = batch_q
                q_queue.put(sess["last_asked_text"])
                return
            # elif intent=="confused" or intent=="need_context":
            #     printLogs("[INFO]Confusion or need context")
            #     q_queue.put("क्या आप कृपया फिर से बता सकते हैं?")
        else:
            # print("tanishka")
            extracted = extract_answer_using_gpt(QUESTIONNAIRE, user_text, sess, "")
            if extracted and "default" in extracted:
                for qid, value in extracted.items():
                    # Save to answers dict
                    sess["answers"][qid] = value
                    sess["unanswered_required"] = [q for q in sess["unanswered_required"] if q != qid]
                    sess["unanswered_optional"] = [q for q in sess["unanswered_optional"] if q != qid]

                    # Basic validation for Rupeezy AP fields
                    if qid == "network_size":
                        try:
                            n = int(value)
                            if n >= 1:
                                sess["answers"]["network_size"] = n
                            else:
                                sess["answers"].pop("network_size", None)
                                if "network_size" not in sess["unanswered_required"]:
                                    sess["unanswered_required"].append("network_size")
                        except (ValueError, TypeError):
                            sess["answers"].pop("network_size", None)
                            if "network_size" not in sess["unanswered_required"]:
                                sess["unanswered_required"].append("network_size")

                    elif qid == "phone":
                        digits = re.sub(r"\D", "", str(value))
                        if len(digits) == 12 and digits.startswith("91"):
                            digits = digits[2:]
                        if len(digits) != 10:
                            sess["answers"].pop("phone", None)
                            if "phone" not in sess["unanswered_required"]:
                                sess["unanswered_required"].append("phone")

                sess["unanswered_required"] = [
                q["id"] for q in QUESTIONNAIRE
                if q["required"] and q["id"] not in sess["answers"]
                        ]
                sess["unanswered_optional"] = [
                q["id"] for q in QUESTIONNAIRE
                if not q["required"] and q["id"] not in sess["answers"]
                        ]
                printLogs(f"[DEBUG] Final answers: {sess['answers']}")
                printLogs(f"[DEBUG] unanswered_required: {sess['unanswered_required']}")
                printLogs(f"[DEBUG] unanswered_optional: {sess['unanswered_optional']}")
            if "default" in extracted and extracted["default"] == "invalid":
                await ws.send("FILLER:ask_again")
                return


    # Questionnaire Flow
    if sess["intent_confirmed"]:
        intent = detect_intent_with_chroma(user_text)
        # Check if it's a repeated "interested" intent and user hasn't answered the last question
        if intent == "interested":
            printLogs("[INFO] Repeated 'interested' intent detected.")

            current_qid = sess.get("last_asked_qid")
            current_q = next((q for q in QUESTIONNAIRE if q["id"] == current_qid), None)

            if current_q:
                printLogs("[INFO] Re-asking last question due to repeated 'interested' intent.")
                lang = sess.get("lang", "hi")
                prompt = qprompt(current_q, lang)
                sess["last_asked_text"] = S("resume_journey", lang, prompt)
                q_queue.put(sess["last_asked_text"])
                return
            else:
                # No question asked yet — ask the first unanswered question
                next_q = get_next_question(sess)
                if next_q:
                    lang = sess.get("lang", "hi")
                    prompt = qprompt(next_q, lang)
                    sess["last_asked_qid"] = next_q["id"]
                    sess["last_asked_text"] = prompt
                    q_queue.put(prompt)
                    return

        if intent == "cancel":
            session_id = ws.data.get("sessionId")
            if session_id in SESSIONS:
                del SESSIONS[session_id]
            sess.clear()
            # Clear queue before cancellation message
            while not q_queue.empty():
                q_queue.get()
            q_queue.put(S("cancel_bye", sess.get("lang", "hi")))
            await ws.send("FLOW_CANCELLED")

            # Clear session data
            session_id = ws.data.get("sessionId")
            if session_id in SESSIONS:
                del SESSIONS[session_id]
            sess.clear()

            # Clear queue
            try:
                with q_queue.mutex:
                    q_queue.queue.clear()
                    q_queue.all_tasks_done.notify_all()
                    q_queue.unfinished_tasks = 0
            except AttributeError:
                while not q_queue.empty():
                    try:
                        q_queue.get_nowait()
                    except queue.Empty:
                        break

            # Close WebSocket
            #await ws.close()
            printLogs("WebSocket connection closed after cancel intent.")
            return

        if intent.startswith("obj_"):
            lang = sess.get("lang", "hi")
            if intent not in sess.get("objections_raised", []):
                sess.setdefault("objections_raised", []).append(intent)
            rebuttal = OBJECTION_REBUTTALS.get(intent)
            if rebuttal:
                q_queue.put(rebuttal)
            # Re-ask the current question after rebuttal
            current_qid_tmp = sess.get("last_asked_qid")
            current_q_tmp = next((q for q in QUESTIONNAIRE if q["id"] == current_qid_tmp), None)
            if current_q_tmp:
                q_queue.put(qprompt(current_q_tmp, lang))
            return

        current_qid = sess.get("last_asked_qid")
        current_q = next((q for q in QUESTIONNAIRE if q["id"] == current_qid), None)

        ## Copied once. ensure to make changes there if changes made here.
        if intent == "confused" or intent == "invalid" or intent in ("not_insurance", "not_relevant"):
            printLogs("[INFO] Confused detected.")
            playContinueJourney = True

            if sess.get("last_answer_was_general_enquiry"):
                printLogs("[INFO] Confusion after general_enquiry. Re-invoking GPT - 1")
                #sess["last_answer_was_general_enquiry"] = False  # Reset #TODO: IF THIS NEEDS TO BE DONE

                printLogs(f"[INFO] last user text : {sess.get('last_user_text', '')}")

                # Trigger GPT response again
                generate_text_and_audio_gpt(sess["last_user_text"], ws, q_queue)  ## Aditya come here
                await ws.send(f"FILLER:back_to_buy")
                lang = sess.get("lang", "hi")
                prompt = qprompt(current_q, lang)
                q_queue.put(S("resume_journey", lang, prompt))
                sess["last_asked_text"] = prompt
            else:
                lang = sess.get("lang", "hi")
                prompt = qprompt(current_q, lang)
                q_queue.put(S("repeat_q", lang, prompt))
                sess["last_asked_text"] = prompt

            return

        # if intent == "continue_journey":
        #     printLogs("[INFO] User asked to continue journey.")

        #     sess["intent_confirmed"] = True
        #     if "cancel" in sess and sess["cancel"] == True:
        #         sess.clear()
        #         sess["cancel"] = False
        #     return

            ##sess.update({
            ##    "intent_confirmed": True,
            ##    "cancel": False,
            ##    "unanswered_required": [q["id"] for q in QUESTIONNAIRE if q.get("required")],
            ##    "unanswered_optional": [q["id"] for q in QUESTIONNAIRE if not q.get("required")],
            ##    "answers": {}
            ##})


        # 1. First scan for ALL possible answers in the input
        extracted = extract_answer_using_gpt(QUESTIONNAIRE, user_text, sess, intent)

        printLogs(f"extracted1 : {extracted}")

        #for q in QUESTIONNAIRE:
        #    if q["id"] not in sess["answers"]:
        #        answer = extract_answer_using_gpt(q, user_text)
        #        if answer not in ("not_found", "not_an_answer", "general_enquiry", "confused"):
        #            extracted[q["id"]] = answer
        #            printLogs(f"[INFO] Extracted {q['id']}: {answer}")

        # Process any found answers
        if extracted and "default" not in extracted:
            sess["last_user_text"] = ""
            sess["last_answer_was_general_enquiry"] = False
            for qid, value in extracted.items():
                sess["answers"][qid] = value
                if qid in sess["unanswered_required"]:
                    sess["unanswered_required"].remove(qid)
                if qid in sess["unanswered_optional"]:
                    sess["unanswered_optional"].remove(qid)

                # Validate and acknowledge each field
                lang = sess.get("lang", "hi")

                if qid == "name":
                    q_queue.put(S("thanks_name", lang, value))

                elif qid == "profession":
                    if value == "not_found":
                        sess["answers"].pop("profession", None)
                        if "profession" not in sess["unanswered_required"]:
                            sess["unanswered_required"].append("profession")
                    elif len(extracted) == 1:
                        await ws.send("FILLER:noted")

                elif qid == "network_size":
                    try:
                        n = int(value)
                        if n < 1:
                            raise ValueError
                        sess["answers"]["network_size"] = n
                        if "network_size" in sess["unanswered_required"]:
                            sess["unanswered_required"].remove("network_size")
                        if len(extracted) == 1:
                            await ws.send("FILLER:noted")
                    except (ValueError, TypeError):
                        sess["answers"].pop("network_size", None)
                        if "network_size" not in sess["unanswered_required"]:
                            sess["unanswered_required"].append("network_size")
                        q_queue.put(S("network_reask", lang))
                        sess["last_asked_qid"] = "network_size"
                        return

                elif qid == "city":
                    if value == "not_found":
                        sess["answers"].pop("city", None)
                        if "city" not in sess["unanswered_required"]:
                            sess["unanswered_required"].append("city")
                    elif len(extracted) == 1:
                        await ws.send("FILLER:noted")

                elif qid == "current_broker":
                    sess["answers"]["current_broker"] = value
                    if "current_broker" in sess["unanswered_optional"]:
                        sess["unanswered_optional"].remove("current_broker")
                    if value.lower() not in ["none", "no", "nahi", "not_found"]:
                        sess.setdefault("objections_raised", [])
                        if "obj_already_broker" not in sess["objections_raised"]:
                            sess["objections_raised"].append("obj_already_broker")

                elif qid == "phone":
                    digits = re.sub(r"\D", "", str(value))
                    if len(digits) == 12 and digits.startswith("91"):
                        digits = digits[2:]
                    if len(digits) == 10:
                        sess["answers"]["phone"] = digits
                        if "phone" in sess["unanswered_required"]:
                            sess["unanswered_required"].remove("phone")
                        if len(extracted) == 1:
                            await ws.send("FILLER:noted")
                    else:
                        sess["answers"].pop("phone", None)
                        if "phone" not in sess["unanswered_required"]:
                            sess["unanswered_required"].append("phone")
                        q_queue.put(S("phone_reask", lang))
                        sess["last_asked_qid"] = "phone"
                        return

                else:
                    if len(extracted) == 1:
                        await ws.send("FILLER:noted")

            # Move to next question
            next_q = get_next_question(sess)
            if next_q:
                sess["last_asked_qid"] = next_q["id"]
                printLogs(f"[INFO] Next Question: {next_q['id']}")
                lang = sess.get("lang", "hi")
                prompt = qprompt(next_q, lang)
                q_queue.put(prompt)
                sess["last_asked_text"] = prompt
                return
            else:
                # All questions answered — score lead and route
                try:
                    session_id = ws.data.get("sessionId", "unknown")
                    summary = generate_post_call_summary(sess, session_id)
                    summary["transcript"] = SESSION_CONVERSATION.get(session_id, [])
                    lead_score = summary["lead_score"]
                    lang = sess.get("lang", "hi")
                    phone = sess["answers"].get("phone", "")
                    name  = sess["answers"].get("name", "")

                    closing = {
                        "Hot": {
                            "hi": f"{name} ji, bahut badhiya! Aapka profile ek dedicated RM ke paas bhej raha hoon — woh jald hi contact karenge.",
                            "en": f"Excellent {name}! Connecting you with a dedicated RM who will reach out shortly.",
                        },
                        "Warm": {
                            "hi": f"{name} ji, thanks! Aapke WhatsApp par sign-up link bhej raha hoon — apni speed se dekh lena.",
                            "en": f"Thanks {name}! Sending you the sign-up link on WhatsApp.",
                        },
                        "Cold": {
                            "hi": f"{name} ji, bilkul samjha. Kuch dino mein ek updated offer ke saath wapas aaunga.",
                            "en": f"Understood {name}. We will reach out again in a few days.",
                        },
                    }
                    msg = closing[lead_score].get(lang, closing[lead_score]["hi"])
                    q_queue.put(msg)

                    # Persist summary + status into the dashboard leads store
                    _lid = ws.data.get("lead_id", "")
                    if _lid:
                        for _lead in LEADS_DB:
                            if _lead["id"] == _lid:
                                _lead["summary"] = summary
                                _lead["status"]  = lead_score.lower()
                                break

                    if lead_score == "Hot":
                        await ws.send(f"HANDOFF_TO_RM:{json.dumps(summary, ensure_ascii=False)}")
                    elif lead_score == "Warm":
                        await ws.send(f"SCHEDULE_FOLLOWUP:{json.dumps(summary, ensure_ascii=False)}")
                    else:
                        await ws.send(f"NURTURE_LEAD:{json.dumps(summary, ensure_ascii=False)}")

                    await ws.send("FILLER:flow_complete")
                except Exception as e:
                    printLogs(f"[ERROR] During lead scoring/routing: {e}")
                    traceback.print_exc()
                finally:
                    SESSIONS.pop(ws.data["sessionId"], None)
                return
        else:
                for qid, value in extracted.items():
                        sess["answers"][qid] = value
                        if qid in sess["unanswered_required"]:
                            sess["unanswered_required"].remove(qid)
                        if qid in sess["unanswered_optional"]:
                            sess["unanswered_optional"].remove(qid)
                
                # Then handle the general enquiry
                printLogs("[INFO] General enquiry detected along with data.")
                sess["last_answer_was_general_enquiry"] = True
                sess["last_user_text"] = user_text
                # generate_text_and_audio_gpt(user_text, ws, q_queue)
                
                # After handling general enquiry, continue with questionnaire
                if sess["unanswered_required"] or sess["unanswered_optional"]:
                    next_k_q = get_next_k_questions(sess)
                    next_ids = [q["id"] for q in next_k_q]
                    shorterNames = [q["short"] for q in QUESTIONNAIRE if q["id"] in next_ids]
                    lastQuestion = next_k_q[-1]
                    sess["last_asked_qid"] = lastQuestion["id"]
                    
                #     shortNamesString = ",".join(shorterNames)
                #     sess["last_asked_text"] = f"आपका {shortNamesString} क्या हैं ?"
                #     q_queue.put(sess["last_asked_text"])
                # # return

        # 2. Handle current question specifically if no answers found in scan
        if current_q:
            extractedObj = extracted
            #extractedObj = extract_answer_using_gpt(QUESTIONNAIRE, user_text)
            #answer = extractedObj[current_q["id"]] # TODO: Check removal

            printLogs(f"extracted2 : {extractedObj}")

            if not extractedObj:
                pass
            else:

                answer = None
                if "default" in extractedObj:
                    answer = extractedObj["default"]
                elif current_q["id"] in extractedObj:
                    answer = extractedObj[current_q["id"]]

                printLogs(f"[INFO] Primary extraction for {current_qid}: {answer}")

                if answer == "general_enquiry":
                    printLogs("[INFO] General enquiry detected.")
                    playContinueJourney = True
                    sess["last_answer_was_general_enquiry"] = True 
                    # Clear queue before handling enquiry
                    await ws.send("PLAY_FILLER")
                    # while not q_queue.empty():
                    #     q_queue.get()
                    generate_text_and_audio_gpt(user_text, ws, q_queue)
                    lang = sess.get("lang", "hi")
                    prompt = qprompt(current_q, lang)
                    sess["last_asked_text"] = S("resume_journey", lang, prompt)
                    q_queue.put(sess["last_asked_text"]) # TODO: Check if this works as expected.

                    return

                elif answer not in ("not_found", "not_an_answer", "invalid", "not_insurance", "not_relevant", "confused"):
                    sess["answers"][current_qid] = answer
                    if current_qid in sess["unanswered_required"]:
                        sess["unanswered_required"].remove(current_qid)
                    if current_qid in sess["unanswered_optional"]:
                        sess["unanswered_optional"].remove(current_qid)

                    # next_q = get_next_question(sess)
                    if next_q:
                        sess["last_asked_qid"] = next_q["id"]
                        lang = sess.get("lang", "hi")
                        prompt = qprompt(next_q, lang)
                        q_queue.put(prompt)
                        sess["last_asked_text"] = prompt
                        return

                elif answer in ["invalid", "not_insurance", "not_relevant"]:
                    printLogs("Going to send ask_again flag to frontend")
                    await ws.send("FILLER:ask_again")
                    lang = sess.get("lang", "hi")
                    sess["last_asked_text"] = qprompt(current_q, lang)
                    return

                ## Copied once. ensure to make changes there if changes made here.
                elif answer == "confused":
                    printLogs("[INFO] Confused detected.")
                    playContinueJourney = True
                    lang = sess.get("lang", "hi")
                    prompt = qprompt(current_q, lang)

                    if sess.get("last_answer_was_general_enquiry"):
                        printLogs("[INFO] Confusion after general_enquiry. Re-invoking GPT - 2")
                        generate_text_and_audio_gpt(sess["last_user_text"], ws, q_queue)
                        q_queue.put(S("resume_confused", lang, prompt))
                        sess["last_asked_text"] = prompt
                    else:
                        q_queue.put(S("repeat_q", lang, prompt))
                        sess["last_asked_text"] = prompt

                    return

                # elif answer == "continue_journey":
                #     printLogs("[INFO] User asked to continue journey - 2")

                #     sess["intent_confirmed"] = True
                #     if "cancel" in sess and sess["cancel"] == True:
                #         sess.clear()
                #         sess["cancel"] = False
                #     return


        # 3. Re-prompt if no valid answer found
        if current_q and "default" not in extractedObj:
            printLogs(f"[WARN] No valid answer for {current_qid}, re-asking")
            while not q_queue.empty():
                q_queue.get()
            await ws.send("FILLER:re_asking")
            lang = sess.get("lang", "hi")
            prompt = qprompt(current_q, lang)
            sess["last_asked_text"] = prompt
            q_queue.put(prompt)
            return
        
        elif "default" in extractedObj and extractedObj["default"] in ("not_insurance", "not_relevant"):
            await ws.send("FILLER:ask_again")
            #generate_text_and_audio_gpt(user_text, ws, q_queue)  ## Aditya come here
            #q_queue.put(f"चलिए आपकी अधूरी यात्रा को शुरू करते हैं। {current_q['prompt']}")
            return

    # Fallback to chit-chat
    printLogs("[INFO] Fallback to ordinary chit-chat")
    await ws.send("PLAY_FILLER")
    # Clear queue before chit-chat
    # while not q_queue.empty():
    #     q_queue.get()
    gpt_done_event = threading.Event()
    threadObj = threading.Thread(
        target=generate_text_and_audio_gpt,
        args=(user_text, ws, q_queue),
        kwargs={"gpt_done_event": gpt_done_event},
        daemon=True
    )
    threadObj.start()
    gpt_done_event.wait()
    lang = get_session(ws).get("lang", "hi")
    followup = {
        "en": "Just to circle back — are you open to exploring the Rupeezy AP partner opportunity?",
        "hi": "Main aapko bata dun — Rupeezy AP program mein zero joining fee hai, 100% brokerage share milta hai, aur daily payouts hote hain. Kya aap iske baare mein aur sunna chahenge?",
    }.get(lang, "Main aapko bata dun — Rupeezy AP program mein zero joining fee hai, 100% brokerage share milta hai, aur daily payouts hote hain. Kya aap iske baare mein aur sunna chahenge?")
    q_queue.put(followup)

def prepareConversationHistory(sess, QUESTIONNAIRE):
    history = []
    found_next_question = False

    for q in QUESTIONNAIRE:
        q_id = q["id"]
        prompt = q["prompt"]

        if q_id in sess.get("answers", {}):
            # Add the question
            history.append({
                "role": "assistant",
                "content": prompt
            })
            # Add the answer
            history.append({
                "role": "user",
                "content": str(sess["answers"][q_id])
            })
        elif not found_next_question:
            # Add the first unanswered question only
            history.append({
                "role": "assistant",
                "content": prompt
            })
            found_next_question = True
            break  # Stop after including the first unanswered question

    return history
  

# --------------------------
# LLM streaming helper
# --------------------------
def generate_text_and_audio_gpt(user_text: str, ws, q_queue: queue.Queue,voice=None,gpt_done_event=None):   
    sess = get_session(ws)
    sess["last_user_text"] = user_text

    '''
    you are dhvani, a female who helps people buy insurance. Be friendly, cheerful, concise, no bullet points. Try and figure out the best possible meaning of the answer given by user, in context, to the previous asked question. only hinglish is allowed. Don't use filler words like "bilkul" to begin with. Should not end with a question and only answer insurance related questions 
    (health).
    '''
    lang = sess.get("lang", "hi")
    lang_instruction = {
        "en": "Reply ONLY in English.",
        "hi": "Reply ONLY in Hinglish (Hindi + English mix). No bullet points.",
        "ta": "Reply ONLY in Tamil. No bullet points.",
        "te": "Reply ONLY in Telugu. No bullet points.",
        "mr": "Reply ONLY in Marathi. No bullet points.",
        "gu": "Reply ONLY in Gujarati. No bullet points.",
        "bn": "Reply ONLY in Bengali. No bullet points.",
    }.get(lang, "Reply in Hinglish.")

    lead_gender   = sess.get("gender", "male")
    gender_note   = (
        "The lead is female — use feminine pronouns and gender-appropriate verb forms when referring to her. "
        if lead_gender == "female" else
        "The lead is male — use masculine pronouns and gender-appropriate verb forms when referring to him. "
    )

    system_instruction = (
        "You are Priya, a senior partner executive at Rupeezy — a SEBI-registered stockbroker. "
        "Your job is to pitch Rupeezy's Authorized Person (AP) partner program to leads who are "
        "MFDs, financial advisors, insurance agents, or finance influencers. "
        f"{gender_note}"
        "CORE BENEFITS to weave in naturally: "
        "(1) Zero joining fee — no upfront cost ever. "
        "(2) 100% brokerage share — vs 60-70% industry standard. "
        "(3) Daily payouts via RISE Portal — no waiting till month-end. "
        "(4) SEBI-registered broker — fully regulated and trustworthy. "
        "(5) Dedicated RM support — full hand-holding from onboarding to first payout. "
        "OBJECTION HANDLING RULES: "
        "If lead says they are already with another broker — frame Rupeezy as ADDITIVE, not a replacement. "
        "If lead has few contacts — reassure that even 5-10 active contacts are enough to start. "
        "If lead asks about support — emphasise the dedicated RM layer, not a generic helpline. "
        "If lead doubts trustworthiness — cite SEBI registration and offer to share certificate. "
        "If lead says 'think about it' — offer to send WhatsApp signup link; joining is free so risk is zero. "
        "TONE & FORMAT: Friendly, confident, sales-oriented. Never use bullet points. "
        "Keep responses under 60 words. Do NOT start with filler words like 'bilkul' or 'sure'. "
        "Do NOT end responses with a question unless collecting a required detail. "
        "Never make unverifiable claims about company size, revenue, or guaranteed earnings. "
        "If asked about earnings, say active APs average ₹20,000–₹1,00,000/month with no cap. "
        "Always guide towards: sign up via WhatsApp link, or confirm callback time. "
        f"{lang_instruction}"
    )


    conversation = [
        {"role": "system", "content": system_instruction},
    ]

    currentUserText = [
        {"role": "user", "content": user_text}
    ]
    
    SESSION_CONVERSATION[ws.data["sessionId"]].append({"role": "user", "content": user_text})

    conversation = conversation + prepareConversationHistory(get_session(ws), QUESTIONNAIRE) + SESSION_CONVERSATION[ws.data["sessionId"]] 
                    #+ currentUserText

    # print("conversation till now : ", conversation)
    gpt_start=time.perf_counter()
    stream = groq_client.chat.completions.create(
        model=textToTextModelId,
        messages=conversation,
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    
    buf = ""
    completeResponse = ""
    for chunk in stream:
        if ws.connOpen == False:
            printLogs("websocket closed. dont add to TTS")
            break
        delta = getattr(chunk.choices[0].delta, "content", "")
        if delta:
            buf += delta
            completeResponse += delta
            if re.search(r"[।.:;!?]$", buf):
                q_queue.put(buf)
                printLogs(f"[LATENCY][TTS chunk flush]Sent chunk to TTS queue-{buf.strip()}({len(buf)}chars)")
                buf = ""
    if buf.strip() and ws.connOpen == True:
        completeResponse += buf
        q_queue.put(buf)
        printLogs(f"[LATENCY][TTS final flush]Sent last chunk to TTS - {buf.strip()}")
    gpt_end=time.perf_counter()
    if ws.connOpen == True:
        SESSION_CONVERSATION[ws.data["sessionId"]].append({"role": "assistant", "content": completeResponse})
        asyncio.run_coroutine_threadsafe(
            ws.send(f"TRANSCRIPT:BOT:{completeResponse}"), ws.main_loop
        )
    else:
        try:
            with q_queue.mutex:  # Thread-safe queue clearing
                q_queue.queue.clear()
                q_queue.all_tasks_done.notify_all()
                q_queue.unfinished_tasks = 0
        except AttributeError:
            # Fallback to manual clearing if mutex method fails
            while not q_queue.empty():
                try:
                    q_queue.get_nowait()
                except queue.Empty:
                    break
        printLogs("Queue cleared")
    printLogs(f"[LATENCY][GPT RESPONSE]Total time :{gpt_end-gpt_start:2f}s")
    if gpt_done_event:
        gpt_done_event.set()
    # gpt_done_event.set()
# --------------------------
# Audio streaming VAD handler
# --------------------------
async def audio_handler(ws):

    receivedInfo = await ws.recv()

    # Parse "SESSION_ID:<sessionId>" or "SESSION_ID:<sessionId>:<leadId>"
    sessionId = ""
    lead_id   = ""
    if receivedInfo.startswith("SESSION_ID"):
        parts     = receivedInfo.split(":", 2)
        sessionId = parts[1] if len(parts) > 1 else ""
        lead_id   = parts[2] if len(parts) > 2 else ""

    printLogs(f"[Server] Client connected sessionId={sessionId} lead_id={lead_id}")
    ws.connOpen = True
    ws.data: dict[str, any] = {}
    ws.data["sessionId"] = sessionId
    ws.data["lead_id"]   = lead_id

    if lead_id:
        SESSION_TO_LEAD[sessionId] = lead_id

    if ws.data["sessionId"] not in SESSION_CONVERSATION:
        printLogs("Creating new session.")
        SESSION_CONVERSATION[ws.data["sessionId"]] = []
    else:
        printLogs(f"Found existing session: {len(SESSION_CONVERSATION[ws.data['sessionId']])} msgs")

    # Pre-load lead data so session language is correct from turn 1
    lead_data = next((l for l in LEADS_DB if l["id"] == lead_id), None)
    if lead_data:
        sess = get_session(ws)
        sess["lang"]      = lead_data.get("language", "hi")
        sess["gender"]    = lead_data.get("gender", "male")
        sess["lead_data"] = lead_data
        lead_data["status"] = "active"
        # Pre-fill answers we already know from the dashboard so questionnaire skips them
        for field in ("name", "city", "profession"):
            val = lead_data.get(field, "")
            if val:
                sess["answers"][field] = val
                sess["unanswered_required"] = [q for q in sess["unanswered_required"] if q != field]
                sess["unanswered_optional"] = [q for q in sess["unanswered_optional"] if q != field]

    vad = webrtcvad.Vad(); vad.set_mode(VAD_MODE)

    read_buf, utt_buf, in_speech, app_running_flag, silent = bytearray(), bytearray(), False, True, 0
    temp_buff = bytearray()

    q_queue: queue.Queue[str] = queue.Queue()

    # Kick off the TTS consumer that will turn strings on `q_queue`
    # into audio and push them back to the client via `ws`.
    loop = asyncio.get_running_loop()
    ws.main_loop = loop  # stored so threaded helpers can schedule coroutines
    threading.Thread(target=tts_consumer,
                     args=(q_queue, loop, ws),
                     daemon=True).start()

    # ── Personalised opening in registered language + language confirmation ──
    lang        = lead_data.get("language", "hi") if lead_data else "hi"
    lead_gender = lead_data.get("gender", "male") if lead_data else "male"
    lead_name   = lead_data.get("name", "").split()[0] if lead_data else ""
    if lead_name:
        if lang == "en":
            name_part = ", " + lead_name
        elif lang in ("hi", "mr", "gu"):
            name_part = " " + lead_name + " ji"
        elif lang == "ta":
            # Tamil: female → Akka (elder sister), male → Anna (elder brother) — respectful
            name_part = " " + lead_name + (" Akka" if lead_gender == "female" else " Anna")
        else:  # te, bn — gender-neutral suffix "ji" works universally
            name_part = " " + lead_name + " ji"
    else:
        name_part = ""
    intro_template = INTRO_TEXTS.get(lang, INTRO_TEXTS["hi"])
    intro_text = intro_template.format(name=name_part)
    q_queue.put(intro_text)
    # Language preference question — ask after intro
    q_queue.put(LANG_CONFIRM_Q.get(lang, LANG_CONFIRM_Q["hi"]))
    # ─────────────────────────────────────────────────────────────────────────

    try:
        while app_running_flag:
            while True:
                try:
                    data = await ws.recv()
                except websockets.exceptions.ConnectionClosed:
                    printLogs("[Server] Connection closed by client.")
                    ws.connOpen = False
                    app_running_flag = False
                    break
                if data == b"DONE":

                    ## Dummy code to write the audio to the file
                    #now = datetime.now()
                    #with wave.open(f"audiohandler_test_{now.strftime("%Y-%m-%d %H:%M:%S")}.wav", 'wb') as wf:
                    #    wf.setnchannels(1)
                    #    wf.setsampwidth(2)
                    #    wf.setframerate(16000)
                    #    wf.writeframes(read_buf)
                    #return

                    break
                else:
                    utt_buf.clear()
                
                read_buf.extend(data)
                # ------------------------------------------

                PRE_ROLL_FRAMES = 36                        # how many silent frames to keep
                pre_roll = collections.deque(maxlen=PRE_ROLL_FRAMES)

                for frame in chunkify_frames(read_buf):    # ── each frame is 10/20/30 ms bytes
                    speech = vad.is_speech(frame, SAMPLE_RATE)

                    if speech:                             # ── we're inside speech
                        if not in_speech:                  # ① transition silence→speech
                            utt_buf.extend(b"".join(pre_roll))  # prepend the 3 stored frames
                            pre_roll.clear()
                            in_speech, silent = True, 0

                        utt_buf.extend(frame)              # ② always keep the current frame

                    else:                                  # ── vad says "silence"
                        if in_speech:                      # ③ inside an utterance, count gap
                            silent += 1
                            utt_buf.extend(frame)

                            # if silent >= PAUSE_THRESHOLD_FRAMES:
                            #     printLogs("silence : " + str(silent) + ", buffersize : " + str(len(utt_buf)))
                            #     await process_utterance(bytes(utt_buf), ws, q_queue)
                            #     printLogs("Going to clear buffers")
                            #     in_speech, utt_buf, silent = False, bytearray(), 0

                        else:                              # ④ still in silence → grow pre-roll
                            pre_roll.append(frame)

            if in_speech and utt_buf:
                printLogs("buffersize : " + str(len(bytes(utt_buf))))
                await process_utterance(bytes(utt_buf), ws, q_queue)
                printLogs("Going to clear buffers - 2nd utterance")
                in_speech, utt_buf, read_buf, silent = False, bytearray(), bytearray(), 0
    except websockets.exceptions.ConnectionClosed:
        printLogs("Connection closed from client")
        ws.connOpen = False
    finally:
        #SESSIONS.pop(ws.data["sessionId"], None)
        printLogs("[Server] audio handler ended. Websocket closed")

# --------------------------
# Utilities
# --------------------------

def chunkify_frames(buffer: bytearray):
    #printLogs(f"[arraySize] {len(buffer)}")
    temp_buffer = buffer.copy()
    chunks = [temp_buffer[i:i + FRAME_SIZE] for i in range(0, len(temp_buffer), FRAME_SIZE)]
    
    # Pad the last chunk if it is smaller than FRAME_SIZE
    if len(chunks) > 0 and len(chunks[-1]) < FRAME_SIZE:
        padding_needed = FRAME_SIZE - len(chunks[-1])
        chunks[-1] += b'\x00' * padding_needed  # Add silence padding

    #printLogs(f"[numberOfChunks] {len(chunks)}")

    # Corrected verification calculation
    #expected_chunks = (len(buffer) + FRAME_SIZE - 1) // FRAME_SIZE  # Round up to the nearest chunk
    #if len(chunks) == expected_chunks:
        #printLogs(f"[VERIFICATION] Array size and chunk calculations are correct")
    #else:
        #printLogs(f"[ERROR] Mismatch in chunk calculation. ArraySize: {len(buffer)}, Expected Chunks: {expected_chunks}, Actual Chunks: {len(chunks)}")
    
    return chunks

    ##temp_buffer = buffer.copy()
    ##while len(temp_buffer) >= FRAME_SIZE:
    ##    yield bytes(temp_buffer[:FRAME_SIZE]); del temp_buffer[:FRAME_SIZE]
    ##if len(temp_buffer) > 0:
    ##    # Pad the last chunk with zeros (16-bit PCM silence = b'\x00\x00')
    ##    pad_length = FRAME_SIZE - len(temp_buffer)
    ##    padded = temp_buffer + b'\x00' * pad_length
    ##    yield bytes(padded)
    ##    #temp_buffer.clear()

# --------------------------
# Static file server (serves React build on a separate port for Replit)
# --------------------------
import http.server

def _start_static_server():
    static_port = int(os.getenv("STATIC_PORT", "5000"))
    dist_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "client", "project", "dist"
    ))

    class RupeezyHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            serve_dir = dist_dir if os.path.isdir(dist_dir) else os.path.dirname(__file__)
            super().__init__(*args, directory=serve_dir, **kwargs)

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _json(self, code: int, data):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(200)
            self._cors()
            self.end_headers()

        def do_GET(self):
            if self.path == "/api/leads":
                self._json(200, LEADS_DB)
            elif self.path.startswith("/api/leads/"):
                lead_id = self.path.rstrip("/").split("/")[-1]
                lead = next((l for l in LEADS_DB if l["id"] == lead_id), None)
                self._json(200, lead) if lead else self._json(404, {"error": "Not found"})
            else:
                if not os.path.isdir(dist_dir):
                    self._json(503, {"error": "Frontend not built yet"})
                    return
                super().do_GET()

        def do_POST(self):
            if self.path.startswith("/api/leads/") and self.path.endswith("/summary"):
                parts   = self.path.rstrip("/").split("/")
                lead_id = parts[-2]
                length  = int(self.headers.get("Content-Length", 0))
                body    = json.loads(self.rfile.read(length))
                for lead in LEADS_DB:
                    if lead["id"] == lead_id:
                        lead["summary"] = body
                        score = body.get("lead_score", "")
                        lead["status"] = score.lower() if score in ("Hot", "Warm", "Cold") else "cold"
                        break
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "Not found"})

        def log_message(self, *_):
            pass  # suppress noisy access logs

    server = http.server.ThreadingHTTPServer(("0.0.0.0", static_port), RupeezyHandler)
    printLogs(f"[STATIC] API + static server at http://0.0.0.0:{static_port}")
    server.serve_forever()

# --------------------------
# Entrypoint
# --------------------------
async def main():
    threading.Thread(target=_start_static_server, daemon=True).start()
    async with websockets.serve(audio_handler, HOST, PORT, max_size=100_000_000):
        printLogs(f"[Server] ws://{HOST}:{PORT}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())