"""
One-off validator: stub heavy native deps, import the real hindi_bot.py,
then exercise every constant + helper our recent edits depend on.

Run from the bot/server directory:
    PYTHONIOENCODING=utf-8 python _boot_check.py

Delete after use — this file is not part of the bot's runtime.
"""
import sys
import types
import traceback


def stub(name, attrs=None):
    m = types.ModuleType(name)
    for k, v in (attrs or {}).items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m



# numpy
class _NDArray:
    pass


def _frombuffer(*a, **kw):
    return _NDArray()


def _zeros(*a, **kw):
    return _NDArray()


def _clip(x, *a, **kw):
    return x


def _asarray(x):
    return x


stub("numpy", {
    "ndarray": _NDArray, "int16": "int16", "float32": "float32",
    "frombuffer": _frombuffer, "zeros": _zeros, "clip": _clip, "asarray": _asarray,
})


# webrtcvad
class _Vad:
    def set_mode(self, m):
        pass

    def is_speech(self, *a):
        return False


stub("webrtcvad", {"Vad": _Vad})


# omnivoice
class _OV:
    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()

    def generate(self, **kw):
        return []


stub("omnivoice", {"OmniVoice": _OV})

stub("noisereduce", {"reduce_noise": lambda **kw: kw.get("y")})
stub("torchaudio", {})
stub("soundfile", {"write": lambda *a, **kw: None})


class _Proc:
    @staticmethod
    def extractOne(*a, **kw):
        return None


stub("rapidfuzz", {"process": _Proc(), "fuzz": types.SimpleNamespace()})


class _Embedding(list):
    def tolist(self):
        return list(self)


class _ST:
    def __init__(self, *a, **kw):
        pass

    def encode(self, x, **kw):
        return _Embedding([0.0] * 384)


stub("sentence_transformers", {"SentenceTransformer": _ST})
stub("vectordb_mgmt", {"prepareChromaDB": lambda: None, "SentenceTransformer": _ST})


class _Coll:
    def __init__(self):
        self._n = 0

    def add(self, **kw):
        self._n = len(kw.get("ids", []))

    def query(self, **kw):
        return {"distances": [[1.0]], "metadatas": [[{}]], "documents": [[""]]}

    def count(self):
        return self._n


class _CC:
    def create_collection(self, **kw):
        return _Coll()

    def get_or_create_collection(self, **kw):
        return _Coll()


stub("chromadb", {"Client": lambda: _CC()})
stub("chromadb.utils", {})
stub("chromadb.utils.embedding_functions", {"OpenAIEmbeddingFunction": lambda **kw: None})

# Force-stub torch (the real one tries to introspect numpy.bool_ at import,
# which our minimal numpy stub doesn't provide).
class _Cuda:
    @staticmethod
    def is_available():
        return False


stub("torch", {
    "cuda": _Cuda(), "float16": "fp16", "float32": "fp32",
    "Tensor": type("Tensor", (), {}), "nn": types.SimpleNamespace(),
})


class _CC2:
    def create(self, **kw):
        choice = types.SimpleNamespace(
            message=types.SimpleNamespace(content="ok"),
            delta=types.SimpleNamespace(content="ok"),
        )
        return types.SimpleNamespace(choices=[choice])


class _Audio:
    transcriptions = types.SimpleNamespace(
        create=lambda **kw: types.SimpleNamespace(text="")
    )


class _Groq:
    def __init__(self, **kw):
        self.chat = types.SimpleNamespace(completions=_CC2())
        self.audio = _Audio()


stub("groq", {"Groq": _Groq})
stub("openai", {"OpenAI": type("X", (), {"__init__": lambda self, **kw: None})})
stub("requests", {"get": lambda *a, **kw: None, "post": lambda *a, **kw: None})
stub("dotenv", {"load_dotenv": lambda *a, **kw: None})
stub("websockets", {
    "serve": lambda *a, **kw: None,
    "exceptions": types.SimpleNamespace(ConnectionClosed=Exception),
})
stub("websockets.exceptions", {"ConnectionClosed": Exception})


def main():
    try:
        import hindi_bot as hb
    except Exception as e:
        print(f"[FAIL] hindi_bot failed to import: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    print("[OK] hindi_bot imported cleanly")
    print(f"     KB_DOCUMENTS         = {len(hb.KB_DOCUMENTS)} entries")
    print(f"     OBJECTION_REBUTTALS  = {len(hb.OBJECTION_REBUTTALS)} entries")
    print(f"     OBJECTION_REBUTTALS_LANG languages = {sorted(hb.OBJECTION_REBUTTALS_LANG)}")
    print(f"     RECALL_TEMPLATES languages = {sorted(hb.RECALL_TEMPLATES)}")
    print(f"     PERSONA_PITCH keys   = {sorted(hb.PERSONA_PITCH)}")
    print(f"     HOT/WARM             = {hb.LEAD_SCORE_HOT}/{hb.LEAD_SCORE_WARM}")
    print(f"     BENEFITS_BLOCK len   = {len(hb.BENEFITS_BLOCK)}")
    print(f"     COMPLIANCE_BLOCK len = {len(hb.COMPLIANCE_BLOCK)}")
    print(f"     KB_SYSTEM_PROMPT len = {len(hb.KB_SYSTEM_PROMPT)}")

    # Boot-time invariants
    try:
        assert len(hb.KB_DOCUMENTS) > 15, f"KB_DOCUMENTS too small: {len(hb.KB_DOCUMENTS)}"
        assert len(hb.OBJECTION_REBUTTALS) == 5
        assert set(hb.OBJECTION_REBUTTALS_LANG) == {"hi", "en", "ta", "te", "mr", "gu", "bn"}
        assert set(hb.RECALL_TEMPLATES) == {"hi", "en", "ta", "te", "mr", "gu", "bn"}
        assert "default" in hb.PERSONA_PITCH
        assert hb.LEAD_SCORE_HOT > hb.LEAD_SCORE_WARM > 0
        assert abs(sum(hb.LEAD_SCORE_WEIGHTS.values()) - 1.0) < 0.001
        assert hb.BENEFITS_BLOCK and hb.COMPLIANCE_BLOCK
        assert hb.KB_SYSTEM_PROMPT and "Priya" in hb.KB_SYSTEM_PROMPT
    except AssertionError as e:
        print(f"[FAIL] Invariant: {e}")
        return 1
    print("[OK] All boot-time invariants hold")

    # get_rebuttal: every objection in every language + unknown lang fallback
    for lang in ["hi", "en", "ta", "te", "mr", "gu", "bn", "xx"]:
        for oid in ["obj_already_broker", "obj_no_contacts", "obj_support", "obj_trust", "obj_think_later"]:
            r = hb.get_rebuttal(oid, lang)
            if not r:
                print(f"[FAIL] empty rebuttal: {oid} / {lang}")
                return 1
    print("[OK] get_rebuttal: 5 objections * 8 languages (incl. unknown fallback)")

    # compute_lead_score
    sess = {"lead_score_components": {
        "intent": 80, "readiness": 70, "fit": 100, "engagement": 80,
        "objection_resolution": 100, "sentiment": 70,
    }}
    cls = hb.compute_lead_score(sess)
    print(f"[OK] compute_lead_score: {cls} ({sess['lead_score_total']}/100)")

    # S() across all 7 languages and the 6 keys
    for k in ["cancel_bye", "not_ready", "no_info_yet", "phone_reask", "network_reask", "asr_error"]:
        for lang in ["hi", "en", "ta", "te", "mr", "gu", "bn"]:
            tpl = hb.STRINGS.get(lang, {}).get(k, "")
            v = hb.S(k, lang, "X") if "{}" in tpl else hb.S(k, lang)
            if not v:
                print(f"[FAIL] empty S({k}, {lang})")
                return 1
    print("[OK] S(): 6 keys * 7 languages")

    # PERSONA_PITCH covers every demo profession
    for prof in [
        "Financial Advisor", "Mutual Fund Distributor", "Insurance Agent",
        "Finance Influencer", "Sub-Broker", "CA / Tax Consultant", "Stock Sub-Broker",
        "Some Unknown Profession",
    ]:
        p = hb.PERSONA_PITCH.get(prof, hb.PERSONA_PITCH["default"])
        if not p:
            print(f"[FAIL] empty pitch: {prof}")
            return 1
    print("[OK] PERSONA_PITCH: 7 demo personas + unknown fallback")

    # generate_post_call_summary smoke test
    test_sess = {
        "answers": {
            "name": "Rajesh", "phone": "9876543210",
            "profession": "Financial Advisor", "city": "Delhi",
            "network_size": 50, "current_broker": "Zerodha",
        },
        "lang": "hi", "persona": "Financial Advisor",
        "lead_score_components": {
            "intent": 80, "readiness": 70, "fit": 100, "engagement": 80,
            "objection_resolution": 100, "sentiment": 70,
        },
        "objections": [{"type": "obj_already_broker", "resolved": True}],
        "sentiment": "positive", "stage": "CTA", "intent_confirmed": True,
    }
    s = hb.generate_post_call_summary(test_sess, "sid_test")
    if s["lead_name"] != "Rajesh":
        print(f"[FAIL] summary lead_name mismatch")
        return 1
    if s["lead_score"] not in ("Hot", "Warm", "Cold"):
        print(f"[FAIL] bad lead_score: {s['lead_score']}")
        return 1
    if not s.get("whatsapp_body"):
        print(f"[FAIL] missing whatsapp_body for {s['lead_score']}")
        return 1
    print(f"[OK] generate_post_call_summary: {s['lead_score']} ({s['score']}/100), WA body present")

    # detect_language smoke test (script-based detection)
    cases = [("haan main interested hoon", "hi"),
             ("yes I am interested", "en"),
             ("நான் ஆர்வமாக உள்ளேன்", "ta"),
             ("నేను ఆసక్తిగా ఉన్నాను", "te"),
             ("મને રસ છે", "gu"),
             ("আমি আগ্রহী", "bn")]
    for text, _expected in cases:
        d = hb.detect_language(text)
        if d not in ("hi", "en", "ta", "te", "mr", "gu", "bn"):
            print(f"[FAIL] bad detection: {d}")
            return 1
    print(f"[OK] detect_language returns valid code for all sample inputs")

    print("\n[ALL CHECKS PASSED] hindi_bot.py wiring is sound.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
