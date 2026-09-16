#!/usr/bin/env python3
"""
Quick endpoint smoke test — verifies all 5 endpoints work correctly
without needing an LLM API key (tests structure, not composition quality).
"""
import json, time, sys
from pathlib import Path
from urllib import request as urlrequest, error as urlerror

BOT_URL = "http://localhost:8080"
DATASET  = Path(__file__).parent / "dataset"

def call(method, path, body=None, timeout=15):
    url = f"{BOT_URL}{path}"
    data = json.dumps(body).encode() if body else None
    req  = urlrequest.Request(url, data=data, method=method,
                              headers={"Content-Type":"application/json"})
    try:
        resp = urlrequest.urlopen(req, timeout=timeout)
        return json.loads(resp.read()), None
    except urlerror.HTTPError as e:
        try:    return json.loads(e.read()), None
        except: return None, f"HTTP {e.code}"
    except Exception as ex:
        return None, str(ex)

def ok(label, cond, detail=""):
    mark = "✓" if cond else "✗"
    print(f"  [{mark}] {label}" + (f"  — {detail}" if detail else ""))
    return cond

passed = failed = 0

print("\n═══════ GET /v1/healthz ═══════")
d, e = call("GET", "/v1/healthz")
ok("status=ok",          not e and d.get("status")=="ok")
ok("uptime_seconds int", not e and isinstance(d.get("uptime_seconds"), int))
ok("contexts_loaded dict", not e and isinstance(d.get("contexts_loaded"), dict))

print("\n═══════ GET /v1/metadata ═══════")
d, e = call("GET", "/v1/metadata")
for k in ("team_name","model","approach","version"):
    ok(f"has '{k}'", not e and k in d)

print("\n═══════ POST /v1/context — push category ═══════")
cat = json.load(open(DATASET / "categories" / "dentists.json"))
d, e = call("POST", "/v1/context", {"scope":"category","context_id":"dentists",
                                     "version":1,"payload":cat,
                                     "delivered_at":"2026-04-26T09:45:00Z"})
ok("accepted=true",  not e and d.get("accepted")==True,  str(d))
ok("ack_id present", not e and "ack_id" in d)

print("\n═══════ POST /v1/context — idempotency (same version) ═══════")
d, e = call("POST", "/v1/context", {"scope":"category","context_id":"dentists",
                                     "version":1,"payload":cat,
                                     "delivered_at":"2026-04-26T09:45:00Z"})
ok("accepted=false (stale)", not e and d.get("accepted")==False and
                               d.get("reason")=="stale_version", str(d))

print("\n═══════ POST /v1/context — version bump replaces ═══════")
d, e = call("POST", "/v1/context", {"scope":"category","context_id":"dentists",
                                     "version":2,"payload":cat,
                                     "delivered_at":"2026-04-26T10:00:00Z"})
ok("accepted=true (v2)", not e and d.get("accepted")==True, str(d))

print("\n═══════ POST /v1/context — push all categories + merchant ═══════")
for slug in ["salons","restaurants","gyms","pharmacies"]:
    c = json.load(open(DATASET/"categories"/f"{slug}.json"))
    d, e = call("POST","/v1/context",{"scope":"category","context_id":slug,
                                       "version":1,"payload":c,"delivered_at":"2026-04-26T09:46:00Z"})
    ok(f"category/{slug}", not e and d.get("accepted"), str(d))

merchants = json.load(open(DATASET/"merchants_seed.json"))["merchants"]
for i, m in enumerate(merchants[:3]):
    d, e = call("POST","/v1/context",{"scope":"merchant","context_id":m["merchant_id"],
                                       "version":1,"payload":m,"delivered_at":"2026-04-26T09:47:00Z"})
    ok(f"merchant/{m['merchant_id'][:25]}", not e and d.get("accepted"), str(d))

print("\n═══════ POST /v1/context — push trigger ═══════")
triggers = json.load(open(DATASET/"triggers_seed.json"))["triggers"]
trg = triggers[0]  # research_digest for dr meera
d, e = call("POST","/v1/context",{"scope":"trigger","context_id":trg["id"],
                                   "version":1,"payload":trg,"delivered_at":"2026-04-26T10:00:00Z"})
ok("trigger accepted", not e and d.get("accepted"), str(d))

print("\n═══════ POST /v1/context — invalid scope ═══════")
d, e = call("POST","/v1/context",{"scope":"invalid","context_id":"x",
                                   "version":1,"payload":{},"delivered_at":"2026-04-26T10:00:00Z"})
ok("accepted=false on bad scope", not e and d.get("accepted")==False, str(d))

print("\n═══════ POST /v1/tick — no triggers (empty) ═══════")
d, e = call("POST","/v1/tick",{"now":"2026-04-26T10:30:00Z","available_triggers":[]})
ok("actions=[]", not e and d.get("actions")==[], str(d))

print("\n═══════ POST /v1/tick — with trigger (structure check, no LLM needed if no key) ═══════")
d, e = call("POST","/v1/tick",{"now":"2026-04-26T10:35:00Z",
                                "available_triggers":[trg["id"]]}, timeout=35)
if e:
    ok("tick responded", False, e)
else:
    actions = d.get("actions", [])
    ok("actions is list", isinstance(actions, list), str(actions)[:80])
    if actions:
        a = actions[0]
        for k in ("conversation_id","merchant_id","send_as","trigger_id","body","cta","suppression_key","rationale"):
            ok(f"  action has '{k}'", k in a, a.get(k,"MISSING")[:60] if isinstance(a.get(k,""),str) else str(a.get(k,"")))
        ok("  body non-empty", bool(a.get("body","").strip()))
        ok("  send_as=vera (merchant trigger)", a.get("send_as")=="vera")
    else:
        print("  [!] Bot chose not to send (no LLM key? fallback might still work)")

print("\n═══════ POST /v1/reply — auto-reply detection ═══════")
auto_msg = "Thank you for contacting us! Our team will respond shortly."
d, e = call("POST","/v1/reply",{"conversation_id":"conv_autoreply_test",
                                 "merchant_id":"m_001_drmeera_dentist_delhi",
                                 "from_role":"merchant","message":auto_msg,
                                 "received_at":"2026-04-26T10:42:00Z","turn_number":2})
ok("auto-reply → send or wait (not end on first)",
   not e and d.get("action") in ("send","wait"), str(d))

# 2nd auto-reply same conv
d, e = call("POST","/v1/reply",{"conversation_id":"conv_autoreply_test",
                                 "merchant_id":"m_001_drmeera_dentist_delhi",
                                 "from_role":"merchant","message":auto_msg,
                                 "received_at":"2026-04-26T10:43:00Z","turn_number":3})
ok("2nd auto-reply → wait or end", not e and d.get("action") in ("wait","end"), str(d))

# 3rd auto-reply
d, e = call("POST","/v1/reply",{"conversation_id":"conv_autoreply_test",
                                 "merchant_id":"m_001_drmeera_dentist_delhi",
                                 "from_role":"merchant","message":auto_msg,
                                 "received_at":"2026-04-26T10:44:00Z","turn_number":4})
ok("3rd auto-reply → end", not e and d.get("action")=="end", str(d))

print("\n═══════ POST /v1/reply — hostile message ═══════")
d, e = call("POST","/v1/reply",{"conversation_id":"conv_hostile_test",
                                 "merchant_id":"m_001_drmeera_dentist_delhi",
                                 "from_role":"merchant",
                                 "message":"Stop messaging me. This is useless spam.",
                                 "received_at":"2026-04-26T10:45:00Z","turn_number":2})
ok("hostile → end", not e and d.get("action")=="end", str(d))

print("\n═══════ POST /v1/reply — positive intent transition ═══════")
d, e = call("POST","/v1/reply",{"conversation_id":"conv_intent_test",
                                 "merchant_id":"m_001_drmeera_dentist_delhi",
                                 "from_role":"merchant",
                                 "message":"Ok let's do it. What's next?",
                                 "received_at":"2026-04-26T10:46:00Z","turn_number":3},
             timeout=35)
ok("intent → action (not qualifying)", not e and d.get("action")=="send", str(d))
if not e and d.get("action")=="send":
    body_l = d.get("body","").lower()
    qualifying = ["would you","do you","can you tell","what if","how about","kya aap"]
    ok("  no qualifying questions in body", not any(q in body_l for q in qualifying), d.get("body","")[:80])

print("\n═══════ POST /v1/reply — curveball (off-topic) ═══════")
d, e = call("POST","/v1/reply",{"conversation_id":"conv_curveball_test",
                                 "merchant_id":"m_001_drmeera_dentist_delhi",
                                 "from_role":"merchant",
                                 "message":"Btw can you help me file my GST this month?",
                                 "received_at":"2026-04-26T10:47:00Z","turn_number":2},
             timeout=35)
ok("curveball → send (not end)", not e and d.get("action") in ("send","wait"), str(d))
if not e and d.get("action")=="send":
    ok("  body non-empty", bool(d.get("body","").strip()))

print("\n═══════ GET /v1/healthz after warmup ═══════")
d, e = call("GET", "/v1/healthz")
counts = d.get("contexts_loaded",{})
ok("categories loaded ≥5", counts.get("category",0)>=5, str(counts))
ok("merchants loaded ≥3",  counts.get("merchant",0)>=3, str(counts))
ok("triggers loaded ≥1",   counts.get("trigger",0)>=1, str(counts))

print("\n" + "═"*50)
print("Smoke test complete. Check [✗] lines above for issues.")
print("NOTE: /v1/tick composition quality requires a valid LLM API key in .env")
