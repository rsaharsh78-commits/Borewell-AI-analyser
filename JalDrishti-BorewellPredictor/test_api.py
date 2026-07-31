import urllib.request
import json
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

PORT = 8001
BASE = f"http://localhost:{PORT}"

def test(label, fn):
    print(f"\n=== {label} ===")
    try:
        fn()
        print("  [PASS]")
    except Exception as e:
        print(f"  [FAIL]: {e}")
        return False
    return True

passed = 0
total = 0

# Test 1: Health
def t_health():
    r = urllib.request.urlopen(f"{BASE}/api/health")
    d = json.loads(r.read())
    assert d['status'] == 'ok', f"Bad status: {d['status']}"
    assert d['models_loaded'] == True, "Models not loaded"
    print(f"  Version: {d['model_version']}")

total += 1
if test("Health Check", t_health): passed += 1

# Test 2: Predict (Kukatpally — over-exploited)
def t_predict_kukatpally():
    body = json.dumps({"latitude": 17.4947, "longitude": 78.3996}).encode()
    req = urllib.request.Request(f"{BASE}/api/predict", data=body,
                                 headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=30)
    d = json.loads(r.read())
    print(f"  Risk: {d['risk_score']} ({d['risk_category']})")
    print(f"  Confidence: {d['confidence']}")
    print(f"  Location: {d['location_info']['mandal']}, {d['location_info']['district']}")
    print(f"  Extraction: {d['nearby_stats']['extraction_stage_pct']}%")
    print(f"  Depth: {d['recommended_depth_ft']['min']}-{d['recommended_depth_ft']['max']} ft")
    print(f"  Factors: {len(d['factors'])}")
    for f in d['factors'][:3]:
        print(f"    - {f['name']}: {f['impact']} ({f['contribution']})")
    adv = d['advisory'][:100].encode('ascii', 'replace').decode('ascii')
    print(f"  Advisory: {adv}...")
    assert 0 <= d['risk_score'] <= 100
    assert len(d['factors']) > 0
    assert d['location_info']['state'] == 'Telangana'

total += 1
if test("Predict: Kukatpally (Over-Exploited)", t_predict_kukatpally): passed += 1

# Test 3: Predict (Maheshwaram — rural, should be lower risk)
def t_predict_maheshwaram():
    body = json.dumps({"latitude": 17.17, "longitude": 78.44}).encode()
    req = urllib.request.Request(f"{BASE}/api/predict", data=body,
                                 headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=30)
    d = json.loads(r.read())
    print(f"  Risk: {d['risk_score']} ({d['risk_category']})")
    print(f"  Location: {d['location_info']['mandal']}")
    print(f"  Extraction: {d['nearby_stats']['extraction_stage_pct']}%")
    assert 0 <= d['risk_score'] <= 100

total += 1
if test("Predict: Maheshwaram (Rural)", t_predict_maheshwaram): passed += 1

# Test 4: Geocode
def t_geocode():
    r = urllib.request.urlopen(f"{BASE}/api/geocode?q=Kukatpally+Hyderabad", timeout=15)
    d = json.loads(r.read())
    print(f"  Results: {len(d)}")
    if d:
        print(f"  First: {d[0]['display_name'][:60]}...")
    assert len(d) > 0

total += 1
if test("Geocode: Kukatpally", t_geocode): passed += 1

# Test 5: Heatmap
def t_heatmap():
    r = urllib.request.urlopen(f"{BASE}/api/heatmap", timeout=15)
    d = json.loads(r.read())
    print(f"  Points: {len(d)}")
    scores = [p['risk_score'] for p in d]
    print(f"  Risk range: {min(scores):.1f} - {max(scores):.1f}")
    assert len(d) > 50

total += 1
if test("Heatmap Grid", t_heatmap): passed += 1

# Test 6: Rainfall History
def t_rainfall():
    r = urllib.request.urlopen(f"{BASE}/api/rainfall-history?lat=17.385&lng=78.486", timeout=15)
    d = json.loads(r.read())
    print(f"  Years: {len(d['years'])}")
    if d['annual_rainfall']:
        print(f"  Rain range: {min(d['annual_rainfall']):.0f} - {max(d['annual_rainfall']):.0f} mm")
    assert len(d['years']) > 0

total += 1
if test("Rainfall History", t_rainfall): passed += 1

# Summary
print(f"\n{'='*50}")
print(f"Results: {passed}/{total} passed")
if passed == total:
    print("ALL TESTS PASSED!")
else:
    print("WARNING: Some tests failed")
    sys.exit(1)
