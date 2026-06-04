# Quick Testing Guide for Fixes

## How to Test the Fixes

### Test 1: Cat Image Rejection (Report Validation)

1. **Start the backend**:
   ```bash
   cd web_interface2/backend
   python main.py
   ```

2. **Open the frontend** (http://localhost:5173 in browser)

3. **Upload a valid chest X-ray** as the image
   - Can use any real chest X-ray image

4. **Upload a CAT IMAGE as the report**
   - **Expected Result**: ✓ Error message appears
   - Error text: "Report image is not a medical document. Detected as: photo of a cat"
   - Backend shows: "Report validation: is_medical_document=False"

5. **Upload other non-medical images as report** and verify rejection:
   - Dog photo → "Detected as: photo of a dog"
   - Selfie → "Detected as: photo of a person's face"
   - Random object → "Detected as: everyday object"

### Test 2: Successful Dual Pipeline (Valid Inputs)

1. **Upload a real chest X-ray** as image
2. **Upload a medical report image** as report
   - Can be a radiology report, handwritten notes, official document

3. **Expected Result**: 
   - ✓ Both pipelines succeed
   - ✓ Shows "X-Ray + Report" badge
   - ✓ Shows confidence scores without NaN
   - ✓ Disease findings merged from both sources

### Test 3: NaN Handling in Results

1. **Run the test script**:
   ```bash
   cd web_interface2/backend
   python test_fixes.py
   ```

2. **Look for output**:
   - ✓ "All NaN validation tests pass"
   - ✓ "SUCCESS: No NaN values in merged results!"

3. **In the frontend**:
   - Upload valid images
   - Verify ALL confidence values display as numbers (never NaN)
   - Verify calculations show: "87.5%" not "NaN%"

### Test 4: Image Only Fallback

1. **Upload a valid chest X-ray**
2. **Upload an invalid report** (cat photo, selfie, etc.)

3. **Expected Result**:
   - ✓ Yellow warning banner appears
   - ✓ Text: "⚠️ Report analysis failed (Report image is not a medical document...). Showing X-ray analysis only."
   - ✓ Disease findings shown from X-ray only
   - ✓ "From X-Ray Image" badge displayed

### Test 5: Both Pipelines Fail

1. **Upload a non-chest-x-ray as image** (cat photo, person, etc.)
2. **Upload an invalid report** (also non-medical)

3. **Expected Result**:
   - ✓ Error message displayed
   - ✓ Shows both pipeline failures
   - ✓ "❌ Analysis Failed" header
   - ✓ Lists errors for both pipelines

---

## What Changed (User Perspective)

### Before Fixes
- ❌ Could upload cat image as report → Would try to process it
- ❌ Results might show "NaN" in confidence scores
- ❌ No clear error messages for wrong image types

### After Fixes
- ✓ Cat images immediately rejected with clear message
- ✓ All confidence scores are valid numbers (0-100%)
- ✓ Clear error messages explaining what went wrong
- ✓ Graceful fallback to single pipeline if one fails

---

## Debugging Commands

### Check if backend is running
```bash
curl http://127.0.0.1:8000/docs
```
(Should open Swagger API docs if backend is running)

### Test backend directly with Python
```python
import requests
import json

# Test 1: Check if backend loaded successfully
response = requests.get("http://127.0.0.1:8000/")
print(response.status_code)  # Should be 200 or similar
```

### View backend logs
- Backend prints validation messages:
  - `"Report validation: is_medical_document=False, confidence=..."`
  - `"Stage 1: Checking if text is medical..."`
  - `"Merged results: Top 3 diseases..."`

---

## Expected Console Output

### Backend startup
```
Loading Authenticity Model...
Loading Main Disease Classification Model...
[ready to accept requests]
```

### When cat image is uploaded as report
```
Validating report image as medical document...
Report validation: is_medical_document=False, confidence=0.92, label=photo of a cat
Report pipeline error: Report image is not a medical document. Detected as: photo of a cat
```

### When valid images are uploaded
```
Validating report image as medical document...
Report validation: is_medical_document=True, confidence=0.85, label=medical chart or report
Extracted Report Text: [text content...]
Stage 2: Checking if text is a chest X-ray report...
=== MERGING BOTH PIPELINES ===
Merged results: Pneumonia (0.75), Edema (0.62), Consolidation (0.55)
```

---

## Common Issues & Solutions

### Issue: Still seeing NaN values
**Solution**: 
- Clear browser cache (Ctrl+Shift+Delete)
- Refresh page (Ctrl+R)
- Ensure backend is using latest code (restart backend)

### Issue: Report always rejected
**Solution**:
- Check if report image is actually medical (radiology reports, screenshots of medical text)
- Try a clear photo of a medical chart or document
- Check backend logs for what it detected the image as

### Issue: Backend doesn't start
**Solution**:
```bash
# Make sure you're in the right directory
cd web_interface2/backend

# Make sure venv is activated
..\...\diffvenv\Scripts\Activate.ps1

# Check for import errors
python -c "import main; print('OK')"

# Try running
python main.py
```

---

## Performance Expectations

- **Report validation**: 200-300ms (additional time)
  - This is the BiomedCLIP model validating the image
  
- **Medical document detection**: Very fast once models are loaded

- **Overall analysis time**: Same as before (added validation is minimal overhead)

---

## Rollback Instructions

If needed to revert changes:

1. **Revert pipeline.py**:
   - Remove: `_is_valid_confidence()`, `_sanitize_confidence()`, `is_medical_document()`
   - Keep: `is_chest_xray()`, `detect_diseases()`

2. **Revert main.py**:
   - Remove medical document validation block
   - Remove `_sanitize_confidence()` calls
   - Restore original NaN handling

3. **Revert App.jsx**:
   - Remove NaN checks (`!isNaN()`)
   - Just use `value * 100` directly

---

## Next Steps

1. Start backend: `python main.py`
2. Start frontend: `npm run dev`
3. Run test script: `python test_fixes.py`
4. Try uploading test images (especially cat photo as report)
5. Verify no NaN values in results
6. Check error messages are clear and helpful
