# ✅ Deployment Checklist — Hugging Face Spaces

## Files Ready for Deployment

- [x] `app.py` — Standalone Gradio application (41 KB)
- [x] `requirements.txt` — Python dependencies (✅ fixed from "requirements .txt")
- [x] `SPACE_README.md` — Hugging Face Spaces documentation
- [x] `global_conflicts_dataset.csv` — Main conflict data (exists)
- [x] `fuel_prices_1970_2026.csv` — Oil price data (exists)
- [x] Other CSVs in `/Dataset/` folder (aramco, tourism, food data)

## Quick Deploy to Hugging Face Spaces

### Step 1: Create Space on Hugging Face
goto: https://huggingface.co/spaces/create
- **Owner:** Your username
- **Space name:** `Impact-of-Global-Conflicts`
- **License:** Openrail
- **SDK:** Gradio
- **Visibility:** Public

### Step 2: Upload/Clone & Push Files

**Option A (Recommended):** Git push
```bash
# Clone the space
git clone https://huggingface.co/spaces/[YOUR-USERNAME]/Impact-of-Global-Conflicts
cd Impact-of-Global-Conflicts

# Copy files from your repo
cp /workspaces/Impact_of_Global_Conflicts/app.py .
cp /workspaces/Impact_of_Global_Conflicts/requirements.txt .
cp /workspaces/Impact_of_Global_Conflicts/*.csv .

# Git push
git add .
git commit -m "Initial Gradio dashboard deployment"
git push
```

**Option B (UI):** Upload via Hugging Face web interface
1. Go to your Space
2. Click **📂 Files** tab
3. Upload `app.py`, `requirements.txt`, and all `.csv` files
4. Space auto-builds

### Step 3: Monitor Build
- Go to **Logs** tab
- Wait for build to complete (~2–3 minutes)
- Once live, you'll see the Gradio interface at the Space URL

### Step 4: Share
- **Space URL:** `https://huggingface.co/spaces/[USERNAME]/Impact-of-Global-Conflicts`
- Share with colleagues, embed in docs, add to portfolio

---

## What Happens on First Load?

When a user visits your Space for the first time:

1. **Data Load** (~5 sec) — CSV files loaded into memory
2. **Model Training** (~15–30 sec):
   - Ridge, Random Forest, Gradient Boosting regression models trained
   - K-means clustering fit
   - Holt-Winters time-series forecasting fitted
3. **Gradio Launch** (~2 sec) — UI becomes interactive

**Total:** ~30–45 seconds first load, then instant for subsequent interactions

---

## Troubleshooting Pre-Deployment

### Issue: CSV files not found
**Fix:** Make sure `global_conflicts_dataset.csv` and `fuel_prices_1970_2026.csv` are in the Space root (not in subfolders). The app looks for them in the current directory and `Dataset/` folder.

### Issue: Dependencies missing
**Fix:** Ensure `requirements.txt` is in the Space root with all packages listed.

### Issue: Slow Space
**Fix:** Hugging Face provides ~16GB RAM for Spaces. Your data is well under that. If slow, it's likely training time (unavoidable first load).

### Issue: CSV Encoding
**Fix:** If you see encoding errors, the CSV reading includes `errors='ignore'` to handle special characters.

---

## Local Testing (Optional)

To test before uploading:

```bash
cd /workspaces/Impact_of_Global_Conflicts

# Install dependencies
pip install -r requirements.txt

# Run app
python app.py

# Browser: http://localhost:7860
```

---

## Post-Deployment

Once live:

1. **Share the link** — https://huggingface.co/spaces/[USERNAME]/Impact-of-Global-Conflicts
2. **Embed in portfolio/docs** — Hugging Face provides embed code
3. **Monitor logs** — Check for any runtime errors
4. **Update CSV data** — If you refresh datasets, re-upload to the Space

---

## Key Features of Your Deployment

✅ **No API Key Required** — Fully FREE Hugging Face Spaces (up to 2 spaces)  
✅ **Auto-Scaling** — Spaces handle traffic automatically  
✅ **Live Updates** — Push new code/data, Space auto-rebuilds  
✅ **Shareable** — One unique URL for all users  
✅ **Mobile-Friendly** — Gradio UI is responsive  
✅ **No Pricing** — Free tier supports unlimited usage  

---

## Support

**Having issues?**
1. Check `SPACE_README.md` for detailed documentation
2. Review Hugging Face Spaces [docs](https://huggingface.co/docs/hub/spaces)
3. Check `app.py` comments for code explanations

---

**Ready to deploy!** 🚀 Go to https://huggingface.co/spaces/create and follow Step 1 above.
