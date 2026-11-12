# Adding this project to GitHub (step by step)

## 1. Create the repository

1. Go to github.com and click **New repository**.
2. Owner: your account. Name: `nifty-forecasting`.
3. Description:
   ```
   NIFTY 50 forecasting: ARIMA/SARIMA, ETS and gradient boosting with rolling-origin CV and prediction intervals, plus GARCH volatility and VaR backtests.
   ```
4. Set it to **Public**. Leave README, .gitignore and license **unchecked**. Click **Create repository**.

## 2. Upload the files

**Without a terminal (easiest)**

1. Unzip `nifty-forecasting.zip` on your computer.
2. On the empty repo page click **uploading an existing file**.
3. Drag in everything from inside the unzipped folder: `src`, `tests`, `results`, `.github`, `README.md`, `requirements.txt`, `pytest.ini`, `LICENSE`, `GITHUB_INSTRUCTIONS.md`.
4. Click **Commit changes**.

The browser upload may skip hidden files. Check that `.github/workflows/ci.yml` appears in the repo (it gives you the green "tests" badge). If it is missing, click **Add file → Create new file**, type `.github/workflows/ci.yml` as the name, paste the contents of that file, and commit. Also add `.gitignore` the same way if it is missing.

**With a terminal** (PowerShell on Windows, Terminal on Mac), from inside the unzipped folder:

```bash
git init
git add .
git commit -m "Add NIFTY 50 forecasting and volatility project"
git branch -M main
git remote add origin https://github.com/Fariha12t/nifty-forecasting.git
git push -u origin main
```

## 3. Run it and commit your results

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_all --data nifty --out results
```

Then, to show the results on GitHub:

1. Open `.gitignore` and delete the two lines under "Generated outputs" (`results/*` and `!results/.gitkeep`).
2. Upload or commit the new files in `results/` (`REPORT.md` and the `figures` folder).
3. Embed two or three figures in the README, for example:
   `![Forecast intervals](results/figures/monthly_forecast_intervals_best.png)`
4. Replace the **Results** paragraph in the README with 3 to 4 sentences of your own findings.

## 4. Polish

- Add topics on the repo page: `time-series`, `forecasting`, `garch`, `arima`, `volatility`, `value-at-risk`, `statsmodels`.
- Pin the repository on your profile.
- Check the **Actions** tab: the tests workflow should turn green.
