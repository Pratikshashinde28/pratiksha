# Instagram Engagement Analysis

This project is a VS Code-friendly Instagram engagement analysis app that predicts what types of posts are likely to get more likes and comments. It includes:

- A reproducible synthetic Instagram post dataset
- Three regression algorithms for comparison
- A FastAPI backend
- A simple frontend dashboard
- Saved training artifacts and charts

## Algorithms Compared

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The app trains separate models for `likes` and `comments`, compares the algorithms on test performance, and uses the best model for predictions.

## Project Structure

```text
instagram_engagement_analysis/
|-- artifacts/
|-- src/
|   |-- api/
|   |   `-- app.py
|   |-- data/
|   |   `-- generate_data.py
|   |-- frontend/
|   |   `-- static/
|   |       |-- app.js
|   |       |-- index.html
|   |       `-- style.css
|   `-- models/
|       `-- train.py
|-- tests/
|   `-- test_smoke.py
|-- requirements.txt
`-- README.md
```

## How To Run In VS Code

1. Open the `instagram_engagement_analysis` folder in VS Code.
2. Open a new terminal in VS Code.
3. Run these commands:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.models.train
python -m uvicorn src.api.app:app --reload
```

4. Open these URLs in your browser:

- Frontend: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`

## What The Project Shows

- Which post types tend to perform better
- How reels, carousels, images, and videos affect engagement
- How caption length, hashtags, posting hour, and account size influence likes and comments
- Which of the three algorithms performs best

## Example Prediction Input

```json
{
  "followers": 120000,
  "post_type": "reel",
  "caption_length": 180,
  "hashtags_count": 8,
  "hour_posted": 20,
  "is_weekend": true,
  "sponsorship": false
}
```

## Notes

- The dataset is synthetic but designed to reflect realistic social media patterns.
- Training generates charts and model files inside `artifacts/`.
- The frontend also shows a summary of which post types usually attract higher likes and comments.
