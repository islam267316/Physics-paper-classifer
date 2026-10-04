# Physics Paper Classifier

A Streamlit application that predicts Astrophysics or Condensed Matter from a paper abstract. The saved classifier uses Logistic Regression and TF-IDF features.

## Run locally

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Keep the two model files alongside app.py. Only load pickle/joblib files from trusted sources. Saved scikit-learn objects may need the same library version as the original training environment; that version has not been verified here.

## Scope and limitations

This repository contains the application and saved models. The original training notebook is not included in the supplied folder. No independently verified accuracy result is claimed. The model can struggle with unfamiliar vocabulary and overlapping subjects. It is a learning project, not a scientific judgement of a paper.

## Development

Built as a learning project with AI assistance. Review and understand the code before presenting it as evidence of your skills.
