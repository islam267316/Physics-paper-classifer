# Physics Paper Classifier

A Streamlit app that classifies paper abstracts using TF-IDF and Logistic Regression.

The broad-category dataset supports **Astrophysics, Condensed Matter, Quantum Physics, and Other Physics**. Retrain the model to use these labels.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Train

Open `train_model.ipynb` in VS Code. Use `physics_papers_broad_categories.csv`, remove blank abstracts, and run the training cells. Save the fitted pipeline:

```python
import joblib
joblib.dump(grid.best_estimator_, "physics_classifier.joblib")
```

## Run

Keep `physics_classifier.joblib` beside `app.py`. The app must load this pipeline and predict directly from the abstract.

```bash
python3 -m streamlit run app.py
```

Open the Local URL shown in the terminal.

## Dataset limitations

The updated CSV preserves 210 original rows and adds six clearly labelled synthetic Quantum Physics paraphrases. Exclude the 50 blank abstracts before training. The broad dataset then has 166 usable examples and supports the checked 80/20 stratified split and three-fold CV with `random_state=42`.

**Do not report scores from this augmented dataset as real-paper generalization.** Quantum Physics and Other Physics have very few usable examples; collect more real abstracts for reliable evaluation.
