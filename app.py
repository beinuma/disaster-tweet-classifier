import streamlit as st
import pandas as pd
import torch
import re

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "beibei4/disaster-tweet-classifier"

@st.cache_resource
def load_model():
    """Load tokenizer + model once and cache them."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()


# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)   # remove URLs
    text = re.sub(r'@\w+', '', text)                     # remove mentions
    text = re.sub(r'#', '', text)                        # remove hash symbol
    text = re.sub(r'[^a-zA-Z\s]', '', text)              # keep only letters
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_tweet(text: str):
    cleaned = clean_text(text)

    encoded = tokenizer(
        cleaned,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids      = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    probs      = torch.softmax(outputs.logits, dim=1)[0]
    prediction = torch.argmax(probs).item()
    confidence = probs[prediction].item() * 100
    label      = "🚨 Disaster" if prediction == 1 else "✅ Non-Disaster"

    return label, confidence


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Disaster Tweet Classifier", page_icon="🚨")
st.title("🚨 Disaster Tweet Classifier")
st.caption("Powered by DistilBERT · Fine-tuned on disaster tweets")

tab1, tab2 = st.tabs(["Single Tweet", "Bulk Classification"])

# ── Tab 1 : single prediction ─────────────────────────────────────────────────
with tab1:
    tweet = st.text_area("Enter a tweet to classify", height=120)

    if st.button("Predict", type="primary"):
        if not tweet.strip():
            st.warning("Please enter a tweet first.")
        else:
            with st.spinner("Classifying…"):
                label, confidence = predict_tweet(tweet)

            color = "red" if "Disaster" in label and "Non" not in label else "green"
            st.markdown(
                f"### Result: :{color}[{label}]"
            )
            st.progress(int(confidence))
            st.metric("Confidence", f"{confidence:.2f}%")

            with st.expander("See cleaned text"):
                st.code(clean_text(tweet))


# ── Tab 2 : bulk prediction ───────────────────────────────────────────────────
with tab2:
    st.markdown(
        "Upload a CSV file with a column named **`text`** containing one tweet per row."
    )
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error("❌ CSV must contain a column named 'text'. "
                     f"Found columns: {list(df.columns)}")
        else:
            st.info(f"Loaded **{len(df)}** tweets. Starting classification…")
            progress_bar = st.progress(0)
            status_text  = st.empty()

            predictions, confidences = [], []
            total = len(df)

            for i, tweet in enumerate(df["text"]):
                label, confidence = predict_tweet(str(tweet))
                predictions.append(label)
                confidences.append(round(confidence, 2))

                # Update progress
                progress_bar.progress((i + 1) / total)
                status_text.text(f"Classified {i + 1} / {total} tweets")

            status_text.success("✅ Classification complete!")

            df["Prediction"]  = predictions
            df["Confidence %"] = confidences

            # Summary stats
            disaster_count     = sum(1 for p in predictions if "Non" not in p)
            non_disaster_count = total - disaster_count

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Tweets",    total)
            col2.metric("🚨 Disaster",     disaster_count)
            col3.metric("✅ Non-Disaster", non_disaster_count)

            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv,
                file_name="classified_tweets.csv",
                mime="text/csv",
                type="primary",
            )