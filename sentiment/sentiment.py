import ast
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

df = pd.read_csv("data/emotion_words_checked.csv")
df_t = pd.read_csv("data/Flow_current.csv")

# Ensure ID types match
df["ID"] = df["ID"].astype(str)
df_t["ID"] = df_t["ID"].astype(str)

# Clean column names in df_t (strip whitespace / NBSP)
df_t.columns = df_t.columns.str.strip()

# Check what the condition column is actually called
# print(df_t.columns)  # uncomment to inspect
# Assume it is now "Exp_Condition" after stripping:
condition_col = "Exp_Condition" if "Exp_Condition" in df_t.columns else "Exp_Condition "

# Merge to reliably attach condition to each row in df
df = df.merge(df_t[["ID", condition_col]], on="ID", how="left")

def remove_not_words(cell):
    try:
        words = ast.literal_eval(cell)
        cleaned = [w for w in words if not w.startswith("not_")]
        return cleaned
    except Exception:
        return cell

df["text"] = df["text"].apply(remove_not_words).astype(str)

df['scores'] = df['text'].apply(lambda text: analyzer.polarity_scores(text))

df['compound']  = df['scores'].apply(lambda score_dict: score_dict['compound'])

df.to_csv("sentiment/vader_sentiment_emotional_only.csv", index=False)
