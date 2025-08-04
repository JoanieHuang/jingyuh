import streamlit as st
import requests

# Title
st.title("Headline Sentiment Analyzer")

# Input area
st.subheader("Enter your headlines (one per line):")
headline_text = st.text_area("Headlines", height=200)

# Button
if st.button("Analyze Sentiment"):
    headlines = [line.strip() for line in headline_text.split('\n') if line.strip()]
    
    if not headlines:
        st.warning("Please enter at least one headline.")
    else:
        try:
            # Call your local API
            response = requests.post(
                "http://localhost:8008/score_headlines",
                json={"headlines": headlines}
            )
            if response.status_code == 200:
                labels = response.json().get("labels", [])
                for h, l in zip(headlines, labels):
                    st.write(f" **{h}** → _{l}_")
            else:
                st.error("Error from API: " + response.text)
        except Exception as e:
            st.error(f"Could not reach API: {e}")
