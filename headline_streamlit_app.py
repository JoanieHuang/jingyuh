import streamlit as st
import requests

# App title
st.title("Headline Sentiment Analyzer")

# Input section
st.subheader("Enter your headlines (one per line):")
headline_text = st.text_area("Headlines", height=200)

# Analyze button
if st.button("Analyze Sentiment"):
    # Split input text into individual headlines, ignoring empty lines
    headlines = [line.strip() for line in headline_text.split('\n') if line.strip()]
    
    if not headlines:
        st.warning("Please enter at least one headline.")
    else:
        try:
            # Send headlines to the local API for sentiment scoring
            response = requests.post(
                "http://localhost:8008/score_headlines",
                json={"headlines": headlines}
            )
            if response.status_code == 200:
                # Retrieve sentiment labels from API response
                labels = response.json().get("labels", [])
                # Display each headline with its corresponding sentiment label
                for h, l in zip(headlines, labels):
                    st.write(f"**{h}** → _{l}_")
            else:
                st.error("API returned an error: " + response.text)
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")