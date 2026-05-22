import streamlit as st

from app.predict import predict_message


st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩"
)

st.title("📩 Spam Email Detector")

st.write("AI-powered spam/ham classifier")

message = st.text_area("Enter your message")


if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message")

    else:
        result, confidence = predict_message(message)

        st.success("Message analyzed successfully!")

        st.subheader(f"Prediction: {result}")

        st.write(
            f"Confidence: {round(confidence * 100, 2)}%"
        )

        if result == "SPAM":
            st.error("Spam Detected!!!")
        else:
            st.success("Legitimate Message. ✅")