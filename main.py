from app.predict import predict_message


message = input("Enter message: ")

result, confidence = predict_message(message)

print("\nPrediction:", result)
print("Confidence:", round(confidence * 100, 2), "%")