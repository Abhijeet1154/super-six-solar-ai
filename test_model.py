from ultralytics import YOLO

# 1. Load the model
model = YOLO("hotspot_model.pt")
print("Hotspot model loaded! Analyzing...")

# 2. Run prediction WITHOUT auto-saving yet (removed save=True)
results = model.predict(source="test_panel.jpg", conf=0.25)

# 3. Intercept the result, force the new name, and save manually
for result in results:
    # This overwrites every single class name in the model's memory to "Hotspot"
    result.names = {key: "Hotspot" for key in result.names.keys()}
    
    # Save the newly labeled image directly to your main project folder
    result.save(filename="hotspot_result_fixed.jpg")

print("\n--- Success! ---")
print("Look in your main VS Code folder for 'hotspot_result_fixed.jpg'")

# use this prompt inn ternminal for runn the webpage python -m streamlit run app.py