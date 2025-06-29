import subprocess

print("🚦 Starting TrafficTelligence Pipeline...\n")

# Step 1: Generate Data
print("📁 Step 1: Generating Synthetic Data...")
subprocess.run(["python", "scripts/generate_data.py"])

# Step 2: Preprocess Data
print("\n🧹 Step 2: Preprocessing Data...")
subprocess.run(["python", "scripts/preprocess.py"])

# Step 3: Train Model
print("\n🤖 Step 3: Training Model...")
subprocess.run(["python", "scripts/train_model.py"])

print("\n✅ All steps completed successfully!")
