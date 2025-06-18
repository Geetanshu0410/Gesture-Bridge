# evaluate.py

from ultralytics import YOLO
import os

def evaluate_model(model_path="C:/Users/singh/Documents/.vscode/Minor - 2/AirCanvas/aircanvas/src/main.py"):
    try:
        # Load YOLOv8 model
        model = YOLO(model_path)
        print("✅ Model loaded successfully.")

        # Run evaluation (uses the data.yaml from training)
        metrics = model.val()

        # YOLO saves results in this directory
        run_dir = "runs/detect/val"
        if not os.path.exists(run_dir):
            print("⚠️ Validation directory not found. Ensure training was completed.")
            return

        # Get latest evaluation folder
        last_run = sorted(os.listdir(run_dir))[-1]
        full_path = os.path.abspath(os.path.join(run_dir, last_run))

        print("\n📁 Evaluation completed!")
        print("📊 Evaluation metrics and plots saved in:", full_path)

        # Print out core evaluation metrics
        print("\n--- Evaluation Metrics ---")
        print(f"🔸 Precision     : {metrics.box.precision:.4f}")
        print(f"🔸 Recall        : {metrics.box.recall:.4f}")
        print(f"🔸 F1 Score      : {metrics.box.f1:.4f}")
        print(f"🔸 mAP@0.5       : {metrics.box.map50:.4f}")
        print(f"🔸 mAP@0.5:0.95  : {metrics.box.map:.4f}")

    except Exception as e:
        print(f"\n❌ Evaluation Error: {e}")

if __name__ == "__main__":
    evaluate_model("C:/Users/singh/Documents/.vscode/Minor - 2/AirCanvas/aircanvas/src/main.py")
