import tkinter as tk
from tkinter import simpledialog, messagebox
import subprocess
import sys
import os

def run_pipeline_for_student(name):
    """Run the face registration pipeline for a student"""
    scripts = [
        "1_capture_images.py",
        "2_crop_faces.py", 
        "3_generate_embeddings.py",
        "insert_embedding.py"
    ]
    
    print(f"🚀 Starting face registration for: {name}")
    
    for script in scripts:
        script_path = os.path.join("python_scripts", script)
        if os.path.exists(script_path):
            print(f"▶️ Running {script}...")
            try:
                result = subprocess.run([
                    sys.executable, script_path, name
                ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
                
                if result.returncode == 0:
                    print(f"✅ {script} completed")
                    print(result.stdout)
                else:
                    print(f"❌ {script} failed: {result.stderr}")
                    return False
            except Exception as e:
                print(f"❌ Error running {script}: {e}")
                return False
        else:
            print(f"❌ Script not found: {script_path}")
            return False
    
    print(f"🎉 Face registration completed for {name}!")
    return True

def main():
    # Create simple GUI
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    # Ask for student name
    name = simpledialog.askstring("Face Registration", "Enter student name:")
    
    if not name:
        messagebox.showwarning("Cancelled", "Registration cancelled.")
        return
    
    # Run the pipeline
    success = run_pipeline_for_student(name)
    
    if success:
        messagebox.showinfo("Success", f"Face registration completed for {name}!")
    else:
        messagebox.showerror("Error", f"Face registration failed for {name}. Check console for details.")
    
    root.destroy()

if __name__ == "__main__":
    main()
