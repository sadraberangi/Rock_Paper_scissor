# 🪨📄✂️ Rock Paper Scissors - AI Powered

An AI-powered **two-player Rock Paper Scissors game** using **YOLOv11** for hand gesture recognition and **MediaPipe** for face detection.

---

## 🎮 Game Stages

1. **GO Rock**  
   Both players must start with the **rock** gesture.
   
   ![Display](https://github.com/sadraberangi/Rock_Paper_scissor/blob/main/readme_assets/Screenshot%202025-02-15%20105357.png?raw=true)

2. **Counting (⏳ 3 Seconds Countdown)**  
   - A **3-second countdown** starts.  
   - **Players must not change their gesture** during this phase.

3. **Playing**  
   - Players **switch to their desired gesture** after the countdown ends.

4. **Winning Condition 👑**  
   - A player wins if they score the required points.  
   - A **crown appears above the winner’s face**.

5. **Cheating Detection 🚨**  
   - If a player **changes their gesture** during the countdown or **after 2 seconds** in the playing phase, a **red mask** appears on their face to indicate cheating.

---

## 🛠️ Machine Learning & Vision Tasks

✅ **Face Detection**: Implemented using **Google's MediaPipe** for accurate facial tracking.  
✅ **Hand Gesture Detection**:  
   - Powered by **YOLOv11**, fine-tuned specifically for **Rock Paper Scissors**.  
   - Model trained using a custom dataset: [YOLOv11 Rock Paper Scissors Detection](https://github.com/Gholamrezadar/yolo11-rock-paper-scissors-detection).  

---

## 🚀 Future Improvements

- Improve **gesture detection accuracy** with a **larger dataset**.  
- Implement **real-time leaderboard tracking**.  
- Add **multi-player online support**.

---
