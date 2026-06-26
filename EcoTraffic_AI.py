# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 23:46:08 2026

@author: Rasulov Sh
"""

import streamlit as st
import cv2
import numpy as np
import time
import csv
import os
import pandas as pd
from ultralytics import YOLO

# Sarlavha va sahifa sozlamalari
st.set_page_config(page_title="Smart Traffic & ECO System", layout="wide")
st.title("🌱 Smart Traffic Management & ECO Monitoring System")
st.markdown("### PhD Research Project: Computer Vision based Vehicle Delay and Emission Tracking")

# 1. Modelni yuklaymiz (Keshlanadi, har safar qayta yuklanmasligi uchun)
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

model = load_yolo_model()

# Yon panel (Sidebar) sozlamalari
st.sidebar.header("⚙️ Tizim Sozlamalari")
dist_meters = st.sidebar.slider("Ikki chiziq orasi masofasi (Metrda):", min_value=5.0, max_value=100.0, value=25.0, step=1.0)
conf_threshold = st.sidebar.slider("YOLO Ishonchlilik chegarasi (Confidence):", min_value=0.1, max_value=1.0, value=0.35, step=0.05)

# Fayl yuklash tugmasi
uploaded_file = st.file_uploader("💻 Video faylni yuklang (MP4 formatda):", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Yuklangan videoni vaqtinchalik saqlaymiz
    tfile = open("temp_video.mp4", "wb")
    tfile.write(uploaded_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture("temp_video.mp4")
    
    # Koordinatalar
    LINE_ENTRY_Y = 90    
    LINE_EXIT_Y = 590    

    # Xotira va o'zgaruvchilar
    entry_times = {}     
    waiting_times = []   
    total_fuel_wasted = 0.0  
    total_co2_emitted = 0.0  
    
    # CSV uchun ro'yxat (Streamlit-da fayl yopilishini kutmasdan ma'lumot saqlash uchun DataFrame ishlatamiz)
    data_records = []

    # Streamlit vizual elementlarini tayyorlash
    st.info("Video qayta ishlanmoqda... Iltimos kuting.")
    
    # Jonli video va statistika panellari uchun joy ajratamiz (Columns)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🎥 Jonli Video Oqimi")
        video_placeholder = st.empty() # Video kadrbama-kadr shu yerda yangilanadi
        
    with col2:
        st.markdown("#### 📊 Jonli Statistika")
        stat_cnt = st.empty()
        stat_time = st.empty()
        stat_fuel = st.empty()
        stat_co2 = st.empty()
        stat_speed = st.empty()

    # Videoni o'qish sikli
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        height, width, _ = frame.shape
        current_time_now = time.time()
        
        # 24 soatlik xotira filtri
        for tid, t_start in list(entry_times.items()):
            if current_time_now - t_start > 180:
                del entry_times[tid]
        
        # YOLO kuzatuvi
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=conf_threshold, iou=0.45)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()                
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)  
            clss = results[0].boxes.cls.cpu().numpy().astype(int)      
            
            for box, track_id, cls in zip(boxes, track_ids, clss):
                if cls in [2, 5, 7]: 
                    x1, y1, x2, y2 = box
                    y_center = int((y1 + y2) / 2) 
                    
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    if LINE_ENTRY_Y <= y_center <= (LINE_ENTRY_Y + 30):
                        if track_id not in entry_times:
                            entry_times[track_id] = current_time_now
                    
                    if y_center >= LINE_EXIT_Y:
                        if track_id in entry_times:
                            start_time = entry_times[track_id]
                            duration = current_time_now - start_time
                            
                            if duration > 0.5:
                                waiting_times.append(duration)
                                speed_kmh = (dist_meters / duration) * 3.6         
                                coef = 0.0003 if cls == 2 else 0.0007  
                                fuel_wasted = duration * coef
                                co2_emitted = fuel_wasted * 2.3
                                
                                total_fuel_wasted += fuel_wasted
                                total_co2_emitted += co2_emitted
                                
                                tur_nomi = "Car" if cls == 2 else ("Bus" if cls == 5 else "Truck")
                                k_vaqt = time.strftime('%H:%M:%S', time.localtime(start_time))
                                ch_vaqt = time.strftime('%H:%M:%S', time.localtime(current_time_now))
                                
                                # Rekordni saqlaymiz
                                data_records.append([track_id, tur_nomi, k_vaqt, ch_vaqt, round(duration, 2), round(speed_kmh, 1), round(fuel_wasted, 4), round(co2_emitted, 4)])
                                
                                # Jonli panellarni yangilash
                                stat_speed.metric("Oxirgi Tezlik", f"{speed_kmh:.1f} km/h")
                            
                            del entry_times[track_id]
        
        # Chiziqlarni chizish
        cv2.line(frame, (0, LINE_ENTRY_Y), (width, LINE_ENTRY_Y), (0, 255, 255), 3)  
        cv2.line(frame, (0, LINE_EXIT_Y), (width, LINE_EXIT_Y), (0, 0, 255), 3)      
        
        # Streamlit orqali kadrni ko'rsatish (BGR to RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Yon statistikani yangilab borish
        if waiting_times:
            stat_cnt.metric("Sanalgan Transportlar", f"{len(waiting_times)} ta")
            stat_time.metric("O'rtacha Kutish Vaqti", f"{sum(waiting_times)/len(waiting_times):.2f} sek")
            stat_fuel.metric("Isrof bo'lgan Yoqilg'i", f"{total_fuel_wasted:.3f} Litr")
            stat_co2.metric("CO2 Emissiya hajmi", f"{total_co2_emitted:.3f} kg")

    cap.release()
    st.success("🎉 Video tahlili yakunlandi!")
    
    # --- YAKUNIY JADVAL VA YUKLAB OLISH (DOWNLOAD) ---
    if data_records:
        st.markdown("### 📋 Eksperiment Natijalari Jadvali")
        df = pd.DataFrame(data_records, columns=["Mashina_ID", "Transport_Turi", "Kirish_Vaqti", "Chiqish_Vaqti", "Kutish_Vaqti_Sekund", "O_rtacha_Tezlik_KMH", "Isrof_Yoqilgi_Litr", "CO2_Emissiya_Kg"])
        st.dataframe(df)
        
        # CSV yuklab olish tugmasi
        csv_buffer = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Excel/CSV formatda yuklab olish",
            data=csv_buffer,
            file_name="trafik_ekologik_natijalari.csv",
            mime="text/csv"
        )
        
    # Vaqtinchalik faylni o'chiramiz
    if os.path.exists("temp_video.mp4"):
        os.remove("temp_video.mp4")