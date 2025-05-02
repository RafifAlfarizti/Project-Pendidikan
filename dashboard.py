import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ---------------------
# Load dan Preprocess
# ---------------------
st.set_page_config(page_title="Dashboard Dropout Mahasiswa", layout="wide")
st.sidebar.title("📊 Dashboard Dropout Mahasiswa")
page = st.sidebar.radio("Pilih Halaman", ["Overview", "Visualisasi", "Prediksi", "Rekomendasi"])

# Load dataset
ip = pd.read_csv('data.csv', delimiter=';')

# Preprocessing dasar
ip.dropna(inplace=True)
ip.drop_duplicates(inplace=True)

# Label encoding jika dibutuhkan
for col in ip.select_dtypes(include='object').columns:
    ip[col] = LabelEncoder().fit_transform(ip[col].astype(str))

# --------------------------------
# HALAMAN 1: OVERVIEW (Revisi)
# --------------------------------
if page == "Overview":
    st.title("🎓 Overview Mahasiswa")

    # Mapping label Target
    label_target = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
    ip['Status'] = ip['Target'].map(label_target)

    # Mapping Course ID ke nama jurusan
    course_mapping = {
        33: "Biofuel Production Technologies",
        171: "Animation and Multimedia Design",
        8014: "Social Service (evening attendance)",
        9003: "Agronomy",
        9070: "Communication Design",
        9085: "Veterinary Nursing",
        9119: "Informatics Engineering",
        9130: "Equinculture",
        9147: "Management",
        9238: "Social Service",
        9254: "Tourism",
        9500: "Nursing",
        9556: "Oral Hygiene",
        9670: "Advertising and Marketing Management",
        9773: "Journalism and Communication",
        9853: "Basic Education",
        9991: "Management (evening attendance)"
    }
    ip['Course Name'] = ip['Course'].map(course_mapping)

    # Metric info
    total_mhs = len(ip)
    dropout_rate = (ip['Target'] == 0).mean() * 100
    col1, col2 = st.columns(2)
    col1.metric("Total Mahasiswa", total_mhs)
    col2.metric("Dropout Rate", f"{dropout_rate:.2f}%")

    # Visualisasi Distribusi berdasarkan Course
    st.markdown("### 📊 Distribusi Dropout Berdasarkan Program Studi")
    fig1 = px.histogram(ip, x='Course Name', color='Status', barmode='group',
                        labels={'Status': 'Status Mahasiswa', 'count': 'Jumlah'}, height=500)
    fig1.update_layout(xaxis_title='Program Studi', yaxis_title='Jumlah Mahasiswa')
    st.plotly_chart(fig1)

    st.info("""
    **Insight:**  
    - Beberapa program studi seperti *Informatics Engineering*, *Nursing*, dan *Management* memiliki jumlah dropout tinggi.
    - Program seperti *Biofuel Production* dan *Oral Hygiene* relatif lebih kecil secara populasi.
    - Visual ini membantu institusi mengidentifikasi jurusan mana yang perlu perhatian khusus dalam hal retensi mahasiswa.
    """)

    # Tabel Rata-rata nilai dan usia
    st.markdown("### 📈 Rata-rata Nilai & Usia berdasarkan Status Mahasiswa")
    stats_avg = ip.groupby('Status')[['Age at enrollment', 'Admission grade', 'Curricular units 1st sem (grade)']].mean().round(2)
    st.dataframe(stats_avg)

    st.info("""
    **Insight:**  
    - Mahasiswa yang **dropout** cenderung memiliki nilai lebih rendah pada semester awal dan usia lebih tua saat masuk.
    - Mahasiswa yang **graduate** memiliki performa akademik semester awal yang lebih baik.
    - Institusi dapat menjadikan nilai semester pertama sebagai indikator awal untuk melakukan intervensi.
    """)


# --------------------------------
# HALAMAN 2: VISUALISASI (Final Revisi)
# --------------------------------
elif page == "Visualisasi":
    st.title("📈 Visualisasi Performa & Demografi Mahasiswa")

    # Mapping Target menjadi label kategori
    label_target = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
    ip['Status'] = ip['Target'].map(label_target)

    # 1. Korelasi Numerik terhadap Target
    st.subheader("📊 Korelasi Fitur Numerik terhadap Status Mahasiswa")
    corr = ip.corr(numeric_only=True)['Target'].drop('Target').sort_values()
    fig1, ax1 = plt.subplots(figsize=(8, 12))
    corr.plot(kind='barh', ax=ax1, color='skyblue')
    ax1.set_title("Korelasi terhadap Target (Dropout/Graduate/Enrolled)")
    ax1.set_xlabel("Nilai Korelasi")
    ax1.set_ylabel("Fitur")
    st.pyplot(fig1)

    st.info("""
    **Insight:**  
    - Fitur dengan korelasi positif tertinggi terhadap kelulusan adalah `Curricular units (approved)` dan `Tuition fees up to date`.
    - Fitur negatif tertinggi terhadap kelulusan (positif terhadap dropout): `Debtor`, `Gender`, `Application mode`, `Age at enrollment`.
    """)

    # 2. Mapping kode course ke nama jurusan
    course_mapping = {
        33: "Biofuel Production Technologies",
        171: "Animation and Multimedia Design",
        8014: "Social Service (evening attendance)",
        9003: "Agronomy",
        9070: "Communication Design",
        9085: "Veterinary Nursing",
        9119: "Informatics Engineering",
        9130: "Equinculture",
        9147: "Management",
        9238: "Social Service",
        9254: "Tourism",
        9500: "Nursing",
        9556: "Oral Hygiene",
        9670: "Advertising and Marketing Management",
        9773: "Journalism and Communication",
        9853: "Basic Education",
        9991: "Management (evening attendance)"
    }
    ip['Course Name'] = ip['Course'].map(course_mapping)

    # 3. Distribusi Mahasiswa berdasarkan Course (Bar Chart)
    st.subheader("🏫 Distribusi Mahasiswa Berdasarkan Program Studi")
    course_count_df = ip['Course Name'].value_counts().reset_index()
    course_count_df.columns = ['Course Name', 'Count']

    fig2 = px.bar(course_count_df, x='Course Name', y='Count',
                  labels={'Course Name': 'Program Studi', 'Count': 'Jumlah Mahasiswa'},
                  color='Course Name', color_discrete_sequence=px.colors.qualitative.Set3)
    fig2.update_layout(title='Jumlah Mahasiswa per Program Studi',
                       xaxis_tickangle=-45, showlegend=False)
    st.plotly_chart(fig2)

    st.info("""
    **Insight:**  
    - Program studi seperti **Informatics Engineering**, **Management**, dan **Nursing** memiliki jumlah mahasiswa tertinggi.
    - Visual ini berguna untuk mengidentifikasi jurusan dengan skala besar yang mungkin perlu prioritas dalam kebijakan retensi.
    """)

    # 4. Proporsi Dropout/Enrolled/Graduate (Pie Chart)
    st.subheader("📊 Proporsi Status Mahasiswa (Dropout vs Enrolled vs Graduate)")
    status_pie_df = ip['Status'].value_counts().reset_index()
    status_pie_df.columns = ['Status', 'Count']
    fig3 = px.pie(status_pie_df, values='Count', names='Status',
                  title='Proporsi Mahasiswa Berdasarkan Status',
                  color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig3)

    st.info("""
    **Insight:**  
    - Mahasiswa yang **lulus (graduate)** mendominasi populasi, namun sekitar **32% mengalami dropout**.
    - Pie chart ini menunjukkan kebutuhan untuk terus meningkatkan sistem pendampingan dan evaluasi akademik.
    """)

# -------------------------------- 
# HALAMAN 3: PREDIKSI (SVM) 
# -------------------------------- 
elif page == "Prediksi":
    st.title("🧠 Prediksi Dropout Mahasiswa (SVM)")
    
    # Menambahkan deskripsi untuk setiap filter
    st.sidebar.markdown("### Parameter Prediksi")
    st.sidebar.markdown("Sesuaikan parameter di bawah untuk memprediksi kemungkinan dropout mahasiswa:")
    
    feature_cols = ['Age at enrollment', 'Admission grade', 'Scholarship holder', 'Curricular units 1st sem (grade)', 'Tuition fees up to date']
    
    feature_descriptions = {
        'Age at enrollment': 'Usia mahasiswa saat mendaftar (dalam tahun). Rentang usia biasanya antara 17-70 tahun.',
        'Admission grade': 'Nilai ujian masuk mahasiswa. Semakin tinggi nilai menunjukkan performa akademik yang lebih baik (skala 95-190).',
        'Scholarship holder': 'Status beasiswa mahasiswa (0 = tidak menerima beasiswa, 1 = menerima beasiswa).',
        'Curricular units 1st sem (grade)': 'Rata-rata nilai akademik pada semester pertama (skala 0-18.88).',
        'Tuition fees up to date': 'Status pembayaran biaya kuliah (0 = terlambat/belum membayar, 1 = tepat waktu/sudah membayar).'
    }
    
    # Input slider dengan deskripsi
    input_data = {}
    for col in feature_cols:
        st.sidebar.markdown(f"**{col}**")
        st.sidebar.markdown(f"*{feature_descriptions[col]}*")
        input_data[col] = st.sidebar.slider(
            label="",
            min_value=float(ip[col].min()),
            max_value=float(ip[col].max()),
            value=float(ip[col].mean()),
            key=col
        )
        st.sidebar.markdown("---")
    
    input_df = pd.DataFrame([input_data])
    
    # Model pipeline
    X = ip[feature_cols]
    y = ip['Target']
    
    scaler = StandardScaler()
    svm_model = SVC(probability=True)
    svm_pipeline = Pipeline([('scaler', scaler), ('svm', svm_model)])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    svm_pipeline.fit(X_train, y_train)
    
    # Simpan model menggunakan joblib
    import joblib
    joblib.dump(svm_pipeline, 'dropout_prediction_svm_model.joblib')
    
    # Prediksi
    pred = svm_pipeline.predict(input_df)[0]
    prob = svm_pipeline.predict_proba(input_df)[0][1]
    
    status = "TIDAK Dropout" if pred == 1 else "Dropout"
    st.success(f"🧾 Prediksi: {status} dengan probabilitas {prob:.2f}")
    
    # Tambahkan insight berdasarkan input yang diberikan
    st.markdown("### Insight Prediksi")
    
    insight_container = st.container()
    with insight_container:
        st.markdown(f"""
        **Analisis Parameter Mahasiswa:**
        
        Berdasarkan parameter yang diatur:
        - Usia Mahasiswa: **{input_data['Age at enrollment']:.2f}** tahun
        - Nilai Masuk: **{input_data['Admission grade']:.2f}**
        - Status Beasiswa: **{"Menerima" if input_data['Scholarship holder'] >= 0.5 else "Tidak Menerima"}**
        - Nilai Semester 1: **{input_data['Curricular units 1st sem (grade)']:.2f}**
        - Status Pembayaran: **{"Tepat Waktu" if input_data['Tuition fees up to date'] >= 0.5 else "Terlambat/Belum"}**
        
        **Hasil Analisis:**
        """)
        
        # Insight dinamis berdasarkan probabilitas dropout
        if prob < 0.3:
            st.markdown(f"""
            Mahasiswa ini memiliki **risiko dropout rendah** ({prob:.2f}). Beberapa faktor yang mungkin mempengaruhi:
            
            {"- Nilai semester pertama yang baik" if input_data['Curricular units 1st sem (grade)'] > 15 else ""}
            {"- Nilai masuk yang tinggi" if input_data['Admission grade'] > 150 else ""}
            {"- Status pembayaran tepat waktu" if input_data['Tuition fees up to date'] > 0.5 else ""}
            {"- Memiliki beasiswa" if input_data['Scholarship holder'] > 0.5 else ""}
            """)
        elif prob < 0.7:
            st.markdown(f"""
            Mahasiswa ini memiliki **risiko dropout sedang** ({prob:.2f}). Area yang perlu diperhatikan:
            
            {"- Nilai semester pertama cukup rendah" if input_data['Curricular units 1st sem (grade)'] < 12 else ""}
            {"- Nilai masuk di bawah rata-rata" if input_data['Admission grade'] < 130 else ""}
            {"- Status pembayaran perlu diperhatikan" if input_data['Tuition fees up to date'] < 0.5 else ""}
            {"- Tidak memiliki beasiswa" if input_data['Scholarship holder'] < 0.5 else ""}
            """)
        else:
            st.markdown(f"""
            Mahasiswa ini memiliki **risiko dropout tinggi** ({prob:.2f}). Faktor risiko utama:
            
            {"- Nilai semester pertama sangat rendah" if input_data['Curricular units 1st sem (grade)'] < 8 else ""}
            {"- Nilai masuk rendah" if input_data['Admission grade'] < 110 else ""}
            {"- Masalah dalam pembayaran biaya kuliah" if input_data['Tuition fees up to date'] < 0.5 else ""}
            {"- Tidak memiliki dukungan beasiswa" if input_data['Scholarship holder'] < 0.5 else ""}
            """)
        
        st.markdown("""
        **Rekomendasi:**
        """)
        
        if prob < 0.3:
            st.markdown("Mahasiswa ini memiliki prospek yang baik untuk menyelesaikan studi. Tetap pantau perkembangan akademiknya.")
        elif prob < 0.7:
            st.markdown("Pertimbangkan untuk memberikan pendampingan akademik dan konseling keuangan jika diperlukan.")
        else:
            st.markdown("Mahasiswa ini membutuhkan intervensi segera. Rekomendasikan program bimbingan intensif dan evaluasi dukungan finansial.")


# --------------------------------
# HALAMAN 4: REKOMENDASI
# --------------------------------
elif page == "Rekomendasi":
    st.title("📌 Rekomendasi Strategis")
    
    st.markdown("""
    Berdasarkan hasil analisis dan prediksi, berikut rekomendasi untuk mengurangi risiko dropout:
    """)
    
    # Tambahkan filter untuk melihat data yang menjadi objek rekomendasi
    st.sidebar.markdown("### Filter Data")
    st.sidebar.markdown("Gunakan filter berikut untuk menyesuaikan visualisasi dan rekomendasi:")
    
    # Filter berdasarkan kriteria risiko
    risk_category = st.sidebar.radio(
        "Kategori Risiko Dropout:",
        ["Semua Mahasiswa", "Risiko Tinggi", "Risiko Sedang", "Risiko Rendah"]
    )
    
    # Load model yang telah disimpan (jika ada)
    try:
        import joblib
        svm_pipeline = joblib.load('dropout_prediction_svm_model.joblib')
        has_model = True
    except:
        has_model = False
    
    # Tambahkan prediksi probabilitas dropout ke dataframe jika model tersedia
    if has_model:
        # Fitur yang digunakan untuk prediksi
        feature_cols = ['Age at enrollment', 'Admission grade', 'Scholarship holder', 
                        'Curricular units 1st sem (grade)', 'Tuition fees up to date']
        
        # Prediksi probabilitas dropout untuk semua mahasiswa
        dropout_probs = svm_pipeline.predict_proba(ip[feature_cols])[:, 1]
        
        # Tambahkan kolom probabilitas dropout
        ip['Dropout Probability'] = dropout_probs
        
        # Kategorikan berdasarkan probabilitas
        ip['Risk Category'] = pd.cut(
            ip['Dropout Probability'], 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=['Risiko Rendah', 'Risiko Sedang', 'Risiko Tinggi']
        )
    
    # Filter data berdasarkan kategori risiko yang dipilih
    filtered_data = ip.copy()
    if risk_category != "Semua Mahasiswa" and has_model:
        filtered_data = ip[ip['Risk Category'] == risk_category]
    
    # Tampilkan jumlah mahasiswa terfilter
    st.sidebar.markdown(f"**Jumlah mahasiswa terfilter:** {len(filtered_data)}")
    
    # Tambahkan filter untuk course/mata kuliah
    available_courses = ['Semua Course', 'Teknologi', 'Ekonomi', 'Kesehatan', 'Seni', 'Hukum']
    if 'Course' not in ip.columns:
        # Simulasi data course jika tidak ada
        np.random.seed(42)
        ip['Course'] = np.random.choice(available_courses[1:], size=len(ip))
        filtered_data['Course'] = np.random.choice(available_courses[1:], size=len(filtered_data))
    
    selected_course = st.sidebar.selectbox(
        "Program Studi:",
        available_courses
    )
    
    if selected_course != 'Semua Course':
        filtered_data = filtered_data[filtered_data['Course'] == selected_course]
        st.sidebar.markdown(f"**Jumlah mahasiswa dalam program {selected_course}:** {len(filtered_data)}")
    
    # Tampilkan rekomendasi strategis dalam 4 kolom
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    with col1:
        st.markdown("### 💸 Evaluasi Pembayaran")
        payment_data = filtered_data.groupby('Tuition fees up to date')['Target'].value_counts().unstack().fillna(0)
        if not payment_data.empty:
            payment_fig = px.pie(
                names=['Belum Bayar', 'Sudah Bayar'], 
                values=payment_data.iloc[:, 0] if 0 in payment_data.columns else [0, 0],
                title="Status Pembayaran Mahasiswa Dropout"
            )
            st.plotly_chart(payment_fig, use_container_width=True)
            
            # Insight untuk pembayaran
            pct_unpaid = filtered_data[filtered_data['Tuition fees up to date'] == 0].shape[0] / max(1, len(filtered_data))
            st.markdown(f"""
            **Insight:**
            - {pct_unpaid:.1%} mahasiswa memiliki status pembayaran terlambat/belum membayar
            - Mahasiswa dengan status pembayaran belum selesai memiliki risiko dropout lebih tinggi
            
            **Rekomendasi:**
            - Buat sistem peringatan dini untuk pembayaran
            - Tawarkan opsi pembayaran fleksibel untuk mahasiswa berisiko
            """)
    
    with col2:
        st.markdown("### 🎓 Intervensi Akademik")
        grade_fig = px.histogram(
            filtered_data, 
            x='Curricular units 1st sem (grade)', 
            color='Target',
            barmode='overlay',
            title="Distribusi Nilai Semester 1"
        )
        st.plotly_chart(grade_fig, use_container_width=True)
        
        # Insight untuk nilai akademik
        avg_grade = filtered_data['Curricular units 1st sem (grade)'].mean()
        risk_threshold = filtered_data['Curricular units 1st sem (grade)'].quantile(0.25)
        st.markdown(f"""
        **Insight:**
        - Rata-rata nilai semester 1: {avg_grade:.2f}
        - Nilai di bawah {risk_threshold:.2f} berpotensi tinggi untuk dropout
        
        **Rekomendasi:**
        - Program bimbingan akademik untuk mahasiswa dengan nilai < {risk_threshold:.2f}
        - Sesi tambahan untuk mata kuliah dengan tingkat kesulitan tinggi
        """)
    
    with col3:
        st.markdown("### 🎯 Alokasi Beasiswa")
        scholarship_data = filtered_data.groupby('Scholarship holder')['Target'].value_counts().unstack().fillna(0)
        if not scholarship_data.empty:
            scholarship_fig = px.bar(
                scholarship_data.reset_index(),
                x='Scholarship holder',
                y=[0, 1] if all(col in scholarship_data.columns for col in [0, 1]) else [0] if 0 in scholarship_data.columns else [1],
                barmode='group',
                title="Pengaruh Beasiswa terhadap Status Dropout",
                labels={'Scholarship holder': 'Status Beasiswa', 'value': 'Jumlah Mahasiswa', 'variable': 'Status'},
                color_discrete_map={0: 'red', 1: 'green'}
            )
            scholarship_fig.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Tanpa Beasiswa', 'Dengan Beasiswa']))
            st.plotly_chart(scholarship_fig, use_container_width=True)
            
            # Insight untuk beasiswa
            pct_scholarship = filtered_data[filtered_data['Scholarship holder'] == 1].shape[0] / max(1, len(filtered_data))
            st.markdown(f"""
            **Insight:**
            - {pct_scholarship:.1%} mahasiswa menerima beasiswa
            - Beasiswa dapat mengurangi tingkat dropout secara signifikan
            
            **Rekomendasi:**
            - Prioritaskan beasiswa untuk mahasiswa berisiko tinggi dropout
            - Tingkatkan jumlah dan cakupan program beasiswa
            """)
    
    with col4:
        st.markdown("### 📚 Konsultasi Usia")
        age_fig = px.box(
            filtered_data, 
            x='Target', 
            y='Age at enrollment',
            color='Target',
            title="Distribusi Usia saat Pendaftaran"
        )
        st.plotly_chart(age_fig, use_container_width=True)
        
        # Insight untuk usia
        avg_age = filtered_data['Age at enrollment'].mean()
        risk_age = filtered_data[filtered_data['Target'] == 0]['Age at enrollment'].mean()
        st.markdown(f"""
        **Insight:**
        - Rata-rata usia: {avg_age:.2f} tahun
        - Mahasiswa dropout rata-rata berusia {risk_age:.2f} tahun
        
        **Rekomendasi:**
        - Program khusus untuk mahasiswa non-tradisional (usia lebih tua)
        - Konseling karir untuk mahasiswa dewasa
        """)
    
    # Content-Based Filtering untuk rekomendasi program intervensi
    st.markdown("### 📊 Rekomendasi Program Intervensi")
    st.markdown("""
    Sistem rekomendasi ini menggunakan metode **Content-Based Filtering** untuk menyarankan program intervensi
    berdasarkan karakteristik mahasiswa. Program ini disesuaikan dengan profil dan kebutuhan spesifik mahasiswa.
    """)
    
    # Simulasi sistem rekomendasi
    if has_model:
        # Contoh sampel mahasiswa untuk rekomendasi
        sample_size = min(5, len(filtered_data))
        sample_students = filtered_data.sample(sample_size) if not filtered_data.empty else ip.sample(sample_size)
        
        # Program intervensi yang tersedia
        programs = {
            "high_risk": [
                "Program Mentor Akademik Intensif",
                "Konseling Keuangan dan Beasiswa",
                "Workshop Manajemen Waktu",
                "Program Remedial Mata Kuliah Dasar"
            ],
            "medium_risk": [
                "Kelompok Belajar Terbimbing",
                "Pemantauan Kemajuan Bulanan",
                "Konseling Akademik",
                "Pelatihan Keterampilan Belajar"
            ],
            "low_risk": [
                "Sesi Orientasi Karir",
                "Workshop Pengembangan Soft Skills",
                "Program Pengayaan Akademik",
                "Kegiatan Ekstrakurikuler"
            ]
        }
        
        # Tampilkan rekomendasi untuk sampel mahasiswa
        st.markdown("#### Contoh Rekomendasi Program untuk Mahasiswa")
        
        for i, (idx, student) in enumerate(sample_students.iterrows()):
            # Tentukan kategori risiko
            if 'Risk Category' in student:
                risk_level = student['Risk Category']
            else:
                prob = student['Dropout Probability'] if 'Dropout Probability' in student else 0.5
                if prob >= 0.7:
                    risk_level = "Risiko Tinggi"
                elif prob >= 0.3:
                    risk_level = "Risiko Sedang"
                else:
                    risk_level = "Risiko Rendah"
            
            # Pilih program yang sesuai
            if "Tinggi" in risk_level:
                recommended_programs = programs["high_risk"]
            elif "Sedang" in risk_level:
                recommended_programs = programs["medium_risk"]
            else:
                recommended_programs = programs["low_risk"]
            
            # Custom recommendations based on specific features
            custom_recommendations = []
            if student['Tuition fees up to date'] == 0:
                custom_recommendations.append("Program Bantuan Keuangan")
            
            if student['Curricular units 1st sem (grade)'] < 10:
                custom_recommendations.append("Program Pendampingan Akademik Intensif")
            
            if student['Age at enrollment'] > 25:
                custom_recommendations.append("Program Dukungan untuk Mahasiswa Dewasa")
                
            # Pilih maksimal 3 program rekomendasi
            final_recommendations = custom_recommendations + [p for p in recommended_programs if p not in custom_recommendations]
            final_recommendations = final_recommendations[:3]
            
            # Tampilkan profil mahasiswa dan rekomendasi
            expander = st.expander(f"Mahasiswa #{i+1} - {risk_level}")
            with expander:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Profil Mahasiswa:**")
                    st.markdown(f"- Usia: {student['Age at enrollment']:.1f} tahun")
                    st.markdown(f"- Nilai Masuk: {student['Admission grade']:.1f}")
                    st.markdown(f"- Nilai Semester 1: {student['Curricular units 1st sem (grade)']:.1f}")
                    st.markdown(f"- Status Beasiswa: {'Ya' if student['Scholarship holder'] == 1 else 'Tidak'}")
                    st.markdown(f"- Status Pembayaran: {'Tepat Waktu' if student['Tuition fees up to date'] == 1 else 'Terlambat/Belum'}")
                    if 'Course' in student:
                        st.markdown(f"- Program Studi: {student['Course']}")
                
                with col2:
                    st.markdown("**Program yang Direkomendasikan:**")
                    for j, program in enumerate(final_recommendations):
                        st.markdown(f"{j+1}. {program}")
    
    # Tambahkan insight visualisasi admission grade dari kode asli
    st.markdown("### 📈 Analisis Nilai Masuk")
    admission_fig = px.histogram(
        filtered_data, 
        x='Admission grade', 
        color='Target', 
        barmode='overlay',
        title="Distribusi Nilai Masuk Berdasarkan Status Dropout",
        labels={'Target': 'Status', 'Admission grade': 'Nilai Masuk'},
        color_discrete_map={0: 'red', 1: 'blue', 2: 'green'}
    )
    
    # Tambahkan legenda yang lebih informatif
    admission_fig.update_layout(
        legend=dict(
            title="Status Mahasiswa",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(admission_fig)
    
    # Insight untuk nilai masuk
    mean_dropout = filtered_data[filtered_data['Target'] == 0]['Admission grade'].mean()
    mean_graduate = filtered_data[filtered_data['Target'] == 1]['Admission grade'].mean()
    threshold = filtered_data['Admission grade'].quantile(0.25)
    
    st.markdown(f"""
    **Insight Nilai Masuk:**
    - Rata-rata nilai masuk mahasiswa dropout: {mean_dropout:.2f}
    - Rata-rata nilai masuk mahasiswa lulus: {mean_graduate:.2f}
    - Nilai masuk di bawah {threshold:.2f} menunjukkan risiko dropout lebih tinggi
    
    **Rekomendasi:**
    - Pertimbangkan program persiapan untuk mahasiswa dengan nilai masuk rendah
    - Evaluasi kembali standar penerimaan untuk program studi tertentu
    - Integrasikan program pengenalan kampus yang lebih komprehensif untuk mahasiswa baru
    """)
    
    # Tambahkan kesimpulan umum
    st.markdown("### 🔍 Kesimpulan dan Rekomendasi Umum")
    st.markdown("""
    Berdasarkan analisis data di atas, sistem rekomendasi kami mengidentifikasi beberapa faktor utama yang memengaruhi risiko dropout mahasiswa:
    
    1. **Faktor Akademik**: Nilai masuk dan performa semester pertama menjadi indikator kuat risiko dropout.
    2. **Faktor Finansial**: Status pembayaran dan dukungan beasiswa sangat memengaruhi kelangsungan studi.
    3. **Faktor Demografis**: Usia saat pendaftaran dapat menjadi indikator tambahan untuk menyesuaikan dukungan.
    
    **Rekomendasi Strategis Komprehensif:**
    
    - **Sistem Peringatan Dini**: Implementasikan sistem monitoring untuk mengidentifikasi mahasiswa berisiko sejak dini
    - **Program Dukungan Terintegrasi**: Gabungkan dukungan akademik, finansial, dan sosial secara personal
    - **Evaluasi Berkala**: Lakukan penilaian rutin terhadap efektivitas intervensi yang dilakukan
    - **Peningkatan Engagement**: Kembangkan program keterlibatan mahasiswa untuk meningkatkan rasa memiliki
    """)

# buat file requirement dengan menjalankan perintah ini di terminal
# pip freeze > requirements.txt