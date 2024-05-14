from flask import Flask, render_template, request, flash, session
import numpy as np
import pickle

app = Flask(__name__)
app.secret_key = 'diur837r3bchs7347r6q099e93'

# Memuat model yang telah dilatih
with open('model/model_dt_c45.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    title = 'Beranda'
    return render_template('index.html', title=title)

@app.route('/prediksi', methods=['GET', 'POST'])  # Tambahkan method 'GET' untuk mengakses halaman prediksi
def prediksi():
    title = 'Prediksi'
    if request.method == 'POST':  # Pastikan kode untuk menangani POST request hanya dijalankan saat metode adalah POST
        try:
            # Mendapatkan data dari formulir
            # age = int(request.form['age'])
            gender = int(request.form['gender'])
            polyuria = int(request.form['polyuria'])
            polydipsia = int(request.form['polydipsia'])
            sudden_weight_loss = int(request.form['sudden_weight_loss'])
            weakness = int(request.form['weakness'])
            polyphagia = int(request.form['polyphagia'])
            visual_blurring = int(request.form['visual_blurring'])
            irritability = int(request.form['irritability'])
            partial_paresis = int(request.form['partial_paresis'])
            alopecia = int(request.form['alopecia'])

            # Mengubah data input menjadi array numpy
            features = np.array([gender, polyuria, polydipsia, sudden_weight_loss, weakness, polyphagia, visual_blurring, irritability, partial_paresis, alopecia]).reshape(1, -1)

            # Melakukan prediksi dengan model
            prediction = model.predict(features)
            probabilities = model.predict_proba(features)[0]  # Probabilitas kelas
            
            # Menampilkan hasil prediksi dalam bentuk teks
            if prediction[0] == 1:
                prediction_text = "Positif"
                positive_level = round(probabilities[1] * 100, 2)
            else:
                prediction_text = "Negatif"
                positive_level = round(probabilities[0] * 100, 2)
            
            feature_names = ['Gender', 'Polyuria', 'Polydipsia', 'Sudden Weight Loss', 'Weakness', 'Polyphagia', 'Visual Blurring', 'Irritability', 'Partial Paresis', 'Alopecia']

            # Mengekstrak fitur yang mempengaruhi hasil dari model
            influential_features = [feature for i, feature in enumerate(feature_names) if features[0][i] == 1]
            influential_count = len(influential_features)
            total_features = 10

            # Informasi tambahan dan saran pencegahan
            prevention_advice = {
                "Polyuria": "Polyuria adalah gejala sering buang air kecil dan bisa menjadi tanda diabetes. Jika Anda mengalami polyuria, segera konsultasikan dengan dokter untuk evaluasi lebih lanjut.",
                "Polydipsia": "Polydipsia, atau rasa haus yang berlebihan, bisa menjadi tanda diabetes. Hindari minuman manis dan konsumsi gula berlebihan untuk mengurangi risiko diabetes.",
                "Age": "Resiko diabetes meningkat seiring bertambahnya usia. Penting untuk menjaga pola makan yang sehat dan aktif secara fisik untuk menjaga kesehatan Anda, terutama saat usia bertambah.",
                "Gender": "Tidak semua jenis kelamin memiliki risiko yang sama terhadap diabetes. Namun, penting bagi semua orang untuk menjaga gaya hidup sehat dan melakukan pemeriksaan rutin untuk mencegah diabetes.",
                "Partial Paresis": "Partial paresis, atau kelumpuhan sebagian tubuh, dapat menjadi komplikasi diabetes. Jaga kontrol gula darah Anda dan konsultasikan dengan dokter Anda untuk pengelolaan yang tepat.",
                "Sudden Weight Loss": "Kehilangan berat badan yang tiba-tiba bisa menjadi tanda diabetes, terutama jika tidak disengaja. Perhatikan pola makan Anda dan konsultasikan dengan dokter jika Anda mengalami kehilangan berat badan yang tidak wajar.",
                "Irritability": "Irritability atau mudah marah bisa menjadi tanda stres yang terkait dengan kondisi kesehatan tertentu, termasuk diabetes. Temukan cara untuk mengelola stres dengan baik, seperti meditasi atau olahraga.",
                "Delayed Healing": "Luka yang lambat sembuh bisa menjadi tanda diabetes, terutama pada luka kecil seperti luka sayat atau lecet. Jaga luka tetap bersih dan konsultasikan dengan dokter jika tidak sembuh dalam waktu yang wajar.",
                "Alopecia": "Alopecia, atau kebotakan, tidak secara langsung terkait dengan diabetes, tetapi dapat menjadi gejala diabetes tipe 2 pada beberapa orang. Lakukan pemeriksaan kesehatan secara rutin untuk memantau kondisi Anda.",
                "Visual Blurring": "Kabur penglihatan dapat menjadi tanda dari komplikasi diabetes, terutama pada tahap lanjut penyakit. Jaga kontrol gula darah Anda dan konsultasikan dengan dokter jika Anda mengalami perubahan penglihatan yang signifikan."
            }

            return render_template('hasil.html', prediction=prediction_text, positive_level=positive_level, influential_features=influential_features, influential_count=influential_count, total_features=total_features, prevention_advice=prevention_advice, title=title)

        except Exception as e:
            flash("Gagal mendiagnosa terdapat pertanyaan yang belum dijawab!", "error")
            return render_template('prediksi.html')
    else:
        # Jika metode bukan POST, render halaman prediksi
        return render_template('prediksi.html', title=title)

@app.route('/pencegahan')
def pencegahan():
    title = 'Pencegahan'
    return render_template('pencegahan.html', title=title)

@app.route('/hidup-sehat')
def hidup_sehat():
    title = 'Hidup Sehat'
    return render_template('hidup_sehat.html', title=title)

if __name__ == '__main__':
    app.run(debug=True)
