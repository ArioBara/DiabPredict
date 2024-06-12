from flask import Flask, render_template, request, flash, session
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

app = Flask(__name__)
app.secret_key = 'diur837r3bchs7347r6q099e93'

# Memuat model yang telah dilatih
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    title = 'Beranda'
    return render_template('index.html', title=title)

@app.route('/prediksi', methods=['GET', 'POST'])
def prediksi():
    title = 'Prediksi'
    if request.method == 'POST':
        try:
            # Mendapatkan data dari formulir
            age = int(request.form['age'])
            gender = int(request.form['gender'])
            polyuria = int(request.form['polyuria'])
            polydipsia = int(request.form['polydipsia'])
            sudden_weight_loss = int(request.form['sudden_weight_loss'])
            weakness = int(request.form['weakness'])
            polyphagia = int(request.form['polyphagia'])
            genital_thrush = int(request.form['genital_thrush'])
            visual_blurring = int(request.form['visual_blurring'])
            itching = int(request.form['itching'])
            irritability = int(request.form['irritability'])
            delayed_healing = int(request.form['delayed_healing'])
            partial_paresis = int(request.form['partial_paresis'])
            muscle_stiffness = int(request.form['muscle_stiffness'])
            alopecia = int(request.form['alopecia'])
            obesity = int(request.form['obesity'])

            # Mengubah data input menjadi array numpy
            features = np.array([
                age, gender, polyuria, polydipsia, sudden_weight_loss, weakness,
                polyphagia, genital_thrush, visual_blurring, itching, irritability,
                delayed_healing, partial_paresis, muscle_stiffness, alopecia, obesity
            ]).reshape(1, -1)

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

            feature_names = [
                'Age', 'Gender', 'Polyuria', 'Polydipsia', 'Sudden Weight Loss',
                'Weakness', 'Polyphagia', 'Genital Thrush', 'Visual Blurring', 'Itching',
                'Irritability', 'Delayed Healing', 'Partial Paresis', 'Muscle Stiffness', 'Alopecia', 'Obesity'
            ]

            # Mengekstrak fitur yang mempengaruhi hasil dari model
            influential_features = [feature for i, feature in enumerate(feature_names) if features[0][i] == 1]
            influential_count = len(influential_features)
            total_features = 16

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
                "Visual Blurring": "Penglihtan kabur dapat menjadi tanda dari komplikasi diabetes, terutama pada tahap lanjut penyakit. Jaga kontrol gula darah Anda dan konsultasikan dengan dokter jika Anda mengalami perubahan penglihatan yang signifikan.",
                "Obesity": "Obesitas adalah faktor risiko utama untuk diabetes tipe 2. Penting untuk menjaga berat badan dalam kisaran yang sehat dengan mengadopsi pola makan seimbang dan rutin berolahraga.",
                "Muscle Stiffness": "Kekejangan otot bisa menjadi tanda dari berbagai kondisi medis, termasuk diabetes. Jika Anda mengalami kekejangan otot yang berulang, sebaiknya berkonsultasi dengan dokter untuk penanganan yang tepat.",
                "Polyphagia": "Polyphagia, atau kebiasaan makan berlebihan, dapat menjadi tanda diabetes. Kontrol pola makan Anda dan hindari makanan tinggi gula dan karbohidrat untuk mengelola gejala ini.",
                "Genital Thrush": "Genital thrush, atau infeksi ragi pada area genital, bisa menjadi tanda dari masalah metabolik seperti diabetes. Jaga kebersihan dan kesehatan area genital Anda serta konsultasikan dengan dokter jika mengalami gejala ini.",
                "Itching": "Gatal-gatal yang terus-menerus bisa menjadi tanda dari beberapa kondisi termasuk diabetes. Jaga kulit Anda tetap bersih dan hindari menggaruk area yang gatal untuk mencegah iritasi dan infeksi.",
                "Weakness": "Kelemahan umum bisa menjadi tanda dari berbagai kondisi termasuk diabetes. Istirahat yang cukup, konsumsi makanan bergizi, dan olahraga ringan dapat membantu mengatasi kelemahan ini. Jika kelemahan tidak membaik, sebaiknya konsultasikan dengan dokter untuk pemeriksaan lebih lanjut."
            }

            return render_template('hasil.html', prediction=prediction_text, positive_level=positive_level, influential_features=influential_features, influential_count=influential_count, total_features=total_features, prevention_advice=prevention_advice, title=title)

        except Exception as e:
            flash("Gagal mendiagnosa terdapat pertanyaan yang belum dijawab!", "error")
            return render_template('prediksi.html', title=title)
    else:
        # Jika metode bukan POST, render halaman prediksi
        return render_template('prediksi.html', title=title)


@app.route('/pencegahan')
def pencegahan():
    title = 'Pencegahan'
    return render_template('pencegahan.html', title=title)

@app.route('/ml')
def machine_learning():
    title = 'Machine Learning Model'
    return render_template('notebooks.html', title=title)

@app.route('/algoritma')
def algoritma():
    title = 'Algoritma'

    #<===========LOAD DATASET===========>
    # Load Dataset
    df = pd.read_csv("data/diabetes_data_upload.csv")
    # Konversi DataFrame menjadi HTML
    data = df.head().to_html(index=False, classes='table table-striped rounded')

    # Distribusi Diagnosis Diabetes
    plt.figure(figsize=(6, 6))
    ax = sns.countplot(df['class'], palette=['#1DCC70', '#1DCC70'])
    plt.title('Distribusi Diagnosis Diabetes')
    plt.xlabel('Diabetes')
    plt.ylabel('Frekuensi')
    # Menambahkan jumlah data di atas setiap batang
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=12, color='#1DCC70', xytext=(0, 5),
                    textcoords='offset points')
    img1 = BytesIO()
    plt.savefig(img1, format='png')
    img1.seek(0)
    plot = base64.b64encode(img1.getvalue()).decode()

    #<===========PREPROCESSING===========>
    # Memeriksa nilai yang hilang
    missing_values = df.isna().sum()

    # Mengubah nilai target: Positive menjadi 1 dan Negative menjadi 0
    # df['class'] = df['class'].apply(lambda x: 0 if x=='Negative' else 1)

    # Mengkodekan kolom objek menjadi numerik
    label_encoder = LabelEncoder()
    for column in df.columns:
        df[column] = label_encoder.fit_transform(df[column])

    # Memeriksa tipe data
    data_types = df.dtypes

    # Tampilan informasi tentang data setelah preprocessing
    df_info = pd.DataFrame({
        'Column': df.columns,
        'Missing Values': missing_values,
        'Data Type': data_types
    })

    preprocessing = df_info.to_html(index=False, classes='table table-striped rounded')

    #<===========PEMILIHAN FITUR===========>
    # Memisahkan kumpulan data menjadi fitur dan variabel target
    X = df.drop('class', axis=1)
    y = df['class']

    # Korelasi antara setiap fitur dengan target
    correlation_with_target = X.corrwith(y)

    # Menyatukan data menjadi dataframe
    correlation_df = pd.DataFrame({
        'Feature': correlation_with_target.index,
        'Correlation with Target': correlation_with_target.values
    }).to_html(index=False, classes='table table-striped rounded')

    # Heatmap
    plt.figure(figsize=(10, 10))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='Greens')
    plt.title('Matriks Korelasi')

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Encode the image data to base64
    heatmap = base64.b64encode(img.getvalue()).decode()

    # Korelasi Fitur
    plt.figure(figsize=(16, 6))
    correlation_with_target = df.drop('class', axis=1).corrwith(df['class'])
    correlation_with_target.plot.bar(title="Korelasi dengan Diabetes", fontsize=15, rot=90, grid=True, color='#1DCC70')
    plt.xlabel("Fitur")
    plt.ylabel("Korelasi")

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Encode the image data to base64
    correlation_plot = base64.b64encode(img.getvalue()).decode()

    #<===========MEMBAGI DATA===========>
    # Memisahkan data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Membuat dataframe untuk dimensi data latih dan data uji
    data_dimensions = {
        'Data': ['Data Latih', 'Data Uji'],
        'Jumlah Fitur': [X_train.shape[1], X_test.shape[1]],
        'Jumlah Target': [y_train.shape[0], y_test.shape[0]]
    }
    df_dimensions = pd.DataFrame(data_dimensions).to_html(index=False, classes='table table-striped rounded')

    # Memeriksa distribusi variabel target dalam pemisahan data latih dan uji
    train_target_distribution = y_train.value_counts().reset_index()
    train_target_distribution.columns = ['Target', 'Jumlah Data']

    test_target_distribution = y_test.value_counts().reset_index()
    test_target_distribution.columns = ['Target', 'Jumlah Data']
    train = train_target_distribution.to_html(index=False, classes='table table-striped rounded')
    test = test_target_distribution.to_html(index=False, classes='table table-striped rounded')

    #<===========PEMBANGUNAN MODEL===========>
    # Membuat model Decision Tree C4.5 dengan kriteria entropi
    decision_tree = DecisionTreeClassifier(criterion='entropy', random_state=0)

    # Melatih model pada data latih
    decision_tree.fit(X_train, y_train)

    # Menampilkan informasi model ke dalam dataframe
    model_info = pd.DataFrame({
        'Feature': X_train.columns,
        'Feature Importance': decision_tree.feature_importances_})
    model_info.sort_values(by='Feature Importance', ascending=False, inplace=True)

    model = model_info.to_html(index=False, classes='table table-striped rounded')

    #<===========PENGUJIAN MODEL===========>
    # Pengujian Model
    kfold = KFold(n_splits=5, shuffle=False, random_state=None)
    scoring = 'accuracy'

    acc_decision_tree = cross_val_score(estimator=decision_tree, X=X_train, y=y_train, cv=kfold, scoring=scoring)

    # Membuat dataframe untuk menampilkan hasil cross-validation
    cv_results = pd.DataFrame({'Fold': range(1, len(acc_decision_tree)+1), 'Accuracy': acc_decision_tree}).to_html(index=False, classes='table table-striped rounded')

    # Rata-rata akurasi cross-validation
    mean_accuracy = acc_decision_tree.mean()

    #<===========EVALUASI MODEL===========>
    y_predict_decision_tree = decision_tree.predict(X_test)
    acc = accuracy_score(y_test, y_predict_decision_tree)
    prec = precision_score(y_test, y_predict_decision_tree)
    rec = recall_score(y_test, y_predict_decision_tree)
    f1 = f1_score(y_test, y_predict_decision_tree)

    results = pd.DataFrame([['Decision Tree C4.5',acc, acc_decision_tree.mean(), prec, rec, f1]],
                        columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score']).to_html(index=False, classes='table table-striped rounded')

    return render_template('algoritma.html',
                           title=title,
                           data=data,
                           plot=plot,
                           preprocessing=preprocessing,
                           correlation_df=correlation_df,
                           heatmap=heatmap,
                           correlation_plot=correlation_plot,
                           df_dimensions=df_dimensions,
                           train=train,
                           test=test,
                           model=model,
                           cv_results=cv_results,
                           mean_accuracy=mean_accuracy,
                           results=results)

if __name__ == '__main__':
    app.run(debug=True)
