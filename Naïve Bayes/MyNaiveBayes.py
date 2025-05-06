import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                             QPushButton, QTextEdit, QFileDialog, QLabel)
from PyQt5.QtGui import QFont
from collections import defaultdict
import math # Import math for round, though built-in round() is used

class NaiveBayesClassifierGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Naive Bayes Classifier") # Judul jendela
        self.setGeometry(100, 100, 900, 700) # Ukuran dan posisi jendela

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.label = QLabel("Pilih file teks data:") # Label instruksi
        self.layout.addWidget(self.label)

        self.btn_select_file = QPushButton("Pilih File Data") # Tombol pilih file
        self.btn_select_file.clicked.connect(self.select_file) # Hubungkan sinyal clicked dengan method select_file
        self.layout.addWidget(self.btn_select_file)

        self.output_text_edit = QTextEdit() # Area teks untuk menampilkan output
        self.output_text_edit.setReadOnly(True) # Buat read-only agar user tidak bisa mengedit
        self.output_text_edit.setFont(QFont("Courier New", 10)) # Gunakan font monospace untuk tampilan tabel/perhitungan yang rapi
        self.layout.addWidget(self.output_text_edit)

        # Variabel untuk menyimpan data dan hasil
        self.data = []
        self.attributes = []
        self.training_data = []
        self.testing_data = []
        self.prior_probs = {}
        self.likelihoods = {}
        self.unique_attribute_values = {} # Menyimpan nilai unik per atribut dari data training

    def select_file(self):
        """Membuka dialog untuk memilih file teks data."""
        options = QFileDialog.Options()
        # Membuka dialog file, filter hanya file .txt
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Pilih File Data", "", "Text Files (*.txt);;All Files (*)", options=options
        )
        if file_path:
            self.output_text_edit.clear() # Bersihkan output sebelumnya
            self.output_text_edit.append(f"File terpilih: {file_path}\n") # Tampilkan nama file terpilih
            self.load_and_process_data(file_path) # Proses data dari file

    def load_and_process_data(self, file_path):
        """Membaca data dari file, memproses, melatih, dan menguji model."""
        # Reset semua variabel data
        self.data = []
        self.attributes = []
        self.training_data = []
        self.testing_data = []
        self.prior_probs = {}
        self.likelihoods = {}
        self.unique_attribute_values = {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f: # Gunakan encoding utf-8
                lines = f.readlines()

            if not lines:
                self.output_text_edit.append("Error: File kosong.")
                return

            # Baca header (baris pertama)
            self.attributes = [attr.strip() for attr in lines[0].strip().split(';')]
            if not self.attributes:
                 self.output_text_edit.append("Error: Header tidak ditemukan atau format salah.")
                 return
            if len(self.attributes) < 2:
                 self.output_text_edit.append("Error: File harus memiliki setidaknya atribut ID dan Hipotesis.")
                 return

            # Baca data (baris selanjutnya)
            for i, line in enumerate(lines[1:]):
                values = [value.strip() for value in line.strip().split(';')]
                if len(values) == len(self.attributes):
                    instance = dict(zip(self.attributes, values))
                    # Pastikan ID data ada
                    if self.attributes[0] not in instance or not instance[self.attributes[0]]:
                         self.output_text_edit.append(f"Peringatan: Baris {i+2} dilewati karena ID data kosong.")
                         continue
                    self.data.append(instance)
                else:
                    self.output_text_edit.append(f"Peringatan: Baris {i+2} dilewati karena jumlah kolom tidak sesuai ({len(values)} vs {len(self.attributes)}).")

            if not self.data:
                self.output_text_edit.append("Error: Tidak ada data valid ditemukan setelah header.")
                return

            # Lakukan pembagian data, training, testing, dan evaluasi
            self.split_data()
            if not self.training_data or not self.testing_data:
                 self.output_text_edit.append("Error: Data training atau testing kosong setelah pembagian. Pastikan setiap kelas memiliki cukup data.")
                 return

            self.train()
            self.test()
            self.evaluate()

        except FileNotFoundError:
            self.output_text_edit.append("Error: File tidak ditemukan.")
        except Exception as e:
            # Tangani error umum lainnya
            self.output_text_edit.append(f"Terjadi kesalahan: {e}")
            # Opsional: Tampilkan traceback untuk debugging
            # import traceback
            # self.output_text_edit.append(f"Traceback:\n{traceback.format_exc()}")

    def format_as_table(self, data, attributes):
        """Memformat data menjadi string menyerupai tabel."""
        if not data:
            return "Tidak ada data untuk ditampilkan."

        # Tentukan lebar maksimum untuk setiap kolom
        column_widths = {attr: len(attr) for attr in attributes}
        for instance in data:
            for attr in attributes:
                column_widths[attr] = max(column_widths[attr], len(str(instance.get(attr, ''))))

        # Buat garis pemisah horizontal
        separator_parts = ["-" * (column_widths[attr] + 2) for attr in attributes]
        separator = "+" + "+".join(separator_parts) + "+"

        # Buat header tabel
        header_parts = [f" {attr:<{column_widths[attr]}} " for attr in attributes]
        header = "|" + "|".join(header_parts) + "|"

        # Buat baris data
        data_rows = []
        for instance in data:
            row_parts = [f" {str(instance.get(attr, '')):<{column_widths[attr]}} " for attr in attributes]
            data_rows.append("|" + "|".join(row_parts) + "|")

        # Gabungkan semua bagian menjadi string tabel
        table_string = [separator, header, separator] + data_rows + [separator]

        return "\n".join(table_string)


    def split_data(self):
        """Membagi data menjadi training (70%) dan testing (30%) per kelas, diurutkan berdasarkan ID."""
        data_by_class = defaultdict(list)
        class_attribute = self.attributes[-1] # Atribut terakhir adalah hipotesis (kelas)

        # Kelompokkan data berdasarkan kelas
        for instance in self.data:
            data_by_class[instance[class_attribute]].append(instance)

        self.training_data = []
        self.testing_data = []

        # Bagi setiap kelompok kelas
        for class_label, instances in data_by_class.items():
            # Urutkan berdasarkan ID data (atribut pertama)
            sorted_instances = sorted(instances, key=lambda x: x[self.attributes[0]])

            total_in_class = len(sorted_instances)
            # Hitung indeks pemisah 70% menggunakan round()
            split_index = round(total_in_class * 0.7)

            # Ambil 70% pertama untuk training, sisanya untuk testing
            self.training_data.extend(sorted_instances[:split_index])
            self.testing_data.extend(sorted_instances[split_index:])

        # Urutkan kembali seluruh data training dan testing berdasarkan ID untuk tampilan
        self.training_data = sorted(self.training_data, key=lambda x: x[self.attributes[0]])
        self.testing_data = sorted(self.testing_data, key=lambda x: x[self.attributes[0]])

        # Tampilkan jumlah data
        self.output_text_edit.append("--- Pembagian Data ---")
        self.output_text_edit.append(f"Jumlah Data Training: {len(self.training_data)}")
        self.output_text_edit.append(f"Jumlah Data Testing: {len(self.testing_data)}\n")

        # Tampilkan data training dalam format tabel
        self.output_text_edit.append("Data Training (urut berdasarkan ID):")
        self.output_text_edit.append(self.format_as_table(self.training_data, self.attributes))
        self.output_text_edit.append("")

        # Tampilkan data testing dalam format tabel
        self.output_text_edit.append("Data Testing (urut berdasarkan ID):")
        self.output_text_edit.append(self.format_as_table(self.testing_data, self.attributes))
        self.output_text_edit.append("")


    def train(self):
        """Menghitung probabilitas prior dan likelihood dari data training."""
        class_attribute = self.attributes[-1] # Atribut kelas
        # Atribut prediktor (selain ID dan kelas)
        attribute_list = self.attributes[1:-1]
        total_training_instances = len(self.training_data)
        class_counts = defaultdict(int) # Hitungan per kelas
        # Hitungan nilai atribut per kelas: class -> attribute -> value -> count
        attribute_value_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        if total_training_instances == 0:
             self.output_text_edit.append("Error: Data training kosong, tidak bisa melakukan training.")
             return

        # Hitung kemunculan setiap kelas
        for instance in self.training_data:
            class_counts[instance[class_attribute]] += 1

        # Hitung probabilitas prior P(class)
        for class_label, count in class_counts.items():
            # P(class) = count(class) / total_training_instances
            self.prior_probs[class_label] = round(count / total_training_instances, 2)

        # Hitung kemunculan nilai atribut untuk setiap kelas
        for instance in self.training_data:
            class_label = instance[class_attribute]
            for attr in attribute_list:
                value = instance[attr]
                attribute_value_counts[class_label][attr][value] += 1

        # Temukan nilai unik untuk setiap atribut di *seluruh* data training (untuk smoothing)
        for attr in attribute_list:
            self.unique_attribute_values[attr] = set()
            for instance in self.training_data:
                 self.unique_attribute_values[attr].add(instance[attr])

        # Hitung likelihood P(attribute_value | class) dengan Laplace smoothing
        # Formula: P(X=v | C=c) = (count(X=v and C=c) + 1) / (count(C=c) + |V_X|)
        # |V_X| = jumlah nilai unik untuk atribut X di data training
        self.likelihoods = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        self.output_text_edit.append("--- Perhitungan Likelihood (dengan Laplace Smoothing) ---")

        # Iterasi melalui setiap kelas yang ada di data training
        for class_label in class_counts:
            N_class = class_counts[class_label] # count(C=c)
            self.output_text_edit.append(f"\nUntuk Hipotesis: {class_label} (Jumlah data training untuk kelas ini: {N_class})")

            # Iterasi melalui setiap atribut prediktor
            for attr in attribute_list:
                # |V_X| = jumlah nilai unik untuk atribut ini di data training
                V_attribute = len(self.unique_attribute_values.get(attr, set()))
                self.output_text_edit.append(f"  Atribut '{attr}' (Jumlah nilai unik di training untuk atribut ini: {V_attribute})")

                # Iterasi melalui semua nilai unik yang terlihat untuk atribut ini di training
                # Ini penting agar likelihood untuk nilai yang count-nya 0 tetap terhitung
                all_values_for_attr = self.unique_attribute_values.get(attr, set())
                for value in all_values_for_attr:
                    # count(X=v and C=c). Gunakan .get(value, 0) jika nilai tidak ada di kelas ini
                    count_attr_value_class = attribute_value_counts[class_label][attr].get(value, 0)

                    # Hitung likelihood dengan Laplace Smoothing
                    denominator = N_class + V_attribute
                    likelihood = (count_attr_value_class + 1) / denominator if denominator > 0 else 0.0
                    # Bulatkan ke 2 desimal
                    self.likelihoods[class_label][attr][value] = round(likelihood, 2)

                    # Tampilkan asal usul nilai yang digunakan
                    self.output_text_edit.append(
                        f"    P('{attr}'='{value}' | '{class_label}') = ({count_attr_value_class} + 1) / ({N_class} + {V_attribute}) = {self.likelihoods[class_label][attr][value]}"
                    )

        self.output_text_edit.append("\n--- Perhitungan Prediksi Data Testing ---")


    def test(self):
        """Memprediksi kelas untuk setiap instance data testing dan menampilkan probabilitas persentase."""
        class_attribute = self.attributes[-1] # Atribut kelas
        attribute_list = self.attributes[1:-1] # Atribut prediktor
        self.predictions = [] # Menyimpan pasangan (aktual, prediksi) untuk evaluasi

        # Iterasi melalui setiap instance di data testing
        for instance in self.testing_data:
            instance_id = instance[self.attributes[0]] # Ambil ID data
            actual_class = instance[class_attribute] # Ambil kelas aktual
            probabilities = {} # Menyimpan probabilitas (unnormalized) untuk setiap kelas

            self.output_text_edit.append(f"\nData Testing ID: {instance_id}")
            # Tampilkan nilai atribut data testing
            attr_values_str = ";".join([f"{a}={instance[a]}" for a in attribute_list])
            self.output_text_edit.append(f"  Atribut: ({attr_values_str})")
            self.output_text_edit.append(f"  Kelas Aktual: {actual_class}")

            # Hitung probabilitas untuk setiap kemungkinan kelas (hipotesis)
            for class_label, prior in self.prior_probs.items():
                self.output_text_edit.append(f"  Menghitung untuk Hipotesis: {class_label}")
                # Mulai dengan probabilitas prior P(class)
                calculated_prob = prior
                calculation_steps = [f"P('{class_label}') = {prior}"] # Simpan langkah perhitungan untuk tampilan

                # Kalikan dengan likelihood untuk setiap atribut
                for attr in attribute_list:
                    value = instance[attr] # Nilai atribut dari instance testing

                    # Ambil likelihood P(attribute_value | class)
                    # N_class = jumlah data training untuk class_label
                    N_class_for_likelihood = len([d for d in self.training_data if d[self.attributes[-1]] == class_label])
                    # V_attribute_train = jumlah nilai unik untuk atribut ini di training
                    V_attribute_train = len(self.unique_attribute_values.get(attr, set()))

                    # Cek apakah nilai atribut testing ada di nilai unik training untuk atribut ini
                    if value in self.unique_attribute_values.get(attr, set()):
                        # Nilai ada di training, ambil likelihood yang sudah dihitung
                        likelihood = self.likelihoods.get(class_label, {}).get(attr, {}).get(value, 0.0)
                        likelihood_origin_display = f"P('{attr}'='{value}' | '{class_label}') = {likelihood}"
                    else:
                        # Nilai atribut testing TIDAK ada di nilai unik training untuk atribut ini
                        # Gunakan formula Laplace smoothing untuk count=0
                        # (0 + 1) / (N_class + V_attribute_train)
                        denominator_fallback = N_class_for_likelihood + V_attribute_train
                        likelihood = (0 + 1) / denominator_fallback if denominator_fallback > 0 else 0.0
                        likelihood = round(likelihood, 2) # Bulatkan nilai fallback
                        likelihood_origin_display = f"P('{attr}'='{value}' | '{class_label}') = (0 + 1) / ({N_class_for_likelihood} + {V_attribute_train}) = {likelihood} (Nilai '{value}' tidak ada di training untuk atribut '{attr}')"


                    calculated_prob *= likelihood # Kalikan probabilitas

                    # Tambahkan langkah perkalian ke tampilan
                    # Ambil nilai likelihood yang sudah dibulatkan yang sebenarnya digunakan dalam perkalian
                    calculation_steps.append(f"* {likelihood} ({likelihood_origin_display.split('=')[-1].strip()})")


                probabilities[class_label] = round(calculated_prob, 6) # Simpan probabilitas akhir (unnormalized), gunakan presisi lebih tinggi untuk persentase

                # Tampilkan perhitungan total untuk kelas ini
                self.output_text_edit.append(f"    P(Data Testing | '{class_label}') * P('{class_label}') = {' '.join(calculation_steps)} = {round(probabilities[class_label], 2)}") # Tampilkan hasil akhir dibulatkan 2 digit

            # Hitung total probabilitas untuk normalisasi
            total_prob_sum = sum(probabilities.values())
            normalized_probabilities = {}
            percentage_display = []

            # Hitung probabilitas persentase
            for class_label, prob in probabilities.items():
                # Hindari pembagian oleh nol jika total_prob_sum adalah 0
                normalized_prob = prob / total_prob_sum if total_prob_sum > 0 else 0.0
                normalized_probabilities[class_label] = normalized_prob
                percentage = round(normalized_prob * 100, 2) # Bulatkan persentase ke 2 desimal
                percentage_display.append(f"{class_label}: {percentage}%")


            # Prediksi kelas: pilih kelas dengan probabilitas tertinggi (dari nilai unnormalized atau normalized, hasilnya sama)
            if probabilities:
                 predicted_class = max(probabilities, key=probabilities.get)
            else:
                 predicted_class = "Tidak Diketahui" # Atau handle sesuai kebutuhan jika tidak ada kelas

            # Tampilkan probabilitas akhir dan persentase
            self.output_text_edit.append(f"  Probabilitas Akhir (Unnormalized): {probabilities}")
            self.output_text_edit.append(f"  Probabilitas Persentase: {', '.join(percentage_display)}")
            self.output_text_edit.append(f"  Prediksi: {predicted_class}\n")

            # Simpan hasil prediksi (menggunakan kelas aktual dan prediksi)
            self.predictions.append((actual_class, predicted_class))


    def evaluate(self):
        """Menghitung dan menampilkan confusion matrix, akurasi, presisi, dan recall."""
        class_attribute = self.attributes[-1] # Atribut kelas
        # Ambil semua kelas unik yang ada di data training (ini akan jadi baris confusion matrix)
        possible_actual_classes = sorted(list(self.prior_probs.keys()))
        # Ambil semua kelas unik yang muncul sebagai prediksi (ini akan jadi kolom confusion matrix)
        all_predicted_classes = sorted(list(set([pred for actual, pred in self.predictions])))

        # Gabungkan semua kemungkinan kelas (aktual dan prediksi) untuk header kolom
        all_unique_classes = sorted(list(set(possible_actual_classes + all_predicted_classes)))


        # Bangun Confusion Matrix (Baris: Aktual, Kolom: Prediksi)
        confusion_matrix = {actual: {predicted: 0 for predicted in all_unique_classes} for actual in possible_actual_classes}

        # Isi confusion matrix berdasarkan hasil prediksi
        for actual, predicted in self.predictions:
            if actual in confusion_matrix and predicted in confusion_matrix[actual]:
                 confusion_matrix[actual][predicted] += 1
            # Jika kelas aktual tidak ada di possible_actual_classes (seharusnya tidak terjadi dengan split per kelas), abaikan atau tambahkan baris baru jika perlu.
            # Jika kelas prediksi tidak ada di all_unique_classes (seharusnya sudah ditambahkan di atas), abaikan.


        self.output_text_edit.append("--- Evaluasi ---")
        self.output_text_edit.append("Confusion Matrix (Baris: Aktual, Kolom: Prediksi):")

        # Tampilkan Confusion Matrix
        # Header kolom
        header_row = ["Aktual"] + all_unique_classes
        # Gunakan format string dengan lebar tetap atau tabulasi untuk perataan
        header_format = "{:<15}" * len(header_row) # Contoh format dengan lebar 15
        self.output_text_edit.append(header_format.format(*header_row))

        # Isi baris matrix
        for actual in possible_actual_classes:
            row_values = [actual]
            for predicted in all_unique_classes:
                row_values.append(str(confusion_matrix[actual].get(predicted, 0))) # Ambil nilai, default 0 jika tidak ada
            self.output_text_edit.append(header_format.format(*row_values))
        self.output_text_edit.append("")

        # Hitung Metrik
        total_test_instances = len(self.testing_data)
        # Hitung jumlah prediksi yang benar (diagonal matrix)
        correct_predictions = sum(confusion_matrix.get(c, {}).get(c, 0) for c in possible_actual_classes)

        # Akurasi
        # Formula: Akurasi = (Jumlah Prediksi Benar) / (Total Data Testing)
        accuracy_calc_str = f"{correct_predictions} / {total_test_instances}"
        accuracy = round(correct_predictions / total_test_instances, 2) if total_test_instances > 0 else 0.0
        accuracy_percentage = round(accuracy * 100, 2) # Hitung persentase akurasi
        self.output_text_edit.append(f"Akurasi = (Jumlah Prediksi Benar) / (Total Data Testing)")
        self.output_text_edit.append(f"Akurasi = {accuracy_calc_str} = {accuracy} ({accuracy_percentage}%)\n") # Tampilkan desimal dan persentase

        # Presisi dan Recall per kelas
        for class_label in possible_actual_classes: # Hitung untuk setiap kelas aktual
            self.output_text_edit.append(f"Untuk Hipotesis: {class_label}")

            # TP (True Positives): Aktual = class_label, Prediksi = class_label
            TP = confusion_matrix.get(class_label, {}).get(class_label, 0)
            # FP (False Positives): Aktual != class_label, Prediksi = class_label
            FP = sum(confusion_matrix.get(other_class, {}).get(class_label, 0)
                     for other_class in possible_actual_classes if other_class != class_label)
            # FN (False Negatives): Aktual = class_label, Prediksi != class_label
            FN = sum(confusion_matrix.get(class_label, {}).get(other_class, 0)
                     for other_class in all_unique_classes if other_class != class_label) # Gunakan all_unique_classes untuk FN

            # Presisi
            # Formula: Presisi = TP / (TP + FP)
            precision_denom = TP + FP
            precision_calc_str = f"{TP} / ({TP} + {FP})"
            precision = round(TP / precision_denom, 2) if precision_denom > 0 else 0.0
            precision_percentage = round(precision * 100, 2) # Hitung persentase presisi
            self.output_text_edit.append(f"  Presisi = TP / (TP + FP)")
            self.output_text_edit.append(f"  Presisi = {precision_calc_str} = {precision} ({precision_percentage}%)") # Tampilkan desimal dan persentase

            # Recall
            # Formula: Recall = TP / (TP + FN)
            recall_denom = TP + FN
            recall_calc_str = f"{TP} / ({TP} + FN)"
            recall = round(TP / recall_denom, 2) if recall_denom > 0 else 0.0
            recall_percentage = round(recall * 100, 2) # Hitung persentase recall
            self.output_text_edit.append(f"  Recall  = TP / (TP + FN)")
            self.output_text_edit.append(f"  Recall  = {recall_calc_str} = {recall} ({recall_percentage}%)\n") # Tampilkan desimal dan persentase


# Blok utama untuk menjalankan aplikasi GUI
if __name__ == '__main__':
    app = QApplication(sys.argv) # Buat instance aplikasi
    main_window = NaiveBayesClassifierGUI() # Buat instance jendela utama
    main_window.show() # Tampilkan jendela
    sys.exit(app.exec_()) # Jalankan event loop aplikasi
