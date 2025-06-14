import csv
from datetime import datetime
from predict import predict_book_cover
from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
predictions_history = []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        genre_label, confidence = predict_book_cover(filepath)

        predictions_history.append({
            "index": len(predictions_history) + 1,
            "filename": filename,
            "prediction": genre_label,
            "confidence": float(confidence)
        })

        file_exists = os.path.exists('predictions_log.csv')
        write_header = not file_exists or os.path.getsize('predictions_log.csv') == 0
        with open('predictions_log.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(['datetime', 'filename', 'prediction', 'confidence', 'index'])
            writer.writerow([datetime.now(), filename, genre_label, confidence, len(predictions_history)])

        return render_template(
            'result.html',
            filename=filename,
            prediction=genre_label,
            confidence=f"{confidence * 100:.2f}%",
            index=len(predictions_history),
        )


@app.route('/api/predictions')
def get_predictions():
    return jsonify(predictions_history)


if __name__ == '__main__':
    app.run(debug=True)

