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
            "filename": filename,
            "prediction": genre_label,
            "confidence": float(confidence)
        })

        print(predictions_history)

        return render_template(
            'result.html',
            filename=filename,
            prediction=genre_label,
            confidence=f"{confidence * 100:.2f}%"
        )


@app.route('/api/predictions')
def get_predictions():
    return jsonify(predictions_history)


if __name__ == '__main__':
    app.run(debug=True)

