from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
import sqlite3
import os
import face_recognition
import numpy as np
import cv2
from datetime import datetime
from PIL import Image
import base64, io

app = Flask(__name__, static_folder="public", static_url_path="/")
app.secret_key = "smartvision_secret_key"
CORS(app)

DB_FILE = "smartvision.db"

# -------------------------------------------------
# DATABASE INITIALIZATION
# -------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS faculty (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    email TEXT UNIQUE,
                    password TEXT,
                    department TEXT,
                    class_name TEXT
                )''')

    c.execute('''CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    reg_no TEXT UNIQUE,
                    class_name TEXT,
                    department TEXT,
                    face_encoding BLOB
                )''')

    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_name TEXT,
                    reg_no TEXT,
                    class_name TEXT,
                    department TEXT,
                    date TEXT,
                    status TEXT
                )''')

    conn.commit()
    conn.close()
    print("✅ Database initialized: smartvision.db")

# Helper to run SQL safely
def run_query(query, params=(), fetch=False):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(query, params)
    data = None
    if fetch:
        data = c.fetchall()
    conn.commit()
    conn.close()
    return data

# -------------------------------------------------
# FRONTEND ROUTES
# -------------------------------------------------
@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)

# -------------------------------------------------
# FACULTY SIGNUP
# -------------------------------------------------
@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    try:
        run_query(
            "INSERT INTO faculty (name, email, password, department, class_name) VALUES (?, ?, ?, ?, ?)",
            (data["name"], data["email"], data["password"], data["department"], data["class_name"])
        )
        print(f"✅ New faculty added: {data['email']}")
        return jsonify({"success": True})
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "message": "Email already exists"})

# -------------------------------------------------
# FACULTY LOGIN
# -------------------------------------------------
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    faculty = run_query(
        "SELECT * FROM faculty WHERE email = ? AND password = ?",
        (data["email"], data["password"]),
        fetch=True
    )
    if faculty:
        f = faculty[0]
        session["faculty"] = {
            "id": f[0], "name": f[1], "email": f[2],
            "department": f[4], "class_name": f[5]
        }
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "message": "Invalid credentials"})

# -------------------------------------------------
# FACULTY INFO
# -------------------------------------------------
@app.route("/faculty-info")
def faculty_info():
    if "faculty" not in session:
        return jsonify({"success": False, "message": "Not logged in"})
    return jsonify({"success": True, **session["faculty"]})

@app.route("/update-faculty", methods=["POST"])
def update_faculty():
    if "faculty" not in session:
        return jsonify({"success": False})
    data = request.get_json()
    f = session["faculty"]
    run_query(
        "UPDATE faculty SET name=?, department=?, class_name=?, password=? WHERE email=?",
        (data["name"], data["department"], data["class_name"], data["password"], f["email"])
    )
    session["faculty"].update(data)
    return jsonify({"success": True})

# -------------------------------------------------
# STUDENT ENROLLMENT
# -------------------------------------------------
@app.route("/capture_face", methods=["POST"])
def capture_face():
    try:
        data = request.get_json()

        # 🧩 Extract data from frontend
        img_data = data.get("image")
        name = data.get("name")
        reg_no = data.get("reg_no")
        class_name = data.get("class_name")
        dept = data.get("department")

        # 🧩 Decode Base64 → Image
        image_bytes = base64.b64decode(img_data.split(",")[1])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)

        # 🧠 Get Face Encodings
        encodings = face_recognition.face_encodings(img_np)
        if len(encodings) == 0:
            print("❌ No face detected during enrollment.")
            return jsonify({"success": False, "message": "No face detected"})

        encoding = encodings[0].astype(np.float64).tobytes()

        # 🧾 Insert into Database
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO students (name, reg_no, class_name, department, face_encoding) VALUES (?, ?, ?, ?, ?)",
            (name, reg_no, class_name, dept, encoding)
        )
        conn.commit()
        conn.close()

        print(f"✅ Enrolled new student: {name} ({reg_no}) [{class_name}]")
        return jsonify({"success": True, "message": "Student enrolled successfully"})

    except Exception as e:
        print("❌ Error in /capture_face:", e)
        return jsonify({"success": False, "message": str(e)})
# -------------------------------------------------
# FACE RECOGNITION + ATTENDANCE MARKING
# -------------------------------------------------
@app.route("/recognize", methods=["POST"])
def recognize():
    try:
        # Load all students
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT name, reg_no, class_name, department, face_encoding FROM students")
        rows = c.fetchall()
        conn.close()

        known_encodings = []
        known_names = []

        for row in rows:
            name, reg_no, cls, dept, encoding_blob = row
            if encoding_blob:
                enc = np.frombuffer(encoding_blob, dtype=np.float64)
                if len(enc) > 0:
                    known_encodings.append(enc)
                    known_names.append(f"{name} ({reg_no})")

        print(f"📚 Loaded {len(known_encodings)} known faces from DB.")

        # If no students are loaded
        if len(known_encodings) == 0:
            return jsonify({"success": False, "message": "No known faces found in database"}), 200

        # Get uploaded image
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)
        print(f"📸 Faces detected: {len(encodings)}")

        recognized = []
        for enc in encodings:
            matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_encodings, enc)
            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else -1

            if best_match_index != -1 and matches[best_match_index]:
                name = known_names[best_match_index]
                recognized.append(name)

                # Mark attendance
                student_info = rows[best_match_index]
                date_str = datetime.now().strftime("%d/%m/%Y")
                run_query(
                    "INSERT OR IGNORE INTO attendance (student_name, reg_no, class_name, department, date, status) VALUES (?, ?, ?, ?, ?, ?)",
                    (student_info[0], student_info[1], student_info[2], student_info[3], date_str, "Present")
                )
                print(f"✅ Marked attendance for {name}")

        if not recognized:
            print("❌ No match found.")
            return jsonify({"success": True, "recognized": []})
        else:
            return jsonify({"success": True, "recognized": recognized})

    except Exception as e:
        print("❌ Error in recognize:", e)
        return jsonify({"success": False, "error": str(e)})

        # Read uploaded image
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image received"}), 400

        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, boxes)
        print(f"📸 Faces detected in frame: {len(encs)}")

        recognized = []
        for enc in encs:
            matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_encodings, enc)
            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else -1

            name = "Unknown"
            if best_match_index != -1 and matches[best_match_index]:
                name = known_names[best_match_index]
                recognized.append(name)

                # Save attendance
                selected = students[best_match_index]
                date_str = datetime.now().strftime("%d/%m/%Y")
                exists = run_query(
                    "SELECT * FROM attendance WHERE reg_no=? AND date=?",
                    (selected[1], date_str),
                    fetch=True
                )
                if not exists:
                    run_query(
                        "INSERT INTO attendance (student_name, reg_no, class_name, department, date, status) VALUES (?, ?, ?, ?, ?, ?)",
                        (selected[0], selected[1], selected[2], selected[3], date_str, "Present")
                    )
                    print(f"🟢 Attendance marked for {selected[0]} ({selected[1]})")

        print(f"✅ Recognized: {recognized if recognized else 'No match found'}")
        return jsonify({"success": True, "recognized": recognized})

    except Exception as e:
        print("❌ Error in recognize:", e)
        return jsonify({"success": False, "error": str(e)})

# -------------------------------------------------
# ATTENDANCE API
# -------------------------------------------------
@app.route("/api/attendance")
def api_attendance():
    cls = request.args.get("class")
    from_date = request.args.get("from")
    to_date = request.args.get("to")

    query = "SELECT student_name, reg_no, class_name, department, date, status FROM attendance WHERE 1=1"
    params = []

    if cls:
        query += " AND class_name=?"
        params.append(cls)
    if from_date and to_date:
        query += " AND date BETWEEN ? AND ?"
        params.extend([from_date, to_date])

    records = run_query(query, tuple(params), fetch=True)
    formatted = [
        {
            "student_name": r[0],
            "reg_no": r[1],
            "class_name": r[2],
            "department": r[3],
            "date": r[4],
            "status": r[5]
        } for r in records
    ]
    return jsonify({"success": True, "records": formatted})

# -------------------------------------------------
# STATUS ROUTE
# -------------------------------------------------
@app.route("/status")
def status():
    students = run_query("SELECT COUNT(*) FROM students", fetch=True)[0][0]
    faculty = run_query("SELECT COUNT(*) FROM faculty", fetch=True)[0][0]
    attendance = run_query("SELECT COUNT(*) FROM attendance", fetch=True)[0][0]
    return jsonify({
        "status": "running",
        "students_loaded": students,
        "faculty_accounts": faculty,
        "attendance_records": attendance
    })

# -------------------------------------------------
# MAIN ENTRY
# -------------------------------------------------
if __name__ == "__main__":
    init_db()
    print("🚀 SmartVision+ Backend (SQLite) running at http://127.0.0.1:5001")
    app.run(port=5001, debug=True)